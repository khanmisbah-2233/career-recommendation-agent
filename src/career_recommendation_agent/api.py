from __future__ import annotations

import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError

from career_recommendation_agent.database import (
    authenticate_user,
    admin_stats,
    consume_password_reset_token,
    create_password_reset_token,
    create_user,
    delete_saved_job,
    get_user_by_id,
    init_database,
    list_recommendation_runs,
    list_saved_jobs,
    save_recommendation_run,
    save_job,
    update_user_profile,
)
from career_recommendation_agent.job_sources import JobSourceError
from career_recommendation_agent.profile_extraction import (
    UploadedFile,
    extract_keyword_profile_fields,
    extract_profile_fields,
)
from career_recommendation_agent.security import create_access_token, verify_access_token
from career_recommendation_agent.web_app import _index_html, _recommend_from_form_payload


class HealthResponse(BaseModel):
    status: str


class ExtractProfileResponse(BaseModel):
    fields: dict[str, str]


class SignupRequest(BaseModel):
    email: str
    password: str
    full_name: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class ProfileUpdateRequest(BaseModel):
    full_name: str | None = None
    headline: str | None = None
    location: str | None = None
    skills: str | None = None


class PasswordResetRequest(BaseModel):
    email: str


class PasswordResetConfirmRequest(BaseModel):
    token: str
    new_password: str


MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
ALLOWED_UPLOAD_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".docx",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "80"))
_request_log: dict[str, deque[float]] = defaultdict(deque)
_started_at = time.time()
_api_metrics: dict[str, int] = defaultdict(int)

app = FastAPI(
    title="Career Recommendation Agent",
    version="0.1.0",
    description="Production API for CV, portfolio, keyword-based career recommendations.",
)

init_database()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def production_safety_middleware(request: Request, call_next):
    started = time.time()
    _api_metrics["requests_total"] += 1
    _api_metrics[f"path:{request.url.path}"] += 1
    if request.url.path.startswith(("/api/recommend", "/api/extract-profile", "/api/auth")):
        _enforce_rate_limit(request)
    response = await call_next(request)
    _api_metrics[f"status:{response.status_code}"] += 1
    response.headers["X-Response-Time-ms"] = str(round((time.time() - started) * 1000, 2))
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    return _index_html()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/api/auth/signup", response_model=AuthResponse)
def signup(payload: SignupRequest) -> AuthResponse:
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    try:
        user = create_user(payload.email, payload.password, payload.full_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AuthResponse(access_token=create_access_token(int(user["id"])), user=_public_user(user))


@app.post("/api/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest) -> AuthResponse:
    user = authenticate_user(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return AuthResponse(access_token=create_access_token(int(user["id"])), user=_public_user(user))


@app.get("/api/auth/me")
def me(authorization: str | None = Header(default=None)) -> dict:
    user = _current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return {"user": _public_user(user)}


@app.put("/api/auth/me")
def update_me(payload: ProfileUpdateRequest, authorization: str | None = Header(default=None)) -> dict:
    user = _require_user(authorization)
    updated = update_user_profile(
        int(user["id"]),
        full_name=_clean_optional(payload.full_name),
        headline=_clean_optional(payload.headline),
        location=_clean_optional(payload.location),
        skills=_clean_optional(payload.skills),
    )
    return {"user": _public_user(updated)}


@app.post("/api/auth/password-reset/request")
def request_password_reset(payload: PasswordResetRequest) -> dict:
    token = secrets.token_urlsafe(32)
    reset = create_password_reset_token(payload.email, token)
    response = {"message": "If the email exists, a password reset token has been created."}
    if reset and os.getenv("RETURN_PASSWORD_RESET_TOKEN", "true").lower() == "true":
        response["reset_token"] = token
    return response


@app.post("/api/auth/password-reset/confirm")
def confirm_password_reset(payload: PasswordResetConfirmRequest) -> dict:
    if len(payload.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    if not consume_password_reset_token(payload.token, payload.new_password):
        raise HTTPException(status_code=400, detail="Invalid or expired reset token.")
    return {"message": "Password updated successfully."}


@app.get("/api/admin/stats")
def get_admin_stats(authorization: str | None = Header(default=None)) -> dict:
    user = _require_user(authorization)
    if not _is_admin(user):
        raise HTTPException(status_code=403, detail="Admin access required.")
    return {"stats": admin_stats()}


@app.get("/api/admin/system")
def get_admin_system(authorization: str | None = Header(default=None)) -> dict:
    user = _require_user(authorization)
    if not _is_admin(user):
        raise HTTPException(status_code=403, detail="Admin access required.")
    return {
        "system": {
            "uptime_seconds": int(time.time() - _started_at),
            "database": _database_backend(),
            "max_upload_mb": round(MAX_UPLOAD_BYTES / (1024 * 1024), 1),
            "rate_limit_window_seconds": RATE_LIMIT_WINDOW_SECONDS,
            "rate_limit_requests": RATE_LIMIT_REQUESTS,
            "providers": _provider_configuration_status(),
            "metrics": dict(_api_metrics),
        }
    }


@app.get("/api/recommendations/history")
def recommendation_history(authorization: str | None = Header(default=None)) -> dict:
    user = _current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return {"history": list_recommendation_runs(int(user["id"]))}


@app.get("/api/saved-jobs")
def saved_jobs(authorization: str | None = Header(default=None)) -> dict:
    user = _require_user(authorization)
    return {"saved_jobs": list_saved_jobs(int(user["id"]))}


@app.post("/api/saved-jobs")
def create_saved_job(payload: dict, authorization: str | None = Header(default=None)) -> dict:
    user = _require_user(authorization)
    job = _normalize_saved_job(payload)
    saved_job_id = save_job(int(user["id"]), job)
    return {"saved_job_id": saved_job_id, "job": job}


@app.delete("/api/saved-jobs/{saved_job_id}")
def remove_saved_job(saved_job_id: int, authorization: str | None = Header(default=None)) -> dict:
    user = _require_user(authorization)
    deleted = delete_saved_job(int(user["id"]), saved_job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Saved job not found.")
    return {"deleted": True}


@app.post("/api/recommend")
def recommend(payload: dict, authorization: str | None = Header(default=None)) -> dict:
    try:
        result = _recommend_from_form_payload(payload)
    except (ValidationError, JobSourceError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}") from exc

    output = result.model_dump(mode="json")
    user = _current_user(authorization)
    if user:
        save_recommendation_run(
            user_id=int(user["id"]),
            target_role=str(payload.get("desired_role") or ""),
            provider=str(payload.get("provider") or "remotive"),
            input_payload=payload,
            output_payload=output,
        )
    return output


@app.post("/api/extract-profile", response_model=ExtractProfileResponse)
async def extract_profile(
    cv_file: Annotated[UploadFile | None, File()] = None,
    portfolio_url: Annotated[str, Form()] = "",
    portfolio_text: Annotated[str, Form()] = "",
    input_type: Annotated[str, Form()] = "",
) -> ExtractProfileResponse:
    uploaded_file = None
    if cv_file and cv_file.filename:
        content = await cv_file.read()
        suffix = Path(cv_file.filename).suffix.lower()
        if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported CV file type.")
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="CV file is too large.")
        _validate_upload_content(suffix, content)
        uploaded_file = UploadedFile(filename=Path(cv_file.filename).name, content=content)

    try:
        if input_type.strip().lower() == "keywords":
            fields = extract_keyword_profile_fields(portfolio_text)
        else:
            fields = extract_profile_fields(
                cv_file=uploaded_file,
                portfolio_text=portfolio_text,
                portfolio_url=portfolio_url,
            )
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}") from exc
    return ExtractProfileResponse(fields=fields)


def _current_user(authorization: str | None) -> dict | None:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    user_id = verify_access_token(token)
    if not user_id:
        return None
    return get_user_by_id(user_id)


def _require_user(authorization: str | None) -> dict:
    user = _current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


def _public_user(user: dict) -> dict:
    return {
        "id": user["id"],
        "email": user["email"],
        "full_name": user.get("full_name"),
        "headline": user.get("headline"),
        "location": user.get("location"),
        "skills": user.get("skills"),
        "is_admin": _is_admin(user),
        "created_at": user.get("created_at"),
    }


def _is_admin(user: dict) -> bool:
    admin_emails = {
        email.strip().lower()
        for email in os.getenv("ADMIN_EMAILS", "").split(",")
        if email.strip()
    }
    return str(user.get("email", "")).lower() in admin_emails


def _clean_optional(value: str | None) -> str:
    return str(value or "").strip()


def _normalize_saved_job(payload: dict) -> dict:
    job = {
        "job_title": str(payload.get("job_title") or payload.get("title") or "").strip(),
        "company": str(payload.get("company") or payload.get("company_name") or "").strip(),
        "source": str(payload.get("source") or payload.get("source_platform") or "").strip(),
        "match_score": _safe_int(payload.get("match_score")),
        "reason": str(payload.get("reason") or "").strip(),
        "apply_link": str(payload.get("apply_link") or payload.get("apply_url") or "").strip(),
    }
    if not job["job_title"] or not job["company"]:
        raise HTTPException(status_code=400, detail="Job title and company are required to save a job.")
    return job


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _enforce_rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    key = f"{client}:{request.url.path}"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    requests = _request_log[key]
    while requests and requests[0] < window_start:
        requests.popleft()
    if len(requests) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="Too many requests. Please try again shortly.")
    requests.append(now)


def _database_backend() -> str:
    database_url = os.getenv("DATABASE_URL", "").strip().lower()
    return "postgresql" if database_url.startswith(("postgres://", "postgresql://")) else "sqlite"


def _provider_configuration_status() -> dict[str, bool]:
    return {
        "remotive": True,
        "serpapi": bool(os.getenv("SERPAPI_API_KEY", "").strip()),
        "jsearch": bool(os.getenv("JSEARCH_API_KEY", "").strip()),
        "jooble": bool(os.getenv("JOOBLE_API_KEY", "").strip()),
        "llama": bool(os.getenv("GROQ_API_KEY", "").strip()),
    }


def _validate_upload_content(suffix: str, content: bytes) -> None:
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded CV file is empty.")
    if content[:2] == b"MZ":
        raise HTTPException(status_code=400, detail="Executable files are not allowed.")
    signatures = {
        ".pdf": (b"%PDF",),
        ".docx": (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"),
        ".png": (b"\x89PNG\r\n\x1a\n",),
        ".jpg": (b"\xff\xd8\xff",),
        ".jpeg": (b"\xff\xd8\xff",),
        ".webp": (b"RIFF",),
        ".bmp": (b"BM",),
        ".tif": (b"II*\x00", b"MM\x00*"),
        ".tiff": (b"II*\x00", b"MM\x00*"),
    }
    expected = signatures.get(suffix)
    if expected and not any(content.startswith(signature) for signature in expected):
        raise HTTPException(status_code=400, detail="Uploaded CV content does not match the file type.")
