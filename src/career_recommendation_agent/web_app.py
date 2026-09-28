from __future__ import annotations

import argparse
import email
import json
import math
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from pydantic import ValidationError

from career_recommendation_agent.agent import recommend_jobs
from career_recommendation_agent.job_sources import (
    JobSourceError,
    LiveJobSearchRequest,
    fetch_live_jobs,
)
from career_recommendation_agent.models import RecommendationInput, UserProfile
from career_recommendation_agent.profile_extraction import UploadedFile, extract_profile_fields
from career_recommendation_agent.scoring import score_job
from career_recommendation_agent.text import domains_are_compatible, infer_domain


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_WEB_JOB_LIMIT = 100
DEFAULT_WEB_MIN_MATCH_SCORE = 50
MIN_WEB_JOB_OPTIONS = 25
WEB_CANDIDATE_POOL_LIMIT = 100
ALL_PROVIDER_ORDER = ("jsearch", "serpapi", "jooble", "remotive")


class CareerRecommendationHandler(BaseHTTPRequestHandler):
    server_version = "CareerRecommendationAgent/0.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(_index_html())
            return
        if path == "/health":
            self._send_json({"status": "ok"})
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/extract-profile":
            self._handle_extract_profile()
            return
        if path != "/api/recommend":
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self._read_json_body()
            result = _recommend_from_form_payload(payload)
        except (json.JSONDecodeError, ValidationError, JobSourceError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:
            self._send_json({"error": f"Unexpected server error: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json(result.model_dump(mode="json"))

    def _handle_extract_profile(self) -> None:
        try:
            form = self._read_multipart_form()
            cv_file = form.get("cv_file")
            portfolio_text = str(form.get("portfolio_text") or "")
            portfolio_url = str(form.get("portfolio_url") or "")
            fields = extract_profile_fields(
                cv_file=cv_file if isinstance(cv_file, UploadedFile) else None,
                portfolio_text=portfolio_text,
                portfolio_url=portfolio_url,
            )
        except (ValueError, ValidationError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:
            self._send_json({"error": f"Unexpected server error: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json({"fields": fields})

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def _read_multipart_form(self) -> dict[str, str | UploadedFile]:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("Expected multipart form data.")

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        message = email.message_from_bytes(
            b"Content-Type: " + content_type.encode("utf-8") + b"\r\n\r\n" + body
        )

        fields: dict[str, str | UploadedFile] = {}
        for part in message.walk():
            if part.is_multipart():
                continue
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue
            filename = part.get_filename()
            content = part.get_payload(decode=True) or b""
            if filename:
                fields[name] = UploadedFile(filename=filename, content=content)
            else:
                fields[name] = content.decode("utf-8", errors="ignore")
        return fields

    def _send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    server = ThreadingHTTPServer((host, port), CareerRecommendationHandler)
    print(f"Career Recommendation Agent running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local career recommendation web app.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    run_server(args.host, args.port)


def _recommend_from_form_payload(payload: dict) -> object:
    skills = _split_lines_or_commas(payload.get("skills", ""))
    desired_role = str(payload.get("desired_role") or "").strip() or None
    experience = _split_lines(payload.get("experience", ""))
    projects = _split_lines(payload.get("projects", ""))
    payload_domain = str(payload.get("domain") or "").strip() or None
    inferred_domain = infer_domain(
        desired_role,
        " ".join(skills),
        " ".join(experience),
        " ".join(projects),
    )
    domain = inferred_domain if inferred_domain != "General Technology" else payload_domain or inferred_domain
    profile = UserProfile.model_validate(
        {
            "skills": skills,
            "experience_years": _optional_float(payload.get("experience_years")),
            "experience": experience,
            "education": _split_lines(payload.get("education", "")),
            "projects": projects,
            "desired_role": desired_role,
            "preferences": {
                "location": str(payload.get("location") or "").strip() or None,
                "seniority": str(payload.get("seniority") or "").strip() or None,
                "domain": domain,
            },
        }
    )
    query = str(payload.get("query") or profile.desired_role or "").strip()
    if not query:
        raise ValueError("Please provide a desired role or search query.")

    provider = str(payload.get("provider") or "remotive").strip().lower()
    limit = int(payload.get("limit") or DEFAULT_WEB_JOB_LIMIT)
    source_filter = str(payload.get("source_filter") or "").strip().lower() or None
    extra_queries = [
        *_split_lines(payload.get("search_queries", "")),
        *_split_lines(payload.get("alternate_roles", "")),
    ]

    jobs = _fetch_web_jobs_with_fallbacks(
        provider=provider,
        profile=profile,
        query=query,
        location=profile.preferences.location,
        limit=max(MIN_WEB_JOB_OPTIONS, min(limit, WEB_CANDIDATE_POOL_LIMIT)),
        source_filter=source_filter,
        extra_queries=extra_queries,
    )
    recommendation_input = RecommendationInput(
        user_profile=profile,
        job_listings=jobs,
        minimum_match_score=DEFAULT_WEB_MIN_MATCH_SCORE,
        fallback_recommendation_count=0,
    )
    return recommend_jobs(recommendation_input)


def _fetch_web_jobs_with_fallbacks(
    provider: str,
    profile: UserProfile,
    query: str,
    location: str | None,
    limit: int,
    source_filter: str | None,
    extra_queries: list[str] | None = None,
) -> list:
    active_providers = _active_providers(provider, source_filter)
    provider_jobs = {active_provider: [] for active_provider in active_providers}
    provider_seen = {active_provider: set() for active_provider in active_providers}
    provider_errors: list[str] = []
    per_provider_target = max(1, math.ceil(limit / max(1, len(active_providers))))

    for search_query in _candidate_job_queries(profile, query, extra_queries or []):
        if all(
            len(provider_jobs[active_provider]) >= per_provider_target
            for active_provider in active_providers
        ):
            break
        for active_provider in active_providers:
            remaining = per_provider_target - len(provider_jobs[active_provider])
            if remaining <= 0:
                continue
            try:
                fetched_jobs = fetch_live_jobs(
                    active_provider,
                    LiveJobSearchRequest(
                        query=search_query,
                        location=location,
                        limit=_provider_fetch_limit(active_provider, remaining),
                        source_filter=source_filter,
                    ),
                )
            except JobSourceError as exc:
                provider_errors.append(f"{active_provider}: {exc}")
                continue
            for job in fetched_jobs:
                if not _is_relevant_live_job(profile, job):
                    continue
                key = _live_job_key(job)
                if key in provider_seen[active_provider]:
                    continue
                provider_seen[active_provider].add(key)
                provider_jobs[active_provider].append(job)
                if len(provider_jobs[active_provider]) >= per_provider_target:
                    break

    if "remotive" in active_providers and len(provider_jobs["remotive"]) < per_provider_target:
        remaining = per_provider_target - len(provider_jobs["remotive"])
        try:
            fetched_jobs = fetch_live_jobs(
                "remotive",
                LiveJobSearchRequest(
                    query="",
                    location=location,
                    limit=_provider_fetch_limit("remotive", remaining),
                    source_filter=source_filter,
                ),
            )
        except JobSourceError as exc:
            provider_errors.append(str(exc))
            fetched_jobs = []
        for job in fetched_jobs:
            if not _is_relevant_live_job(profile, job):
                continue
            key = _live_job_key(job)
            if key in provider_seen["remotive"]:
                continue
            provider_seen["remotive"].add(key)
            provider_jobs["remotive"].append(job)
            if len(provider_jobs["remotive"]) >= per_provider_target:
                break

    jobs = _round_robin_provider_jobs(provider_jobs, active_providers, limit)
    if not jobs and provider_errors:
        raise JobSourceError(provider_errors[-1])

    return jobs


def _active_providers(provider: str, source_filter: str | None) -> tuple[str, ...]:
    if provider != "all":
        return (provider,)
    if source_filter:
        return tuple(item for item in ALL_PROVIDER_ORDER if item in {"jsearch", "serpapi"})
    return ALL_PROVIDER_ORDER


def _provider_fetch_limit(provider: str, remaining: int) -> int:
    if provider == "remotive":
        return max(remaining, WEB_CANDIDATE_POOL_LIMIT)
    return max(remaining, MIN_WEB_JOB_OPTIONS)


def _live_job_key(job) -> tuple[str, str, str]:
    return (
        _dedupe_text(job.job_title),
        _dedupe_text(job.company),
        _dedupe_text(job.location or ""),
    )


def _is_relevant_live_job(profile: UserProfile, job) -> bool:
    score = score_job(profile, job)
    profile_domain = profile.preferences.domain or infer_domain(
        profile.desired_role,
        " ".join(profile.skills),
        " ".join(profile.experience),
        " ".join(profile.projects),
    )
    job_domain = infer_domain(
        job.job_title,
        job.job_description,
        " ".join(job.required_skills),
    )
    same_or_related_domain = domains_are_compatible(profile_domain, job_domain)
    has_unknown_domain = "General Technology" in {profile_domain, job_domain}
    if same_or_related_domain:
        has_meaningful_relevance = (
            score.role_relevance >= 0.35
            or score.skills_match >= 0.35
            or score.role_relevance >= 0.2
        )
    elif has_unknown_domain:
        has_meaningful_relevance = (
            score.role_relevance >= 0.5
            or score.skills_match >= 0.5
        )
    else:
        has_meaningful_relevance = False
    return (
        has_meaningful_relevance
        and (same_or_related_domain or has_unknown_domain)
        and score.final_score >= DEFAULT_WEB_MIN_MATCH_SCORE
    )


def _round_robin_provider_jobs(
    provider_jobs: dict[str, list],
    active_providers: tuple[str, ...],
    limit: int,
) -> list:
    merged = []
    seen: set[tuple[str, str, str]] = set()
    index = 0
    while len(merged) < limit:
        added = False
        for provider in active_providers:
            bucket = provider_jobs.get(provider, [])
            if index >= len(bucket):
                continue
            job = bucket[index]
            key = _live_job_key(job)
            if key not in seen:
                seen.add(key)
                merged.append(job)
                added = True
                if len(merged) >= limit:
                    break
        if not added and all(index >= len(provider_jobs.get(provider, [])) for provider in active_providers):
            break
        index += 1
    return merged


def _dedupe_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _candidate_job_queries(
    profile: UserProfile,
    primary_query: str,
    extra_queries: list[str] | None = None,
) -> list[str]:
    candidates = [
        primary_query,
        profile.desired_role or "",
    ]
    candidates.extend(extra_queries or [])
    candidates.extend(_role_related_queries(profile.desired_role or primary_query))
    candidates.extend(_domain_related_queries(profile))
    candidates.extend(_skill_based_queries(profile.skills))

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = candidate.strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            unique.append(cleaned)
    return unique


def _role_related_queries(role: str) -> list[str]:
    lowered = role.lower()
    role_query_groups = [
        (("video", "editor", "3d", "modeler", "modeller", "animator", "artist"), [
            "Video Editor",
            "3D Modeler",
            "Motion Graphics Designer",
            "Blender Artist",
            "Video Production Specialist",
        ]),
        (("machine learning", "ai", "data scientist"), [
            "Machine Learning",
            "AI Engineer",
            "Data Scientist",
            "Python Developer",
            "Data Engineer",
        ]),
        (("backend", "python developer", "software engineer"), [
            "Backend Developer",
            "Software Engineer",
            "Python Developer",
            "API Developer",
        ]),
        (("frontend", "react", "ui ux", "designer"), [
            "Frontend Developer",
            "React Developer",
            "UI UX Designer",
            "Product Designer",
        ]),
        (("marketing", "seo", "content"), [
            "Digital Marketing",
            "Marketing Specialist",
            "SEO Specialist",
            "Content Marketing",
        ]),
        (("sales", "business development", "account executive"), [
            "Sales Executive",
            "Business Development",
            "Account Executive",
            "Sales Representative",
        ]),
        (("accountant", "finance", "bookkeeper"), [
            "Accountant",
            "Finance Analyst",
            "Bookkeeper",
            "Accounts Assistant",
        ]),
        (("hr", "recruiter", "talent"), [
            "HR Officer",
            "Recruiter",
            "Talent Acquisition",
            "HR Assistant",
        ]),
        (("teacher", "tutor", "education", "trainer"), [
            "Teacher",
            "Tutor",
            "Education Coordinator",
            "Trainer",
        ]),
        (("nurse", "medical", "doctor", "pharmacist", "healthcare"), [
            "Nurse",
            "Medical Assistant",
            "Healthcare Assistant",
            "Pharmacist",
        ]),
        (("civil", "mechanical", "electrical", "engineer", "architect"), [
            "Civil Engineer",
            "Mechanical Engineer",
            "Electrical Engineer",
            "Site Engineer",
            "Architect",
        ]),
        (("customer support", "customer service", "admin", "operations"), [
            "Customer Support",
            "Customer Service Representative",
            "Administrative Assistant",
            "Operations Assistant",
        ]),
        (("logistics", "supply chain", "warehouse", "procurement"), [
            "Logistics Coordinator",
            "Supply Chain Specialist",
            "Warehouse Supervisor",
            "Procurement Officer",
        ]),
        (("legal", "lawyer", "paralegal"), [
            "Legal Assistant",
            "Paralegal",
            "Legal Officer",
            "Compliance Officer",
        ]),
        (("hotel", "restaurant", "chef", "hospitality"), [
            "Hotel Manager",
            "Restaurant Manager",
            "Chef",
            "Hospitality Supervisor",
        ]),
    ]
    for keywords, queries in role_query_groups:
        if any(keyword in lowered for keyword in keywords):
            return queries
    return []


def _domain_related_queries(profile: UserProfile) -> list[str]:
    domain = profile.preferences.domain or infer_domain(
        profile.desired_role,
        " ".join(profile.skills),
        " ".join(profile.experience),
        " ".join(profile.projects),
    )
    domain_queries = {
        "Creative Media / Video Production": [
            "Video Editor",
            "Motion Graphics Designer",
            "3D Artist",
        ],
        "Healthcare": ["Healthcare Assistant", "Nurse", "Medical Assistant"],
        "Education / Training": ["Teacher", "Tutor", "Trainer"],
        "Finance / Accounting": ["Accountant", "Finance Assistant", "Bookkeeper"],
        "Marketing / Growth": ["Digital Marketing Specialist", "SEO Specialist"],
        "Sales / Business Development": ["Sales Executive", "Business Development Executive"],
        "HR / Recruitment": ["HR Officer", "Recruiter"],
        "Engineering / Construction": ["Site Engineer", "Civil Engineer", "Project Engineer"],
        "Customer Support / Operations": ["Customer Support Representative", "Operations Assistant"],
        "Logistics / Supply Chain": ["Logistics Coordinator", "Supply Chain Specialist"],
        "Legal": ["Legal Assistant", "Paralegal"],
        "Hospitality": ["Hotel Manager", "Restaurant Manager", "Chef"],
    }
    return domain_queries.get(domain, [])


def _skill_based_queries(skills: list[str]) -> list[str]:
    important_skills = [
        skill
        for skill in skills
        if skill.lower()
        in {
            "python",
            "sql",
            "javascript",
            "react",
            "django",
            "flask",
            "machine learning",
            "tensorflow",
            "pytorch",
            "docker",
            "seo",
            "sales",
            "accounting",
            "figma",
            "adobe premiere pro",
            "adobe after effects",
            "blender",
            "unreal engine",
            "video editing",
            "3d modeling",
            "nursing",
            "patient care",
            "pharmacy",
            "teaching",
            "recruitment",
            "customer service",
            "logistics",
            "supply chain",
            "legal research",
            "civil engineering",
            "mechanical engineering",
            "electrical engineering",
        }
    ]
    return [f"{skill} jobs" for skill in important_skills[:5]]


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _split_lines(value: object) -> list[str]:
    text = str(value or "")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _split_lines_or_commas(value: object) -> list[str]:
    text = str(value or "")
    items: list[str] = []
    for line in text.splitlines():
        items.extend(part.strip() for part in line.split(","))
    return [item for item in items if item]


def _index_html() -> str:
    return Path(__file__).with_name("web_index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
