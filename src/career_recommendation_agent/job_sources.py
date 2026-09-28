from __future__ import annotations

import json
import os
import re
import socket
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from career_recommendation_agent.models import JobListing


load_dotenv()

SERPAPI_API_KEY_ENV_VAR = "SERPAPI_API_KEY"
SERPAPI_PLACEHOLDER = "your_serpapi_api_key_here"
JSEARCH_API_KEY_ENV_VAR = "JSEARCH_API_KEY"
RAPIDAPI_KEY_ENV_VAR = "RAPIDAPI_KEY"
JSEARCH_PLACEHOLDER = "your_jsearch_api_key_here"
JSEARCH_RAPIDAPI_HOST = "jsearch.p.rapidapi.com"
JSEARCH_NUM_PAGES_ENV_VAR = "JSEARCH_NUM_PAGES"
JOOBLE_API_KEY_ENV_VAR = "JOOBLE_API_KEY"
JOOBLE_PLACEHOLDER = "your_jooble_api_key_here"
JOOBLE_ENDPOINT_TEMPLATE = "https://jooble.org/api/{api_key}"
TECH_SKILL_KEYWORDS = [
    "Python",
    "FastAPI",
    "Django",
    "Flask",
    "Machine Learning",
    "Deep Learning",
    "SQL",
    "PostgreSQL",
    "MySQL",
    "Docker",
    "Kubernetes",
    "AWS",
    "Azure",
    "GCP",
    "JavaScript",
    "TypeScript",
    "React",
    "Node.js",
    "Java",
    "Go",
    "Rust",
]


class JobSourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class LiveJobSearchRequest:
    query: str
    location: str | None = None
    limit: int = 10
    source_filter: str | None = None


def fetch_live_jobs(provider: str, request: LiveJobSearchRequest) -> list[JobListing]:
    if provider == "serpapi":
        return fetch_serpapi_google_jobs(request)
    if provider == "jsearch":
        return fetch_jsearch_jobs(request)
    if provider == "jooble":
        return fetch_jooble_jobs(request)
    if provider == "remotive":
        return fetch_remotive_jobs(request)
    raise JobSourceError(f"Unsupported live job provider: {provider}")


def fetch_serpapi_google_jobs(request: LiveJobSearchRequest) -> list[JobListing]:
    api_key = _env_secret(SERPAPI_API_KEY_ENV_VAR)
    if not api_key or api_key == SERPAPI_PLACEHOLDER:
        raise JobSourceError(
            "SERPAPI_API_KEY is missing. Add it to .env to enable live job search."
        )

    params = {
        "engine": "google_jobs",
        "q": request.query,
        "api_key": api_key,
    }
    if request.location:
        params["location"] = request.location

    url = "https://serpapi.com/search.json?" + urlencode(params)
    try:
        with urlopen(_request(url), timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise JobSourceError(f"Live job search HTTP error: {exc.code}") from exc
    except URLError as exc:
        raise JobSourceError(f"Live job search network error: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise JobSourceError("Live job search returned invalid JSON.") from exc

    if "error" in payload:
        raise JobSourceError(f"Live job search provider error: {payload['error']}")

    return parse_serpapi_jobs(payload, request.limit, request.source_filter)


def fetch_remotive_jobs(request: LiveJobSearchRequest) -> list[JobListing]:
    if request.source_filter:
        raise JobSourceError(
            "Remotive does not provide LinkedIn/Indeed source filtering. "
            "Use --job-provider serpapi or --job-provider jsearch for --source-filter."
        )

    params = {"limit": str(max(1, request.limit))}
    if request.query.strip():
        params["search"] = request.query
    url = "https://remotive.com/api/remote-jobs?" + urlencode(params)
    try:
        with urlopen(_request(url), timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise JobSourceError(f"Remotive job search HTTP error: {exc.code}") from exc
    except URLError as exc:
        raise JobSourceError(f"Remotive job search network error: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise JobSourceError("Remotive job search returned invalid JSON.") from exc

    return parse_remotive_jobs(payload, request.limit)


def fetch_jsearch_jobs(request: LiveJobSearchRequest) -> list[JobListing]:
    api_key = _env_secret(JSEARCH_API_KEY_ENV_VAR) or _env_secret(RAPIDAPI_KEY_ENV_VAR)
    if not api_key or api_key == JSEARCH_PLACEHOLDER:
        raise JobSourceError(
            "JSEARCH_API_KEY is missing. Add it to .env to enable JSearch live job search."
        )

    params = {
        "query": _jsearch_query(request),
        "num_pages": str(_jsearch_num_pages()),
        "date_posted": "all",
    }
    url = f"https://{JSEARCH_RAPIDAPI_HOST}/search-v2?" + urlencode(params)
    try:
        with urlopen(_request(url, rapidapi_key=api_key), timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 429:
            raise JobSourceError(
                "JSearch rate limit hit. Please wait a few minutes, reduce JSEARCH_NUM_PAGES, "
                "or use another provider while the quota resets."
            ) from exc
        raise JobSourceError(f"JSearch job search HTTP error: {exc.code}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise JobSourceError(
            "JSearch job search timed out. Try again, use Source Filter Any, "
            "or reduce JSEARCH_NUM_PAGES."
        ) from exc
    except URLError as exc:
        reason = str(exc.reason)
        if "timed out" in reason.lower():
            raise JobSourceError(
                "JSearch job search timed out. Try again, use Source Filter Any, "
                "or reduce JSEARCH_NUM_PAGES."
            ) from exc
        raise JobSourceError(f"JSearch job search network error: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise JobSourceError("JSearch job search returned invalid JSON.") from exc

    if payload.get("status") == "ERROR":
        raise JobSourceError(f"JSearch provider error: {payload.get('message', 'unknown error')}")

    return parse_jsearch_jobs(payload, request.limit, request.source_filter)


def fetch_jooble_jobs(request: LiveJobSearchRequest) -> list[JobListing]:
    api_key = _env_secret(JOOBLE_API_KEY_ENV_VAR)
    if not api_key or api_key == JOOBLE_PLACEHOLDER:
        raise JobSourceError(
            "JOOBLE_API_KEY is missing. Add it to .env to enable Jooble live job search."
        )
    if request.source_filter:
        raise JobSourceError(
            "Jooble does not provide LinkedIn/Indeed source filtering. Use Source Filter Any."
        )

    body = {
        "keywords": request.query,
        "location": request.location or "",
        "page": 1,
    }
    try:
        with urlopen(_request(JOOBLE_ENDPOINT_TEMPLATE.format(api_key=api_key), data=body), timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 429:
            raise JobSourceError("Jooble rate limit hit. Please wait a few minutes and try again.") from exc
        raise JobSourceError(f"Jooble job search HTTP error: {exc.code}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise JobSourceError("Jooble job search timed out. Please try again.") from exc
    except URLError as exc:
        raise JobSourceError(f"Jooble job search network error: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise JobSourceError("Jooble job search returned invalid JSON.") from exc

    return parse_jooble_jobs(payload, request.limit)


def parse_serpapi_jobs(
    payload: dict[str, Any],
    limit: int,
    source_filter: str | None = None,
) -> list[JobListing]:
    jobs: list[JobListing] = []
    normalized_filter = source_filter.lower().strip() if source_filter else None

    for item in payload.get("jobs_results", []):
        apply_source, apply_link = _best_apply_option(item, normalized_filter)
        source = apply_source or str(item.get("via") or "Google Jobs")

        if normalized_filter and normalized_filter not in source.lower():
            continue

        title = str(item.get("title") or "").strip()
        company = str(item.get("company_name") or "").strip()
        if not title or not company:
            continue

        jobs.append(
            JobListing(
                job_title=title,
                company=company,
                job_description=str(item.get("description") or ""),
                required_skills=[],
                experience_level=_infer_experience_from_text(str(item.get("description") or "")),
                location=str(item.get("location") or ""),
                apply_url=apply_link,
                source_platform=source,
            )
        )
        if len(jobs) >= limit:
            break

    return jobs


def parse_remotive_jobs(payload: dict[str, Any], limit: int) -> list[JobListing]:
    jobs: list[JobListing] = []

    for item in payload.get("jobs", []):
        title = str(item.get("title") or "").strip()
        company = str(item.get("company_name") or "").strip()
        if not title or not company:
            continue

        jobs.append(
            JobListing(
                job_title=title,
                company=company,
                job_description=_clean_html(str(item.get("description") or "")),
                required_skills=_extract_skills(str(item.get("description") or "")),
                experience_level=_infer_experience_from_text(str(item.get("description") or "")),
                location=str(item.get("candidate_required_location") or "Remote"),
                apply_url=str(item.get("url") or ""),
                source_platform="Remotive",
            )
        )
        if len(jobs) >= limit:
            break

    return jobs


def parse_jsearch_jobs(
    payload: dict[str, Any],
    limit: int,
    source_filter: str | None = None,
) -> list[JobListing]:
    jobs: list[JobListing] = []
    normalized_filter = source_filter.lower().strip() if source_filter else None

    for item in _jsearch_job_items(payload):
        if item.get("job_is_expired") is True:
            continue
        title = str(item.get("job_title") or "").strip()
        company = str(item.get("employer_name") or "").strip()
        if not title or not company:
            continue

        description = str(item.get("job_description") or "")
        location = _jsearch_location(item)
        apply_url = str(item.get("job_apply_link") or item.get("job_google_link") or "")
        source = _jsearch_source(item, apply_url)
        if normalized_filter and normalized_filter not in source.lower():
            continue
        required_skills = item.get("job_required_skills") or []
        if not isinstance(required_skills, list):
            required_skills = []

        jobs.append(
            JobListing(
                job_title=title,
                company=company,
                job_description=description,
                required_skills=[str(skill) for skill in required_skills] or _extract_skills(description),
                experience_level=_infer_jsearch_experience(item, description),
                location=location,
                apply_url=apply_url,
                source_platform=source,
            )
        )
        if len(jobs) >= limit:
            break

    return jobs


def parse_jooble_jobs(payload: dict[str, Any], limit: int) -> list[JobListing]:
    jobs: list[JobListing] = []
    items = payload.get("jobs", [])
    if not isinstance(items, list):
        return jobs

    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        company = str(item.get("company") or "").strip()
        if not title or not company:
            continue
        description = _clean_html(str(item.get("snippet") or item.get("description") or ""))
        jobs.append(
            JobListing(
                job_title=title,
                company=company,
                job_description=description,
                required_skills=_extract_skills(description),
                experience_level=_infer_experience_from_text(description),
                location=str(item.get("location") or ""),
                apply_url=str(item.get("link") or ""),
                source_platform="Jooble",
            )
        )
        if len(jobs) >= limit:
            break

    return jobs


def _jsearch_job_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data", [])
    if isinstance(data, dict):
        data = data.get("jobs", [])
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _best_apply_option(
    item: dict[str, Any],
    source_filter: str | None,
) -> tuple[str | None, str | None]:
    apply_options = item.get("apply_options") or []
    if not isinstance(apply_options, list):
        return None, None

    selected = None
    if source_filter:
        for option in apply_options:
            title = str(option.get("title") or "")
            if source_filter in title.lower():
                selected = option
                break

    if selected is None and apply_options:
        selected = apply_options[0]

    if not selected:
        return None, None
    return str(selected.get("title") or ""), str(selected.get("link") or "")


def _jsearch_location(item: dict[str, Any]) -> str:
    parts = [
        item.get("job_city"),
        item.get("job_state"),
        item.get("job_country"),
    ]
    location = ", ".join(str(part).strip() for part in parts if str(part or "").strip())
    return location or str(item.get("job_location") or "")


def _jsearch_source(item: dict[str, Any], apply_url: str) -> str:
    publisher = str(item.get("job_publisher") or "").strip()
    lowered_url = apply_url.lower()
    if "linkedin." in lowered_url or "linkedin.com" in lowered_url:
        return "LinkedIn"
    if "indeed." in lowered_url or "indeed.com" in lowered_url:
        return "Indeed"
    if publisher:
        return publisher
    hostname = urlparse(apply_url).hostname or ""
    return hostname.replace("www.", "") or "JSearch"


def _jsearch_query(request: LiveJobSearchRequest) -> str:
    query_parts = [request.query]
    if request.location:
        query_parts.append(f"in {request.location}")
    return " ".join(part.strip() for part in query_parts if part and part.strip())


def _jsearch_num_pages() -> int:
    raw_value = os.getenv(JSEARCH_NUM_PAGES_ENV_VAR, "2").strip()
    try:
        value = int(raw_value)
    except ValueError:
        value = 2
    return max(1, min(value, 5))


def _infer_jsearch_experience(item: dict[str, Any], description: str) -> str | None:
    experience = item.get("job_required_experience") or {}
    if isinstance(experience, dict):
        months = experience.get("required_experience_in_months")
        if isinstance(months, (int, float)):
            if months < 36:
                return "Junior"
            if months < 72:
                return "Mid"
            return "Senior"
        if experience.get("no_experience_required") is True:
            return "Junior"
    return _infer_experience_from_text(description)


def _infer_experience_from_text(description: str) -> str | None:
    lowered = description.lower()
    if any(term in lowered for term in ("senior", "sr.", "lead", "principal")):
        return "Senior"
    if any(term in lowered for term in ("junior", "entry level", "graduate", "intern")):
        return "Junior"
    if any(term in lowered for term in ("mid-level", "mid level", "intermediate")):
        return "Mid"
    return None


def _clean_html(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", without_tags).strip()


def _extract_skills(value: str) -> list[str]:
    text = _clean_html(value).lower()
    skills: list[str] = []
    for skill in TECH_SKILL_KEYWORDS:
        pattern = r"(?<![a-z0-9.+#-])" + re.escape(skill.lower()) + r"(?![a-z0-9.+#-])"
        if re.search(pattern, text):
            skills.append(skill)
    return skills


def _env_secret(name: str) -> str:
    return os.getenv(name, "").strip().strip('"').strip("'")


def _request(url: str, rapidapi_key: str | None = None, data: dict[str, Any] | None = None) -> Request:
    headers = {
        "Accept": "application/json",
        "User-Agent": "career-recommendation-agent/0.1",
    }
    encoded_data = None
    if data is not None:
        headers["Content-Type"] = "application/json"
        encoded_data = json.dumps(data).encode("utf-8")
    if rapidapi_key:
        headers["X-RapidAPI-Key"] = rapidapi_key
        headers["X-RapidAPI-Host"] = JSEARCH_RAPIDAPI_HOST
    return Request(
        url,
        data=encoded_data,
        headers=headers,
    )
