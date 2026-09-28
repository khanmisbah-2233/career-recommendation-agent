from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

from career_recommendation_agent.models import RecommendedJob, RecommendationInput, UserProfileSummary


load_dotenv()

DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
MODEL_ENV_VAR = "CAREER_AGENT_LLM_MODEL"
API_KEY_ENV_VAR = "GROQ_API_KEY"
USE_LLM_ENV_VAR = "CAREER_AGENT_USE_LLM"


def llm_is_configured() -> bool:
    enabled = os.getenv(USE_LLM_ENV_VAR, "true").strip().lower()
    api_key = os.getenv(API_KEY_ENV_VAR, "").strip()
    return (
        enabled not in {"0", "false", "no", "off"}
        and bool(api_key)
        and api_key != "your_groq_api_key_here"
    )


def enhance_recommendation_reasons(
    payload: RecommendationInput,
    summary: UserProfileSummary,
    recommendations: list[RecommendedJob],
) -> list[RecommendedJob]:
    if not recommendations or not llm_is_configured():
        return recommendations

    try:
        response = _call_llama_recruiter_review(payload, summary, recommendations)
        reasons = _parse_reason_map(response)
    except Exception:
        return recommendations

    enhanced: list[RecommendedJob] = []
    for recommendation in recommendations:
        key = _job_key(recommendation.job_title, recommendation.company)
        reason = reasons.get(key)
        if reason:
            enhanced.append(recommendation.model_copy(update={"reason": reason[:500]}))
        else:
            enhanced.append(recommendation)
    return enhanced


def _call_llama_recruiter_review(
    payload: RecommendationInput,
    summary: UserProfileSummary,
    recommendations: list[RecommendedJob],
) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model=os.getenv(MODEL_ENV_VAR, DEFAULT_LLM_MODEL),
        temperature=0,
        max_tokens=1200,
    )
    messages = [
        SystemMessage(
            content=(
                "You are a professional recruitment analyst. Use only the provided "
                "profile and job listing data. Do not search, invent jobs, invent URLs, "
                "or change match scores. Return only valid JSON."
            )
        ),
        HumanMessage(content=json.dumps(_build_llm_payload(payload, summary, recommendations))),
    ]
    result = llm.invoke(messages)
    return str(result.content)


def _build_llm_payload(
    payload: RecommendationInput,
    summary: UserProfileSummary,
    recommendations: list[RecommendedJob],
) -> dict[str, Any]:
    return {
        "task": (
            "Rewrite each recommendation reason in one concise recruiter-style sentence. "
            "Base it only on skill overlap, experience alignment, and role fit. "
            "Return schema: {\"reasons\":[{\"job_title\":\"\",\"company\":\"\",\"reason\":\"\"}]}."
        ),
        "user_profile_summary": summary.model_dump(),
        "user_profile": payload.user_profile.model_dump(mode="json"),
        "provided_job_listings": [job.model_dump(mode="json") for job in payload.job_listings],
        "current_recommendations": [
            recommendation.model_dump(mode="json") for recommendation in recommendations
        ],
    }


def _parse_reason_map(raw_response: str) -> dict[str, str]:
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    data = json.loads(cleaned)
    reason_map: dict[str, str] = {}
    for item in data.get("reasons", []):
        title = str(item.get("job_title", "")).strip()
        company = str(item.get("company", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if title and company and reason:
            reason_map[_job_key(title, company)] = reason
    return reason_map


def _job_key(title: str, company: str) -> str:
    return f"{title.strip().lower()}::{company.strip().lower()}"
