from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from career_recommendation_agent.models import (
    JobListing,
    RecommendedJob,
    RecommendationInput,
    RecommendationOutput,
    ScoreBreakdown,
    UserProfileSummary,
)
from career_recommendation_agent.llm import enhance_recommendation_reasons
from career_recommendation_agent.scoring import score_job
from career_recommendation_agent.text import infer_domain, infer_experience_level, unique_preserving_order


MIN_RECOMMENDATION_SCORE = 50


class CareerRecommendationState(TypedDict, total=False):
    payload: RecommendationInput
    user_profile_summary: UserProfileSummary
    analyzed_jobs: list[tuple[JobListing, ScoreBreakdown]]
    ranked_jobs: list[tuple[JobListing, ScoreBreakdown]]
    recommended_jobs: list[RecommendedJob]
    output: RecommendationOutput


def build_career_recommendation_graph() -> Any:
    graph = StateGraph(CareerRecommendationState)

    graph.add_node("analyze_user_profile", analyze_user_profile)
    graph.add_node("analyze_job_listings", analyze_job_listings)
    graph.add_node("rank_and_filter_jobs", rank_and_filter_jobs)
    graph.add_node("llama_recruiter_review", llama_recruiter_review)
    graph.add_node("format_strict_json_output", format_strict_json_output)

    graph.set_entry_point("analyze_user_profile")
    graph.add_edge("analyze_user_profile", "analyze_job_listings")
    graph.add_edge("analyze_job_listings", "rank_and_filter_jobs")
    graph.add_edge("rank_and_filter_jobs", "llama_recruiter_review")
    graph.add_edge("llama_recruiter_review", "format_strict_json_output")
    graph.add_edge("format_strict_json_output", END)

    return graph.compile()


def run_career_recommendation_graph(payload: RecommendationInput) -> RecommendationOutput:
    app = build_career_recommendation_graph()
    final_state = app.invoke({"payload": payload})
    return final_state["output"]


def analyze_user_profile(state: CareerRecommendationState) -> CareerRecommendationState:
    profile = state["payload"].user_profile
    summary = UserProfileSummary(
        target_role=profile.desired_role or "Not specified",
        skills=unique_preserving_order(profile.skills),
        experience_level=infer_experience_level(
            profile.experience_years,
            profile.preferences.seniority,
        ),
        domain=profile.preferences.domain
        or infer_domain(
            profile.desired_role,
            " ".join(profile.skills),
            " ".join(profile.experience),
            " ".join(profile.projects),
        ),
    )
    return {"user_profile_summary": summary}


def analyze_job_listings(state: CareerRecommendationState) -> CareerRecommendationState:
    payload = state["payload"]
    analyzed_jobs = [(job, score_job(payload.user_profile, job)) for job in payload.job_listings]
    return {"analyzed_jobs": analyzed_jobs}


def rank_and_filter_jobs(state: CareerRecommendationState) -> CareerRecommendationState:
    ranked_jobs = sorted(
        state["analyzed_jobs"],
        key=lambda item: (
            item[1].final_score,
            item[1].skills_match,
            item[1].role_relevance,
        ),
        reverse=True,
    )

    configured_threshold = max(0, min(100, state["payload"].minimum_match_score))
    threshold = configured_threshold if len(ranked_jobs) > 2 else 0
    recommendations = [
        RecommendedJob(
            job_title=job.job_title,
            company=job.company,
            source=job.source_platform or "Provided Input",
            match_score=score.final_score,
            reason=_build_reason(score.matched_skills, score.final_score),
            apply_link=str(job.apply_url or ""),
        )
        for job, score in ranked_jobs
        if score.final_score >= threshold
    ]
    if state["payload"].fallback_recommendation_count > 0:
        target_count = state["payload"].fallback_recommendation_count
        already_selected = {
            _recommendation_key(recommendation)
            for recommendation in recommendations
        }
        for job, score in ranked_jobs:
            if len(recommendations) >= target_count:
                break
            key = _job_key(job)
            if key in already_selected or score.final_score >= threshold:
                continue
            recommendations.append(
                RecommendedJob(
                    job_title=job.job_title,
                    company=job.company,
                    source=job.source_platform or "Provided Input",
                    match_score=score.final_score,
                    reason=_build_fallback_reason(score.matched_skills, score.final_score),
                    apply_link=str(job.apply_url or ""),
                )
            )
            already_selected.add(key)

    return {
        "ranked_jobs": ranked_jobs,
        "recommended_jobs": recommendations,
    }


def format_strict_json_output(state: CareerRecommendationState) -> CareerRecommendationState:
    return {
        "output": RecommendationOutput(
            user_profile_summary=state["user_profile_summary"],
            recommended_jobs=state["recommended_jobs"],
        )
    }


def llama_recruiter_review(state: CareerRecommendationState) -> CareerRecommendationState:
    enhanced_recommendations = enhance_recommendation_reasons(
        payload=state["payload"],
        summary=state["user_profile_summary"],
        recommendations=state["recommended_jobs"],
    )
    return {"recommended_jobs": enhanced_recommendations}


def _build_reason(matched_skills: list[str], score: int) -> str:
    if matched_skills:
        skills = ", ".join(matched_skills[:5])
        return (
            f"Strong role fit with matching skills: {skills}. "
            f"Overall alignment score is {score}/100."
        )
    return (
        "Limited explicit skill overlap, but available role, experience, and domain signals "
        f"produce an alignment score of {score}/100."
    )


def _job_key(job: JobListing) -> tuple[str, str, str]:
    return (
        job.job_title.strip().lower(),
        job.company.strip().lower(),
        str(job.apply_url or "").strip().lower(),
    )


def _recommendation_key(recommendation: RecommendedJob) -> tuple[str, str, str]:
    return (
        recommendation.job_title.strip().lower(),
        recommendation.company.strip().lower(),
        recommendation.apply_link.strip().lower(),
    )


def _build_fallback_reason(matched_skills: list[str], score: int) -> str:
    if matched_skills:
        skills = ", ".join(matched_skills[:5])
        return (
            f"Lower-confidence live result with matching skills: {skills}. "
            f"Alignment score is {score}/100."
        )
    return (
        "Lower-confidence live result. It is shown because no stronger live jobs crossed "
        f"the relevance threshold; alignment score is {score}/100."
    )
