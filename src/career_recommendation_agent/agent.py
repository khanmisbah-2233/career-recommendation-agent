from __future__ import annotations

from career_recommendation_agent.models import RecommendationInput, RecommendationOutput
from career_recommendation_agent.workflow import run_career_recommendation_graph


def recommend_jobs(payload: RecommendationInput) -> RecommendationOutput:
    return run_career_recommendation_graph(payload)
