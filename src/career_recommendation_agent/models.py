from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator


class UserPreferences(BaseModel):
    location: str | None = None
    role_type: str | None = None
    seniority: str | None = None
    domain: str | None = None


class UserProfile(BaseModel):
    skills: list[str] = Field(default_factory=list)
    experience_years: float | None = None
    experience: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    desired_role: str | None = None
    preferences: UserPreferences = Field(default_factory=UserPreferences)

    @field_validator("skills", "experience", "education", "projects", mode="before")
    @classmethod
    def coerce_string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)


class JobListing(BaseModel):
    job_title: str
    company: str
    job_description: str = ""
    required_skills: list[str] = Field(default_factory=list)
    experience_level: str | None = None
    location: str | None = None
    apply_url: HttpUrl | str | None = None
    source_platform: str | None = None

    @field_validator("required_skills", mode="before")
    @classmethod
    def coerce_required_skills(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return list(value)


class RecommendationInput(BaseModel):
    user_profile: UserProfile
    job_listings: list[JobListing] = Field(default_factory=list)
    minimum_match_score: int = 50
    fallback_recommendation_count: int = 0


class UserProfileSummary(BaseModel):
    target_role: str
    skills: list[str]
    experience_level: str
    domain: str


class RecommendedJob(BaseModel):
    job_title: str
    company: str
    source: str
    match_score: int
    reason: str
    apply_link: str


class RecommendationOutput(BaseModel):
    user_profile_summary: UserProfileSummary
    recommended_jobs: list[RecommendedJob]


class ScoreBreakdown(BaseModel):
    skills_match: float
    experience_match: float
    role_relevance: float
    domain_match: float
    final_score: int
    matched_skills: list[str]
