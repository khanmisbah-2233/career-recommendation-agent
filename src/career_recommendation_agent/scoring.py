from __future__ import annotations

from career_recommendation_agent.models import JobListing, ScoreBreakdown, UserProfile
from career_recommendation_agent.text import (
    SENIORITY_ORDER,
    domains_are_compatible,
    infer_domain,
    infer_experience_level,
    normalize_skill,
    token_overlap_score,
)


def score_job(profile: UserProfile, job: JobListing) -> ScoreBreakdown:
    profile_skill_lookup = {normalize_skill(skill): skill for skill in profile.skills}
    profile_skills = set(profile_skill_lookup)
    expanded_profile_skills = _expanded_skill_set(profile.skills)
    required_skills = _expanded_skill_set(job.required_skills)

    matched_skills = sorted(profile_skill_lookup[skill] for skill in profile_skills & required_skills)
    if required_skills:
        skills_match = len(expanded_profile_skills & required_skills) / len(required_skills)
    else:
        skills_match = _description_skill_signal(expanded_profile_skills, job.job_description)

    experience_match = _score_experience_match(profile, job)
    role_relevance = _score_role_relevance(profile, job)
    domain_match = _score_domain_match(profile, job)

    final_score = round(
        (skills_match * 50)
        + (experience_match * 25)
        + (role_relevance * 15)
        + (domain_match * 10)
    )

    return ScoreBreakdown(
        skills_match=skills_match,
        experience_match=experience_match,
        role_relevance=role_relevance,
        domain_match=domain_match,
        final_score=max(0, min(100, final_score)),
        matched_skills=matched_skills,
    )


def _description_skill_signal(profile_skills: set[str], description: str) -> float:
    if not profile_skills or not description:
        return 0.25
    description_normalized = description.lower()
    hits = sum(1 for skill in profile_skills if skill in description_normalized)
    return min(1.0, hits / max(3, len(profile_skills)))


def _expanded_skill_set(skills: list[str]) -> set[str]:
    expanded: set[str] = set()
    for skill in skills:
        normalized = normalize_skill(skill)
        if not normalized:
            continue
        expanded.add(normalized)
        expanded.update(_skill_aliases(normalized))
    return expanded


def _skill_aliases(skill: str) -> set[str]:
    alias_groups = [
        {"adobe premiere pro", "premiere pro", "video editing", "video editor"},
        {"adobe after effects", "after effects", "motion graphics", "video production"},
        {"blender", "3d modeling", "3d modeler", "3d artist"},
        {"unreal engine", "game development", "game asset", "3d artist"},
        {"patient care", "nursing", "healthcare", "medical"},
        {"pharmacy", "pharmacist", "healthcare"},
        {"teaching", "teacher", "tutor", "education", "training"},
        {"bookkeeping", "accounting", "finance", "quickbooks"},
        {"seo", "digital marketing", "marketing", "social media marketing"},
        {"crm", "sales", "business development", "account executive"},
        {"recruitment", "hr", "human resources", "talent acquisition"},
        {"supply chain", "logistics", "warehouse", "procurement"},
        {"legal research", "legal", "paralegal", "compliance"},
        {"machine learning", "ml", "ai", "artificial intelligence", "deep learning"},
        {"python", "django", "flask", "backend", "api"},
        {"react", "javascript", "typescript", "frontend"},
    ]
    for group in alias_groups:
        if skill in group:
            return set(group)
    return set()


def _score_experience_match(profile: UserProfile, job: JobListing) -> float:
    profile_level = infer_experience_level(profile.experience_years, profile.preferences.seniority)
    job_level = infer_experience_level(None, job.experience_level)
    difference = abs(SENIORITY_ORDER[profile_level] - SENIORITY_ORDER[job_level])

    if difference == 0:
        return 1.0
    if difference == 1:
        return 0.75
    if difference == 2:
        return 0.45
    return 0.2


def _score_role_relevance(profile: UserProfile, job: JobListing) -> float:
    if not profile.desired_role:
        return 0.6
    title_score = token_overlap_score(job.job_title, profile.desired_role)
    description_score = token_overlap_score(job.job_description, profile.desired_role)
    return min(1.0, max(title_score, description_score * 0.8))


def _score_domain_match(profile: UserProfile, job: JobListing) -> float:
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
    if profile_domain == job_domain:
        return 1.0
    if domains_are_compatible(profile_domain, job_domain):
        return 0.7
    if profile_domain == "General Technology" or job_domain == "General Technology":
        return 0.35
    return 0.25
