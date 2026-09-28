from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import career_recommendation_agent.profile_extraction as profile_extraction
from career_recommendation_agent.agent import recommend_jobs
from career_recommendation_agent.api import app
from career_recommendation_agent.job_sources import (
    parse_jsearch_jobs,
    parse_jooble_jobs,
    parse_remotive_jobs,
    parse_serpapi_jobs,
)
from career_recommendation_agent.llm import DEFAULT_LLM_MODEL
from career_recommendation_agent.models import RecommendationInput
from career_recommendation_agent.profile_extraction import (
    UploadedFile,
    extract_profile_fields,
    fetch_portfolio_text,
)
from career_recommendation_agent.text import infer_domain
from career_recommendation_agent.web_app import _is_relevant_live_job
from career_recommendation_agent.web_app import _active_providers
from career_recommendation_agent.web_app import _recommend_from_form_payload
from career_recommendation_agent.web_app import _round_robin_provider_jobs
from career_recommendation_agent.workflow import build_career_recommendation_graph


@pytest.fixture(autouse=True)
def disable_llm_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAREER_AGENT_USE_LLM", "false")


def test_langgraph_workflow_compiles() -> None:
    graph = build_career_recommendation_graph()

    assert graph is not None


def test_default_llm_model_is_llama_3_3_70b() -> None:
    assert DEFAULT_LLM_MODEL == "llama-3.3-70b-versatile"


def test_machine_learning_role_keeps_ai_domain_with_data_skills() -> None:
    assert (
        infer_domain("Machine Learning Engineer", "Python, SQL, Pandas, Data, Analytics")
        == "AI / Machine Learning"
    )


def test_parse_serpapi_jobs_supports_linkedin_filter() -> None:
    payload = {
        "jobs_results": [
            {
                "title": "Machine Learning Engineer",
                "company_name": "AI Co",
                "description": "Build Python machine learning APIs.",
                "location": "Remote",
                "via": "via Google Jobs",
                "apply_options": [
                    {"title": "LinkedIn", "link": "https://linkedin.com/jobs/view/1"},
                    {"title": "Company Careers", "link": "https://example.com/jobs/1"},
                ],
            },
            {
                "title": "Frontend Developer",
                "company_name": "Web Co",
                "description": "React UI work.",
                "location": "Remote",
                "via": "via Google Jobs",
                "apply_options": [
                    {"title": "Company Careers", "link": "https://example.com/jobs/2"},
                ],
            },
        ]
    }

    jobs = parse_serpapi_jobs(payload, limit=10, source_filter="linkedin")

    assert len(jobs) == 1
    assert jobs[0].source_platform == "LinkedIn"
    assert str(jobs[0].apply_url) == "https://linkedin.com/jobs/view/1"


def test_parse_remotive_jobs() -> None:
    payload = {
        "jobs": [
            {
                "title": "Python Developer",
                "company_name": "Remote Co",
                "description": "Build backend services with Python.",
                "candidate_required_location": "Worldwide",
                "url": "https://remotive.com/remote-jobs/software-dev/python-developer-1",
            }
        ]
    }

    jobs = parse_remotive_jobs(payload, limit=5)

    assert len(jobs) == 1
    assert jobs[0].job_title == "Python Developer"
    assert jobs[0].source_platform == "Remotive"


def test_parse_jsearch_jobs_supports_source_filter() -> None:
    payload = {
        "data": [
            {
                "job_title": "AI Engineer",
                "employer_name": "AI Labs",
                "job_description": "Build Python machine learning systems.",
                "job_city": "Islamabad",
                "job_country": "PK",
                "job_apply_link": "https://www.linkedin.com/jobs/view/100",
                "job_publisher": "LinkedIn",
                "job_required_experience": {"required_experience_in_months": 24},
            },
            {
                "job_title": "Sales Manager",
                "employer_name": "Sales Co",
                "job_description": "Manage enterprise accounts.",
                "job_apply_link": "https://example.com/jobs/200",
                "job_publisher": "Company Careers",
            },
        ]
    }

    jobs = parse_jsearch_jobs(payload, limit=10, source_filter="linkedin")

    assert len(jobs) == 1
    assert jobs[0].job_title == "AI Engineer"
    assert jobs[0].source_platform == "LinkedIn"
    assert jobs[0].experience_level == "Junior"
    assert str(jobs[0].apply_url) == "https://www.linkedin.com/jobs/view/100"


def test_parse_jsearch_jobs_detects_source_from_apply_link() -> None:
    payload = {
        "data": {
            "jobs": [
                {
                    "job_title": "Video Editor",
                    "employer_name": "Creative Co",
                    "job_description": "Edit commercial videos using Adobe Premiere Pro.",
                    "job_apply_link": "https://www.linkedin.com/jobs/view/200",
                    "job_publisher": "Company Site",
                }
            ]
        }
    }

    jobs = parse_jsearch_jobs(payload, limit=10, source_filter="linkedin")

    assert len(jobs) == 1
    assert jobs[0].source_platform == "LinkedIn"


def test_parse_jooble_jobs() -> None:
    payload = {
        "jobs": [
            {
                "title": "Video Editor",
                "company": "Creative Studio",
                "snippet": "Edit commercial videos using Adobe Premiere Pro and After Effects.",
                "location": "Remote",
                "link": "https://example.com/jobs/video-editor",
            }
        ]
    }

    jobs = parse_jooble_jobs(payload, limit=10)

    assert len(jobs) == 1
    assert jobs[0].job_title == "Video Editor"
    assert jobs[0].company == "Creative Studio"
    assert jobs[0].source_platform == "Jooble"
    assert str(jobs[0].apply_url) == "https://example.com/jobs/video-editor"


def test_all_provider_mode_skips_sources_without_source_filter_support() -> None:
    assert _active_providers("all", None) == ("jsearch", "serpapi", "jooble", "remotive")
    assert _active_providers("all", "linkedin") == ("jsearch", "serpapi")
    assert _active_providers("jooble", None) == ("jooble",)


def test_extract_profile_fields_from_text_cv() -> None:
    cv = UploadedFile(
        filename="cv.txt",
        content=(
            "Machine Learning Engineer\n"
            "Skills: Python, FastAPI, Machine Learning, SQL, Docker\n"
            "3 years of experience\n"
            "Experience\n"
            "Built production APIs for machine learning services.\n"
            "Education\n"
            "BS Computer Science\n"
            "Projects\n"
            "AI career recommendation system\n"
        ).encode("utf-8"),
    )

    fields = extract_profile_fields(cv, "")

    assert fields["desired_role"] == "Machine Learning Engineer"
    assert "Python" in fields["skills"]
    assert fields["experience_years"] == "3"


def test_extract_profile_experience_from_date_ranges() -> None:
    cv = UploadedFile(
        filename="cv.txt",
        content=(
            "AI Engineer\n"
            "Skills: Python, Machine Learning, SQL\n"
            "Experience\n"
            "ML Intern | Company A | June 2024-Nov 2024\n"
            "AI Engineer | Company B | Jan 2025 - Jun 2025\n"
        ).encode("utf-8"),
    )

    fields = extract_profile_fields(cv, "")

    assert fields["experience_years"] == "1"
    assert fields["seniority"] == "Junior"


def test_project_date_ranges_do_not_count_as_experience_years() -> None:
    cv = UploadedFile(
        filename="cv.txt",
        content=(
            "AI Engineer\n"
            "Skills: Python, Machine Learning, SQL\n"
            "Projects\n"
            "House Price Prediction June 2024 - Aug 2024\n"
            "Machine Learning Project Dec 2024 - May 2025\n"
        ).encode("utf-8"),
    )

    fields = extract_profile_fields(cv, "")

    assert fields["experience_years"] == ""


def test_recommendations_are_ranked_by_match_score() -> None:
    payload = RecommendationInput.model_validate(
        {
            "user_profile": {
                "skills": ["Python", "FastAPI", "Machine Learning"],
                "experience_years": 3,
                "desired_role": "Machine Learning Engineer",
            },
            "job_listings": [
                {
                    "job_title": "Frontend Developer",
                    "company": "Web Co",
                    "required_skills": ["React", "CSS"],
                    "experience_level": "Mid",
                    "source_platform": "Provided",
                },
                {
                    "job_title": "Machine Learning Engineer",
                    "company": "AI Co",
                    "required_skills": ["Python", "FastAPI", "Machine Learning"],
                    "experience_level": "Mid",
                    "source_platform": "Provided",
                },
            ],
        }
    )

    output = recommend_jobs(payload)

    assert output.recommended_jobs[0].job_title == "Machine Learning Engineer"
    assert output.recommended_jobs[0].match_score > output.recommended_jobs[1].match_score


def test_large_job_set_filters_low_relevance_jobs() -> None:
    payload = RecommendationInput.model_validate(
        {
            "user_profile": {
                "skills": ["Python", "SQL"],
                "experience_years": 2,
                "desired_role": "Backend Developer",
            },
            "job_listings": [
                {
                    "job_title": "Backend Developer",
                    "company": "API Co",
                    "required_skills": ["Python", "SQL"],
                    "experience_level": "Junior",
                },
                {
                    "job_title": "Chef",
                    "company": "Food Co",
                    "required_skills": ["Menu Planning"],
                    "experience_level": "Senior",
                },
                {
                    "job_title": "Accountant",
                    "company": "Books Co",
                    "required_skills": ["Tax"],
                    "experience_level": "Senior",
                },
                {
                    "job_title": "Driver",
                    "company": "Transport Co",
                    "required_skills": ["Logistics"],
                    "experience_level": "Senior",
                },
            ],
        }
    )

    output = recommend_jobs(payload)

    assert [job.job_title for job in output.recommended_jobs] == ["Backend Developer"]


def test_fallback_recommendations_show_ranked_live_jobs_when_threshold_not_met() -> None:
    payload = RecommendationInput.model_validate(
        {
            "user_profile": {
                "skills": ["Python"],
                "experience_years": 1,
                "desired_role": "AI Engineer",
            },
            "minimum_match_score": 80,
            "fallback_recommendation_count": 2,
            "job_listings": [
                {
                    "job_title": "Data Labeling Specialist",
                    "company": "Remote Co",
                    "job_description": "Review AI training data.",
                    "required_skills": [],
                    "experience_level": "Junior",
                },
                {
                    "job_title": "Support Assistant",
                    "company": "Ops Co",
                    "job_description": "Customer operations work.",
                    "required_skills": [],
                    "experience_level": "Junior",
                },
                {
                    "job_title": "Video Editor",
                    "company": "Media Co",
                    "job_description": "AI video editing role.",
                    "required_skills": [],
                    "experience_level": "Junior",
                },
            ],
        }
    )

    output = recommend_jobs(payload)

    assert len(output.recommended_jobs) == 2
    assert "Lower-confidence" in output.recommended_jobs[0].reason


def test_fallback_recommendations_supplement_strong_matches() -> None:
    payload = RecommendationInput.model_validate(
        {
            "user_profile": {
                "skills": ["Python", "Machine Learning"],
                "experience_years": 2,
                "desired_role": "Machine Learning Engineer",
            },
            "minimum_match_score": 80,
            "fallback_recommendation_count": 3,
            "job_listings": [
                {
                    "job_title": "Machine Learning Engineer",
                    "company": "AI Co",
                    "job_description": "Build machine learning systems with Python.",
                    "required_skills": ["Python", "Machine Learning"],
                    "experience_level": "Junior",
                },
                {
                    "job_title": "Data Analyst",
                    "company": "Data Co",
                    "job_description": "Analyze model reports and SQL dashboards.",
                    "required_skills": ["SQL"],
                    "experience_level": "Junior",
                },
                {
                    "job_title": "AI Content Reviewer",
                    "company": "Review Co",
                    "job_description": "Review AI generated data and model outputs.",
                    "required_skills": [],
                    "experience_level": "Junior",
                },
            ],
        }
    )

    output = recommend_jobs(payload)

    assert len(output.recommended_jobs) == 3
    assert output.recommended_jobs[0].match_score >= 80
    assert any("Lower-confidence" in job.reason for job in output.recommended_jobs[1:])


def test_portfolio_url_blocks_localhost() -> None:
    with pytest.raises(ValueError, match="public website"):
        fetch_portfolio_text("http://127.0.0.1:8000")


def test_extract_profile_fields_for_non_technical_cv() -> None:
    cv = UploadedFile(
        filename="cv.txt",
        content=(
            "Digital Marketing Specialist\n"
            "Skills: SEO, Social Media Marketing, Google Analytics, Content Writing, Canva\n"
            "Experience\n"
            "Marketing Intern | Brand Co | Jan 2024 - Jun 2024\n"
            "Projects\n"
            "Managed Instagram campaigns and improved engagement through market research.\n"
        ).encode("utf-8"),
    )

    fields = extract_profile_fields(cv, "")

    assert fields["desired_role"] == "Digital Marketing Specialist"
    assert "SEO" in fields["skills"]
    assert "Google Analytics" in fields["skills"]


def test_extract_profile_fields_for_creative_video_cv() -> None:
    cv = UploadedFile(
        filename="cv.txt",
        content=(
            "FAHAD ALI\n"
            "Video Editor\n"
            "&\n"
            "3d Modeler\n"
            "About Me\n"
            "I am a 2D and 3d Artist & Video Editor\n"
            "Experience\n"
            "Studio: Sr4webservices Islamabad 2020 - 2022\n"
            "Video Ads and Commercial Video Production Ads\n"
            "Studio: Proveyourself Islamabad 2022 - 2024\n"
            "Video Production Manager leading a team of video editors\n"
            "Expertise\n"
            "video editor\n"
            "3d modeler\n"
            "2d animator\n"
            "Game Development\n"
            "Skills Summary\n"
            "Adobe Premiere Pro 90%\n"
            "Adobe After Effects 75%\n"
            "Blender 80%\n"
            "Adobe Illustrator 70%\n"
            "Adobe Photoshop 65%\n"
            "Unreal Engine 50%\n"
        ).encode("utf-8"),
    )

    fields = extract_profile_fields(cv, "")

    assert fields["desired_role"] == "Video Editor & 3D Modeler"
    assert "Adobe Premiere Pro" in fields["skills"]
    assert "Blender" in fields["skills"]
    assert fields["domain"] == "Creative Media / Video Production"
    assert (
        infer_domain(fields["desired_role"], fields["skills"], fields["experience"])
        == "Creative Media / Video Production"
    )


def test_top_section_role_overrides_generic_llm_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_extraction, "llm_is_configured", lambda: True)
    monkeypatch.setattr(
        profile_extraction,
        "_extract_with_llama",
        lambda _text: {
            "desired_role": "Professional",
            "skills": "",
            "experience_years": "",
            "seniority": "",
            "experience": "",
            "education": "",
            "projects": "",
            "location": "",
            "domain": "General Technology",
            "query": "Professional",
            "alternate_roles": "",
            "search_queries": "",
            "confidence": "0.35",
        },
    )
    cv = UploadedFile(
        filename="cv.txt",
        content=(
            "Usman Ahmed Rauf\n"
            "Video editor\n"
            "Education\n"
            "2019-2021 High School\n"
            "Skills\n"
            "Adobe Premiere Pro\n"
            "After Effects\n"
        ).encode("utf-8"),
    )

    fields = extract_profile_fields(cv, "")

    assert fields["desired_role"] == "Video Editor"
    assert fields["domain"] == "Creative Media / Video Production"


def test_low_confidence_pdf_uses_ocr_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        profile_extraction,
        "_extract_pdf_text",
        lambda _content: "Usman Ahmed Rauf\nEducation\n2019-2021 High School\n",
    )
    monkeypatch.setattr(
        profile_extraction,
        "_extract_scanned_pdf_text",
        lambda _content: (
            "Usman Ahmed Rauf\n"
            "Video editor\n"
            "Skills\n"
            "Motion Graphics video\n"
            "Adobe Premiere Pro\n"
            "Adobe After effects\n"
        ),
    )
    cv = UploadedFile(filename="resume.pdf", content=b"%PDF test")

    fields = extract_profile_fields(cv, "")

    assert fields["desired_role"] == "Video Editor"
    assert fields["domain"] == "Creative Media / Video Production"


def test_invalid_long_llm_role_falls_back_to_ai_title(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_extraction, "llm_is_configured", lambda: True)
    monkeypatch.setattr(
        profile_extraction,
        "_extract_with_llama",
        lambda _text: {
            "desired_role": "Networking, Monitoring, And Scalable Deployment Architecture",
            "skills": "Python, TensorFlow, Machine Learning, Docker",
            "experience_years": "2",
            "seniority": "Junior",
            "experience": "Built machine learning services and deployment pipelines.",
            "education": "",
            "projects": "",
            "location": "",
            "domain": "AI / Machine Learning",
            "query": "Networking, Monitoring, And Scalable Deployment Architecture",
            "alternate_roles": "",
            "search_queries": "",
            "confidence": "0.95",
        },
    )
    cv = UploadedFile(
        filename="cv.txt",
        content=(
            "Skills: Python, TensorFlow, Machine Learning, Docker\n"
            "Experience\n"
            "Built machine learning models and scalable deployment workflows.\n"
        ).encode("utf-8"),
    )

    fields = extract_profile_fields(cv, "")

    assert fields["desired_role"] == "Machine Learning Engineer"
    assert fields["domain"] == "AI / Machine Learning"


def test_extract_profile_fields_for_healthcare_cv() -> None:
    cv = UploadedFile(
        filename="cv.txt",
        content=(
            "Ayesha Khan\n"
            "Skills: Nursing, Patient Care, Medical Records, Communication\n"
            "Experience\n"
            "Nurse | City Hospital | Jan 2022 - Dec 2024\n"
            "Provided patient care, maintained medical records, and supported clinical teams.\n"
        ).encode("utf-8"),
    )

    fields = extract_profile_fields(cv, "")

    assert fields["desired_role"] == "Nurse"
    assert "Patient Care" in fields["skills"]
    assert fields["domain"] == "Healthcare"
    assert infer_domain(fields["desired_role"], fields["skills"], fields["experience"]) == "Healthcare"


def test_extract_profile_fields_for_logistics_cv() -> None:
    cv = UploadedFile(
        filename="cv.txt",
        content=(
            "Logistics Coordinator\n"
            "Skills: Logistics, Supply Chain, Excel, Inventory\n"
            "Experience\n"
            "Managed warehouse dispatch, shipping records, and procurement follow-up.\n"
        ).encode("utf-8"),
    )

    fields = extract_profile_fields(cv, "")

    assert fields["desired_role"] == "Logistics Coordinator"
    assert "Supply Chain" in fields["skills"]
    assert fields["domain"] == "Logistics / Supply Chain"
    assert (
        infer_domain(fields["desired_role"], fields["skills"], fields["experience"])
        == "Logistics / Supply Chain"
    )


@pytest.mark.parametrize(
    ("keywords", "expected_domain"),
    [
        (
            "Python, FastAPI, Machine Learning, TensorFlow",
            "AI / Machine Learning",
        ),
        (
            "Digital Marketing, SEO, Google Ads, Social Media",
            "Marketing / Growth",
        ),
    ],
)
def test_keyword_domains_use_specific_canonical_domain(
    keywords: str,
    expected_domain: str,
) -> None:
    assert infer_domain(keywords) == expected_domain


def test_live_job_relevance_rejects_wrong_domain_for_creative_profile() -> None:
    from career_recommendation_agent.models import JobListing, UserPreferences, UserProfile

    profile = UserProfile(
        desired_role="Video Editor & 3D Modeler",
        skills=["Adobe Premiere Pro", "Blender", "Video Production"],
        preferences=UserPreferences(domain="Creative Media / Video Production"),
    )
    job = JobListing(
        job_title="Software Development Engineer",
        company="Tech Co",
        job_description="Build backend APIs with Java and cloud services.",
        required_skills=["Java", "AWS", "APIs"],
        source_platform="LinkedIn",
    )

    assert _is_relevant_live_job(profile, job) is False


def test_live_job_relevance_allows_exact_role_when_domain_is_unknown() -> None:
    from career_recommendation_agent.models import JobListing, UserProfile

    profile = UserProfile(desired_role="Product Manager")
    job = JobListing(
        job_title="Senior Product Manager",
        company="Product Co",
        job_description="Own the product roadmap and coordinate product delivery.",
        source_platform="LinkedIn",
    )

    assert _is_relevant_live_job(profile, job) is True


def test_provider_results_are_round_robin_and_deduplicated() -> None:
    from career_recommendation_agent.models import JobListing

    duplicate = JobListing(
        job_title="Video Editor",
        company="Media Co",
        location="Remote",
        source_platform="LinkedIn",
    )
    provider_jobs = {
        "jsearch": [
            duplicate,
            JobListing(job_title="Motion Designer", company="Studio A"),
        ],
        "serpapi": [
            duplicate.model_copy(update={"source_platform": "Indeed"}),
            JobListing(job_title="Video Producer", company="Studio B"),
        ],
        "jooble": [
            JobListing(job_title="3D Artist", company="Studio C"),
        ],
    }

    merged = _round_robin_provider_jobs(
        provider_jobs,
        ("jsearch", "serpapi", "jooble"),
        limit=10,
    )

    assert [job.job_title for job in merged] == [
        "Video Editor",
        "3D Artist",
        "Motion Designer",
        "Video Producer",
    ]


def test_recommend_payload_recomputes_domain_from_keywords_over_stale_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from career_recommendation_agent.models import JobListing

    monkeypatch.setattr(
        "career_recommendation_agent.web_app.fetch_live_jobs",
        lambda *_args, **_kwargs: [
            JobListing(
                job_title="Cybersecurity Analyst",
                company="Security Co",
                job_description="Monitor SOC alerts, SIEM logs, and vulnerability reports.",
                required_skills=["SOC", "SIEM"],
                experience_level="Senior",
                source_platform="Test",
            )
        ],
    )
    output = _recommend_from_form_payload(
        {
            "desired_role": "Cybersecurity Analyst",
            "skills": "cyber security, SOC, SIEM, penetration testing",
            "experience_years": "5",
            "seniority": "Senior",
            "domain": "Creative Media / Video Production",
            "query": "Cybersecurity Analyst",
            "provider": "remotive",
        }
    )

    assert output.user_profile_summary.domain == "Cybersecurity"


def test_fastapi_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_fastapi_extract_profile_from_text_upload() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/extract-profile",
        files={
            "cv_file": (
                "cv.txt",
                b"Digital Marketing Specialist\nSkills: SEO, Google Analytics, Canva\n",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["fields"]["desired_role"] == "Digital Marketing Specialist"


def test_fastapi_keyword_profile_uses_canonical_marketing_domain() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/extract-profile",
        data={
            "portfolio_text": "Digital Marketing, SEO, Google Ads, Social Media",
            "input_type": "keywords",
        },
    )

    assert response.status_code == 200
    fields = response.json()["fields"]
    assert fields["desired_role"] == "Digital Marketing Specialist"
    assert fields["domain"] == "Marketing / Growth"


def test_fastapi_keyword_profile_detects_cybersecurity() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/extract-profile",
        data={
            "portfolio_text": "Cybersecurity, SOC, SIEM, Penetration Testing",
            "input_type": "keywords",
        },
    )

    assert response.status_code == 200
    fields = response.json()["fields"]
    assert fields["desired_role"] == "Cybersecurity Analyst"
    assert fields["domain"] == "Cybersecurity"


@pytest.mark.parametrize(
    ("keywords", "expected_role", "expected_domain"),
    [
        (
            "Content Writing, Copywriting, Blogging, SEO",
            "Content Writer",
            "Content / Writing",
        ),
        (
            "Customer Support, CRM, Email Support, Communication",
            "Customer Support Representative",
            "Customer Support / Operations",
        ),
        (
            "Graphic Design, Photoshop, Illustrator, Branding",
            "Graphic Designer",
            "Design / Creative",
        ),
        (
            "UI/UX Design, Figma, Wireframing, Prototyping",
            "UI/UX Designer",
            "Design / Product Design",
        ),
        (
            "Project Management, Agile, Scrum, Jira",
            "Project Manager",
            "Project Management",
        ),
        (
            "Hotel Management, Guest Services, Front Desk, Hospitality",
            "Hotel Manager",
            "Hospitality",
        ),
    ],
)
def test_fastapi_keyword_profile_handles_content_and_support_roles(
    keywords: str,
    expected_role: str,
    expected_domain: str,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/api/extract-profile",
        data={"portfolio_text": keywords, "input_type": "keywords"},
    )

    assert response.status_code == 200
    fields = response.json()["fields"]
    assert fields["desired_role"] == expected_role
    assert fields["domain"] == expected_domain


@pytest.mark.parametrize(
    ("keywords", "expected_role", "expected_domain"),
    [
        (
            "Cybersecurity, SOC, SIEM, Penetration Testing",
            "Cybersecurity Analyst",
            "Cybersecurity",
        ),
        (
            "Python, FastAPI, Machine Learning, TensorFlow",
            "Machine Learning Engineer",
            "AI / Machine Learning",
        ),
        (
            "Data Analysis, Power BI, Tableau, Dashboard",
            "Data Analyst",
            "Data / Analytics",
        ),
        (
            "Accounting, Bookkeeping, QuickBooks, Financial Reporting",
            "Accountant",
            "Finance / Accounting",
        ),
        (
            "Teaching, Curriculum, Classroom, Training",
            "Teacher",
            "Education / Training",
        ),
        (
            "Logistics, Supply Chain, Warehouse, Procurement",
            "Logistics Coordinator",
            "Logistics / Supply Chain",
        ),
    ],
)
def test_keyword_profile_supports_universal_domain_detection(
    keywords: str,
    expected_role: str,
    expected_domain: str,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/api/extract-profile",
        data={"portfolio_text": keywords, "input_type": "keywords"},
    )

    assert response.status_code == 200
    fields = response.json()["fields"]
    assert fields["desired_role"] == expected_role
    assert fields["domain"] == expected_domain


def test_fastapi_signup_login_and_me() -> None:
    client = TestClient(app)
    email = "test-user-auth@example.com"

    signup_response = client.post(
        "/api/auth/signup",
        json={"email": email, "password": "strong-password", "full_name": "Test User"},
    )
    if signup_response.status_code == 400:
        login_response = client.post(
            "/api/auth/login",
            json={"email": email, "password": "strong-password"},
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
    else:
        assert signup_response.status_code == 200
        token = signup_response.json()["access_token"]

    me_response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert me_response.status_code == 200
    assert me_response.json()["user"]["email"] == email


def test_fastapi_saved_jobs_requires_auth_and_persists_job() -> None:
    client = TestClient(app)
    email = "test-user-saved-jobs@example.com"
    signup_response = client.post(
        "/api/auth/signup",
        json={"email": email, "password": "strong-password", "full_name": "Saved User"},
    )
    if signup_response.status_code == 400:
        signup_response = client.post(
            "/api/auth/login",
            json={"email": email, "password": "strong-password"},
        )
    token = signup_response.json()["access_token"]

    unauthenticated_response = client.get("/api/saved-jobs")
    assert unauthenticated_response.status_code == 401

    save_response = client.post(
        "/api/saved-jobs",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "job_title": "Machine Learning Engineer",
            "company": "AI Co",
            "source": "Remotive",
            "match_score": 88,
            "reason": "Strong Python and ML overlap.",
            "apply_link": "https://example.com/jobs/ml",
        },
    )
    assert save_response.status_code == 200
    saved_job_id = save_response.json()["saved_job_id"]

    list_response = client.get("/api/saved-jobs", headers={"Authorization": f"Bearer {token}"})
    assert list_response.status_code == 200
    assert list_response.json()["saved_jobs"][0]["job_title"] == "Machine Learning Engineer"

    delete_response = client.delete(
        f"/api/saved-jobs/{saved_job_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert delete_response.status_code == 200


def test_fastapi_profile_update_and_password_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETURN_PASSWORD_RESET_TOKEN", "true")
    client = TestClient(app)
    email = f"test-user-profile-reset-{uuid4().hex}@example.com"
    password = "strong-password"
    signup_response = client.post(
        "/api/auth/signup",
        json={"email": email, "password": password, "full_name": "Profile User"},
    )
    if signup_response.status_code == 400:
        signup_response = client.post(
            "/api/auth/login",
            json={"email": email, "password": password},
        )
    token = signup_response.json()["access_token"]

    profile_response = client.put(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "full_name": "Updated User",
            "headline": "AI Engineer",
            "location": "Remote",
            "skills": "Python, ML",
        },
    )
    assert profile_response.status_code == 200
    assert profile_response.json()["user"]["headline"] == "AI Engineer"

    reset_request = client.post("/api/auth/password-reset/request", json={"email": email})
    assert reset_request.status_code == 200
    reset_token = reset_request.json()["reset_token"]

    reset_confirm = client.post(
        "/api/auth/password-reset/confirm",
        json={"token": reset_token, "new_password": "new-strong-password"},
    )
    assert reset_confirm.status_code == 200

    login_response = client.post(
        "/api/auth/login",
        json={"email": email, "password": "new-strong-password"},
    )
    assert login_response.status_code == 200


def test_fastapi_admin_stats_requires_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    email = f"test-admin-{uuid4().hex}@example.com"
    password = "strong-password"
    monkeypatch.setenv("ADMIN_EMAILS", email)
    signup_response = client.post(
        "/api/auth/signup",
        json={"email": email, "password": password, "full_name": "Admin User"},
    )
    if signup_response.status_code == 400:
        signup_response = client.post(
            "/api/auth/login",
            json={"email": email, "password": password},
        )
    token = signup_response.json()["access_token"]

    stats_response = client.get("/api/admin/stats", headers={"Authorization": f"Bearer {token}"})

    assert stats_response.status_code == 200
    assert "users" in stats_response.json()["stats"]


def test_fastapi_admin_system_reports_production_status(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    email = f"test-admin-system-{uuid4().hex}@example.com"
    monkeypatch.setenv("ADMIN_EMAILS", email)
    monkeypatch.setenv("JSEARCH_API_KEY", "test-key")

    signup_response = client.post(
        "/api/auth/signup",
        json={"email": email, "password": "strong-password", "full_name": "System Admin"},
    )
    token = signup_response.json()["access_token"]

    response = client.get("/api/admin/system", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    system = response.json()["system"]
    assert system["database"] in {"sqlite", "postgresql"}
    assert system["providers"]["remotive"] is True
    assert system["providers"]["jsearch"] is True
    assert system["rate_limit_requests"] > 0


def test_fastapi_extract_profile_rejects_file_type_mismatch() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/extract-profile",
        files={
            "cv_file": (
                "resume.pdf",
                b"MZ executable payload",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert "Executable files are not allowed" in response.json()["detail"]
