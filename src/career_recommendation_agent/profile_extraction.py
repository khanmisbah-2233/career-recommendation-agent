from __future__ import annotations

import io
import json
import os
import re
import socket
import zipfile
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from html import unescape
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from career_recommendation_agent.job_sources import TECH_SKILL_KEYWORDS
from career_recommendation_agent.llm import llm_is_configured
from career_recommendation_agent.text import infer_domain


GENERIC_ROLES = {"", "Professional", "Candidate", "Worker", "Employee"}

COMMON_ROLES = [
    "Video Editor & 3D Modeler",
    "Video Editor",
    "3D Modeler",
    "2D Animator",
    "3D Artist",
    "Video Production Manager",
    "Animator",
    "Lab Technician",
    "Medical Billing Specialist",
    "Machine Learning Engineer",
    "AI Engineer",
    "Software Engineer",
    "Data Scientist",
    "Data Analyst",
    "Backend Developer",
    "Node.js Developer",
    "Java Backend Developer",
    "Python Developer",
    "Full Stack Developer",
    "Frontend Developer",
    "Flutter Developer",
    "Android Developer",
    "iOS Developer",
    "DevOps Engineer",
    "Cloud Engineer",
    "Project Manager",
    "Product Manager",
    "Business Analyst",
    "Marketing Manager",
    "Digital Marketing Specialist",
    "Sales Executive",
    "Accountant",
    "Finance Analyst",
    "HR Officer",
    "Recruiter",
    "Customer Support Representative",
    "Customer Success Specialist",
    "Graphic Designer",
    "UI UX Designer",
    "UI/UX Designer",
    "Content Writer",
    "Copywriter",
    "Teacher",
    "Administrative Assistant",
    "Operations Manager",
    "Nurse",
    "Doctor",
    "Medical Assistant",
    "Pharmacist",
    "Civil Engineer",
    "Mechanical Engineer",
    "Electrical Engineer",
    "Architect",
    "Legal Assistant",
    "Lawyer",
    "Paralegal",
    "Logistics Coordinator",
    "Supply Chain Specialist",
    "Warehouse Manager",
    "Chef",
    "Hotel Manager",
    "Front Desk Officer",
    "Restaurant Manager",
    "Tutor",
    "Trainer",
    "Instructor",
    "Videographer",
    "Motion Graphics Designer",
    "Editor",
    "Social Media Manager",
    "SEO Specialist",
    "Cybersecurity Analyst",
    "Network Security Engineer",
    "SOC Analyst",
    "Security Analyst",
    "Business Development Executive",
    "Accounts Assistant",
    "Bookkeeper",
    "Data Entry Operator",
    "Virtual Assistant",
    "Office Assistant",
    "Site Engineer",
    "Medical Officer",
    "Healthcare Assistant",
    "Delivery Driver",
    "Electrician",
    "Plumber",
    "Project Management Professional",
    "Product Management Professional",
    "Business Analyst",
    "Procurement Officer",
    "Banking Officer",
    "Auditor",
    "Travel Consultant",
    "Warehouse Supervisor",
]

ROLE_TAXONOMY = [
    ("Video Editor", "Creative Media / Video Production", ("video editor", "video editing", "premiere pro", "after effects")),
    ("3D Artist", "Creative Media / Video Production", ("3d artist", "3d modeler", "3d modeller", "3d modeling", "blender", "texturing", "game assets")),
    ("Animator", "Creative Media / Video Production", ("animation", "animator", "storyboarding", "character design")),
    ("Motion Graphics Designer", "Creative Media / Video Production", ("motion graphics", "after effects")),
    ("Graphic Designer", "Design / Creative", ("graphic designer", "graphic design", "photoshop", "illustrator", "canva", "branding")),
    ("UI/UX Designer", "Design / Product Design", ("ui/ux", "ui ux", "ux design", "wireframing", "prototyping", "figma")),
    ("Flutter Developer", "Mobile App Development", ("flutter", "dart", "firebase", "mobile app development")),
    ("Android Developer", "Mobile App Development", ("android", "kotlin", "mobile ui")),
    ("iOS Developer", "Mobile App Development", ("ios", "swift", "uikit", "xcode")),
    ("Product Manager", "Product Management", ("product management", "roadmap", "user stories", "saas")),
    ("Business Analyst", "Business / Analysis", ("business analysis", "requirements", "stakeholders", "documentation")),
    ("Machine Learning Engineer", "AI / Machine Learning", ("machine learning engineer", "ml engineer", "tensorflow", "pytorch")),
    ("AI Engineer", "AI / Machine Learning", ("ai engineer", "artificial intelligence", "llm")),
    ("Data Analyst", "Data / Analytics", ("data analyst", "data analytics", "dashboard", "power bi")),
    ("Software Engineer", "General Technology", ("software engineer", "software developer")),
    ("Node.js Developer", "Backend Engineering", ("node.js", "nodejs", "node", "express", "mongodb")),
    ("Java Backend Developer", "Backend Engineering", ("java", "spring boot", "microservices", "mysql")),
    ("Backend Developer", "Backend Engineering", ("backend developer", "api developer", "django", "flask", "fastapi")),
    ("Frontend Developer", "Frontend Engineering", ("frontend developer", "react developer", "javascript")),
    ("Cloud Engineer", "DevOps / Cloud", ("azure", "cloud infrastructure", "terraform", "monitoring")),
    ("DevOps Engineer", "DevOps / Cloud", ("devops engineer", "cloud engineer", "docker", "kubernetes")),
    ("Network Security Engineer", "Cybersecurity", ("network security", "firewalls", "vpn", "cisco")),
    ("Cybersecurity Analyst", "Cybersecurity", ("cybersecurity analyst", "security analyst", "soc analyst")),
    ("SOC Analyst", "Cybersecurity", ("soc analyst", "soc", "siem")),
    ("Digital Marketing Specialist", "Marketing / Growth", ("digital marketing", "seo", "social media marketing")),
    ("Content Writer", "Content / Writing", ("content writer", "content writing", "copywriting", "blogging")),
    ("Sales Executive", "Sales / Business Development", ("sales executive", "sales representative")),
    ("Business Development Executive", "Sales / Business Development", ("business development", "account executive")),
    ("Accountant", "Finance / Accounting", ("accountant", "accounting", "finance", "quickbooks")),
    ("Auditor", "Finance / Accounting", ("auditing", "audit", "ifrs", "risk assessment")),
    ("Banking Officer", "Finance / Accounting", ("banking operations", "kyc", "account opening")),
    ("Bookkeeper", "Finance / Accounting", ("bookkeeper", "bookkeeping")),
    ("HR Officer", "HR / Recruitment", ("hr officer", "human resources", "hr assistant")),
    ("Recruiter", "HR / Recruitment", ("recruiter", "talent acquisition")),
    ("Teacher", "Education / Training", ("teacher", "teaching", "classroom")),
    ("Tutor", "Education / Training", ("tutor", "tuition")),
    ("Medical Billing Specialist", "Healthcare", ("medical billing", "insurance claims", "icd coding")),
    ("Lab Technician", "Healthcare", ("lab testing", "diagnostics", "blood samples")),
    ("Nurse", "Healthcare", ("nurse", "nursing", "patient care")),
    ("Doctor", "Healthcare", ("doctor", "physician", "medical officer")),
    ("Pharmacist", "Healthcare", ("pharmacist", "pharmacy")),
    ("Civil Engineer", "Engineering / Construction", ("civil engineer", "civil engineering")),
    ("Mechanical Engineer", "Engineering / Construction", ("mechanical engineer", "mechanical engineering", "mechanical design", "solidworks", "manufacturing", "cad")),
    ("Electrical Engineer", "Engineering / Construction", ("electrical engineer", "electrical engineering")),
    ("Architect", "Engineering / Construction", ("architect", "architecture")),
    ("Customer Support Representative", "Customer Support / Operations", ("customer support", "customer service")),
    ("Customer Success Specialist", "Customer Support / Operations", ("customer success", "crm", "client support")),
    ("Administrative Assistant", "Customer Support / Operations", ("administrative assistant", "office assistant")),
    ("Data Entry Operator", "Customer Support / Operations", ("data entry", "typing")),
    ("Warehouse Supervisor", "Logistics / Supply Chain", ("warehouse operations", "warehouse", "dispatch", "inventory control", "erp")),
    ("Logistics Coordinator", "Logistics / Supply Chain", ("logistics coordinator", "logistics", "dispatch")),
    ("Procurement Officer", "Logistics / Supply Chain", ("procurement", "vendor management", "purchase orders")),
    ("Supply Chain Specialist", "Logistics / Supply Chain", ("supply chain", "procurement")),
    ("Legal Assistant", "Legal", ("legal assistant", "legal research")),
    ("Lawyer", "Legal", ("lawyer", "advocate", "attorney")),
    ("Paralegal", "Legal", ("paralegal",)),
    ("Chef", "Hospitality", ("chef", "cook")),
    ("Hotel Manager", "Hospitality", ("hotel manager", "hospitality")),
    ("Front Desk Officer", "Hospitality", ("front desk", "guest services", "hotel reception")),
    ("Restaurant Manager", "Hospitality", ("restaurant manager", "restaurant management", "food safety", "staff scheduling")),
    ("Travel Consultant", "Hospitality", ("travel consultant", "ticketing", "itinerary planning", "gds")),
    ("Project Manager", "Project Management", ("project manager", "project management", "agile", "scrum", "jira")),
]

GENERAL_SKILL_KEYWORDS = [
    "Accounting",
    "Administration",
    "Adobe After Effects",
    "Adobe Illustrator",
    "Adobe Premiere Pro",
    "Adobe Photoshop",
    "Analytics",
    "Bookkeeping",
    "Branding",
    "Blender",
    "Budgeting",
    "Business Analysis",
    "Canva",
    "Civil Engineering",
    "Communication",
    "Content Writing",
    "CRM",
    "Customer Service",
    "Data Entry",
    "Digital Marketing",
    "Electrical Engineering",
    "Email Marketing",
    "Excel",
    "Figma",
    "Finance",
    "Financial Reporting",
    "Google Analytics",
    "Graphic Design",
    "HR",
    "Leadership",
    "Legal Research",
    "Logistics",
    "Mobile App Development",
    "Management",
    "Market Research",
    "Marketing",
    "Mechanical Engineering",
    "Medical Records",
    "Microsoft Office",
    "Negotiation",
    "Nursing",
    "Operations",
    "Payroll",
    "Patient Care",
    "Pharmacy",
    "PowerPoint",
    "Project Management",
    "Product Management",
    "Scrum",
    "Agile",
    "Jira",
    "QuickBooks",
    "Recruitment",
    "Sales",
    "SEO",
    "Supply Chain",
    "Social Media Marketing",
    "Teaching",
    "Teamwork",
    "Time Management",
    "Training",
    "Unreal Engine",
    "Video Editing",
    "Video Editor",
    "Video Production",
    "2D Animation",
    "2D Animator",
    "3D Modeling",
    "3D Modeler",
    "3D Artist",
    "Game Development",
    "WordPress",
    "Customer Success",
    "Front Desk",
    "Guest Services",
    "Hospitality",
    "Penetration Testing",
    "SIEM",
    "SOC",
    "UI/UX Design",
    "Wireframing",
    "Prototyping",
]

KEYWORD_PROFILE_RULES = [
    {
        "role": "Flutter Developer",
        "domain": "Mobile App Development",
        "keywords": ("flutter", "dart", "firebase", "mobile app development"),
    },
    {
        "role": "Android Developer",
        "domain": "Mobile App Development",
        "keywords": ("android", "kotlin", "mobile ui"),
    },
    {
        "role": "iOS Developer",
        "domain": "Mobile App Development",
        "keywords": ("ios", "swift", "uikit", "xcode"),
    },
    {
        "role": "Product Manager",
        "domain": "Product Management",
        "keywords": ("product management", "roadmap", "user stories", "saas", "product strategy"),
    },
    {
        "role": "Business Analyst",
        "domain": "Business / Analysis",
        "keywords": ("business analysis", "requirements", "stakeholders", "documentation", "process mapping"),
    },
    {
        "role": "SEO Specialist",
        "domain": "Marketing / Growth",
        "keywords": ("seo", "keyword research", "on-page seo", "link building"),
    },
    {
        "role": "Social Media Manager",
        "domain": "Marketing / Growth",
        "keywords": ("social media management", "content strategy", "instagram ads"),
    },
    {
        "role": "3D Artist",
        "domain": "Creative Media / Video Production",
        "keywords": ("3d modeling", "blender", "texturing", "game assets"),
    },
    {
        "role": "Animator",
        "domain": "Creative Media / Video Production",
        "keywords": ("animation", "storyboarding", "character design"),
    },
    {
        "role": "Medical Billing Specialist",
        "domain": "Healthcare",
        "keywords": ("medical billing", "insurance claims", "icd coding"),
    },
    {
        "role": "Lab Technician",
        "domain": "Healthcare",
        "keywords": ("lab testing", "diagnostics", "blood samples"),
    },
    {
        "role": "Mechanical Engineer",
        "domain": "Engineering / Construction",
        "keywords": ("mechanical design", "solidworks", "manufacturing", "cad"),
    },
    {
        "role": "Restaurant Manager",
        "domain": "Hospitality",
        "keywords": ("restaurant management", "food safety", "staff scheduling"),
    },
    {
        "role": "Travel Consultant",
        "domain": "Hospitality",
        "keywords": ("travel consultant", "ticketing", "itinerary planning", "gds"),
    },
    {
        "role": "Warehouse Supervisor",
        "domain": "Logistics / Supply Chain",
        "keywords": ("warehouse operations", "dispatch", "inventory control", "erp"),
    },
    {
        "role": "Procurement Officer",
        "domain": "Logistics / Supply Chain",
        "keywords": ("procurement", "vendor management", "purchase orders", "negotiation"),
    },
    {
        "role": "Banking Officer",
        "domain": "Finance / Accounting",
        "keywords": ("banking operations", "kyc", "account opening"),
    },
    {
        "role": "Auditor",
        "domain": "Finance / Accounting",
        "keywords": ("auditing", "financial reporting", "ifrs", "risk assessment"),
    },
    {
        "role": "Machine Learning Engineer",
        "domain": "AI / Machine Learning",
        "keywords": (
            "machine learning",
            "deep learning",
            "tensorflow",
            "pytorch",
            "keras",
            "computer vision",
            "nlp",
            "model training",
        ),
    },
    {
        "role": "AI Engineer",
        "domain": "AI / Machine Learning",
        "keywords": ("ai engineer", "artificial intelligence", "llm", "generative ai", "prompt engineering"),
    },
    {
        "role": "Data Analyst",
        "domain": "Data / Analytics",
        "keywords": ("data analysis", "data analyst", "power bi", "tableau", "dashboard", "reporting"),
    },
    {
        "role": "Data Engineer",
        "domain": "Data / Analytics",
        "keywords": ("data engineering", "etl", "data pipeline", "warehouse", "spark"),
    },
    {
        "role": "Node.js Developer",
        "domain": "Backend Engineering",
        "keywords": ("node.js", "nodejs", "node", "express", "mongodb"),
    },
    {
        "role": "Java Backend Developer",
        "domain": "Backend Engineering",
        "keywords": ("java", "spring boot", "microservices", "mysql"),
    },
    {
        "role": "Backend Developer",
        "domain": "Backend Engineering",
        "keywords": ("backend", "api", "django", "flask", "fastapi", "node", "spring"),
    },
    {
        "role": "Frontend Developer",
        "domain": "Frontend Engineering",
        "keywords": ("frontend", "react", "vue", "angular", "javascript", "typescript"),
    },
    {
        "role": "Cloud Engineer",
        "domain": "DevOps / Cloud",
        "keywords": ("azure", "cloud infrastructure", "terraform", "monitoring"),
    },
    {
        "role": "DevOps Engineer",
        "domain": "DevOps / Cloud",
        "keywords": ("devops", "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd"),
    },
    {
        "role": "Network Security Engineer",
        "domain": "Cybersecurity",
        "keywords": ("network security", "firewalls", "vpn", "cisco"),
    },
    {
        "role": "Cybersecurity Analyst",
        "domain": "Cybersecurity",
        "keywords": ("cybersecurity", "cyber security", "soc", "siem", "penetration testing", "vulnerability"),
    },
    {
        "role": "Graphic Designer",
        "domain": "Design / Creative",
        "keywords": ("graphic design", "graphic designer", "photoshop", "illustrator", "branding"),
    },
    {
        "role": "UI/UX Designer",
        "domain": "Design / Product Design",
        "keywords": ("ui/ux", "ui ux", "ux design", "figma", "wireframing", "prototyping"),
    },
    {
        "role": "Digital Marketing Specialist",
        "domain": "Marketing / Growth",
        "keywords": ("digital marketing", "seo", "google ads", "social media", "campaign", "email marketing"),
    },
    {
        "role": "Content Writer",
        "domain": "Content / Writing",
        "keywords": ("content writing", "content writer", "copywriting", "blogging", "blog writer"),
    },
    {
        "role": "Sales Executive",
        "domain": "Sales / Business Development",
        "keywords": ("sales", "lead generation", "business development", "account executive", "negotiation"),
    },
    {
        "role": "Customer Support Representative",
        "domain": "Customer Support / Operations",
        "keywords": ("customer support", "customer service", "email support", "call center", "crm"),
    },
    {
        "role": "Project Manager",
        "domain": "Project Management",
        "keywords": ("project management", "project manager", "agile", "scrum", "jira", "kanban"),
    },
    {
        "role": "HR Officer",
        "domain": "HR / Recruitment",
        "keywords": ("hr", "human resources", "recruitment", "talent acquisition", "onboarding"),
    },
    {
        "role": "Accountant",
        "domain": "Finance / Accounting",
        "keywords": ("accounting", "accountant", "bookkeeping", "quickbooks", "financial reporting"),
    },
    {
        "role": "Legal Assistant",
        "domain": "Legal",
        "keywords": ("legal", "lawyer", "paralegal", "legal research", "contract"),
    },
    {
        "role": "Nurse",
        "domain": "Healthcare",
        "keywords": ("nursing", "nurse", "patient care", "clinical", "medical records"),
    },
    {
        "role": "Teacher",
        "domain": "Education / Training",
        "keywords": ("teaching", "teacher", "tutor", "curriculum", "classroom", "training"),
    },
    {
        "role": "Logistics Coordinator",
        "domain": "Logistics / Supply Chain",
        "keywords": ("logistics", "supply chain", "warehouse", "inventory", "procurement", "dispatch"),
    },
    {
        "role": "Civil Engineer",
        "domain": "Engineering / Construction",
        "keywords": ("civil engineering", "construction", "autocad", "site engineer"),
    },
    {
        "role": "Mechanical Engineer",
        "domain": "Engineering / Construction",
        "keywords": ("mechanical engineering", "mechanical engineer", "solidworks"),
    },
    {
        "role": "Electrical Engineer",
        "domain": "Engineering / Construction",
        "keywords": ("electrical engineering", "electrical engineer", "power systems"),
    },
    {
        "role": "Hotel Manager",
        "domain": "Hospitality",
        "keywords": ("hotel management", "hospitality", "guest services", "front desk", "restaurant"),
    },
]


@dataclass(frozen=True)
class UploadedFile:
    filename: str
    content: bytes


def extract_profile_fields(
    cv_file: UploadedFile | None,
    portfolio_text: str = "",
    portfolio_url: str = "",
) -> dict[str, str]:
    text_parts: list[str] = []
    if cv_file and cv_file.content:
        cv_text = _clean_text(extract_text_from_file(cv_file.filename, cv_file.content))
        if not cv_text and not portfolio_url.strip():
            raise ValueError(
                "This CV does not contain readable text. Please upload a text-based PDF, "
                "TXT, or DOCX file, or add a GitHub/portfolio link."
            )
        text_parts.append(cv_text)
    if portfolio_url.strip():
        text_parts.append(fetch_portfolio_text(portfolio_url.strip()))
    if portfolio_text.strip():
        text_parts.append(portfolio_text.strip())

    combined_text = _clean_text("\n".join(text_parts))
    if not combined_text:
        raise ValueError("Please upload a readable CV or enter a portfolio/GitHub link.")

    rule_fields = _extract_with_rules(combined_text)
    if cv_file and _should_try_pdf_ocr_fallback(cv_file.filename, rule_fields):
        ocr_text = _try_pdf_ocr_fallback(cv_file.content)
        if ocr_text and ocr_text not in combined_text:
            combined_text = _clean_text("\n".join([combined_text, ocr_text]))
            rule_fields = _extract_with_rules(combined_text)
    if llm_is_configured():
        llm_fields = _extract_with_llama(combined_text)
        if llm_fields:
            return _merge_rule_and_llm_fields(rule_fields, llm_fields)

    return rule_fields


def extract_keyword_profile_fields(keyword_text: str) -> dict[str, str]:
    skills = [
        item.strip()
        for item in re.split(r"[,\n]", keyword_text)
        if item.strip()
    ]
    if not skills:
        raise ValueError("Please add at least one job keyword or skill.")

    keyword_match = _keyword_profile_match(keyword_text)
    if keyword_match:
        desired_role = keyword_match["role"]
        domain = keyword_match["domain"]
    else:
        domain = infer_domain(keyword_text)
        desired_role = _fallback_role_from_domain_and_skills(domain, skills, "", "")
    if not desired_role:
        desired_role = f"{skills[0]} Professional"
    related_roles = _related_roles_for_profile(desired_role, skills)

    return _normalize_field_dict(
        {
            "desired_role": desired_role,
            "skills": ", ".join(skills),
            "experience_years": "",
            "seniority": "",
            "experience": "",
            "education": "",
            "projects": "",
            "location": "",
            "domain": domain,
            "query": desired_role,
            "alternate_roles": "\n".join(related_roles),
            "search_queries": "\n".join(related_roles),
            "confidence": "0.85" if keyword_match else ("0.75" if domain != "General Technology" else "0.55"),
        }
    )


def _keyword_profile_match(keyword_text: str) -> dict[str, str] | None:
    signal = keyword_text.lower()
    best_rule: dict[str, object] | None = None
    best_score = 0
    for rule in KEYWORD_PROFILE_RULES:
        score = _keyword_rule_score(signal, rule["keywords"])
        if score > best_score:
            best_rule = rule
            best_score = score

    if not best_rule or best_score < 2:
        return None
    return {
        "role": str(best_rule["role"]),
        "domain": str(best_rule["domain"]),
    }


def _keyword_rule_score(signal: str, keywords: tuple[str, ...]) -> int:
    score = 0
    for keyword in keywords:
        if re.search(r"(?<![a-z0-9])" + re.escape(keyword) + r"(?![a-z0-9])", signal):
            score += 3 if " " in keyword or "/" in keyword else 1
    return score


def _should_try_pdf_ocr_fallback(filename: str, fields: dict[str, str]) -> bool:
    if Path(filename).suffix.lower() != ".pdf":
        return False
    role = fields.get("desired_role", "")
    confidence = _confidence_value(fields.get("confidence", ""))
    domain = fields.get("domain", "")
    return _is_generic_role(role) or confidence < 0.55 or domain == "General Technology"


def _try_pdf_ocr_fallback(content: bytes) -> str:
    try:
        return _clean_text(_extract_scanned_pdf_text(content))
    except ValueError:
        return ""


def extract_text_from_file(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md", ".csv"}:
        return content.decode("utf-8", errors="ignore")
    if suffix == ".pdf":
        return _extract_pdf_text(content)
    if suffix == ".docx":
        return _extract_docx_text(content)
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        return _extract_image_text(content)
    raise ValueError(
        "Unsupported CV format. Please upload .txt, .pdf, .docx, .png, .jpg, or .jpeg."
    )


def fetch_portfolio_text(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Portfolio URL must start with http:// or https://.")
    _validate_public_hostname(parsed.hostname or "")

    if parsed.netloc.lower() in {"github.com", "www.github.com"}:
        github_text = _fetch_github_portfolio_text(parsed)
        if github_text:
            return github_text

    request = Request(
        url,
        headers={
            "Accept": "text/html, text/plain;q=0.9, */*;q=0.5",
            "User-Agent": "career-recommendation-agent/0.1",
        },
    )
    with urlopen(request, timeout=20) as response:
        content_type = response.headers.get("Content-Type", "")
        raw = response.read(1_000_000)

    text = raw.decode("utf-8", errors="ignore")
    if "html" in content_type.lower() or "<html" in text[:500].lower():
        text = _html_to_text(text)
    return _clean_text(text)


def _fetch_github_portfolio_text(parsed_url) -> str:
    parts = [part for part in parsed_url.path.strip("/").split("/") if part]
    if not parts:
        return ""

    username = parts[0]
    if len(parts) >= 2:
        repo = parts[1]
        repo_payload = _fetch_json(f"https://api.github.com/repos/{username}/{repo}")
        return _github_repo_to_text(repo_payload)

    repos = _fetch_json(
        f"https://api.github.com/users/{username}/repos?sort=updated&per_page=12"
    )
    if not isinstance(repos, list):
        return ""

    lines = [f"GitHub portfolio for {username}", "Projects"]
    for repo_payload in repos:
        lines.append(_github_repo_to_text(repo_payload))
    return _clean_text("\n".join(lines))


def _github_repo_to_text(repo_payload) -> str:
    if not isinstance(repo_payload, dict):
        return ""

    lines = [
        str(repo_payload.get("name") or ""),
        str(repo_payload.get("description") or ""),
        str(repo_payload.get("language") or ""),
    ]
    topics = repo_payload.get("topics")
    if isinstance(topics, list):
        lines.extend(str(topic) for topic in topics)

    languages_url = repo_payload.get("languages_url")
    if languages_url:
        languages = _fetch_json(str(languages_url))
        if isinstance(languages, dict):
            lines.extend(languages.keys())

    return "\n".join(line for line in lines if line)


def _fetch_json(url: str):
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "career-recommendation-agent/0.1",
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def _extract_pdf_text(content: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ValueError("PDF parsing needs pypdf installed.") from exc

    reader = PdfReader(io.BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = _clean_text("\n".join(pages))
    if text:
        return text
    return _extract_scanned_pdf_text(content)


def _extract_scanned_pdf_text(content: bytes) -> str:
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise ValueError(
            "This looks like a scanned/image-based PDF. OCR support needs pymupdf, "
            "pytesseract, and pillow installed."
        ) from exc

    tesseract_cmd = os.getenv("TESSERACT_CMD", "").strip()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError("Could not open scanned PDF for OCR.") from exc

    ocr_pages: list[str] = []
    for page in document:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        try:
            ocr_pages.append(pytesseract.image_to_string(image))
        except pytesseract.TesseractNotFoundError as exc:
            raise ValueError(
                "This scanned PDF needs OCR, but Tesseract is not installed. "
                "Install Tesseract OCR and restart the server."
            ) from exc

    text = _clean_text("\n".join(ocr_pages))
    if not text:
        raise ValueError("OCR could not read text from this scanned PDF.")
    return text


def _extract_image_text(content: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise ValueError(
            "Image CV OCR needs pytesseract and pillow installed."
        ) from exc

    tesseract_cmd = os.getenv("TESSERACT_CMD", "").strip()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        image = Image.open(io.BytesIO(content))
    except Exception as exc:
        raise ValueError("Could not open CV image file.") from exc

    try:
        text = pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError as exc:
        raise ValueError(
            "This image CV needs OCR, but Tesseract is not installed. "
            "Install Tesseract OCR and restart the server."
        ) from exc

    text = _clean_text(text)
    if not text:
        raise ValueError("OCR could not read text from this CV image.")
    return text


def _extract_docx_text(content: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            xml = archive.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise ValueError("Invalid DOCX file.") from exc

    root = ElementTree.fromstring(xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", namespace))
        if text.strip():
            paragraphs.append(text.strip())
    return "\n".join(paragraphs)


def _validate_public_hostname(hostname: str) -> None:
    try:
        addresses = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError("Portfolio URL host could not be resolved.") from exc

    for address in addresses:
        ip = ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("Portfolio URL must be a public website URL.")


def _html_to_text(html: str) -> str:
    parser = _ReadableHTMLParser()
    parser.feed(html)
    return parser.text


def _extract_with_llama(text: str) -> dict[str, str] | None:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_groq import ChatGroq

        llm = ChatGroq(temperature=0, max_tokens=1000)
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a universal career-profile extraction engine for every industry. "
                        "Analyze CV, resume, portfolio, or GitHub text from any field: technology, healthcare, "
                        "education, finance, legal, engineering, construction, creative media, hospitality, "
                        "logistics, HR, sales, marketing, administration, trades, and other professions. "
                        "Identify the candidate's most likely target role, industry/domain, seniority, "
                        "hard skills, transferable skills, experience, education, projects, and job-search queries. "
                        "Prefer the candidate's explicit title/designation when present. If the title is unclear, "
                        "infer it from repeated skills, experience headings, tools, and responsibilities. "
                        "Return only valid JSON with keys: desired_role, skills, experience_years, seniority, "
                        "experience, education, projects, location, domain, query, alternate_roles, "
                        "search_queries, confidence. Do not invent facts; use empty strings when unknown."
                    )
                ),
                HumanMessage(content=text[:12000]),
            ]
        )
        return _normalize_field_dict(json.loads(str(response.content)))
    except Exception:
        return None


def _merge_rule_and_llm_fields(rule_fields: dict[str, str], llm_fields: dict[str, str]) -> dict[str, str]:
    rule_role = rule_fields.get("desired_role", "").strip()
    llm_role = llm_fields.get("desired_role", "").strip()
    if not _is_valid_job_title(rule_role):
        rule_role = ""
    if not _is_valid_job_title(llm_role):
        llm_role = ""
    rule_confidence = _confidence_value(rule_fields.get("confidence", ""))
    llm_confidence = _confidence_value(llm_fields.get("confidence", ""))

    merged = dict(rule_fields)
    for key in ("experience", "education", "projects", "location", "experience_years", "seniority"):
        if not merged.get(key) and llm_fields.get(key):
            merged[key] = llm_fields[key]

    if llm_fields.get("skills"):
        merged["skills"] = _merge_csv_like_values(merged.get("skills", ""), llm_fields["skills"])

    should_use_llm_role = (
        _is_generic_role(rule_role)
        and not _is_generic_role(llm_role)
    ) or (
        not _is_generic_role(llm_role)
        and llm_confidence >= rule_confidence + 0.2
    )
    if should_use_llm_role:
        merged["desired_role"] = llm_role
    else:
        merged["desired_role"] = rule_role or llm_role
    if not _is_valid_job_title(merged.get("desired_role", "")):
        merged["desired_role"] = ""

    domain = infer_domain(
        merged.get("desired_role", ""),
        merged.get("skills", ""),
        merged.get("experience", ""),
        merged.get("projects", ""),
    )
    merged["domain"] = domain if domain != "General Technology" else (
        llm_fields.get("domain") or rule_fields.get("domain") or domain
    )
    merged["desired_role"] = _validated_or_fallback_role(
        merged.get("desired_role", ""),
        merged.get("domain", ""),
        _csv_to_list(merged.get("skills", "")),
        merged.get("experience", ""),
        merged.get("projects", ""),
    )

    merged["alternate_roles"] = _merge_line_values(
        "\n".join(_related_roles_for_profile(merged.get("desired_role", ""), _csv_to_list(merged.get("skills", "")))),
        rule_fields.get("alternate_roles", ""),
        llm_fields.get("alternate_roles", ""),
    )
    merged["search_queries"] = _merge_line_values(
        "\n".join(_related_roles_for_profile(merged.get("desired_role", ""), _csv_to_list(merged.get("skills", "")))),
        rule_fields.get("search_queries", ""),
        llm_fields.get("search_queries", ""),
    )
    merged["query"] = merged.get("desired_role") or first_non_empty(
        rule_fields.get("query", ""),
        llm_fields.get("query", ""),
    )
    merged["confidence"] = str(round(max(rule_confidence, llm_confidence), 2))
    return _normalize_field_dict(merged)


def _confidence_value(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if parsed > 1:
        parsed = parsed / 100
    return max(0.0, min(parsed, 1.0))


def _is_generic_role(role: str) -> bool:
    return role.strip() in GENERIC_ROLES


def _is_valid_job_title(role: str) -> bool:
    clean = re.sub(r"\s+", " ", role or "").strip()
    if _is_generic_role(clean):
        return False
    if not clean:
        return False
    words = re.findall(r"[a-zA-Z0-9+#/&-]+", clean)
    if len(words) >= 6:
        return False
    lowered = clean.lower()
    invalid_terms = {
        "architecture",
        "deployment",
        "monitoring",
        "networking",
        "workflow",
        "pipeline",
        "system",
        "application",
        "project",
        "platform",
        "solution",
    }
    role_terms = {
        "engineer",
        "developer",
        "scientist",
        "analyst",
        "manager",
        "specialist",
        "officer",
        "executive",
        "assistant",
        "designer",
        "editor",
        "modeler",
        "modeller",
        "artist",
        "teacher",
        "nurse",
        "doctor",
        "pharmacist",
        "accountant",
        "recruiter",
        "coordinator",
        "consultant",
        "architect",
        "administrator",
        "trainer",
        "tutor",
        "lawyer",
        "paralegal",
        "chef",
        "operator",
    }
    lowered_terms = set(re.findall(r"[a-z0-9+#/&-]+", lowered))
    has_invalid_term = any(term in lowered_terms for term in invalid_terms)
    has_role_term = any(term in lowered_terms for term in role_terms)
    if has_invalid_term and not has_role_term:
        return False
    if "," in clean and not any(term in lowered for term in role_terms):
        return False
    return True


def _validated_or_fallback_role(
    role: str,
    domain: str,
    skills: list[str],
    experience: str,
    projects: str,
) -> str:
    if _is_valid_job_title(role):
        return role
    fallback = _fallback_role_from_domain_and_skills(domain, skills, experience, projects)
    return fallback or "Professional"


def _fallback_role_from_domain_and_skills(
    domain: str,
    skills: list[str],
    experience: str,
    projects: str,
) -> str:
    signal = " ".join([domain, *skills, experience, projects]).lower()
    if domain == "AI / Machine Learning":
        if any(term in signal for term in ("machine learning", "tensorflow", "pytorch", "keras", "deep learning")):
            return "Machine Learning Engineer"
        return "AI Engineer"
    if domain == "Data / Analytics":
        if "data engineer" in signal or "etl" in signal:
            return "Data Engineer"
        return "Data Analyst"
    if domain == "Backend Engineering":
        if "node" in signal or "express" in signal or "mongodb" in signal:
            return "Node.js Developer"
        if "java" in signal or "spring boot" in signal or "microservices" in signal:
            return "Java Backend Developer"
        return "Backend Developer"
    if domain == "Frontend Engineering":
        return "Frontend Developer"
    if domain == "DevOps / Cloud":
        if "azure" in signal or "cloud infrastructure" in signal:
            return "Cloud Engineer"
        return "DevOps Engineer"
    if domain == "Cybersecurity":
        if "network security" in signal or "firewall" in signal or "vpn" in signal or "cisco" in signal:
            return "Network Security Engineer"
        return "Cybersecurity Analyst"
    if domain == "Design / Creative":
        return "Graphic Designer"
    if domain == "Design / Product Design":
        return "UI/UX Designer"
    if domain == "Creative Media / Video Production":
        if "3d" in signal or "blender" in signal:
            return "Video Editor & 3D Modeler" if "video" in signal else "3D Modeler"
        return "Video Editor"
    if domain == "Healthcare":
        if "pharmacy" in signal or "pharmacist" in signal:
            return "Pharmacist"
        if "doctor" in signal or "medical officer" in signal:
            return "Medical Officer"
        return "Nurse" if "nursing" in signal else "Healthcare Assistant"
    if domain == "Education / Training":
        return "Teacher"
    if domain == "Finance / Accounting":
        return "Accountant"
    if domain == "Marketing / Growth":
        return "Digital Marketing Specialist"
    if domain == "Content / Writing":
        return "Content Writer"
    if domain == "Sales / Business Development":
        return "Sales Executive"
    if domain == "Project Management":
        return "Project Manager"
    if domain == "HR / Recruitment":
        return "HR Officer"
    if domain == "Logistics / Supply Chain":
        return "Logistics Coordinator"
    if domain == "Legal":
        return "Legal Assistant"
    if domain == "Engineering / Construction":
        if "mechanical" in signal:
            return "Mechanical Engineer"
        if "electrical" in signal:
            return "Electrical Engineer"
        return "Civil Engineer" if "civil" in signal else "Site Engineer"
    if domain == "Customer Support / Operations":
        return "Customer Support Representative"
    if domain == "Hospitality":
        return "Chef" if "chef" in signal else "Hotel Manager"
    if "python" in signal:
        return "Python Developer"
    if "react" in signal or "javascript" in signal:
        return "Frontend Developer"
    return ""


def _merge_csv_like_values(*values: str) -> str:
    items: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in re.split(r"[,\n]", value or ""):
            clean = item.strip()
            key = clean.lower()
            if clean and key not in seen:
                seen.add(key)
                items.append(clean)
    return ", ".join(items)


def _merge_line_values(*values: str) -> str:
    items: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in str(value or "").splitlines():
            clean = item.strip()
            key = clean.lower()
            if clean and key not in seen:
                seen.add(key)
                items.append(clean)
    return "\n".join(items)


def _csv_to_list(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,\n]", value or "") if item.strip()]


def first_non_empty(*values: str) -> str:
    return next((value for value in values if str(value or "").strip()), "")


def _extract_with_rules(text: str) -> dict[str, str]:
    skills = _extract_skills(text)
    desired_role = _extract_role(text, skills)
    experience_years = _extract_years(text)
    sections = _extract_sections(text)
    if not experience_years:
        experience_years = _calculate_experience_years("\n".join(sections.get("experience", [])))

    experience = sections.get("experience") or _top_relevant_lines(
        text,
        ["experience", "developed", "built", "managed", "designed", "implemented", "worked"],
        limit=4,
    )
    education = sections.get("education") or _top_relevant_lines(
        text,
        ["university", "college", "bachelor", "master", "bs ", "ms ", "degree"],
        limit=3,
    )
    projects = sections.get("projects") or _top_relevant_lines(
        text,
        ["project", "portfolio", "github", "app", "system", "website"],
        limit=4,
    )
    domain = infer_domain(
        desired_role,
        ", ".join(skills),
        "\n".join(experience),
        "\n".join(projects),
    )
    desired_role = _validated_or_fallback_role(
        desired_role,
        domain,
        skills,
        "\n".join(experience),
        "\n".join(projects),
    )

    return _normalize_field_dict(
        {
            "desired_role": desired_role,
            "skills": ", ".join(skills),
            "experience_years": experience_years,
            "seniority": _seniority_from_years(experience_years),
            "experience": "\n".join(experience),
            "education": "\n".join(education),
            "projects": "\n".join(projects),
            "location": _extract_location(text) or "",
            "domain": domain,
            "query": desired_role or (skills[0] + " Developer" if skills else ""),
            "alternate_roles": "\n".join(_related_roles_for_profile(desired_role, skills)),
            "search_queries": "\n".join(_related_roles_for_profile(desired_role, skills)),
            "confidence": _extraction_confidence(desired_role, skills, experience),
        }
    )


def _extract_skills(text: str) -> list[str]:
    normalized = text.lower()
    found: list[str] = []
    skill_bank = TECH_SKILL_KEYWORDS + GENERAL_SKILL_KEYWORDS + [
        "TensorFlow",
        "PyTorch",
        "Pandas",
        "NumPy",
        "Keras",
        "LangChain",
        "FAISS",
    ]
    for skill in sorted(set(skill_bank)):
        pattern = r"(?<![a-z0-9.+#-])" + re.escape(skill.lower()) + r"(?![a-z0-9.+#-])"
        if re.search(pattern, normalized):
            found.append(skill)
    return found


def _extract_role(text: str, skills: list[str]) -> str:
    lowered = text.lower()
    top_section_role = _extract_role_from_top_section(text)
    if top_section_role:
        return top_section_role
    taxonomy_role = _extract_role_from_taxonomy(text, skills)
    if taxonomy_role:
        return taxonomy_role
    if "video editor" in lowered and ("3d modeler" in lowered or "3d modeller" in lowered):
        return "Video Editor & 3D Modeler"
    for role in COMMON_ROLES:
        if role.lower() in lowered:
            return role
    title = _extract_role_from_title_line(text)
    if title:
        return title
    if any(skill in skills for skill in ["Machine Learning", "TensorFlow", "PyTorch"]):
        return "Machine Learning Engineer"
    if any(skill in skills for skill in ["Digital Marketing", "SEO", "Social Media Marketing"]):
        return "Digital Marketing Specialist"
    if any(skill in skills for skill in ["Accounting", "Bookkeeping", "QuickBooks"]):
        return "Accountant"
    if any(skill in skills for skill in ["Recruitment", "HR", "Payroll"]):
        return "HR Officer"
    if any(skill in skills for skill in ["Sales", "CRM", "Negotiation"]):
        return "Sales Executive"
    if any(skill in skills for skill in ["Nursing", "Patient Care", "Medical Records"]):
        return "Nurse"
    if "Pharmacy" in skills:
        return "Pharmacist"
    if any(skill in skills for skill in ["Teaching", "Training"]):
        return "Teacher"
    if any(skill in skills for skill in ["Supply Chain", "Logistics"]):
        return "Logistics Coordinator"
    if "Legal Research" in skills:
        return "Legal Assistant"
    if "Civil Engineering" in skills:
        return "Civil Engineer"
    if "Mechanical Engineering" in skills:
        return "Mechanical Engineer"
    if "Electrical Engineering" in skills:
        return "Electrical Engineer"
    if any(skill in skills for skill in ["Video Editing", "Adobe Premiere Pro", "Adobe After Effects"]):
        if any(skill in skills for skill in ["3D Modeling", "Blender"]):
            return "Video Editor & 3D Modeler"
        return "Video Editor"
    if any(skill in skills for skill in ["3D Modeling", "Blender", "Unreal Engine"]):
        return "3D Modeler"
    if any(skill in skills for skill in ["Graphic Design", "Adobe Photoshop", "Canva", "Figma"]):
        return "Graphic Designer"
    if "Python" in skills:
        return "Python Developer"
    if "React" in skills or "JavaScript" in skills:
        return "Frontend Developer"
    return "Professional"


def _extract_role_from_top_section(text: str) -> str:
    lines = _candidate_profile_lines(text, limit=25)
    combined_top = " ".join(lines).lower()
    if "video editor" in combined_top and (
        "3d modeler" in combined_top
        or "3d modeller" in combined_top
        or "3d artist" in combined_top
    ):
        return "Video Editor & 3D Modeler"

    for line in lines:
        normalized_line = _normalize_role_line(line)
        if not normalized_line or _looks_like_name_or_noise(normalized_line):
            continue
        exact_role = _canonical_role(normalized_line)
        if exact_role:
            return exact_role
        if _contains_role_word(normalized_line):
            return _title_case_role(normalized_line)
    return ""


def _extract_role_from_taxonomy(text: str, skills: list[str]) -> str:
    signal = " ".join([text, *skills]).lower()
    top_signal = " ".join(_candidate_profile_lines(text, limit=30)).lower()
    best_role = ""
    best_score = 0
    for role, _domain, phrases in ROLE_TAXONOMY:
        score = 0
        role_lower = role.lower()
        if role_lower in top_signal:
            score += 5
        elif role_lower in signal:
            score += 3
        for phrase in phrases:
            if phrase in top_signal:
                score += 3
            elif phrase in signal:
                score += 1
        if score > best_score:
            best_role = role
            best_score = score
    return best_role if best_score >= 3 else ""


def _canonical_role(value: str) -> str:
    normalized = value.lower().strip()
    for role in COMMON_ROLES:
        if normalized == role.lower():
            return role
    for role, _domain, phrases in ROLE_TAXONOMY:
        if normalized == role.lower() or normalized in phrases:
            return role
    return ""


def _candidate_profile_lines(text: str, limit: int) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip(" -|•\t")
        if not line:
            continue
        if len(line) > 80:
            continue
        lowered = line.lower()
        if lowered in {
            "profile",
            "resume",
            "cv",
            "curriculum vitae",
            "education",
            "experience",
            "skills",
            "contact",
            "about me",
        }:
            continue
        if "@" in line or re.search(r"\+?\d[\d\s().-]{6,}", line):
            continue
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def _normalize_role_line(line: str) -> str:
    clean = re.sub(r"[_*/\\]+", " ", line)
    clean = re.sub(r"\s*&\s*", " & ", clean)
    clean = re.sub(r"\s+", " ", clean).strip(" :-|")
    return clean


def _looks_like_name_or_noise(line: str) -> bool:
    lowered = line.lower()
    if any(char.isdigit() for char in line):
        return True
    if lowered.startswith(("hi my name", "my name is", "i am ", "i'm ")):
        return True
    if lowered in {"english", "urdu", "hindi", "punjabi", "pashto"}:
        return True
    words = re.findall(r"[a-zA-Z]+", line)
    if 2 <= len(words) <= 4 and not _contains_role_word(line):
        uppercase_words = sum(1 for word in words if word[:1].isupper())
        if uppercase_words >= len(words) - 1:
            return True
    return False


def _contains_role_word(line: str) -> bool:
    role_words = {
        "engineer",
        "developer",
        "manager",
        "analyst",
        "specialist",
        "executive",
        "officer",
        "designer",
        "editor",
        "modeler",
        "modeller",
        "animator",
        "artist",
        "writer",
        "teacher",
        "accountant",
        "recruiter",
        "assistant",
        "consultant",
        "coordinator",
        "administrator",
        "nurse",
        "doctor",
        "pharmacist",
        "lawyer",
        "paralegal",
        "chef",
        "architect",
        "tutor",
        "trainer",
        "operator",
        "electrician",
        "plumber",
        "driver",
    }
    lowered = line.lower()
    return any(word in lowered for word in role_words)


def _title_case_role(line: str) -> str:
    acronyms = {"hr", "ui", "ux", "seo", "ai", "ml", "qa", "3d", "2d"}
    words = []
    for word in line.split():
        cleaned = word.strip()
        if cleaned.lower() in acronyms:
            words.append(cleaned.upper())
        elif cleaned == "&":
            words.append(cleaned)
        else:
            words.append(cleaned[:1].upper() + cleaned[1:].lower())
    return " ".join(words)


def _extract_role_from_title_line(text: str) -> str:
    lines = [line.strip(" -|") for line in text.splitlines() if line.strip()]
    role_words = {
        "engineer",
        "developer",
        "manager",
        "analyst",
        "specialist",
        "executive",
        "officer",
        "designer",
        "editor",
        "modeler",
        "modeller",
        "animator",
        "artist",
        "writer",
        "teacher",
        "accountant",
        "recruiter",
        "assistant",
        "consultant",
        "coordinator",
        "administrator",
        "nurse",
        "doctor",
        "pharmacist",
        "lawyer",
        "paralegal",
        "chef",
        "architect",
        "tutor",
        "trainer",
    }
    for line in lines[:12]:
        clean = re.sub(r"\s+", " ", line)
        if len(clean) > 60 or "@" in clean or re.search(r"\+?\d[\d\s-]{6,}", clean):
            continue
        if any(word in clean.lower() for word in role_words):
            return clean
    return ""


def _related_roles_for_profile(role: str, skills: list[str]) -> list[str]:
    signal = " ".join([role, *skills]).lower()
    groups: list[tuple[tuple[str, ...], list[str]]] = [
        (("machine learning", "tensorflow", "pytorch", "deep learning", "ai engineer"), [
            "Machine Learning Engineer",
            "AI Engineer",
            "Deep Learning Engineer",
            "Computer Vision Engineer",
        ]),
        (("data analyst", "analytics", "power bi", "tableau", "dashboard"), [
            "Data Analyst",
            "Business Intelligence Analyst",
            "Reporting Analyst",
            "Data Specialist",
        ]),
        (("cybersecurity", "cyber security", "soc", "siem", "penetration testing"), [
            "Cybersecurity Analyst",
            "SOC Analyst",
            "Security Analyst",
            "Information Security Analyst",
        ]),
        (("ui/ux", "ui ux", "wireframing", "prototyping", "figma"), [
            "UI/UX Designer",
            "Product Designer",
            "UX Designer",
            "UI Designer",
        ]),
        (("graphic design", "photoshop", "illustrator", "branding"), [
            "Graphic Designer",
            "Brand Designer",
            "Visual Designer",
            "Creative Designer",
        ]),
        (("video", "premiere", "after effects", "blender", "3d", "unreal"), [
            "Video Editor",
            "3D Modeler",
            "Motion Graphics Designer",
            "Video Production Specialist",
        ]),
        (("nursing", "patient care", "medical", "pharmacy"), [
            "Nurse",
            "Medical Assistant",
            "Healthcare Assistant",
            "Pharmacist",
        ]),
        (("teacher", "teaching", "tutor", "training"), [
            "Teacher",
            "Tutor",
            "Education Coordinator",
            "Trainer",
        ]),
        (("accounting", "finance", "quickbooks", "bookkeeping"), [
            "Accountant",
            "Finance Analyst",
            "Bookkeeper",
            "Accounts Assistant",
        ]),
        (("sales", "crm", "negotiation", "business development"), [
            "Sales Executive",
            "Business Development Executive",
            "Account Executive",
            "Sales Representative",
        ]),
        (("marketing", "seo", "social media", "google analytics"), [
            "Digital Marketing Specialist",
            "Marketing Specialist",
            "SEO Specialist",
            "Content Marketing Specialist",
        ]),
        (("content writing", "copywriting", "blogging", "blog writer"), [
            "Content Writer",
            "Copywriter",
            "Blog Writer",
            "Content Specialist",
        ]),
        (("project management", "project manager", "agile", "scrum", "jira"), [
            "Project Manager",
            "Project Coordinator",
            "Scrum Master",
            "Program Coordinator",
        ]),
        (("hr", "recruitment", "payroll", "talent"), [
            "HR Officer",
            "Recruiter",
            "Talent Acquisition Specialist",
            "HR Assistant",
        ]),
        (("logistics", "supply chain", "warehouse", "procurement"), [
            "Logistics Coordinator",
            "Supply Chain Specialist",
            "Warehouse Supervisor",
            "Procurement Officer",
        ]),
        (("hospitality", "hotel", "front desk", "guest services", "restaurant"), [
            "Hotel Manager",
            "Front Desk Officer",
            "Guest Services Representative",
            "Restaurant Manager",
        ]),
        (("legal", "paralegal", "contract", "compliance"), [
            "Legal Assistant",
            "Paralegal",
            "Legal Officer",
            "Compliance Officer",
        ]),
        (("civil engineering", "mechanical engineering", "electrical engineering", "autocad"), [
            "Civil Engineer",
            "Mechanical Engineer",
            "Electrical Engineer",
            "Site Engineer",
        ]),
        (("python", "machine learning", "react", "sql", "docker", "django", "flask"), [
            "Software Engineer",
            "Python Developer",
            "Data Analyst",
            "Backend Developer",
        ]),
    ]
    for keywords, roles in groups:
        if any(keyword in signal for keyword in keywords):
            return roles
    return [role] if role and role != "Professional" else []


def _extraction_confidence(role: str, skills: list[str], experience: list[str]) -> str:
    score = 0.35
    if role and role != "Professional":
        score += 0.3
    if len(skills) >= 3:
        score += 0.2
    elif skills:
        score += 0.1
    if experience:
        score += 0.15
    return str(round(min(score, 0.95), 2))


def _extract_years(text: str) -> str:
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\s+(?:of\s+)?experience",
        r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _calculate_experience_years(text: str) -> str:
    ranges = _extract_date_ranges(text)
    if not ranges:
        return ""

    merged = _merge_ranges(ranges)
    total_months = sum(_month_distance(start, end) for start, end in merged)
    if total_months <= 0:
        return ""

    years = total_months / 12
    if years < 1:
        return "0.5" if total_months >= 6 else "0"
    if years.is_integer():
        return str(int(years))
    return f"{years:.1f}".rstrip("0").rstrip(".")


def _extract_date_ranges(text: str) -> list[tuple[date, date]]:
    month_names = (
        "jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        "jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
        "nov(?:ember)?|dec(?:ember)?"
    )
    date_token = rf"(?:{month_names})\s+\d{{4}}|\d{{4}}"
    pattern = re.compile(
        rf"(?P<start>{date_token})\s*(?:-|–|—|to|until)\s*(?P<end>{date_token}|present|current|now)",
        re.IGNORECASE,
    )
    ranges: list[tuple[date, date]] = []
    for match in pattern.finditer(text):
        start = _parse_date_token(match.group("start"))
        end = _parse_date_token(match.group("end"), end=True)
        if start and end and end >= start:
            ranges.append((start, end))
    return ranges


def _parse_date_token(token: str, end: bool = False) -> date | None:
    token = token.strip().lower()
    if token in {"present", "current", "now"}:
        today = date.today()
        return date(today.year, today.month, 1)

    year_match = re.search(r"\d{4}", token)
    if not year_match:
        return None
    year = int(year_match.group(0))
    month = _month_number(token)
    if month is None:
        month = 12 if end else 1
    return date(year, month, 1)


def _month_number(token: str) -> int | None:
    month_lookup = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "sept": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    match = re.search(r"[a-z]+", token.lower())
    if not match:
        return None
    return month_lookup.get(match.group(0)[:4]) or month_lookup.get(match.group(0)[:3])


def _merge_ranges(ranges: list[tuple[date, date]]) -> list[tuple[date, date]]:
    sorted_ranges = sorted(ranges, key=lambda item: item[0])
    merged: list[tuple[date, date]] = []
    for start, end in sorted_ranges:
        if not merged:
            merged.append((start, end))
            continue
        last_start, last_end = merged[-1]
        if _month_distance(last_end, start) <= 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _month_distance(start: date, end: date) -> int:
    return ((end.year - start.year) * 12) + (end.month - start.month) + 1


def _seniority_from_years(years: str) -> str:
    if not years:
        return ""
    value = float(years)
    if value < 3:
        return "Junior"
    if value < 6:
        return "Mid"
    return "Senior"


def _extract_sections(text: str) -> dict[str, list[str]]:
    headings = {
        "experience": ["experience", "work experience", "employment"],
        "education": ["education"],
        "projects": ["projects", "portfolio"],
    }
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    current: str | None = None
    sections: dict[str, list[str]] = {key: [] for key in headings}
    for line in lines:
        normalized = re.sub(r"[^a-z ]", "", line.lower()).strip()
        matched = next(
            (key for key, names in headings.items() if normalized in names),
            None,
        )
        if matched:
            current = matched
            continue
        if current and len(sections[current]) < 6:
            sections[current].append(line)
    return {key: value for key, value in sections.items() if value}


def _top_relevant_lines(text: str, keywords: list[str], limit: int) -> list[str]:
    lines = [line.strip(" -•\t") for line in text.splitlines() if len(line.strip()) > 12]
    matches = [
        line
        for line in lines
        if any(keyword in line.lower() for keyword in keywords)
    ]
    return matches[:limit]


def _extract_location(text: str) -> str:
    match = re.search(r"(?:location|address)\s*[:\-]\s*([^\n\r]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _normalize_field_dict(data: dict) -> dict[str, str]:
    keys = [
        "desired_role",
        "skills",
        "experience_years",
        "seniority",
        "experience",
        "education",
        "projects",
        "location",
        "domain",
        "query",
        "alternate_roles",
        "search_queries",
        "confidence",
    ]
    normalized: dict[str, str] = {}
    for key in keys:
        value = data.get(key, "")
        if isinstance(value, list):
            value = "\n".join(str(item) for item in value if str(item).strip())
        normalized[key] = str(value or "").strip()
    return normalized


def _clean_text(text: str) -> str:
    text = unescape(text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


class _ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    @property
    def text(self) -> str:
        return "\n".join(part.strip() for part in self._parts if part.strip())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag in {"p", "div", "section", "article", "li", "br", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            cleaned = re.sub(r"\s+", " ", data).strip()
            if cleaned:
                self._parts.append(cleaned)
