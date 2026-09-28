from __future__ import annotations

import re


DOMAIN_KEYWORDS = {
    "AI / Machine Learning": {
        "ai",
        "artificial intelligence",
        "machine learning",
        "ml",
        "deep learning",
        "nlp",
        "computer vision",
        "llm",
        "predictive model",
        "tensorflow",
        "pytorch",
    },
    "Data / Analytics": {
        "data",
        "analytics",
        "sql",
        "dashboard",
        "bi",
        "etl",
        "warehouse",
        "reporting",
    },
    "Backend Engineering": {
        "backend",
        "api",
        "microservices",
        "django",
        "fastapi",
        "node",
        "spring",
        "server",
    },
    "Frontend Engineering": {
        "frontend",
        "react",
        "vue",
        "angular",
        "typescript",
        "javascript",
        "ui",
        "ux",
    },
    "Design / Product Design": {
        "ui/ux",
        "ui ux",
        "ux design",
        "ui design",
        "product design",
        "figma",
        "wireframing",
        "prototyping",
    },
    "Mobile App Development": {
        "mobile app",
        "mobile app development",
        "flutter",
        "dart",
        "android",
        "kotlin",
        "ios",
        "swift",
        "uikit",
        "xcode",
        "firebase",
    },
    "Product Management": {
        "product management",
        "product manager",
        "roadmap",
        "user stories",
        "saas",
        "product strategy",
        "backlog",
    },
    "Business / Analysis": {
        "business analysis",
        "business analyst",
        "requirements",
        "stakeholders",
        "documentation",
        "process mapping",
    },
    "DevOps / Cloud": {
        "devops",
        "cloud",
        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes",
        "ci/cd",
        "terraform",
    },
    "Cybersecurity": {
        "security",
        "cybersecurity",
        "soc",
        "siem",
        "penetration",
        "vulnerability",
    },
    "Design / Creative": {
        "graphic design",
        "graphic designer",
        "photoshop",
        "illustrator",
        "branding",
        "visual design",
    },
    "Creative Media / Video Production": {
        "video editor",
        "video editing",
        "video production",
        "adobe premiere",
        "premiere pro",
        "after effects",
        "motion graphics",
        "2d animation",
        "3d modeler",
        "3d modeller",
        "3d modeling",
        "3d artist",
        "blender",
        "unreal engine",
        "commercial video",
        "youtube channel",
    },
    "Marketing / Growth": {
        "marketing",
        "digital marketing",
        "seo",
        "social media",
        "content marketing",
        "google analytics",
        "campaign",
        "branding",
        "email marketing",
    },
    "Content / Writing": {
        "content writing",
        "content writer",
        "copywriting",
        "blogging",
        "blog writer",
        "technical writing",
    },
    "Sales / Business Development": {
        "sales",
        "business development",
        "account executive",
        "lead generation",
        "crm",
        "negotiation",
        "client relationship",
    },
    "Finance / Accounting": {
        "accounting",
        "finance",
        "bookkeeping",
        "quickbooks",
        "financial reporting",
        "payroll",
        "budgeting",
        "audit",
    },
    "HR / Recruitment": {
        "hr",
        "human resources",
        "recruitment",
        "talent acquisition",
        "payroll",
        "employee relations",
        "onboarding",
    },
    "Education / Training": {
        "teacher",
        "teaching",
        "tutor",
        "education",
        "curriculum",
        "training",
        "classroom",
        "instructor",
    },
    "Healthcare": {
        "nurse",
        "nursing",
        "doctor",
        "patient care",
        "medical",
        "clinical",
        "pharmacy",
        "pharmacist",
        "healthcare",
    },
    "Engineering / Construction": {
        "civil engineering",
        "mechanical engineering",
        "electrical engineering",
        "construction",
        "autocad",
        "site engineer",
        "architecture",
        "architect",
    },
    "Customer Support / Operations": {
        "customer support",
        "customer service",
        "customer success",
        "email support",
        "operations",
        "administration",
        "data entry",
        "call center",
        "virtual assistant",
    },
    "Logistics / Supply Chain": {
        "logistics",
        "supply chain",
        "warehouse",
        "inventory",
        "procurement",
        "shipping",
        "dispatch",
    },
    "Legal": {
        "legal",
        "lawyer",
        "paralegal",
        "legal assistant",
        "legal research",
        "contract",
        "compliance",
    },
    "Hospitality": {
        "hotel",
        "hotel management",
        "restaurant",
        "hospitality",
        "chef",
        "food service",
        "front desk",
        "guest service",
        "guest services",
    },
    "Project Management": {
        "project management",
        "project manager",
        "agile",
        "scrum",
        "jira",
        "kanban",
        "pmp",
    },
}

DOMAIN_FAMILIES = (
    {
        "AI / Machine Learning",
        "Data / Analytics",
        "Backend Engineering",
        "Mobile App Development",
        "DevOps / Cloud",
    },
    {
        "Product Management",
        "Business / Analysis",
        "Project Management",
    },
    {
        "Creative Media / Video Production",
        "Design / Creative",
        "Design / Product Design",
        "Marketing / Growth",
        "Content / Writing",
    },
    {
        "Sales / Business Development",
        "Marketing / Growth",
        "Customer Support / Operations",
    },
    {
        "Finance / Accounting",
        "HR / Recruitment",
        "Legal",
    },
    {
        "Healthcare",
        "Education / Training",
    },
    {
        "Engineering / Construction",
        "Logistics / Supply Chain",
        "Project Management",
    },
)

SENIORITY_ORDER = {
    "Intern": 0,
    "Junior": 1,
    "Mid": 2,
    "Senior": 3,
    "Lead": 4,
}


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.strip().lower())


def normalize_skill(value: str) -> str:
    value = normalize_text(value)
    aliases = {
        "js": "javascript",
        "ts": "typescript",
        "py": "python",
        "postgres": "postgresql",
        "postgresql": "postgresql",
        "sklearn": "scikit-learn",
        "ci cd": "ci/cd",
    }
    return aliases.get(value, value)


def unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        key = normalize_skill(cleaned)
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def infer_experience_level(years: float | None, explicit: str | None = None) -> str:
    if explicit:
        normalized = normalize_text(explicit)
        if "intern" in normalized:
            return "Intern"
        if "junior" in normalized or "entry" in normalized:
            return "Junior"
        if "senior" in normalized or "sr" in normalized:
            return "Senior"
        if "lead" in normalized or "principal" in normalized or "staff" in normalized:
            return "Lead"
        if "mid" in normalized or "intermediate" in normalized:
            return "Mid"

    if years is None:
        return "Mid"
    if years < 1:
        return "Intern"
    if years < 3:
        return "Junior"
    if years < 6:
        return "Mid"
    if years < 10:
        return "Senior"
    return "Lead"


def infer_domain(*texts: str | None) -> str:
    combined = normalize_text(" ".join(text for text in texts if text))
    explicit_domain = _explicit_domain_from_role(combined)
    if explicit_domain:
        return explicit_domain

    best_domain = "General Technology"
    best_hits = 0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if _contains_keyword(combined, keyword))
        if hits > best_hits:
            best_domain = domain
            best_hits = hits
    return best_domain


def domains_are_compatible(left: str, right: str) -> bool:
    if left == right:
        return True
    if not left or not right or "General Technology" in {left, right}:
        return False
    return any({left, right}.issubset(family) for family in DOMAIN_FAMILIES)


def _contains_keyword(text: str, keyword: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(keyword) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _explicit_domain_from_role(combined: str) -> str | None:
    explicit_rules = [
        (
            "AI / Machine Learning",
            (
                "machine learning engineer",
                "ml engineer",
                "ai engineer",
                "artificial intelligence engineer",
                "deep learning engineer",
                "computer vision engineer",
                "nlp engineer",
            ),
        ),
        (
            "Data / Analytics",
            (
                "data analyst",
                "business intelligence analyst",
                "bi analyst",
                "reporting analyst",
                "data engineer",
            ),
        ),
        (
            "Backend Engineering",
            ("backend developer", "backend engineer", "api developer"),
        ),
        (
            "Frontend Engineering",
            ("frontend developer", "frontend engineer", "react developer"),
        ),
        (
            "Design / Product Design",
            ("ui/ux designer", "ux designer", "ui designer", "product designer"),
        ),
        (
            "Design / Creative",
            ("graphic designer", "brand designer", "visual designer"),
        ),
        (
            "Project Management",
            ("project manager", "scrum master", "project coordinator"),
        ),
    ]
    for domain, phrases in explicit_rules:
        if any(phrase in combined for phrase in phrases):
            return domain
    return None


def token_overlap_score(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9+#./-]+", normalize_text(left)))
    right_tokens = set(re.findall(r"[a-z0-9+#./-]+", normalize_text(right)))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(right_tokens)
