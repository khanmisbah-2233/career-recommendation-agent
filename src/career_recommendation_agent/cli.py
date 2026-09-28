from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from career_recommendation_agent.agent import recommend_jobs
from career_recommendation_agent.job_sources import (
    JobSourceError,
    LiveJobSearchRequest,
    fetch_live_jobs,
)
from career_recommendation_agent.models import RecommendationInput


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank provided job listings against a supplied professional profile."
    )
    parser.add_argument("input_file", type=Path, help="Path to input JSON file.")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument(
        "--live-jobs",
        action="store_true",
        help="Fetch live jobs from the configured job provider before ranking.",
    )
    parser.add_argument(
        "--job-provider",
        default="serpapi",
        choices=["remotive", "serpapi", "jsearch"],
        help="Live job provider to use.",
    )
    parser.add_argument(
        "--query",
        help="Live job search query. Defaults to the profile desired_role.",
    )
    parser.add_argument(
        "--location",
        help="Live job search location. Defaults to profile preferences.location.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of live jobs to fetch.",
    )
    parser.add_argument(
        "--source-filter",
        choices=["linkedin", "indeed"],
        help="Keep only jobs whose apply source matches LinkedIn or Indeed.",
    )
    parser.add_argument(
        "--include-provided-jobs",
        action="store_true",
        help="Append live jobs to job_listings already present in the input file.",
    )
    parser.add_argument(
        "--live-min-score",
        type=int,
        default=25,
        help="Minimum score for live job recommendations.",
    )
    args = parser.parse_args()

    try:
        raw_payload = json.loads(args.input_file.read_text(encoding="utf-8"))
        payload = RecommendationInput.model_validate(raw_payload)
        if args.live_jobs:
            payload = _payload_with_live_jobs(payload, args)
        result = recommend_jobs(payload)
    except FileNotFoundError:
        print(json.dumps({"error": f"Input file not found: {args.input_file}"}), file=sys.stderr)
        raise SystemExit(1)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid JSON: {exc.msg}"}), file=sys.stderr)
        raise SystemExit(1)
    except ValidationError as exc:
        print(json.dumps({"error": "Input validation failed", "details": exc.errors()}), file=sys.stderr)
        raise SystemExit(1)
    except JobSourceError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(1)

    indent = 2 if args.pretty else None
    print(result.model_dump_json(indent=indent))


def _payload_with_live_jobs(payload: RecommendationInput, args: argparse.Namespace) -> RecommendationInput:
    profile = payload.user_profile
    query = args.query or profile.desired_role
    if not query:
        raise JobSourceError("Live job search needs --query or user_profile.desired_role.")

    location = args.location or profile.preferences.location
    live_jobs = fetch_live_jobs(
        args.job_provider,
        LiveJobSearchRequest(
            query=query,
            location=location,
            limit=max(1, args.limit),
            source_filter=args.source_filter,
        ),
    )
    job_listings = [*payload.job_listings, *live_jobs] if args.include_provided_jobs else live_jobs
    return payload.model_copy(
        update={
            "job_listings": job_listings,
            "minimum_match_score": max(0, min(100, args.live_min_score)),
            "fallback_recommendation_count": 5,
        }
    )


if __name__ == "__main__":
    main()
