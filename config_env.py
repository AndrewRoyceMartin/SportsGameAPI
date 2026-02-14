from __future__ import annotations

import os
from typing import List, Tuple

REQUIRED = [
    "APIFY_TOKEN",
]

STALE = [
    "SPORTS_API_KEY",
    "SPORTS_API_BASE_URL",
]


def get_env_report() -> Tuple[List[str], List[str]]:
    missing = [k for k in REQUIRED if not os.getenv(k)]
    stale_present = [k for k in STALE if os.getenv(k)]
    return missing, stale_present


def assert_required_env() -> List[str]:
    missing, stale_present = get_env_report()
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )
    return stale_present
