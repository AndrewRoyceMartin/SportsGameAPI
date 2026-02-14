import os
import requests
from typing import Any, Dict, List, Optional

APIFY_TOKEN = os.getenv("APIFY_TOKEN")


def run_actor_get_items(
    actor_id: str,
    actor_input: Dict[str, Any],
    limit: Optional[int] = None,
    timeout: int = 600,
) -> List[Dict[str, Any]]:
    if not APIFY_TOKEN:
        raise RuntimeError("Missing APIFY_TOKEN in environment")

    url = f"https://api.apify.com/v2/acts/{actor_id}/run-sync-get-dataset-items"
    params: Dict[str, Any] = {
        "token": APIFY_TOKEN,
        "format": "json",
        "clean": "true",
    }
    if limit is not None:
        params["limit"] = limit

    r = requests.post(url, params=params, json=actor_input, timeout=timeout)
    r.raise_for_status()
    return r.json()
