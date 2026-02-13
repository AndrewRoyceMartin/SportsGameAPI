from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def _get_token() -> str:
    token = os.getenv("APIFY_TOKEN", "")
    if not token:
        raise RuntimeError(
            "Missing APIFY_TOKEN in Replit Secrets. "
            "Add it via Tools -> Secrets."
        )
    return token


def run_actor(
    actor_id: str,
    run_input: Dict[str, Any],
    timeout_secs: int = 300,
    memory_mbytes: Optional[int] = None,
    max_items: Optional[int] = None,
) -> List[Dict[str, Any]]:
    from apify_client import ApifyClient

    client = ApifyClient(_get_token())
    call_kwargs: Dict[str, Any] = {
        "run_input": run_input,
        "timeout_secs": timeout_secs,
    }
    if memory_mbytes:
        call_kwargs["memory_mbytes"] = memory_mbytes

    run_info = client.actor(actor_id).call(**call_kwargs)
    if run_info is None:
        raise RuntimeError(f"Actor {actor_id} run failed (returned None).")

    dataset_id = run_info["defaultDatasetId"]
    return fetch_dataset(dataset_id, max_items=max_items)


def fetch_dataset(
    dataset_id: str,
    max_items: Optional[int] = None,
) -> List[Dict[str, Any]]:
    from apify_client import ApifyClient

    client = ApifyClient(_get_token())
    kwargs: Dict[str, Any] = {}
    if max_items:
        kwargs["limit"] = max_items

    result = client.dataset(dataset_id).list_items(**kwargs)
    return result.items
