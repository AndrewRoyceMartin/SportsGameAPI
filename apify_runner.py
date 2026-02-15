from __future__ import annotations

import os
import re
import logging
from typing import Any, Dict, List, Optional

from apify_client import ApifyClient as _ApifyClient


def _redact(msg: str) -> str:
    return re.sub(r"token=[A-Za-z0-9_\-]+", "token=***", str(msg))


class ApifyError(RuntimeError):
    pass


class ApifyAuthError(ApifyError):
    pass


class ApifyTransientError(ApifyError):
    pass


logger = logging.getLogger(__name__)


def run_actor_get_items(
    actor_id: str,
    actor_input: Dict[str, Any],
    limit: Optional[int] = None,
    timeout: int = 120,
    max_retries: int = 3,
    backoff_base: float = 0.6,
    backoff_cap: float = 12.0,
    session: Any = None,
    logger: Any = None,
) -> List[Dict[str, Any]]:
    token = os.getenv("APIFY_TOKEN")
    if not token:
        raise ApifyAuthError("Missing APIFY_TOKEN in environment")

    client = _ApifyClient(token)

    try:
        run = client.actor(actor_id).call(
            run_input=actor_input,
            timeout_secs=timeout,
        )
    except Exception as e:
        msg = _redact(str(e))
        if "401" in msg or "403" in msg:
            raise ApifyAuthError(f"Apify auth/access error for {actor_id}: {msg}") from e
        raise ApifyError(f"Apify actor run failed for {actor_id}: {msg}") from e

    if not run:
        raise ApifyError(f"Apify actor run returned no result for {actor_id}")

    dataset_id = run.get("defaultDatasetId")
    if not dataset_id:
        raise ApifyError(f"No dataset ID in Apify run result for {actor_id}")

    try:
        items = list(client.dataset(dataset_id).iterate_items())
    except Exception as e:
        raise ApifyError(f"Failed to fetch dataset items for {actor_id}: {_redact(str(e))}") from e

    return items
