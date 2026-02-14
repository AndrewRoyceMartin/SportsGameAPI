from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, List, Optional

import requests


class ApifyError(RuntimeError):
    pass


class ApifyAuthError(ApifyError):
    pass


class ApifyTransientError(ApifyError):
    pass


DEFAULT_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE = 0.6
DEFAULT_BACKOFF_CAP = 12.0


def _sleep_backoff(attempt: int, base: float, cap: float) -> None:
    delay = min(cap, base * (2 ** attempt))
    jitter = random.uniform(0, delay * 0.25)
    time.sleep(delay + jitter)


def run_actor_get_items(
    actor_id: str,
    actor_input: Dict[str, Any],
    limit: Optional[int] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    backoff_cap: float = DEFAULT_BACKOFF_CAP,
    session: Optional[requests.Session] = None,
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    token = os.getenv("APIFY_TOKEN")
    if not token:
        raise ApifyAuthError("Missing APIFY_TOKEN in environment")

    url = f"https://api.apify.com/v2/acts/{actor_id}/run-sync-get-dataset-items"
    params: Dict[str, Any] = {
        "token": token,
        "format": "json",
        "clean": "true",
    }
    if limit is not None:
        params["limit"] = limit

    sess = session or requests.Session()
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            if logger:
                logger.info(
                    f"Apify run actor (attempt {attempt+1}/{max_retries+1}): {actor_id}"
                )

            r = sess.post(url, params=params, json=actor_input, timeout=timeout)

            if r.status_code in (401, 403):
                raise ApifyAuthError(
                    f"Apify auth failed (HTTP {r.status_code}). Check APIFY_TOKEN."
                )

            if r.status_code == 400:
                body = r.text
                body = body[:800] + ("..." if len(body) > 800 else "")
                raise ApifyError(
                    f"Apify 400 Bad Request. Actor input likely invalid. Response: {body}"
                )

            if r.status_code == 429 or 500 <= r.status_code < 600:
                if attempt < max_retries:
                    if logger:
                        logger.warning(
                            f"Apify transient HTTP {r.status_code}; backing off then retrying."
                        )
                    _sleep_backoff(attempt, backoff_base, backoff_cap)
                    continue
                raise ApifyTransientError(
                    f"Apify transient HTTP {r.status_code} after retries."
                )

            r.raise_for_status()

            data = r.json()
            if not isinstance(data, list):
                raise ApifyError("Unexpected Apify response shape (expected list).")
            return data

        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
            if attempt < max_retries:
                if logger:
                    logger.warning(f"Apify network timeout/connection error; retrying: {e}")
                _sleep_backoff(attempt, backoff_base, backoff_cap)
                continue
            raise ApifyTransientError(
                f"Apify request failed after retries: {e}"
            ) from e

        except ApifyError:
            raise

        except Exception as e:
            last_exc = e
            raise ApifyError(f"Apify unexpected failure: {e}") from e

    raise ApifyError(f"Apify failed: {last_exc}")
