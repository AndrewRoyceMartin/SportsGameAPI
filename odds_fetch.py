from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Tuple

from apify_client import run_actor_get_items, ApifyAuthError, ApifyTransientError, ApifyError

logger = logging.getLogger(__name__)

_last_fetch_errors: List[Tuple[str, str]] = []
_last_fatal_error: Optional[str] = None


def _build_apify_input(league_key: str, sportsbook: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"league": league_key}
    if sportsbook:
        payload["sportsbook"] = sportsbook
    return payload


def fetch_odds_for_window(
    harvest_league: str,
    lookahead_days: int,
    sportsbook: Optional[str] = None,
    actor_id: str = "harvest~sportsbook-odds-scraper",
    timeout: int = 120,
) -> List[Dict[str, Any]]:
    global _last_fetch_errors, _last_fatal_error
    _last_fetch_errors = []
    _last_fatal_error = None

    all_items: List[Dict[str, Any]] = []
    seen: set = set()

    actor_input = _build_apify_input(harvest_league, sportsbook)

    try:
        items = run_actor_get_items(actor_id, actor_input, timeout=timeout)
    except ApifyTransientError as exc:
        reason = str(exc)
        logger.warning("Odds fetch transient error: %s", reason)
        _last_fetch_errors.append(("apify", reason))
        return []
    except (ApifyAuthError, ApifyError) as exc:
        reason = str(exc)
        logger.error("Odds fetch fatal error: %s", reason)
        _last_fetch_errors.append(("apify", reason))
        _last_fatal_error = reason
        return []
    except Exception as exc:
        reason = str(exc)
        logger.warning("Odds fetch unexpected error: %s", reason)
        _last_fetch_errors.append(("apify", reason))
        return []

    for g in items:
        home = (g.get("homeTeam") or {}).get("mediumName", "")
        away = (g.get("awayTeam") or {}).get("mediumName", "")
        t = g.get("scheduledTime", "")
        key = (t, home, away)
        if key in seen:
            continue
        seen.add(key)
        all_items.append(g)

    return all_items


def get_fetch_errors() -> List[Tuple[str, str]]:
    return list(_last_fetch_errors)


def get_fatal_error() -> Optional[str]:
    return _last_fatal_error
