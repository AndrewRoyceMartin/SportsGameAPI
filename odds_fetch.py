from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Tuple

from apify_client import run_actor_get_items, ApifyAuthError, ApifyTransientError, ApifyError
from sportsbet_odds import is_sportsbet_league, parse_sportsbet_items, SPORTSBET_LEAGUES

logger = logging.getLogger(__name__)

_last_fetch_errors: List[Tuple[str, str]] = []
_last_fatal_error: Optional[str] = None
_last_odds_source: Optional[str] = None

SPORTSBET_ACTOR_ID = "canadesk~sportsbet-scraper"


def _build_harvest_input(league_key: str, sportsbook: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"league": league_key}
    if sportsbook:
        payload["sportsbook"] = sportsbook
    return payload


def _build_sportsbet_input(league: str) -> Dict[str, Any]:
    config = SPORTSBET_LEAGUES[league]
    return {"urls": [config["url"]]}


def _run_apify(actor_id: str, actor_input: Dict[str, Any], timeout: int) -> Optional[List[Dict[str, Any]]]:
    global _last_fetch_errors, _last_fatal_error
    try:
        return run_actor_get_items(actor_id, actor_input, timeout=timeout)
    except ApifyTransientError as exc:
        reason = str(exc)
        logger.warning("Odds fetch transient error: %s", reason)
        _last_fetch_errors.append(("apify", reason))
        return None
    except (ApifyAuthError, ApifyError) as exc:
        reason = str(exc)
        logger.error("Odds fetch fatal error: %s", reason)
        _last_fetch_errors.append(("apify", reason))
        _last_fatal_error = reason
        return None
    except Exception as exc:
        reason = str(exc)
        logger.warning("Odds fetch unexpected error: %s", reason)
        _last_fetch_errors.append(("apify", reason))
        return None


def fetch_odds_for_window(
    harvest_league: str,
    lookahead_days: int,
    sportsbook: Optional[str] = None,
    actor_id: str = "harvest~sportsbook-odds-scraper",
    timeout: int = 120,
    league_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    global _last_fetch_errors, _last_fatal_error, _last_odds_source
    _last_fetch_errors = []
    _last_fatal_error = None
    _last_odds_source = None

    effective_league = league_label or harvest_league

    if is_sportsbet_league(effective_league):
        _last_odds_source = "Sportsbet (AU)"
        sb_input = _build_sportsbet_input(effective_league)
        items = _run_apify(SPORTSBET_ACTOR_ID, sb_input, timeout)
        if items is None:
            return []
        return parse_sportsbet_items(items, effective_league)

    _last_odds_source = "Harvest"
    actor_input = _build_harvest_input(harvest_league, sportsbook)
    items = _run_apify(actor_id, actor_input, timeout)
    if items is None:
        return []

    all_items: List[Dict[str, Any]] = []
    seen: set = set()
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


def get_odds_source() -> Optional[str]:
    return _last_odds_source
