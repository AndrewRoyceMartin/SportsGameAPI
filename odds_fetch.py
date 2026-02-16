from __future__ import annotations

import logging
import os
from typing import List, Dict, Any, Optional, Tuple

from apify_runner import run_actor_get_items, ApifyAuthError, ApifyTransientError, ApifyError
from sportsbet_odds import is_sportsbet_league, parse_sportsbet_items, SPORTSBET_LEAGUES

logger = logging.getLogger(__name__)

_last_fetch_errors: List[Tuple[str, str]] = []
_last_fatal_error: Optional[str] = None
_last_odds_source: Optional[str] = None
_last_raw_count: int = 0
_last_provider_detail: Optional[str] = None

SPORTSBET_ACTOR_ID = "lexis-solutions~sportsbet-com-au-scraper"


def _build_harvest_input(league_key: str, sportsbook: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"league": league_key}
    if sportsbook:
        payload["sportsbook"] = sportsbook
    return payload


def _build_sportsbet_input(league: str) -> Dict[str, Any]:
    config = SPORTSBET_LEAGUES[league]
    return {"startUrls": [{"url": config["url"]}]}


def _run_apify(actor_id: str, actor_input: Dict[str, Any], timeout: int) -> Optional[List[Dict[str, Any]]]:
    global _last_fetch_errors, _last_fatal_error, _last_raw_count
    try:
        return run_actor_get_items(actor_id, actor_input, timeout=timeout)
    except ApifyTransientError as exc:
        reason = str(exc)
        logger.warning("Odds fetch transient error: %s", reason)
        _last_fetch_errors.append(("apify", reason))
        _last_raw_count = -1
        return None
    except (ApifyAuthError, ApifyError) as exc:
        reason = str(exc)
        logger.error("Odds fetch fatal error: %s", reason)
        _last_fetch_errors.append(("apify", reason))
        _last_fatal_error = reason
        _last_raw_count = -1
        return None
    except Exception as exc:
        reason = str(exc)
        logger.warning("Odds fetch unexpected error: %s", reason)
        _last_fetch_errors.append(("apify", reason))
        _last_raw_count = -1
        return None


def fetch_odds_for_window(
    harvest_league: str,
    lookahead_days: int,
    sportsbook: Optional[str] = None,
    actor_id: str = "harvest~sportsbook-odds-scraper",
    timeout: int = 120,
    league_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    global _last_fetch_errors, _last_fatal_error, _last_odds_source, _last_raw_count, _last_provider_detail
    _last_fetch_errors = []
    _last_fatal_error = None
    _last_odds_source = None
    _last_raw_count = 0
    _last_provider_detail = None

    effective_league = league_label or harvest_league
    has_token = bool(os.getenv("APIFY_TOKEN"))

    logger.info("[ODDS] harvest_key=%s, effective_league=%s, token_present=%s",
                harvest_league, effective_league, has_token)

    if is_sportsbet_league(effective_league):
        _last_odds_source = "Sportsbet (AU)"
        _last_provider_detail = f"actor={SPORTSBET_ACTOR_ID}, league={effective_league}"
        logger.info("[ODDS] Routing to Sportsbet for %s", effective_league)
        sb_input = _build_sportsbet_input(effective_league)
        items = _run_apify(SPORTSBET_ACTOR_ID, sb_input, timeout)
        if items is None:
            logger.warning("[ODDS] Sportsbet fetch returned None (error)")
            return []
        _last_raw_count = len(items)
        logger.info("[ODDS] Sportsbet raw items=%d", len(items))
        parsed = parse_sportsbet_items(items, effective_league)
        logger.info("[ODDS] Sportsbet parsed events=%d", len(parsed))
        return parsed

    _last_odds_source = "Harvest"
    _last_provider_detail = f"actor={actor_id}, input={{league: {harvest_league}}}"
    logger.info("[ODDS] Routing to Harvest for %s (key=%s)", effective_league, harvest_league)
    actor_input = _build_harvest_input(harvest_league, sportsbook)
    items = _run_apify(actor_id, actor_input, timeout)
    if items is None:
        logger.warning("[ODDS] Harvest fetch returned None (error)")
        return []

    _last_raw_count = len(items)
    logger.info("[ODDS] Harvest raw items=%d", len(items))

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

    logger.info("[ODDS] After dedup: %d events (removed %d dupes)",
                len(all_items), len(items) - len(all_items))
    return all_items


def get_fetch_errors() -> List[Tuple[str, str]]:
    return list(_last_fetch_errors)


def get_fatal_error() -> Optional[str]:
    return _last_fatal_error


def get_odds_source() -> Optional[str]:
    return _last_odds_source


def get_raw_count() -> int:
    return _last_raw_count


def get_provider_detail() -> Optional[str]:
    return _last_provider_detail


def get_odds_provider_name(league_label: str) -> str:
    if is_sportsbet_league(league_label):
        return "Sportsbet (AU)"
    return "Harvest"


def probe_odds_provider(
    league_label: str,
    harvest_league_key: str,
    lookahead_days: int = 7,
    timeout: int = 60,
) -> Dict[str, Any]:
    import os
    has_token = bool(os.getenv("APIFY_TOKEN"))
    provider = get_odds_provider_name(league_label)

    if not has_token:
        return {
            "provider": provider,
            "ok": False,
            "items": 0,
            "error": "APIFY_TOKEN not set",
        }

    try:
        if is_sportsbet_league(league_label):
            actor_id = SPORTSBET_ACTOR_ID
            actor_input = _build_sportsbet_input(league_label)
        else:
            actor_id = "harvest~sportsbook-odds-scraper"
            actor_input = _build_harvest_input(harvest_league_key)

        items = run_actor_get_items(actor_id, actor_input, timeout=timeout)
        count = len(items) if items else 0
        return {
            "provider": provider,
            "ok": count > 0,
            "items": count,
            "error": "",
        }
    except Exception as exc:
        return {
            "provider": provider,
            "ok": False,
            "items": 0,
            "error": str(exc)[:120],
        }
