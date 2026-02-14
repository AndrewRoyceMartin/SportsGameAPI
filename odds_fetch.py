from __future__ import annotations

import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

from apify_client import run_actor_get_items

logger = logging.getLogger(__name__)

_last_fetch_errors: List[Tuple[str, str]] = []


def fetch_odds_for_window(
    harvest_league: str,
    lookahead_days: int,
    sportsbook: Optional[str] = None,
    actor_id: str = "harvest~sportsbook-odds-scraper",
    timeout: int = 120,
) -> List[Dict[str, Any]]:
    global _last_fetch_errors
    _last_fetch_errors = []

    all_items: List[Dict[str, Any]] = []
    seen: set = set()

    start = datetime.utcnow().date()
    days = max(1, lookahead_days)
    for i in range(days):
        d = start + timedelta(days=i)
        date_str = d.strftime("%Y-%m-%d")
        actor_input: Dict[str, Any] = {
            "league": harvest_league,
            "date": date_str,
        }
        if sportsbook:
            actor_input["sportsbook"] = sportsbook

        try:
            items = run_actor_get_items(actor_id, actor_input, timeout=timeout)
        except Exception as exc:
            reason = str(exc)
            logger.warning("Odds fetch skipped %s: %s", date_str, reason)
            _last_fetch_errors.append((date_str, reason))
            if "429" in reason or "500" in reason or "502" in reason or "503" in reason:
                time.sleep(2)
            continue

        for g in items:
            home = (g.get("homeTeam") or {}).get("mediumName", "")
            away = (g.get("awayTeam") or {}).get("mediumName", "")
            t = g.get("scheduledTime", "")
            key = (t, home, away)
            if key in seen:
                continue
            seen.add(key)
            all_items.append(g)

        if i < days - 1:
            time.sleep(0.5)

    return all_items


def get_fetch_errors() -> List[Tuple[str, str]]:
    return list(_last_fetch_errors)
