from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from apify_client import run_actor_get_items


def fetch_odds_for_window(
    harvest_league: str,
    lookahead_days: int,
    sportsbook: Optional[str] = None,
    actor_id: str = "harvest~sportsbook-odds-scraper",
    timeout: int = 120,
) -> List[Dict[str, Any]]:
    all_items: List[Dict[str, Any]] = []
    seen: set = set()

    start = datetime.utcnow().date()
    for i in range(max(1, lookahead_days)):
        d = start + timedelta(days=i)
        actor_input: Dict[str, Any] = {
            "league": harvest_league,
            "date": d.strftime("%Y-%m-%d"),
        }
        if sportsbook:
            actor_input["sportsbook"] = sportsbook

        try:
            items = run_actor_get_items(actor_id, actor_input, timeout=timeout)
        except Exception:
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

    return all_items
