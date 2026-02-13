from __future__ import annotations

import re
import os
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from apify_runner import run_actor

ACTOR_ID = os.getenv(
    "SPORTSBET_ACTOR_ID",
    "lexis-solutions/sportsbet-com-au-scraper",
)

DEFAULT_URLS = [
    "https://www.sportsbet.com.au/betting/soccer",
]


def _clean_name(raw: str) -> str:
    raw = raw.replace("\u00a0", " ")
    raw = re.sub(r"\s*\(.*?\)\s*", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def _parse_event_name(raw: str) -> tuple[str, str]:
    raw = raw.replace("\u00a0", " ")
    for sep in [" v ", " vs ", " @ "]:
        if sep in raw.lower():
            idx = raw.lower().index(sep)
            home = raw[:idx]
            away = raw[idx + len(sep):]
            return _clean_name(home), _clean_name(away)
    parts = raw.split("  ")
    if len(parts) >= 2:
        mid = len(parts) // 2
        return _clean_name(" ".join(parts[:mid])), _clean_name(" ".join(parts[mid:]))
    return _clean_name(raw), ""


def _parse_odds(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_event_time(time_str: str, ref_date: Optional[date] = None) -> str:
    if not time_str:
        return ""
    ref = ref_date or date.today()
    try:
        parts = time_str.strip().split(":")
        if len(parts) == 2:
            h, m = int(parts[0]), int(parts[1])
            dt = datetime(ref.year, ref.month, ref.day, h, m)
            return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, IndexError):
        pass
    return time_str


def fetch_sportsbet_odds(
    start_urls: Optional[List[str]] = None,
    timeout_secs: int = 300,
) -> List[Dict[str, Any]]:
    urls = start_urls or DEFAULT_URLS
    run_input = {
        "startUrls": [{"url": u} for u in urls],
    }

    raw_items = run_actor(ACTOR_ID, run_input, timeout_secs=timeout_secs)

    normalised: List[Dict[str, Any]] = []
    for item in raw_items:
        results = item.get("results", [])
        for evt in results:
            event_name = evt.get("eventName", "")
            competition = evt.get("competition", "")
            event_time_raw = evt.get("eventTime", "")
            outcomes = evt.get("outcomes", [])

            home, away = _parse_event_name(event_name)
            if not home:
                continue

            event_time = _parse_event_time(event_time_raw)

            if outcomes:
                for oc in outcomes:
                    selection = oc.get("name", oc.get("label", ""))
                    odds_val = _parse_odds(oc.get("odds", oc.get("price")))
                    market = oc.get("market", oc.get("marketName", "Head to Head"))

                    normalised.append({
                        "event": event_name,
                        "home_team": home,
                        "away_team": away,
                        "start_time": event_time,
                        "competition": competition,
                        "market": market,
                        "selection": _clean_name(selection),
                        "odds_decimal": odds_val,
                    })
            else:
                normalised.append({
                    "event": event_name,
                    "home_team": home,
                    "away_team": away,
                    "start_time": event_time,
                    "competition": competition,
                    "market": "Head to Head",
                    "selection": "",
                    "odds_decimal": None,
                })

    return normalised
