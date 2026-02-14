from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_SYDNEY_TZ = ZoneInfo("Australia/Sydney")
_UTC = ZoneInfo("UTC")

SPORTSBET_LEAGUES: Dict[str, Dict[str, str]] = {
    "AFL": {
        "url": "https://www.sportsbet.com.au/betting/australian-rules/afl",
        "competition": "AFL",
    },
    "NRL": {
        "url": "https://www.sportsbet.com.au/betting/rugby-league/nrl",
        "competition": "NRL",
    },
    "NBL": {
        "url": "https://www.sportsbet.com.au/betting/basketball-aus-other/australian-nbl",
        "competition": "NBL",
    },
}


def is_sportsbet_league(league: str) -> bool:
    return league in SPORTSBET_LEAGUES


def _parse_sportsbet_datetime(event_time: str) -> Optional[datetime]:
    if not event_time or not isinstance(event_time, str):
        return None

    event_time = event_time.strip()

    if re.match(r"^\d{1,2}:\d{2}$", event_time):
        return None

    patterns = [
        (r"^[A-Za-z]+,\s*(\d{1,2}\s+[A-Za-z]+\s+\d{1,2}:\d{2})$", "%d %b %H:%M"),
        (r"^[A-Za-z]+,\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4}\s+\d{1,2}:\d{2})$", "%d %b %Y %H:%M"),
        (r"^(\d{1,2}\s+[A-Za-z]+\s+\d{1,2}:\d{2})$", "%d %b %H:%M"),
        (r"^(\d{1,2}\s+[A-Za-z]+\s+\d{4}\s+\d{1,2}:\d{2})$", "%d %b %Y %H:%M"),
    ]

    for regex, fmt in patterns:
        m = re.match(regex, event_time)
        if m:
            date_str = m.group(1)
            try:
                parsed = datetime.strptime(date_str, fmt)
                if parsed.year == 1900:
                    now = datetime.now(_SYDNEY_TZ)
                    parsed = parsed.replace(year=now.year)
                    if (now - parsed).days > 330:
                        parsed = parsed.replace(year=now.year + 1)
                    elif (parsed - now).days > 330:
                        parsed = parsed.replace(year=now.year - 1)
                local = parsed.replace(tzinfo=_SYDNEY_TZ)
                return local.astimezone(_UTC).replace(tzinfo=None)
            except ValueError:
                continue

    return None


def _parse_odds_value(raw: str) -> Optional[float]:
    try:
        return float(raw)
    except (ValueError, TypeError):
        pass
    m = re.search(r"(\d+\.\d+)\s*$", raw)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass
    return None


def _extract_head_to_head(participants: List[Dict[str, Any]]) -> Optional[tuple]:
    if not participants or len(participants) < 2:
        return None

    home_name = participants[0].get("name", "").strip()
    away_name = participants[1].get("name", "").strip()
    if not home_name or not away_name:
        return None

    home_odds = None
    away_odds = None

    _H2H_LABELS = {
        "head to head", "h2h", "match winner", "moneyline", "money line",
        "match betting", "win",
    }

    for p_idx, p in enumerate(participants[:2]):
        for d in p.get("data", []):
            label = (d.get("label") or "").lower()
            if label in _H2H_LABELS:
                raw = str(d.get("value", "")).strip()
                val = _parse_odds_value(raw)
                if val is not None and val > 1.0:
                    if p_idx == 0:
                        home_odds = val
                    else:
                        away_odds = val

    if home_odds is None and away_odds is None:
        return None

    return home_name, away_name, home_odds, away_odds


def _decimal_to_american(dec: float) -> Optional[int]:
    if dec is None or dec <= 1.0:
        return None
    if dec >= 2.0:
        return int(round((dec - 1) * 100))
    return int(round(-100 / (dec - 1)))


def _to_harvest_format(
    home: str,
    away: str,
    home_odds: Optional[float],
    away_odds: Optional[float],
    start_utc: Optional[datetime],
) -> Dict[str, Any]:
    scheduled_time = ""
    if start_utc:
        scheduled_time = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    odds_entry: Dict[str, Any] = {
        "sportsbook": "Sportsbet",
        "moneyLine": {},
    }
    if home_odds is not None:
        american = _decimal_to_american(home_odds)
        odds_entry["moneyLine"]["currentHomeOdds"] = american
    if away_odds is not None:
        american = _decimal_to_american(away_odds)
        odds_entry["moneyLine"]["currentAwayOdds"] = american

    return {
        "homeTeam": {"mediumName": home, "name": home},
        "awayTeam": {"mediumName": away, "name": away},
        "scheduledTime": scheduled_time,
        "odds": [odds_entry],
    }


def parse_sportsbet_items(
    items: List[Dict[str, Any]],
    league: str,
) -> List[Dict[str, Any]]:
    league_config = SPORTSBET_LEAGUES.get(league)
    if not league_config:
        return []

    target_comp = league_config["competition"].lower()
    target_url = league_config["url"].lower()

    harvest_events: List[Dict[str, Any]] = []
    seen: set = set()

    for item in items:
        item_url = (item.get("url") or "").lower()
        if target_url not in item_url:
            continue

        results_list = item.get("results", [])
        if not isinstance(results_list, list):
            results_list = [item]

        for row in results_list:
            comp = (row.get("competition") or "").lower()
            if comp and target_comp not in comp:
                continue

            event_time = row.get("eventTime", "")
            start_utc = _parse_sportsbet_datetime(event_time)

            participants = row.get("participants", [])
            result = _extract_head_to_head(participants)
            if result is None:
                continue

            home, away, home_odds, away_odds = result

            key = (home, away)
            if key in seen:
                continue
            seen.add(key)

            event = _to_harvest_format(home, away, home_odds, away_odds, start_utc)
            harvest_events.append(event)

    logger.info(
        "Sportsbet parsed %d events for %s from %d raw items",
        len(harvest_events), league, len(items),
    )
    return harvest_events
