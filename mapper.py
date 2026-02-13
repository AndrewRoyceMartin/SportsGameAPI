from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from thefuzz import fuzz

TIME_WINDOW_HOURS = 2
FUZZY_THRESHOLD = 65
NAME_ONLY_THRESHOLD = 80


def _normalise(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"\bfc\b", "", name)
    name = re.sub(r"\bafc\b", "", name)
    name = re.sub(r"\bsc\b", "", name)
    name = re.sub(r"\bcf\b", "", name)
    name = re.sub(r"\bunited\b", "utd", name)
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _parse_dt(s: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def _time_close(dt1: Optional[datetime], dt2: Optional[datetime], hours: float = TIME_WINDOW_HOURS) -> bool:
    if dt1 is None or dt2 is None:
        return False
    return abs((dt1 - dt2).total_seconds()) <= hours * 3600


def _name_score(a: str, b: str) -> int:
    na, nb = _normalise(a), _normalise(b)
    if na == nb:
        return 100
    if na in nb or nb in na:
        return 90
    return fuzz.token_sort_ratio(na, nb)


def _infer_odds_date(odds_time: str, fixture_dates: List[str]) -> Optional[str]:
    if not odds_time or not fixture_dates:
        return None
    unique_dates = sorted(set(d for d in fixture_dates if d))
    if len(unique_dates) == 1:
        return unique_dates[0]
    return None


def match_fixtures_to_odds(
    fixtures: List[Dict[str, Any]],
    odds_events: List[Dict[str, Any]],
    time_window_hours: float = TIME_WINDOW_HOURS,
    fuzzy_threshold: int = FUZZY_THRESHOLD,
) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []

    fx_dates = [fx.get("date_utc", "") for fx in fixtures]

    for fx in fixtures:
        fx_home = fx.get("home_team", "")
        fx_away = fx.get("away_team", "")
        fx_date = fx.get("date_utc", "")
        fx_time = fx.get("time", "")

        fx_dt_str = fx_date
        if fx_time:
            fx_dt_str = f"{fx_date} {fx_time}"
        fx_dt = _parse_dt(fx_dt_str)

        best_score = 0
        best_odds_events: List[Dict[str, Any]] = []

        for od in odds_events:
            od_start = od.get("start_time", "")
            od_dt = _parse_dt(od_start)

            both_have_time = fx_dt is not None and od_dt is not None
            if both_have_time:
                if not _time_close(fx_dt, od_dt, hours=time_window_hours):
                    continue
            else:
                pass

            home_sc = _name_score(fx_home, od.get("home_team", ""))
            away_sc = _name_score(fx_away, od.get("away_team", ""))
            avg_score = (home_sc + away_sc) / 2

            threshold = fuzzy_threshold if both_have_time else NAME_ONLY_THRESHOLD

            if avg_score >= threshold and avg_score > best_score:
                best_score = avg_score
                best_odds_events = [od]
            elif avg_score == best_score and avg_score >= threshold:
                best_odds_events.append(od)

        if best_odds_events:
            for od in best_odds_events:
                matched.append({
                    **fx,
                    "odds_event": od.get("event", ""),
                    "odds_competition": od.get("competition", ""),
                    "market": od.get("market", ""),
                    "selection": od.get("selection", ""),
                    "odds_decimal": od.get("odds_decimal"),
                    "match_confidence": best_score,
                })

    return matched
