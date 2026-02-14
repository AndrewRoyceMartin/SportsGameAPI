from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Set

import requests


@dataclass
class Game:
    league: str
    start_time_utc: datetime
    home: str
    away: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    status: str = "upcoming"


_SOFASCORE_BASE = "https://api.sofascore.com/api/v1"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

LEAGUE_SPORT_MAP = {
    "champions league": "football",
    "nba": "basketball",
    "nhl": "ice-hockey",
    "nfl": "american-football",
    "college football": "american-football",
    "college basketball": "basketball",
    "ufc": "mma",
}


def _sport_for_league(league: str) -> str:
    league_lower = league.lower()
    for key, sport in LEAGUE_SPORT_MAP.items():
        if key in league_lower:
            return sport
    return "football"


def _fetch_events_for_date(d: date, sport: str = "football") -> List[Dict[str, Any]]:
    url = f"{_SOFASCORE_BASE}/sport/{sport}/scheduled-events/{d.isoformat()}"
    r = requests.get(url, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json().get("events", [])


def _parse_game(ev: Dict[str, Any]) -> Optional[Game]:
    try:
        home_team = ev.get("homeTeam", {})
        away_team = ev.get("awayTeam", {})
        home_name = home_team.get("name") or home_team.get("shortName", "")
        away_name = away_team.get("name") or away_team.get("shortName", "")
        if not home_name or not away_name:
            return None

        ts = ev.get("startTimestamp")
        if ts is None:
            return None
        start_time = datetime.utcfromtimestamp(ts)

        tournament = ev.get("tournament", {})
        league = tournament.get("name", "Unknown")
        category = tournament.get("category", {})
        country = category.get("name", "")
        if country:
            league = f"{league} ({country})"

        status_obj = ev.get("status", {})
        status_type = (
            status_obj.get("type", "")
            if isinstance(status_obj, dict)
            else str(status_obj)
        ).lower()

        if status_type == "finished":
            status = "completed"
            home_score_obj = ev.get("homeScore", {})
            away_score_obj = ev.get("awayScore", {})
            hs = (
                home_score_obj.get("current")
                if isinstance(home_score_obj, dict)
                else None
            )
            aws = (
                away_score_obj.get("current")
                if isinstance(away_score_obj, dict)
                else None
            )
            home_score = int(hs) if hs is not None else None
            away_score = int(aws) if aws is not None else None
        else:
            status = "upcoming"
            home_score = None
            away_score = None

        return Game(
            league=league,
            start_time_utc=start_time,
            home=home_name,
            away=away_name,
            home_score=home_score,
            away_score=away_score,
            status=status,
        )
    except Exception:
        return None


_last_fetch_failures: int = 0


def get_fetch_failure_count() -> int:
    return _last_fetch_failures


def _collect_games(
    date_from: date,
    date_to: date,
    league: str,
    wanted_status: str,
) -> List[Game]:
    global _last_fetch_failures
    _last_fetch_failures = 0
    sport = _sport_for_league(league)
    games: List[Game] = []
    seen_ids: Set[int] = set()

    d = date_from
    while d <= date_to:
        try:
            events = _fetch_events_for_date(d, sport=sport)
        except Exception:
            _last_fetch_failures += 1
            d += timedelta(days=1)
            continue

        for ev in events:
            ev_id = ev.get("id")
            if ev_id is not None and ev_id in seen_ids:
                continue

            game = _parse_game(ev)
            if game and game.status == wanted_status:
                if league and league.lower() not in game.league.lower():
                    continue
                games.append(game)
                if ev_id is not None:
                    seen_ids.add(ev_id)

        d += timedelta(days=1)

    games.sort(key=lambda g: g.start_time_utc)
    return games


def get_upcoming_games(
    league: str = "",
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> List[Game]:
    if date_from is None:
        date_from = date.today()
    if date_to is None:
        date_to = date_from + timedelta(days=7)

    return _collect_games(date_from, date_to, league, "upcoming")


def get_results_history(
    league: str = "",
    since_days: int = 30,
) -> List[Game]:
    date_to = date.today() - timedelta(days=1)
    date_from = date_to - timedelta(days=since_days)

    return _collect_games(date_from, date_to, league, "completed")
