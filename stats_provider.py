from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Set, Tuple
import random
import time

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
_HEADERS = {"User-Agent": "Mozilla/5.0 SportsGameAPI/1.0"}

DEFAULT_TIMEOUT = 20
DEFAULT_MAX_RETRIES = 4
DEFAULT_BACKOFF_BASE = 0.4
DEFAULT_BACKOFF_CAP = 6.0


@dataclass
class _LeagueRoute:
    sport: str
    accept: list


_LEAGUE_ROUTING = {
    "NBA": _LeagueRoute(sport="basketball", accept=["nba"]),
    "NHL": _LeagueRoute(sport="ice-hockey", accept=["nhl"]),
    "NFL": _LeagueRoute(sport="american-football", accept=["nfl"]),
    "College Football": _LeagueRoute(
        sport="american-football", accept=["ncaa", "college football", "cfp", "fbs"]
    ),
    "College Basketball": _LeagueRoute(
        sport="basketball", accept=["ncaa", "college basketball", "march madness"]
    ),
    "Champions League": _LeagueRoute(sport="football", accept=["champions league", "ucl"]),
    "UFC": _LeagueRoute(sport="mma", accept=["ufc"]),
    "AFL": _LeagueRoute(sport="aussie-rules", accept=["afl", "australian football"]),
    "NRL": _LeagueRoute(sport="rugby", accept=["nrl", "national rugby league"]),
    "NBL": _LeagueRoute(sport="basketball", accept=["nbl", "australia"]),
}


def _route_for_league(league: str) -> _LeagueRoute:
    if league in _LEAGUE_ROUTING:
        return _LEAGUE_ROUTING[league]
    league_lower = league.lower()
    for key, route in _LEAGUE_ROUTING.items():
        if key.lower() in league_lower:
            return route
    return _LeagueRoute(sport="football", accept=[league.lower()])


def _matches_league(game_league_str: str, accept: list) -> bool:
    gl = game_league_str.lower()
    return any(pat in gl for pat in accept)


def _sleep_backoff(attempt: int, base: float, cap: float) -> None:
    delay = min(cap, base * (2 ** attempt))
    jitter = random.uniform(0, delay * 0.25)
    time.sleep(delay + jitter)


_events_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

_last_fetch_failures: int = 0
_last_http_429: int = 0
_last_http_5xx: int = 0
_last_http_404: int = 0
_last_last_status: Optional[int] = None


def get_fetch_failure_count() -> int:
    return _last_fetch_failures


def get_http_429_count() -> int:
    return _last_http_429


def get_http_5xx_count() -> int:
    return _last_http_5xx


def get_http_404_count() -> int:
    return _last_http_404


def get_last_status_code() -> Optional[int]:
    return _last_last_status


def clear_events_cache() -> None:
    _events_cache.clear()


def reset_fetch_diagnostics() -> None:
    global _last_fetch_failures, _last_http_429, _last_http_5xx, _last_http_404, _last_last_status
    _last_fetch_failures = 0
    _last_http_429 = 0
    _last_http_5xx = 0
    _last_http_404 = 0
    _last_last_status = None


def _fetch_events_for_date(
    d: date,
    sport: str,
    *,
    session: requests.Session,
    timeout: int,
    max_retries: int,
    backoff_base: float,
    backoff_cap: float,
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    global _last_fetch_failures, _last_http_429, _last_http_5xx, _last_http_404, _last_last_status

    cache_key = (sport, d.isoformat())
    if cache_key in _events_cache:
        return _events_cache[cache_key]

    url = f"{_SOFASCORE_BASE}/sport/{sport}/scheduled-events/{d.isoformat()}"

    for attempt in range(max_retries + 1):
        try:
            r = session.get(url, headers=_HEADERS, timeout=timeout)
            _last_last_status = r.status_code

            if r.status_code == 404:
                _last_http_404 += 1
                _last_fetch_failures += 1
                return []

            if r.status_code == 429:
                _last_http_429 += 1
                if attempt < max_retries:
                    if logger:
                        logger.warning("SofaScore 429 rate-limited; backing off then retrying.")
                    _sleep_backoff(attempt, backoff_base, backoff_cap)
                    continue
                _last_fetch_failures += 1
                return []

            if 500 <= r.status_code < 600:
                _last_http_5xx += 1
                if attempt < max_retries:
                    if logger:
                        logger.warning(f"SofaScore HTTP {r.status_code}; backing off then retrying.")
                    _sleep_backoff(attempt, backoff_base, backoff_cap)
                    continue
                _last_fetch_failures += 1
                return []

            r.raise_for_status()
            events = r.json().get("events", [])
            if not isinstance(events, list):
                _last_fetch_failures += 1
                return []

            _events_cache[cache_key] = events
            return events

        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt < max_retries:
                if logger:
                    logger.warning(f"SofaScore network error; retrying: {e}")
                _sleep_backoff(attempt, backoff_base, backoff_cap)
                continue
            _last_fetch_failures += 1
            return []

        except Exception as e:
            if logger:
                logger.error(f"SofaScore fetch error for {sport} {d}: {e}")
            _last_fetch_failures += 1
            return []

    _last_fetch_failures += 1
    return []


_BAD_STATUSES = {"canceled", "cancelled", "postponed", "interrupted", "suspended", "abandoned"}

_TOURNAMENT_DENY = [
    "rising stars",
    "all-star",
    "all star",
    "celebrity",
    "summer league",
    "preseason",
    "pre-season",
    "exhibition",
    "skills challenge",
    "slam dunk",
    "three-point",
    "3-point",
    "pro bowl",
    "combine",
    "draft",
]


def _parse_game(ev: Dict[str, Any]) -> Optional[Game]:
    try:
        status_obj = ev.get("status", {})
        status_type = (
            status_obj.get("type", "")
            if isinstance(status_obj, dict)
            else str(status_obj)
        ).lower()

        if status_type in _BAD_STATUSES:
            return None

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

        league_lower = league.lower()
        if any(deny in league_lower for deny in _TOURNAMENT_DENY):
            return None

        if status_type == "finished":
            status = "completed"
            home_score_obj = ev.get("homeScore", {})
            away_score_obj = ev.get("awayScore", {})
            hs = home_score_obj.get("current") if isinstance(home_score_obj, dict) else None
            aws = away_score_obj.get("current") if isinstance(away_score_obj, dict) else None
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


def _collect_games(
    date_from: date,
    date_to: date,
    league: str,
    wanted_status: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    backoff_cap: float = DEFAULT_BACKOFF_CAP,
    session: Optional[requests.Session] = None,
    logger: Optional[Any] = None,
) -> List[Game]:
    reset_fetch_diagnostics()

    route = _route_for_league(league)
    games: List[Game] = []
    seen_ids: Set[int] = set()

    sess = session or requests.Session()

    d = date_from
    while d <= date_to:
        events = _fetch_events_for_date(
            d,
            sport=route.sport,
            session=sess,
            timeout=timeout,
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_cap=backoff_cap,
            logger=logger,
        )

        for ev in events:
            ev_id = ev.get("id")
            if ev_id is not None and ev_id in seen_ids:
                continue

            game = _parse_game(ev)
            if game and game.status == wanted_status:
                if league and not _matches_league(game.league, route.accept):
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
    *,
    timeout: int = DEFAULT_TIMEOUT,
    session: Optional[requests.Session] = None,
    logger: Optional[Any] = None,
) -> List[Game]:
    if date_from is None:
        date_from = date.today()
    if date_to is None:
        date_to = date_from + timedelta(days=7)

    return _collect_games(
        date_from,
        date_to,
        league,
        "upcoming",
        timeout=timeout,
        session=session,
        logger=logger,
    )


def get_results_history(
    league: str = "",
    since_days: int = 30,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    session: Optional[requests.Session] = None,
    logger: Optional[Any] = None,
) -> List[Game]:
    date_to = date.today() - timedelta(days=1)
    date_from = date_to - timedelta(days=since_days)

    return _collect_games(
        date_from,
        date_to,
        league,
        "completed",
        timeout=timeout,
        session=session,
        logger=logger,
    )
