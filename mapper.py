from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from thefuzz import fuzz

TIME_WINDOW_HOURS = 4
FUZZY_THRESHOLD = 60
NAME_ONLY_THRESHOLD = 75

_ALIASES = {
    "psg": "paris saint germain",
    "man city": "manchester city",
    "man utd": "manchester utd",
    "inter": "internazionale",
    "inter milan": "internazionale",
    "atletico madrid": "atletico de madrid",
    "atlético madrid": "atletico de madrid",
    "rb leipzig": "rasenballsport leipzig",
    "ac milan": "milan",
    "spurs": "tottenham",
    "wolves": "wolverhampton",
    "bayern": "bayern munich",
    "bayern münchen": "bayern munich",
    "barca": "barcelona",
    "real": "real madrid",
    "juve": "juventus",
    "dortmund": "borussia dortmund",
    "gladbach": "borussia monchengladbach",
    "leverkusen": "bayer leverkusen",
    "gws": "greater western sydney giants",
    "gws giants": "greater western sydney giants",
    "brisbane lions": "brisbane lions",
    "west coast": "west coast eagles",
    "north melbourne": "north melbourne kangaroos",
    "kangaroos": "north melbourne kangaroos",
    "port adelaide": "port adelaide power",
    "crows": "adelaide crows",
    "cats": "geelong cats",
    "geelong": "geelong cats",
    "pies": "collingwood magpies",
    "collingwood": "collingwood magpies",
    "bombers": "essendon bombers",
    "essendon": "essendon bombers",
    "hawks": "hawthorn hawks",
    "hawthorn": "hawthorn hawks",
    "blues": "carlton blues",
    "carlton": "carlton blues",
    "tigers": "richmond tigers",
    "richmond": "richmond tigers",
    "demons": "melbourne demons",
    "swans": "sydney swans",
    "dockers": "fremantle dockers",
    "fremantle": "fremantle dockers",
    "suns": "gold coast suns",
    "saints": "st kilda saints",
    "st kilda": "st kilda saints",
    "sea eagles": "manly warringah sea eagles",
    "manly": "manly warringah sea eagles",
    "roosters": "sydney roosters",
    "rabbitohs": "south sydney rabbitohs",
    "south sydney": "south sydney rabbitohs",
    "souths": "south sydney rabbitohs",
    "storm": "melbourne storm",
    "broncos": "brisbane broncos",
    "panthers": "penrith panthers",
    "penrith": "penrith panthers",
    "eels": "parramatta eels",
    "parramatta": "parramatta eels",
    "raiders": "canberra raiders",
    "sharks": "cronulla-sutherland sharks",
    "cronulla": "cronulla-sutherland sharks",
    "warriors": "new zealand warriors",
    "cowboys": "north queensland cowboys",
    "titans": "gold coast titans",
    "knights": "newcastle knights",
    "dragons": "st george illawarra dragons",
    "bulldogs": "canterbury-bankstown bulldogs",
    "canterbury": "canterbury-bankstown bulldogs",
    "wests tigers": "wests tigers",
    "dolphins": "dolphins",
    "36ers": "adelaide 36ers",
    "wildcats": "perth wildcats",
    "kings": "sydney kings",
    "breakers": "new zealand breakers",
    "taipans": "cairns taipans",
    "bullets": "brisbane bullets",
    "illawarra hawks": "illawarra hawks",
    "united": "melbourne united",
    "jackjumpers": "tasmania jackjumpers",
    "phoenix nbl": "south east melbourne phoenix",
}


def _normalise(name: str) -> str:
    name = name.lower().strip()
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        name = f"{parts[1]} {parts[0]}"
    name = re.sub(r"\bfc\b", "", name)
    name = re.sub(r"\bafc\b", "", name)
    name = re.sub(r"\bsc\b", "", name)
    name = re.sub(r"\bcf\b", "", name)
    name = re.sub(r"\bunited\b", "utd", name)
    name = re.sub(r"\bjr\.?\b", "jr", name)
    name = re.sub(r"\bsr\.?\b", "sr", name)
    name = name.replace("-", " ")
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _expand(name: str) -> str:
    low = name.lower().strip()
    return _ALIASES.get(low, low)


def _name_score(a: str, b: str) -> int:
    na, nb = _normalise(a), _normalise(b)
    if na == nb:
        return 100
    if na in nb or nb in na:
        return 90

    ea, eb = _normalise(_expand(a)), _normalise(_expand(b))
    if ea == eb:
        return 95
    if ea in eb or eb in ea:
        return 88

    score_raw = fuzz.token_sort_ratio(na, nb)
    score_exp = fuzz.token_sort_ratio(ea, eb)
    return max(score_raw, score_exp)


def _parse_iso(s: str) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    cleaned = re.sub(r"\.\d+", "", s.strip())
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def match_games_to_odds(
    fixtures: list,
    harvest_games: List[Dict[str, Any]],
    time_window_hours: float = TIME_WINDOW_HOURS,
    fuzzy_threshold: int = FUZZY_THRESHOLD,
) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []

    for fx in fixtures:
        fx_home = fx.home
        fx_away = fx.away
        fx_dt = fx.start_time_utc

        best_score = 0
        best_odds_game = None

        for og in harvest_games:
            og_home = og.get("homeTeam", {}).get("mediumName", "")
            og_away = og.get("awayTeam", {}).get("mediumName", "")
            og_time_str = og.get("scheduledTime", "")
            og_dt = _parse_iso(og_time_str)

            both_have_time = fx_dt is not None and og_dt is not None
            if both_have_time:
                diff = abs((fx_dt - og_dt).total_seconds())
                if diff > time_window_hours * 3600:
                    continue

            home_sc = _name_score(fx_home, og_home)
            away_sc = _name_score(fx_away, og_away)
            avg_normal = (home_sc + away_sc) / 2

            home_sc_flip = _name_score(fx_home, og_away)
            away_sc_flip = _name_score(fx_away, og_home)
            avg_flipped = (home_sc_flip + away_sc_flip) / 2

            avg_score = max(avg_normal, avg_flipped)

            threshold = fuzzy_threshold if both_have_time else NAME_ONLY_THRESHOLD

            if avg_score >= threshold and avg_score > best_score:
                best_score = avg_score
                best_odds_game = og

        if best_odds_game:
            matched.append(
                {
                    "fixture": fx,
                    "odds_game": best_odds_game,
                    "match_confidence": best_score,
                }
            )

    return matched
