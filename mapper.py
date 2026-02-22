from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional

from thefuzz import fuzz

import logging as _logging

from time_utils import parse_iso_utc, to_naive_utc

_mapper_log = _logging.getLogger(__name__)

TIME_WINDOW_HOURS = 4
FUZZY_THRESHOLD = 60
NAME_ONLY_THRESHOLD = 75

_AU_TIME_WINDOW_HOURS = 24
_AU_FUZZY_THRESHOLD = 55
_AU_NAME_ONLY_THRESHOLD = 70

_AU_LEAGUES = {"AFL", "NRL", "NBL", "A-League Men", "Super Rugby Pacific"}

_TEAM_SPORT_LEAGUES = {
    "AFL", "NRL", "NBL", "NBA", "NHL", "NFL",
    "A-League Men", "Super Rugby Pacific",
    "College Football", "College Basketball",
}

_ALIASES = {
    "psg": "paris saint germain",
    "man city": "manchester city",
    "man utd": "manchester utd",
    "inter": "internazionale",
    "inter milan": "internazionale",
    "atletico madrid": "atletico de madrid",
    "rb leipzig": "rasenballsport leipzig",
    "ac milan": "milan",
    "spurs": "tottenham",
    "wolves": "wolverhampton",
    "bayern": "bayern munich",
    "bayern munchen": "bayern munich",
    "barca": "barcelona",
    "real": "real madrid",
    "juve": "juventus",
    "dortmund": "borussia dortmund",
    "gladbach": "borussia monchengladbach",
    "leverkusen": "bayer leverkusen",
    "gws": "greater western sydney giants",
    "gws giants": "greater western sydney giants",
    "greater western sydney": "greater western sydney giants",
    "brisbane": "brisbane lions",
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
    "melbourne": "melbourne demons",
    "demons": "melbourne demons",
    "swans": "sydney swans",
    "dockers": "fremantle dockers",
    "fremantle": "fremantle dockers",
    "gold coast": "gold coast suns",
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
    "jackjumpers": "tasmania jackjumpers",
    "wb": "western bulldogs",
    "w bulldogs": "western bulldogs",
    "western dogs": "western bulldogs",
    "phoenix nbl": "south east melbourne phoenix",
}


_STRIP_SUFFIXES = re.compile(
    r"\b(fc|afc|sc|cf|the)\b", re.IGNORECASE
)


def _normalise(name: str) -> str:
    name = name.lower().strip()
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        name = f"{parts[1]} {parts[0]}"
    name = _STRIP_SUFFIXES.sub("", name)
    name = re.sub(r"\bunited\b", "utd", name)
    name = re.sub(r"\bjr\.?\b", "jr", name)
    name = re.sub(r"\bsr\.?\b", "sr", name)
    name = name.replace("-", " ")
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _expand(name: str) -> str:
    key = _normalise(name)
    return _ALIASES.get(key, key)


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
    dt = parse_iso_utc(s)
    return to_naive_utc(dt)


_MIN_PAIRS_FOR_OFFSET = 4


def detect_time_offset(
    fixtures: list,
    harvest_games: List[Dict[str, Any]],
    league: str = "",
) -> Dict[str, Any]:
    is_au = league in _AU_LEAGUES
    name_threshold = 75 if is_au else 70

    fx_times = []
    for fx in fixtures:
        dt = to_naive_utc(fx.start_time_utc)
        if dt is not None:
            fx_times.append(dt)
    if not fx_times:
        return {"offset_hours": 0.0, "confidence": "none", "pairs_checked": 0, "applied": False, "pairs": []}
    fx_min = min(fx_times)
    fx_max = max(fx_times)
    from datetime import timedelta as _td
    window_start = fx_min - _td(days=2)
    window_end = fx_max + _td(days=2)

    pairs = []
    seen_fx = set()
    for fx in fixtures:
        fx_dt = to_naive_utc(fx.start_time_utc)
        if fx_dt is None:
            continue
        fx_key = (fx.home, fx.away)
        if fx_key in seen_fx:
            continue

        best_pair = None
        best_name = 0
        for og in harvest_games:
            og_home = og.get("homeTeam", {}).get("mediumName", "")
            og_away = og.get("awayTeam", {}).get("mediumName", "")
            og_dt = _parse_iso(og.get("scheduledTime", ""))
            if og_dt is None:
                continue
            if og_dt < window_start or og_dt > window_end:
                continue

            home_sc = _name_score(fx.home, og_home)
            away_sc = _name_score(fx.away, og_away)
            avg = (home_sc + away_sc) / 2

            home_sc_flip = _name_score(fx.home, og_away)
            away_sc_flip = _name_score(fx.away, og_home)
            avg_flip = (home_sc_flip + away_sc_flip) / 2

            best = max(avg, avg_flip)
            if best >= name_threshold and best > best_name:
                best_name = best
                delta_h = (og_dt - fx_dt).total_seconds() / 3600
                best_pair = {
                    "fixture": f"{fx.home} vs {fx.away}",
                    "odds": f"{og_home} vs {og_away}",
                    "fx_time": str(fx_dt)[:19],
                    "odds_time": str(og_dt)[:19],
                    "delta_h": round(delta_h, 2),
                    "name_score": best,
                }

        if best_pair:
            pairs.append(best_pair)
            seen_fx.add(fx_key)

    if not pairs:
        return {"offset_hours": 0.0, "confidence": "none", "pairs_checked": 0, "applied": False, "pairs": []}

    deltas = [p["delta_h"] for p in pairs]
    deltas.sort()
    median_offset = deltas[len(deltas) // 2]

    within_2h = sum(1 for d in deltas if abs(d - median_offset) <= 2)
    consistency = within_2h / len(deltas) if deltas else 0

    enough_pairs = len(pairs) >= _MIN_PAIRS_FOR_OFFSET

    if abs(median_offset) < 1.5:
        confidence = "low"
        should_apply = False
    elif consistency >= 0.6 and abs(median_offset) >= 1.5 and enough_pairs:
        confidence = "high"
        should_apply = True
    elif consistency >= 0.4 and enough_pairs:
        confidence = "medium"
        should_apply = abs(median_offset) >= 3
    else:
        confidence = "low"
        should_apply = False

    return {
        "offset_hours": round(median_offset, 2),
        "confidence": confidence,
        "consistency": round(consistency, 2),
        "pairs_checked": len(pairs),
        "should_apply": should_apply,
        "applied": False,
        "pairs": pairs[:20],
    }


def apply_time_offset(
    harvest_games: List[Dict[str, Any]],
    offset_hours: float,
) -> List[Dict[str, Any]]:
    from datetime import timedelta as _td
    corrected = []
    for og in harvest_games:
        og_copy = dict(og)
        og_time_str = og_copy.get("scheduledTime", "")
        og_dt = _parse_iso(og_time_str)
        if og_dt is not None:
            corrected_dt = og_dt - _td(hours=offset_hours)
            og_copy["scheduledTime"] = corrected_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            og_copy["_original_scheduledTime"] = og_time_str
        corrected.append(og_copy)
    return corrected


def match_games_to_odds(
    fixtures: list,
    harvest_games: List[Dict[str, Any]],
    time_window_hours: float = TIME_WINDOW_HOURS,
    fuzzy_threshold: int = FUZZY_THRESHOLD,
    league: str = "",
) -> List[Dict[str, Any]]:
    is_au = league in _AU_LEAGUES
    tw = _AU_TIME_WINDOW_HOURS if is_au else time_window_hours
    ft = _AU_FUZZY_THRESHOLD if is_au else fuzzy_threshold
    no_time_ft = _AU_NAME_ONLY_THRESHOLD if is_au else NAME_ONLY_THRESHOLD

    _mapper_log.info(
        "match start league=%s is_au=%s fixtures=%d odds=%d tw=%sh ft=%s no_time_ft=%s",
        league or "-", is_au, len(fixtures), len(harvest_games), tw, ft, no_time_ft,
    )

    matched: List[Dict[str, Any]] = []
    unmatched_diagnostics: List[Dict[str, Any]] = []

    for fx in fixtures:
        fx_home = fx.home
        fx_away = fx.away
        fx_dt = to_naive_utc(fx.start_time_utc)

        best_score = 0
        best_odds_game = None
        best_delta_h: Optional[float] = None

        best_rejected_time: Optional[Dict[str, Any]] = None
        best_rejected_name: Optional[Dict[str, Any]] = None
        best_rejected_au_tier: Optional[Dict[str, Any]] = None
        considered = 0

        for og in harvest_games:
            og_home = og.get("homeTeam", {}).get("mediumName", "")
            og_away = og.get("awayTeam", {}).get("mediumName", "")
            og_time_str = og.get("scheduledTime", "")
            og_dt = _parse_iso(og_time_str)
            considered += 1

            both_have_time = fx_dt is not None and og_dt is not None
            if not both_have_time and league in _TEAM_SPORT_LEAGUES:
                continue
            diff_h: Optional[float] = None
            if both_have_time and fx_dt is not None and og_dt is not None:
                diff_h = abs((fx_dt - og_dt).total_seconds()) / 3600
                if diff_h > tw:
                    if best_rejected_time is None or diff_h < best_rejected_time["diff_h"]:
                        best_rejected_time = {
                            "diff_h": round(diff_h, 2),
                            "window": tw,
                            "odds_home": og_home,
                            "odds_away": og_away,
                            "odds_time": og_time_str,
                        }
                    continue

            home_sc = _name_score(fx_home, og_home)
            away_sc = _name_score(fx_away, og_away)
            avg_normal = (home_sc + away_sc) / 2

            home_sc_flip = _name_score(fx_home, og_away)
            away_sc_flip = _name_score(fx_away, og_home)
            avg_flipped = (home_sc_flip + away_sc_flip) / 2

            avg_score = max(avg_normal, avg_flipped)

            threshold = ft if both_have_time else no_time_ft

            if is_au and both_have_time and diff_h is not None:
                if avg_score >= 85 and diff_h <= 24:
                    pass
                elif avg_score >= 70 and diff_h <= 12:
                    pass
                elif avg_score >= ft and diff_h <= 6:
                    pass
                else:
                    if best_rejected_au_tier is None or avg_score > best_rejected_au_tier["name_score"]:
                        best_rejected_au_tier = {
                            "name_score": round(avg_score, 1),
                            "diff_h": round(diff_h, 2) if diff_h else None,
                            "odds_home": og_home,
                            "odds_away": og_away,
                            "reason": f"AU tier: score {avg_score:.0f} needs diff<={'6' if avg_score < 70 else '12' if avg_score < 85 else '24'}h, got {diff_h:.1f}h",
                        }
                    continue

            if avg_score < threshold:
                if best_rejected_name is None or avg_score > best_rejected_name["name_score"]:
                    best_rejected_name = {
                        "name_score": round(avg_score, 1),
                        "threshold": threshold,
                        "diff_h": round(diff_h, 2) if diff_h is not None else None,
                        "odds_home": og_home,
                        "odds_away": og_away,
                    }
                continue

            if avg_score > best_score:
                best_score = avg_score
                best_odds_game = og
                best_delta_h = diff_h

        if best_odds_game:
            matched.append(
                {
                    "fixture": fx,
                    "odds_game": best_odds_game,
                    "match_confidence": best_score,
                    "time_delta_hours": best_delta_h,
                }
            )
            _mapper_log.debug(
                "MATCH '%s vs %s' -> '%s vs %s' score=%s delta=%s",
                fx_home, fx_away,
                best_odds_game.get("homeTeam", {}).get("mediumName", ""),
                best_odds_game.get("awayTeam", {}).get("mediumName", ""),
                best_score, f"{best_delta_h:.1f}h" if best_delta_h is not None else "n/a",
            )
        else:
            diag: Dict[str, Any] = {
                "fixture": f"{fx_home} vs {fx_away}",
                "fixture_time": str(fx_dt)[:19] if fx_dt else None,
                "considered": considered,
            }
            if best_rejected_time:
                diag["nearest_time_reject"] = best_rejected_time
                diag["reject_reason"] = f"Time: nearest odds {best_rejected_time['diff_h']:.1f}h away (window {tw}h)"
            elif best_rejected_au_tier:
                diag["nearest_au_reject"] = best_rejected_au_tier
                diag["reject_reason"] = best_rejected_au_tier["reason"]
            elif best_rejected_name:
                diag["nearest_name_reject"] = best_rejected_name
                diag["reject_reason"] = f"Name: best score {best_rejected_name['name_score']:.0f} < threshold {best_rejected_name['threshold']}"
            else:
                diag["reject_reason"] = "No candidates (no time data or empty odds)"
            unmatched_diagnostics.append(diag)
            _mapper_log.info(
                "NO MATCH '%s vs %s' t=%s reason=%s",
                fx_home, fx_away, fx_dt, diag.get("reject_reason", "unknown"),
            )

    _mapper_log.info("match end matched=%d/%d", len(matched), len(fixtures))

    _last_match_diagnostics.clear()
    _last_match_diagnostics.extend(unmatched_diagnostics)

    return matched


_last_match_diagnostics: List[Dict[str, Any]] = []


def get_match_diagnostics() -> List[Dict[str, Any]]:
    return list(_last_match_diagnostics)
