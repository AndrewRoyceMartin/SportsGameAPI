from __future__ import annotations

from typing import Any, Dict, List
from thefuzz import fuzz
from features import elo_win_prob

DRAW_LABELS = {"draw", "x", "tie"}
HOME_LABELS = {"home", "1", "h"}
AWAY_LABELS = {"away", "2", "a"}


def implied_probability(odds_decimal: float) -> float:
    if odds_decimal <= 0:
        return 0.0
    return 1.0 / odds_decimal


def edge(model_prob: float, implied_prob: float) -> float:
    if implied_prob <= 0:
        return 0.0
    return model_prob - implied_prob


def expected_value(model_prob: float, odds_decimal: float, stake: float = 1.0) -> float:
    return (model_prob * (odds_decimal - 1.0) * stake) - ((1.0 - model_prob) * stake)


def _identify_selection(selection: str, home: str, away: str) -> str | None:
    sel = selection.lower().strip()

    if sel in DRAW_LABELS:
        return None

    if sel in HOME_LABELS:
        return "home"
    if sel in AWAY_LABELS:
        return "away"

    home_score = fuzz.token_sort_ratio(sel, home.lower())
    away_score = fuzz.token_sort_ratio(sel, away.lower())

    if home_score > away_score and home_score >= 50:
        return "home"
    if away_score > home_score and away_score >= 50:
        return "away"

    return None


def compute_value_bets(
    matched_events: List[Dict[str, Any]],
    elo_ratings: Dict[str, float],
    home_adv: float = 65.0,
    min_edge: float = 0.05,
    odds_min: float = 1.80,
    odds_max: float = 3.50,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    value_bets: List[Dict[str, Any]] = []

    for evt in matched_events:
        odds_dec = evt.get("odds_decimal")
        if odds_dec is None or odds_dec <= 1.0:
            continue

        if not (odds_min <= odds_dec <= odds_max):
            continue

        home = evt.get("home_team", "")
        away = evt.get("away_team", "")
        selection = evt.get("selection", "")

        if not selection:
            continue

        side = _identify_selection(selection, home, away)
        if side is None:
            continue

        r_home = float(elo_ratings.get(home, 1500.0))
        r_away = float(elo_ratings.get(away, 1500.0))
        p_home = elo_win_prob(r_home, r_away, home_adv=home_adv)
        p_away = 1.0 - p_home

        model_p = p_home if side == "home" else p_away

        imp_p = implied_probability(odds_dec)
        e = edge(model_p, imp_p)
        ev = expected_value(model_p, odds_dec)

        if e < min_edge:
            continue

        value_bets.append({
            "date": evt.get("date_utc", ""),
            "time": evt.get("time", ""),
            "home_team": home,
            "away_team": away,
            "league": evt.get("league_name", ""),
            "country": evt.get("country_name", ""),
            "market": evt.get("market", ""),
            "selection": selection,
            "odds_decimal": odds_dec,
            "implied_prob": imp_p,
            "model_prob": model_p,
            "edge": e,
            "ev_per_unit": ev,
            "home_elo": r_home,
            "away_elo": r_away,
            "match_confidence": evt.get("match_confidence", 0),
        })

    value_bets.sort(key=lambda x: x["edge"], reverse=True)
    return value_bets[:top_n]
