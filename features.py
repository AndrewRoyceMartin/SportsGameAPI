from __future__ import annotations
from typing import Dict, Any, List, Tuple


def match_outcome(home_score: int, away_score: int) -> float:
    if home_score > away_score:
        return 1.0
    if home_score < away_score:
        return 0.0
    return 0.5


def expected_score(r_home: float, r_away: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_away - r_home) / 400.0))


def update_elo(r_home: float, r_away: float, actual_home: float, k: float = 20.0) -> Tuple[float, float]:
    exp_home = expected_score(r_home, r_away)
    r_home_new = r_home + k * (actual_home - exp_home)
    r_away_new = r_away + k * ((1.0 - actual_home) - (1.0 - exp_home))
    return r_home_new, r_away_new


def build_elo_ratings(
    games: List[Dict[str, Any]],
    base_rating: float = 1500.0,
    k: float = 20.0,
    home_adv: float = 65.0,
) -> Dict[str, float]:
    games_sorted = sorted(games, key=lambda g: g["date_utc"] or "")
    ratings: Dict[str, float] = {}

    for g in games_sorted:
        h, a = g["home_team"], g["away_team"]
        ratings.setdefault(h, base_rating)
        ratings.setdefault(a, base_rating)

        r_home = ratings[h] + home_adv
        r_away = ratings[a]

        actual_home = match_outcome(g["home_score"], g["away_score"])
        new_home, new_away = update_elo(r_home, r_away, actual_home, k=k)

        ratings[h] = new_home - home_adv
        ratings[a] = new_away

    return ratings


def elo_win_prob(r_home: float, r_away: float, home_adv: float = 65.0) -> float:
    return expected_score(r_home + home_adv, r_away)
