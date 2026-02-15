from __future__ import annotations
import math
from typing import Dict, Any, List, Tuple
from datetime import datetime


def match_outcome(home_score: int, away_score: int) -> float:
    if home_score > away_score:
        return 1.0
    if home_score < away_score:
        return 0.0
    return 0.5


def expected_score(r_home: float, r_away: float, scale: float = 400.0) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_away - r_home) / scale))


def _mov_multiplier(
    home_score: int, away_score: int, elo_diff: float
) -> float:
    mov = abs(home_score - away_score)
    if mov <= 0:
        return 1.0
    return math.log(mov + 1) * (2.2 / (0.001 * abs(elo_diff) + 2.2))


def _dynamic_k(
    base_k: float,
    home_games: int,
    away_games: int,
) -> Tuple[float, float]:
    def _team_k(games_played: int) -> float:
        if games_played < 10:
            return base_k * 1.8
        if games_played < 20:
            return base_k * 1.3
        return base_k

    return _team_k(home_games), _team_k(away_games)


def update_elo(
    r_home: float, r_away: float, actual_home: float,
    k_home: float = 20.0, k_away: float = 20.0,
    scale: float = 400.0,
) -> Tuple[float, float]:
    exp_home = expected_score(r_home, r_away, scale=scale)
    r_home_new = r_home + k_home * (actual_home - exp_home)
    r_away_new = r_away + k_away * ((1.0 - actual_home) - (1.0 - exp_home))
    return r_home_new, r_away_new


def build_elo_ratings(
    games: List[Dict[str, Any]],
    base_rating: float = 1500.0,
    k: float = 20.0,
    home_adv: float = 65.0,
    scale: float = 400.0,
    use_mov: bool = True,
    recency_half_life: float = 0.0,
    use_dynamic_k: bool = True,
) -> Dict[str, float]:
    games_sorted = sorted(games, key=lambda g: g["date_utc"] or "")
    ratings: Dict[str, float] = {}
    game_counts: Dict[str, int] = {}

    ref_date = None
    if recency_half_life > 0 and games_sorted:
        last_date_str = games_sorted[-1]["date_utc"]
        if last_date_str:
            try:
                ref_date = datetime.strptime(last_date_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                ref_date = None

    for g in games_sorted:
        h, a = g["home_team"], g["away_team"]
        ratings.setdefault(h, base_rating)
        ratings.setdefault(a, base_rating)
        game_counts.setdefault(h, 0)
        game_counts.setdefault(a, 0)

        r_home = ratings[h] + home_adv
        r_away = ratings[a]

        actual_home = match_outcome(g["home_score"], g["away_score"])

        if use_dynamic_k:
            k_h, k_a = _dynamic_k(k, game_counts[h], game_counts[a])
        else:
            k_h, k_a = k, k

        if use_mov:
            mov_mult = _mov_multiplier(
                g["home_score"], g["away_score"],
                r_home - r_away,
            )
            k_h *= mov_mult
            k_a *= mov_mult

        if recency_half_life > 0 and ref_date and g["date_utc"]:
            try:
                game_date = datetime.strptime(g["date_utc"], "%Y-%m-%d")
                days_ago = (ref_date - game_date).days
                if days_ago > 0:
                    decay = math.exp(-days_ago * math.log(2) / recency_half_life)
                    k_h *= decay
                    k_a *= decay
            except (ValueError, TypeError):
                pass

        new_home, new_away = update_elo(
            r_home, r_away, actual_home,
            k_home=k_h, k_away=k_a,
            scale=scale,
        )

        ratings[h] = new_home - home_adv
        ratings[a] = new_away
        game_counts[h] += 1
        game_counts[a] += 1

    return ratings


def elo_win_prob(
    r_home: float, r_away: float,
    home_adv: float = 65.0, scale: float = 400.0,
) -> float:
    return expected_score(r_home + home_adv, r_away, scale=scale)
