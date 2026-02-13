from __future__ import annotations

import math
from typing import Any


K_FACTOR = 32.0
INITIAL_ELO = 1500.0


def elo_win_prob(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def _update_elo(
    rating_w: float, rating_l: float, drawn: bool = False
) -> tuple[float, float]:
    expected_w = elo_win_prob(rating_w, rating_l)
    if drawn:
        score_w = 0.5
    else:
        score_w = 1.0
    delta = K_FACTOR * (score_w - expected_w)
    return rating_w + delta, rating_l - delta


def build_elo_ratings(games: list[dict[str, Any]]) -> dict[str, float]:
    elo: dict[str, float] = {}

    for g in games:
        home = g.get("home_team", "")
        away = g.get("away_team", "")
        if not home or not away:
            continue

        home_score = g.get("home_score")
        away_score = g.get("away_score")
        if home_score is None or away_score is None:
            continue

        home_score = int(home_score)
        away_score = int(away_score)

        r_h = elo.get(home, INITIAL_ELO)
        r_a = elo.get(away, INITIAL_ELO)

        if home_score > away_score:
            r_h, r_a = _update_elo(r_h, r_a, drawn=False)
        elif away_score > home_score:
            r_a, r_h = _update_elo(r_a, r_h, drawn=False)
        else:
            r_h, r_a = _update_elo(r_h, r_a, drawn=True)

        elo[home] = r_h
        elo[away] = r_a

    return elo
