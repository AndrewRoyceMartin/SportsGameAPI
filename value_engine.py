from __future__ import annotations


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
