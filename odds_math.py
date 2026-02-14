from __future__ import annotations

from typing import Optional


def american_to_decimal(american: Optional[float]) -> Optional[float]:
    """
    Convert American odds to decimal odds.
    Examples:
      +150 -> 2.50
      -200 -> 1.50
    """
    if american is None:
        return None
    try:
        a = float(american)
    except (TypeError, ValueError):
        return None

    if a == 0:
        return None

    if a > 0:
        return 1.0 + (a / 100.0)
    else:
        return 1.0 + (100.0 / abs(a))


def implied_probability_from_decimal(decimal_odds: Optional[float]) -> Optional[float]:
    if decimal_odds is None or decimal_odds <= 1.0:
        return None
    return 1.0 / decimal_odds
