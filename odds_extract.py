from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from statistics import median

from odds_math import american_to_decimal


@dataclass
class MoneylineSnapshot:
    sportsbook: str
    home_decimal: Optional[float]
    away_decimal: Optional[float]


def extract_moneylines(game: Dict[str, Any]) -> List[MoneylineSnapshot]:
    snaps: List[MoneylineSnapshot] = []
    for o in game.get("odds", []):
        book = str(o.get("sportsbook", "")).strip()
        ml = o.get("moneyLine") or {}
        home = american_to_decimal(ml.get("currentHomeOdds"))
        away = american_to_decimal(ml.get("currentAwayOdds"))
        # Keep only rows where we actually have a price
        if home is not None or away is not None:
            snaps.append(MoneylineSnapshot(book, home, away))
    return snaps


def consensus_decimal(snaps: List[MoneylineSnapshot], side: str) -> Optional[float]:
    """
    Compute a 'consensus' decimal price using the median across sportsbooks.
    side: 'home' or 'away'
    """
    vals: List[float] = []
    for s in snaps:
        v = s.home_decimal if side == "home" else s.away_decimal
        if v is not None:
            vals.append(v)
    return median(vals) if vals else None
