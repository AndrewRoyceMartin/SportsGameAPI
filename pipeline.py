from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from stats_provider import get_upcoming_games, get_results_history, Game, clear_events_cache
from features import build_elo_ratings, elo_win_prob
from odds_fetch import fetch_odds_for_window
from odds_extract import extract_moneylines, consensus_decimal
from mapper import match_games_to_odds, _parse_iso
from value_engine import implied_probability, edge, expected_value


def _game_to_elo_dict(g: Game) -> dict:
    return {
        "date_utc": g.start_time_utc.strftime("%Y-%m-%d"),
        "home_team": g.home,
        "away_team": g.away,
        "home_score": g.home_score or 0,
        "away_score": g.away_score or 0,
    }


def fetch_elo_ratings(league_label: str, history_days: int) -> Tuple[dict, int]:
    results = get_results_history(league=league_label, since_days=history_days)
    elo_dicts = [_game_to_elo_dict(g) for g in results]
    ratings = build_elo_ratings(elo_dicts) if elo_dicts else {}
    return ratings, len(results)


def fetch_harvest_odds(harvest_key: str, lookahead_days: int) -> List[Dict[str, Any]]:
    return fetch_odds_for_window(
        harvest_league=harvest_key,
        lookahead_days=lookahead_days,
    )


def count_games_with_odds(harvest_games: List[Dict[str, Any]]) -> int:
    count = 0
    for g in harvest_games:
        snaps = extract_moneylines(g)
        if snaps:
            count += 1
    return count


def harvest_date_range(
    harvest_games: List[Dict[str, Any]],
) -> Tuple[Optional[Any], Optional[Any]]:
    earliest = None
    latest = None
    for g in harvest_games:
        dt = _parse_iso(g.get("scheduledTime", ""))
        if dt:
            if earliest is None or dt < earliest:
                earliest = dt
            if latest is None or dt > latest:
                latest = dt
    return earliest, latest


def fetch_fixtures(
    league_label: str, lookahead_days: int, latest_game_date: Optional[date] = None
) -> Tuple[List[Game], date]:
    fixture_end = date.today() + timedelta(days=lookahead_days)
    if latest_game_date:
        fixture_end = max(fixture_end, latest_game_date)
    return get_upcoming_games(league=league_label, date_to=fixture_end), fixture_end


def match_fixtures_to_odds(
    upcoming: List[Game], harvest_games: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    return match_games_to_odds(upcoming, harvest_games)


def compute_values(
    matched: List[Dict[str, Any]],
    elo_ratings: dict,
    min_edge: float,
    odds_range: Tuple[float, float],
) -> List[Dict[str, Any]]:
    value_bets = []

    for m in matched:
        fx = m["fixture"]
        og = m["odds_game"]
        confidence = m["match_confidence"]

        snaps = extract_moneylines(og)
        if not snaps:
            continue

        home_odds = consensus_decimal(snaps, "home")
        away_odds = consensus_decimal(snaps, "away")

        r_home = float(elo_ratings.get(fx.home, 1500.0))
        r_away = float(elo_ratings.get(fx.away, 1500.0))
        p_home = elo_win_prob(r_home, r_away)
        p_away = 1.0 - p_home

        for side, odds_dec, model_p, selection_name in [
            ("home", home_odds, p_home, fx.home),
            ("away", away_odds, p_away, fx.away),
        ]:
            if odds_dec is None or odds_dec <= 1.0:
                continue
            if not (odds_range[0] <= odds_dec <= odds_range[1]):
                continue

            imp_p = implied_probability(odds_dec)
            e = edge(model_p, imp_p)
            ev = expected_value(model_p, odds_dec)

            if e < min_edge:
                continue

            value_bets.append(
                {
                    "date": fx.start_time_utc.strftime("%Y-%m-%d"),
                    "time": fx.start_time_utc.strftime("%H:%M UTC"),
                    "home_team": fx.home,
                    "away_team": fx.away,
                    "league": fx.league,
                    "market": "Moneyline",
                    "selection": selection_name,
                    "side": side,
                    "odds_decimal": odds_dec,
                    "implied_prob": imp_p,
                    "model_prob": model_p,
                    "edge": e,
                    "ev_per_unit": ev,
                    "home_elo": r_home,
                    "away_elo": r_away,
                    "match_confidence": confidence,
                }
            )

    value_bets.sort(key=lambda x: x["edge"], reverse=True)
    return value_bets


def dedup_best_side(value_bets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best = {}
    for vb in value_bets:
        key = (vb["home_team"], vb["away_team"])
        if key not in best or vb["edge"] > best[key]["edge"]:
            best[key] = vb
    deduped = list(best.values())
    deduped.sort(key=lambda x: x["edge"], reverse=True)
    return deduped
