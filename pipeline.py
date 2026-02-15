from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from stats_provider import get_upcoming_games, get_results_history, Game, clear_events_cache
from features import build_elo_ratings, elo_win_prob
from odds_fetch import fetch_odds_for_window
from odds_extract import extract_moneylines, consensus_decimal
from mapper import match_games_to_odds, _parse_iso
from value_engine import implied_probability, edge, expected_value
from league_defaults import get_elo_params


def _game_to_elo_dict(g: Game) -> dict:
    return {
        "date_utc": g.start_time_utc.strftime("%Y-%m-%d"),
        "home_team": g.home,
        "away_team": g.away,
        "home_score": g.home_score or 0,
        "away_score": g.away_score or 0,
    }


def fetch_elo_ratings(league_label: str, history_days: int) -> Tuple[dict, dict, int]:
    results = get_results_history(league=league_label, since_days=history_days)
    elo_dicts = [
        _game_to_elo_dict(g) for g in results
        if g.home_score is not None and g.away_score is not None
    ]
    ep = get_elo_params(league_label)
    if elo_dicts:
        ratings, game_counts = build_elo_ratings(
            elo_dicts,
            k=ep["k"], home_adv=ep["home_adv"],
            scale=ep["scale"], recency_half_life=ep["recency_half_life"],
        )
    else:
        ratings, game_counts = {}, {}
    return ratings, game_counts, len(elo_dicts)


def fetch_harvest_odds(harvest_key: str, lookahead_days: int, league_label: str = "") -> List[Dict[str, Any]]:
    return fetch_odds_for_window(
        harvest_league=harvest_key,
        lookahead_days=lookahead_days,
        league_label=league_label or None,
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
    upcoming: List[Game], harvest_games: List[Dict[str, Any]], league: str = ""
) -> List[Dict[str, Any]]:
    return match_games_to_odds(upcoming, harvest_games, league=league)


def get_unmatched(
    upcoming: List[Game],
    harvest_games: List[Dict[str, Any]],
    matched: List[Dict[str, Any]],
) -> tuple:
    matched_fixture_keys = set()
    matched_odds_keys = set()
    for m in matched:
        fx = m["fixture"]
        matched_fixture_keys.add((fx.home, fx.away))
        og = m["odds_game"]
        h = og.get("homeTeam", {}).get("mediumName", "")
        a = og.get("awayTeam", {}).get("mediumName", "")
        matched_odds_keys.add((h, a))

    unmatched_fixtures = []
    for fx in upcoming:
        if (fx.home, fx.away) not in matched_fixture_keys:
            unmatched_fixtures.append({
                "home": fx.home,
                "away": fx.away,
                "time": fx.start_time_utc.strftime("%Y-%m-%d %H:%M UTC") if fx.start_time_utc else "?",
            })

    unmatched_odds = []
    for og in harvest_games:
        h = og.get("homeTeam", {}).get("mediumName", "")
        a = og.get("awayTeam", {}).get("mediumName", "")
        if (h, a) not in matched_odds_keys:
            t = og.get("scheduledTime", "?")
            unmatched_odds.append({"home": h, "away": a, "time": str(t)[:19]})

    return unmatched_fixtures[:10], unmatched_odds[:10]


def compute_values(
    matched: List[Dict[str, Any]],
    elo_ratings: dict,
    min_edge: float,
    odds_range: Tuple[float, float],
    league: str = "",
    game_counts: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    ep = get_elo_params(league) if league else {"home_adv": 65, "scale": 400}
    gc = game_counts or {}
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
        p_home = elo_win_prob(r_home, r_away, home_adv=ep["home_adv"], scale=ep["scale"])
        p_away = 1.0 - p_home

        home_games = gc.get(fx.home, 0)
        away_games = gc.get(fx.away, 0)
        min_games = min(home_games, away_games)

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
                    "home_games": home_games,
                    "away_games": away_games,
                    "min_games": min_games,
                }
            )

    value_bets.sort(key=lambda x: x["edge"], reverse=True)
    return value_bets


def run_backtest(
    league_label: str,
    history_days: int = 120,
    test_days: int = 30,
) -> Dict[str, Any]:
    ep = get_elo_params(league_label)
    total_days = history_days + test_days
    all_results = get_results_history(league=league_label, since_days=total_days)

    if not all_results:
        return {"error": "No historical results found.", "games": []}

    all_results.sort(key=lambda g: g.start_time_utc)

    cutoff = all_results[-1].start_time_utc - timedelta(days=test_days)

    train_games = [g for g in all_results if g.start_time_utc <= cutoff]
    test_games = [g for g in all_results if g.start_time_utc > cutoff]

    if not test_games:
        return {"error": "Not enough recent games to test against.", "games": []}

    if not train_games:
        return {"error": "Not enough training data. Try increasing history days.", "games": []}

    train_dicts = [
        _game_to_elo_dict(g) for g in train_games
        if g.home_score is not None and g.away_score is not None
    ]

    elo_kwargs = dict(
        k=ep["k"], home_adv=ep["home_adv"],
        scale=ep["scale"], recency_half_life=ep["recency_half_life"],
    )

    results = []
    correct = 0
    total = 0
    draws = 0
    confident_correct = 0
    confident_total = 0
    brier_sum = 0.0
    log_loss_sum = 0.0

    rolling_dicts = list(train_dicts)

    for g in test_games:
        if g.home_score is None or g.away_score is None:
            continue

        current_ratings, _ = build_elo_ratings(rolling_dicts, **elo_kwargs)

        r_home = float(current_ratings.get(g.home, 1500.0))
        r_away = float(current_ratings.get(g.away, 1500.0))
        p_home = elo_win_prob(r_home, r_away, home_adv=ep["home_adv"], scale=ep["scale"])
        p_away = 1.0 - p_home

        predicted_winner = g.home if p_home >= 0.5 else g.away
        predicted_prob = max(p_home, p_away)

        is_draw = g.home_score == g.away_score
        actual_outcome = 0.5

        if is_draw:
            actual_winner = "Draw"
            is_correct = None
            draws += 1
        elif g.home_score > g.away_score:
            actual_winner = g.home
            is_correct = (predicted_winner == g.home)
            actual_outcome = 1.0
        else:
            actual_winner = g.away
            is_correct = (predicted_winner == g.away)
            actual_outcome = 0.0

        if is_correct is not None:
            total += 1
            if is_correct:
                correct += 1
            if predicted_prob >= 0.60:
                confident_total += 1
                if is_correct:
                    confident_correct += 1

            brier_sum += (p_home - actual_outcome) ** 2

            p_clipped = max(1e-10, min(1.0 - 1e-10, p_home))
            log_loss_sum -= (
                actual_outcome * math.log(p_clipped)
                + (1.0 - actual_outcome) * math.log(1.0 - p_clipped)
            )

        results.append({
            "date": g.start_time_utc.strftime("%Y-%m-%d"),
            "home_team": g.home,
            "away_team": g.away,
            "home_score": g.home_score,
            "away_score": g.away_score,
            "predicted_winner": predicted_winner,
            "predicted_prob": predicted_prob,
            "actual_winner": actual_winner,
            "correct": is_correct,
            "is_draw": is_draw,
            "home_elo": r_home,
            "away_elo": r_away,
            "p_home": p_home,
        })

        rolling_dicts.append(_game_to_elo_dict(g))

    accuracy = correct / total if total > 0 else 0
    confident_accuracy = confident_correct / confident_total if confident_total > 0 else 0
    brier_score = brier_sum / total if total > 0 else 0
    log_loss = log_loss_sum / total if total > 0 else 0

    overall_acc = accuracy
    confident_acc = confident_accuracy
    bucket_lift = (confident_acc - overall_acc) if confident_total > 0 else None

    return {
        "league": league_label,
        "train_games": len(train_dicts),
        "test_games": total,
        "draws": draws,
        "correct": correct,
        "accuracy": accuracy,
        "confident_total": confident_total,
        "confident_correct": confident_correct,
        "confident_accuracy": confident_accuracy,
        "brier_score": brier_score,
        "log_loss": log_loss,
        "bucket_lift": bucket_lift,
        "elo_params": ep,
        "games": results,
    }


def _score_fold(
    all_games: List,
    train_end_idx: int,
    test_end_idx: int,
    elo_kwargs: dict,
    ep: dict,
) -> Optional[Dict[str, Any]]:
    train_dicts = [
        _game_to_elo_dict(g) for g in all_games[:train_end_idx]
        if g.home_score is not None and g.away_score is not None
    ]
    test_games = [
        g for g in all_games[train_end_idx:test_end_idx]
        if g.home_score is not None and g.away_score is not None
    ]
    if not train_dicts or not test_games:
        return None

    rolling_dicts = list(train_dicts)
    correct = 0
    total = 0
    draws = 0
    brier_sum = 0.0
    log_loss_sum = 0.0
    conf_buckets = {60: [0, 0], 65: [0, 0], 70: [0, 0]}
    naive_correct = 0
    fav_correct = 0

    for g in test_games:
        is_draw = g.home_score == g.away_score
        if is_draw:
            draws += 1
            rolling_dicts.append(_game_to_elo_dict(g))
            continue

        current_ratings, _ = build_elo_ratings(rolling_dicts, **elo_kwargs)
        r_home = float(current_ratings.get(g.home, 1500.0))
        r_away = float(current_ratings.get(g.away, 1500.0))
        p_home = elo_win_prob(r_home, r_away, home_adv=ep["home_adv"], scale=ep["scale"])

        predicted_winner = g.home if p_home >= 0.5 else g.away
        predicted_prob = max(p_home, 1.0 - p_home)
        actual_outcome = 1.0 if g.home_score > g.away_score else 0.0
        actual_winner = g.home if actual_outcome == 1.0 else g.away
        is_correct = predicted_winner == actual_winner

        total += 1
        if is_correct:
            correct += 1

        if p_home >= 0.5:
            naive_correct += (1 if actual_outcome == 1.0 else 0)
        else:
            naive_correct += (1 if actual_outcome == 0.0 else 0)

        fav_correct += (1 if actual_outcome == 1.0 else 0)

        brier_sum += (p_home - actual_outcome) ** 2
        p_clipped = max(1e-10, min(1.0 - 1e-10, p_home))
        log_loss_sum -= (
            actual_outcome * math.log(p_clipped)
            + (1.0 - actual_outcome) * math.log(1.0 - p_clipped)
        )

        for threshold in conf_buckets:
            if predicted_prob >= threshold / 100.0:
                conf_buckets[threshold][1] += 1
                if is_correct:
                    conf_buckets[threshold][0] += 1

        rolling_dicts.append(_game_to_elo_dict(g))

    if total == 0:
        return None

    accuracy = correct / total
    brier = brier_sum / total
    ll = log_loss_sum / total
    conf_acc = {}
    for t, (c, n) in conf_buckets.items():
        conf_acc[t] = c / n if n > 0 else None
    bucket_lift_60 = (conf_acc[60] - accuracy) if conf_acc[60] is not None else None

    return {
        "train_games": len(train_dicts),
        "test_games": total,
        "draws": draws,
        "accuracy": accuracy,
        "brier_score": brier,
        "log_loss": ll,
        "bucket_lift": bucket_lift_60,
        "conf_acc": conf_acc,
        "conf_counts": {t: n for t, (_, n) in conf_buckets.items()},
        "naive_baseline_acc": naive_correct / total,
        "always_home_acc": fav_correct / total,
    }


def run_walkforward_backtest(
    league_label: str,
    total_days: int = 240,
    train_days: int = 180,
    test_days: int = 30,
    folds: int = 4,
) -> Dict[str, Any]:
    ep = get_elo_params(league_label)
    all_results = get_results_history(league=league_label, since_days=total_days)

    if not all_results:
        return {"error": "No historical results found.", "folds": []}

    all_results.sort(key=lambda g: g.start_time_utc)

    elo_kwargs = dict(
        k=ep["k"], home_adv=ep["home_adv"],
        scale=ep["scale"], recency_half_life=ep["recency_half_life"],
    )

    naive_elo_kwargs = dict(k=20, home_adv=65, scale=400, recency_half_life=0)
    naive_ep = {"home_adv": 65, "scale": 400}

    fold_results = []
    naive_fold_results = []

    total_games = len(all_results)

    for fold_i in range(folds):
        test_end_offset = fold_i * test_days
        test_end_dt = all_results[-1].start_time_utc - timedelta(days=test_end_offset)
        test_start_dt = test_end_dt - timedelta(days=test_days)
        train_start_dt = test_start_dt - timedelta(days=train_days)

        train_end_idx = 0
        test_end_idx = 0
        for idx, g in enumerate(all_results):
            if g.start_time_utc <= test_start_dt:
                train_end_idx = idx + 1
            if g.start_time_utc <= test_end_dt:
                test_end_idx = idx + 1

        train_start_idx = 0
        for idx, g in enumerate(all_results):
            if g.start_time_utc >= train_start_dt:
                train_start_idx = idx
                break

        fold_games = all_results[train_start_idx:test_end_idx]
        local_train_end = train_end_idx - train_start_idx
        local_test_end = test_end_idx - train_start_idx

        fold_score = _score_fold(fold_games, local_train_end, local_test_end, elo_kwargs, ep)
        if fold_score:
            fold_score["fold"] = fold_i + 1
            fold_score["test_start"] = test_start_dt.strftime("%Y-%m-%d")
            fold_score["test_end"] = test_end_dt.strftime("%Y-%m-%d")
            fold_results.append(fold_score)

        naive_score = _score_fold(fold_games, local_train_end, local_test_end, naive_elo_kwargs, naive_ep)
        if naive_score:
            naive_fold_results.append(naive_score)

    if not fold_results:
        return {"error": "Not enough data for walk-forward validation.", "folds": []}

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0

    def _std(vals):
        if len(vals) < 2:
            return 0
        m = _mean(vals)
        return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

    accs = [f["accuracy"] for f in fold_results]
    briers = [f["brier_score"] for f in fold_results]
    lls = [f["log_loss"] for f in fold_results]
    lifts = [f["bucket_lift"] for f in fold_results if f["bucket_lift"] is not None]

    naive_accs = [f["accuracy"] for f in naive_fold_results] if naive_fold_results else []
    naive_briers = [f["brier_score"] for f in naive_fold_results] if naive_fold_results else []

    return {
        "league": league_label,
        "folds": fold_results,
        "num_folds": len(fold_results),
        "elo_params": ep,
        "summary": {
            "accuracy_mean": _mean(accs),
            "accuracy_std": _std(accs),
            "brier_mean": _mean(briers),
            "brier_std": _std(briers),
            "log_loss_mean": _mean(lls),
            "log_loss_std": _std(lls),
            "bucket_lift_mean": _mean(lifts) if lifts else None,
            "bucket_lift_std": _std(lifts) if lifts else None,
            "conf_60_accs": [f["conf_acc"].get(60) for f in fold_results if f["conf_acc"].get(60) is not None],
            "conf_65_accs": [f["conf_acc"].get(65) for f in fold_results if f["conf_acc"].get(65) is not None],
            "conf_70_accs": [f["conf_acc"].get(70) for f in fold_results if f["conf_acc"].get(70) is not None],
        },
        "baselines": {
            "naive_elo_accuracy_mean": _mean(naive_accs) if naive_accs else None,
            "naive_elo_brier_mean": _mean(naive_briers) if naive_briers else None,
            "always_home_acc": _mean([f["always_home_acc"] for f in fold_results]) if fold_results else None,
        },
    }


def dedup_best_side(value_bets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best = {}
    for vb in value_bets:
        key = (vb["home_team"], vb["away_team"])
        if key not in best or vb["edge"] > best[key]["edge"]:
            best[key] = vb
    deduped = list(best.values())
    deduped.sort(key=lambda x: x["edge"], reverse=True)
    return deduped
