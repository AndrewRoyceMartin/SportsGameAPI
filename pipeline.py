from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from stats_provider import get_upcoming_games, get_results_history, Game, clear_events_cache
from features import build_elo_ratings, elo_win_prob
from odds_fetch import fetch_odds_for_window, get_odds_provider_name, probe_odds_provider
from odds_extract import extract_moneylines, consensus_decimal
from mapper import match_games_to_odds, _parse_iso, detect_time_offset, apply_time_offset
from time_utils import to_naive_utc, parse_iso_utc
from value_engine import implied_probability, edge, expected_value
from league_defaults import get_elo_params, DEFAULTS
from league_map import LEAGUE_MAP
import time as _time
import logging as _logging

_preflight_logger = _logging.getLogger(__name__)


def preflight_availability(
    league_key: str,
    *,
    lookahead_override: int | None = None,
) -> Dict[str, Any]:
    defaults = DEFAULTS.get(league_key, {})
    lookahead = lookahead_override or defaults.get("lookahead_days", 7)
    today = date.today()
    date_to = today + timedelta(days=lookahead)

    earliest_utc = None
    try:
        fixtures = get_upcoming_games(league=league_key, date_from=today, date_to=date_to)
        fixtures_count = len(fixtures)
        stats_error = False
        timed = [g for g in fixtures if getattr(g, "start_time_utc", None) is not None]
        if timed:
            timed.sort(key=lambda g: g.start_time_utc)
            earliest_utc = timed[0].start_time_utc
    except Exception as exc:
        _preflight_logger.warning("Preflight stats error for %s: %s", league_key, exc)
        fixtures_count = 0
        stats_error = True

    odds_provider = get_odds_provider_name(league_key)

    if stats_error:
        status = "error"
    elif fixtures_count > 0:
        status = "ready"
    else:
        status = "no_fixtures"

    return {
        "league": league_key,
        "fixtures_count": fixtures_count,
        "earliest_utc": earliest_utc,
        "lookahead_days": lookahead,
        "window": (str(today), str(date_to)),
        "status": status,
        "odds_provider": odds_provider,
        "odds_status": "not_probed",
        "odds_raw_items": None,
        "odds_usable": None,
        "odds_sample": None,
        "odds_error": None,
    }


def soft_match_preview(
    fixtures: List[Game],
    odds_events: List[Dict[str, Any]],
    league: str = "",
) -> Dict[str, Any]:
    from mapper import _name_score, _parse_iso, _AU_LEAGUES
    from time_utils import to_naive_utc

    if not fixtures or not odds_events:
        return {
            "potential_matches": 0,
            "total_fixtures": len(fixtures),
            "total_odds": len(odds_events),
            "issue": "missing_data",
            "offset_info": None,
            "preview": [],
        }

    is_au = league in _AU_LEAGUES
    name_thresh = 55 if is_au else 60
    soft_window_h = 36 if is_au else 12

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
        for og in odds_events:
            og_home = (og.get("homeTeam") or {}).get("mediumName", "")
            og_away = (og.get("awayTeam") or {}).get("mediumName", "")
            og_dt = _parse_iso(og.get("scheduledTime", ""))

            home_sc = _name_score(fx.home, og_home)
            away_sc = _name_score(fx.away, og_away)
            avg = (home_sc + away_sc) / 2
            home_flip = _name_score(fx.home, og_away)
            away_flip = _name_score(fx.away, og_home)
            avg_flip = (home_flip + away_flip) / 2
            best = max(avg, avg_flip)

            if best >= name_thresh and best > best_name:
                delta_h = None
                if og_dt is not None and fx_dt is not None:
                    delta_h = (og_dt - fx_dt).total_seconds() / 3600
                best_name = best
                best_pair = {
                    "fixture": f"{fx.home} vs {fx.away}",
                    "odds": f"{og_home} vs {og_away}",
                    "fx_time": str(fx_dt)[:19] if fx_dt else "?",
                    "odds_time": str(og_dt)[:19] if og_dt else "?",
                    "delta_h": round(delta_h, 2) if delta_h is not None else None,
                    "name_score": best,
                }

        if best_pair:
            pairs.append(best_pair)
            seen_fx.add(fx_key)

    preview = []
    for p in pairs:
        preview.append({
            "fixture": p["fixture"],
            "odds_event": p["odds"],
            "fixture_time": p["fx_time"],
            "odds_time": p["odds_time"],
            "time_gap_h": p["delta_h"],
            "name_score": p["name_score"],
            "diagnosis": _diagnose_pair(p),
        })

    deltas_with_time = [p["delta_h"] for p in pairs if p["delta_h"] is not None]
    if deltas_with_time:
        deltas_with_time.sort()
        median_offset = deltas_with_time[len(deltas_with_time) // 2]
        within_2h = sum(1 for d in deltas_with_time if abs(d - median_offset) <= 2)
        consistency = within_2h / len(deltas_with_time)
    else:
        median_offset = 0
        consistency = 0

    offset_info = {
        "offset_hours": round(median_offset, 2),
        "consistency": round(consistency, 2),
        "pairs_checked": len(pairs),
    }

    abs_offset = abs(median_offset)
    time_mismatched = sum(1 for d in deltas_with_time if abs(d) > soft_window_h)

    if len(pairs) == 0:
        issue = "no_name_matches"
    elif abs_offset > 3 and consistency >= 0.5:
        issue = "time_mismatch"
    elif abs_offset > 1.5 and len(deltas_with_time) >= 2:
        issue = "time_mismatch_uncertain"
    elif time_mismatched > 0 and time_mismatched >= len(deltas_with_time) * 0.5:
        issue = "time_mismatch"
    else:
        issue = None

    return {
        "potential_matches": len(pairs),
        "total_fixtures": len(fixtures),
        "total_odds": len(odds_events),
        "issue": issue,
        "offset_info": offset_info,
        "preview": preview[:15],
    }


def _diagnose_pair(pair: Dict[str, Any]) -> str:
    delta_raw = pair.get("delta_h")
    if delta_raw is None:
        return "No time data"
    delta = abs(delta_raw)
    name_sc = pair.get("name_score", 0)
    if name_sc >= 70 and delta <= 4:
        return "OK"
    if name_sc >= 70 and delta <= 12:
        return "Slight time gap"
    if name_sc >= 70 and delta > 12:
        return "Time mismatch"
    if name_sc >= 55 and delta <= 6:
        return "Weak name, close time"
    if name_sc >= 55:
        return "Weak name match"
    return "Poor match"


def _cached_odds_probe(league_key: str, harvest_key: str, lookahead_days: int) -> Dict[str, Any]:
    return probe_odds_provider(
        league_key,
        harvest_league_key=harvest_key,
        lookahead_days=lookahead_days,
        timeout=60,
    )


def preflight_with_odds_probe(
    league_key: str,
    *,
    lookahead_override: int | None = None,
) -> Dict[str, Any]:
    result = preflight_availability(league_key, lookahead_override=lookahead_override)

    if result["status"] == "no_fixtures":
        result["odds_status"] = "skipped"
        return result

    harvest_key = LEAGUE_MAP.get(league_key, league_key)

    cache_key = f"odds_probe_{league_key}"
    import streamlit as st
    import time as _t
    cached = st.session_state.get(cache_key)
    ttl = 300
    odds_events: List[Dict[str, Any]] = []
    if cached and (_t.time() - cached.get("_ts", 0)) < ttl:
        probe = cached
        odds_events = cached.get("_events", [])
    else:
        probe = _cached_odds_probe(league_key, harvest_key, result["lookahead_days"])
        odds_events = probe.pop("events", [])
        probe["_ts"] = _t.time()
        probe["_events"] = odds_events
        st.session_state[cache_key] = probe

    result["odds_provider"] = probe["provider"]
    result["odds_raw_items"] = probe.get("raw_items", 0)
    result["odds_usable"] = probe.get("usable_odds_events", 0)
    result["odds_sample"] = probe.get("sample_matchup")
    result["odds_error"] = probe.get("error", "")

    if probe["error"]:
        result["odds_status"] = "error"
        if result["status"] == "ready":
            result["status"] = "odds_error"
    elif probe["ok"]:
        result["odds_status"] = "ok"
    else:
        result["odds_status"] = "empty"
        if result["status"] == "ready":
            result["status"] = "odds_empty"

    fixtures: List[Game] = []
    if result.get("fixtures_count", 0) > 0 and odds_events:
        try:
            today = date.today()
            lookahead = result.get("lookahead_days", 7)
            date_to = today + timedelta(days=lookahead)
            fixtures = get_upcoming_games(league=league_key, date_to=date_to)
        except Exception:
            fixtures = []

    if fixtures and odds_events:
        match_preview = soft_match_preview(fixtures, odds_events, league=league_key)
        result["match_preview"] = match_preview

        if match_preview.get("issue") == "time_mismatch" and result["status"] in ("ready",):
            result["status"] = "time_mismatch"
        elif match_preview.get("issue") == "no_name_matches" and result["status"] in ("ready",):
            result["status"] = "name_mismatch"

    return result


def preflight_scan(
    leagues: List[str],
    *,
    lookahead_days: int | None = None,
    delay: float = 0.25,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for lg in leagues:
        results.append(preflight_availability(lg, lookahead_override=lookahead_days))
        if delay > 0 and lg != leagues[-1]:
            _time.sleep(delay)
    return results


def _game_to_elo_dict(g: Game) -> dict:
    return {
        "date_utc": g.start_time_utc.strftime("%Y-%m-%d"),
        "home_team": g.home,
        "away_team": g.away,
        "home_score": g.home_score or 0,
        "away_score": g.away_score or 0,
    }


def fetch_elo_ratings(
    league_label: str,
    history_days: int,
    elo_overrides: dict | None = None,
) -> Tuple[dict, dict, int]:
    results = get_results_history(league=league_label, since_days=history_days)
    elo_dicts = [
        _game_to_elo_dict(g) for g in results
        if g.home_score is not None and g.away_score is not None
    ]
    ep = get_elo_params(league_label, overrides=elo_overrides)
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
    _preflight_logger.info(
        "[PIPE] fetch_harvest_odds: harvest_key=%s, lookahead=%d, league_label=%s",
        harvest_key, lookahead_days, league_label,
    )
    result = fetch_odds_for_window(
        harvest_league=harvest_key,
        lookahead_days=lookahead_days,
        league_label=league_label or None,
    )
    _preflight_logger.info("[PIPE] odds fetched = %d events", len(result))
    return result


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


def summarize_match_time_deltas(
    matched: List[Dict[str, Any]],
) -> Dict[str, Any]:
    deltas = []
    suspicious = []
    for m in matched:
        fx = m["fixture"]
        og = m["odds_game"]
        td = m.get("time_delta_hours")
        if td is not None:
            deltas.append(td)
            if td > 12:
                fx_time = to_naive_utc(fx.start_time_utc)
                og_time = _parse_iso(og.get("scheduledTime", ""))
                suspicious.append({
                    "fixture": f"{fx.home} vs {fx.away}",
                    "fx_time": str(fx_time)[:19] if fx_time else "?",
                    "odds_time": og.get("scheduledTime", "?")[:19],
                    "delta_h": round(td, 1),
                    "confidence": m.get("match_confidence", 0),
                })
        else:
            fx_dt = to_naive_utc(fx.start_time_utc)
            og_dt = _parse_iso(og.get("scheduledTime", ""))
            if fx_dt and og_dt:
                td_calc = abs((fx_dt - og_dt).total_seconds()) / 3600
                deltas.append(td_calc)
                if td_calc > 12:
                    suspicious.append({
                        "fixture": f"{fx.home} vs {fx.away}",
                        "fx_time": str(fx_dt)[:19],
                        "odds_time": og.get("scheduledTime", "?")[:19],
                        "delta_h": round(td_calc, 1),
                        "confidence": m.get("match_confidence", 0),
                    })

    if not deltas:
        return {"count": len(matched), "median_h": None, "max_h": None, "gt_12h": 0, "gt_24h": 0, "suspicious": []}

    deltas.sort()
    median = deltas[len(deltas) // 2]
    return {
        "count": len(matched),
        "with_times": len(deltas),
        "median_h": round(median, 2),
        "max_h": round(max(deltas), 2),
        "gt_12h": sum(1 for d in deltas if d > 12),
        "gt_24h": sum(1 for d in deltas if d > 24),
        "suspicious": suspicious[:10],
    }


def compute_values(
    matched: List[Dict[str, Any]],
    elo_ratings: dict,
    min_edge: float,
    odds_range: Tuple[float, float],
    league: str = "",
    game_counts: Optional[Dict[str, int]] = None,
    elo_overrides: dict | None = None,
) -> List[Dict[str, Any]]:
    ep = get_elo_params(league, overrides=elo_overrides) if league else {"home_adv": 65, "scale": 400}
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

    diagnostics = _compute_backtest_diagnostics(results)

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
        "diagnostics": diagnostics,
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
    fold_game_results = []

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

        fold_game_results.append({
            "home_team": g.home,
            "away_team": g.away,
            "home_score": g.home_score,
            "away_score": g.away_score,
            "predicted_winner": predicted_winner,
            "predicted_prob": predicted_prob,
            "correct": is_correct,
            "home_elo": r_home,
            "away_elo": r_away,
            "p_home": p_home,
        })

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
        "fold_games": fold_game_results,
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

    all_fold_games = []
    for f in fold_results:
        all_fold_games.extend(f.get("fold_games", []))
    wf_diagnostics = _compute_backtest_diagnostics(all_fold_games)

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
        "diagnostics": wf_diagnostics,
    }


def _compute_backtest_diagnostics(games: List[Dict[str, Any]]) -> Dict[str, Any]:
    decisive = [g for g in games if g.get("correct") is not None]
    if not decisive:
        return {}

    total = len(decisive)
    picked_home_count = sum(1 for g in decisive if g["predicted_winner"] == g["home_team"])
    actual_home_count = sum(
        1 for g in decisive
        if g["home_score"] > g["away_score"]
    )
    home_pick_rate = picked_home_count / total if total else 0
    actual_home_rate = actual_home_count / total if total else 0
    home_bias_gap = home_pick_rate - actual_home_rate

    thresholds = [55, 60, 65, 70]
    acc_by_conf = {}
    for t in thresholds:
        t_frac = t / 100.0
        bucket = [g for g in decisive if g["predicted_prob"] >= t_frac]
        if bucket:
            correct = sum(1 for g in bucket if g["correct"])
            acc_by_conf[f"{t}%+"] = {
                "games": len(bucket),
                "correct": correct,
                "accuracy": correct / len(bucket),
            }

    from collections import Counter
    wrong_confident = [
        g for g in decisive
        if not g["correct"] and g["predicted_prob"] >= 0.65
    ]
    team_wrong = Counter()
    team_prob_sum = Counter()
    for g in wrong_confident:
        team_wrong[g["predicted_winner"]] += 1
        team_prob_sum[g["predicted_winner"]] += g["predicted_prob"]
    overconfident = []
    for team, count in team_wrong.most_common(10):
        overconfident.append({
            "team": team,
            "wrong_count": count,
            "avg_prob": team_prob_sum[team] / count,
        })

    prob_buckets = [
        ("50-55%", 0.50, 0.55),
        ("55-60%", 0.55, 0.60),
        ("60-65%", 0.60, 0.65),
        ("65-70%", 0.65, 0.70),
        ("70-80%", 0.70, 0.80),
        ("80%+", 0.80, 1.01),
    ]
    calibration = []
    for label, lo, hi in prob_buckets:
        bucket = [g for g in decisive if lo <= g["predicted_prob"] < hi]
        if bucket:
            empirical = sum(1 for g in bucket if g["correct"]) / len(bucket)
            midpoint = (lo + hi) / 2
            calibration.append({
                "bucket": label,
                "predicted_avg": midpoint,
                "empirical_win_rate": empirical,
                "games": len(bucket),
                "gap": empirical - midpoint,
            })

    return {
        "home_pick_rate": home_pick_rate,
        "actual_home_rate": actual_home_rate,
        "home_bias_gap": home_bias_gap,
        "acc_by_confidence": acc_by_conf,
        "overconfident_losses": overconfident,
        "calibration": calibration,
        "total_decisive": total,
    }


def tune_elo_params(
    league_label: str,
    total_days: int = 240,
    train_days: int = 180,
    test_days: int = 30,
) -> Dict[str, Any]:
    import streamlit as st

    ep_base = get_elo_params(league_label)
    all_results = get_results_history(league=league_label, since_days=total_days)

    if not all_results:
        return {"error": "No historical results found."}

    all_results.sort(key=lambda g: g.start_time_utc)

    k_values = [12, 18, 24, 32]
    home_adv_values = [0, 20, 40, 60, 80]

    best_score = float("inf")
    best_params = {"k": ep_base["k"], "home_adv": ep_base["home_adv"]}
    best_accuracy = 0.0
    grid_results: List[Dict[str, Any]] = []

    cutoff_dt = all_results[-1].start_time_utc - timedelta(days=test_days)
    train_start_dt = cutoff_dt - timedelta(days=train_days)

    train_games = [
        g for g in all_results
        if g.start_time_utc >= train_start_dt and g.start_time_utc <= cutoff_dt
        and g.home_score is not None and g.away_score is not None
    ]
    test_games = [
        g for g in all_results
        if g.start_time_utc > cutoff_dt
        and g.home_score is not None and g.away_score is not None
    ]

    if not train_games or not test_games:
        return {"error": "Not enough data for tuning."}

    train_dicts = [_game_to_elo_dict(g) for g in train_games]

    grid = [(k_val, ha_val) for k_val in k_values for ha_val in home_adv_values]
    total_combos = len(grid)
    progress = st.progress(0)
    status = st.empty()

    for combo_idx, (k_val, ha_val) in enumerate(grid, start=1):
        status.caption(f"Quick calibrate: {combo_idx}/{total_combos} — K={k_val} HA={ha_val}")
        progress.progress(combo_idx / total_combos)

        ratings, game_counts = build_elo_ratings(
            train_dicts,
            k=float(k_val),
            home_adv=float(ha_val),
            scale=ep_base["scale"],
            recency_half_life=ep_base["recency_half_life"],
        )

        correct = 0
        total = 0
        log_loss_sum = 0.0

        for g in test_games:
            hs = g.home_score or 0
            as_ = g.away_score or 0
            if hs == as_:
                continue

            r_h = float(ratings.get(g.home, 1500.0))
            r_a = float(ratings.get(g.away, 1500.0))
            p_home = elo_win_prob(r_h, r_a, home_adv=ha_val, scale=ep_base["scale"])

            predicted = g.home if p_home >= 0.5 else g.away
            actual_outcome = 1.0 if hs > as_ else 0.0
            actual_winner = g.home if actual_outcome == 1.0 else g.away

            total += 1
            if predicted == actual_winner:
                correct += 1

            p_clip = max(1e-10, min(1.0 - 1e-10, p_home))
            log_loss_sum -= (
                actual_outcome * math.log(p_clip)
                + (1.0 - actual_outcome) * math.log(1.0 - p_clip)
            )

        if total == 0:
            continue

        ll = log_loss_sum / total
        acc = correct / total

        grid_results.append({
            "K": k_val,
            "Home Adv": ha_val,
            "Log Loss": round(ll, 4),
            "Accuracy": round(acc, 4),
            "Games": total,
        })

        if ll < best_score:
            best_score = ll
            best_params = {"k": k_val, "home_adv": ha_val}
            best_accuracy = acc

    progress.empty()
    status.empty()

    grid_results.sort(key=lambda x: x["Log Loss"])

    return {
        "league": league_label,
        "best_params": best_params,
        "best_log_loss": best_score,
        "best_accuracy": best_accuracy,
        "current_params": {"k": ep_base["k"], "home_adv": ep_base["home_adv"]},
        "grid_results": grid_results[:20],
        "total_combos": len(grid_results),
        "train_games": len(train_dicts),
        "test_games": len(test_games),
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
