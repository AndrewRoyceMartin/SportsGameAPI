from __future__ import annotations

import os
import streamlit as st
import pandas as pd
from datetime import date, timedelta

from store import save_picks, get_recent_picks, init_db
from league_map import LEAGUE_MAP, sofascore_to_harvest, available_leagues, is_two_outcome, is_separator
from league_defaults import DEFAULTS
from stats_provider import get_upcoming_games, get_results_history, Game
from features import build_elo_ratings, elo_win_prob
from apify_client import run_actor_get_items
from odds_extract import extract_moneylines, consensus_decimal
from mapper import match_games_to_odds
from value_engine import implied_probability, edge, expected_value

SOFASCORE_LEAGUE_FILTERS = {
    "Champions League": "Champions League",
    "NBA": "NBA",
    "NHL": "NHL",
    "NFL": "NFL",
    "College Football": "College Football",
    "College Basketball": "College Basketball",
    "UFC": "UFC",
}


def _game_to_elo_dict(g: Game) -> dict:
    return {
        "date_utc": g.start_time_utc.strftime("%Y-%m-%d"),
        "home_team": g.home,
        "away_team": g.away,
        "home_score": g.home_score or 0,
        "away_score": g.away_score or 0,
    }


def main():
    st.set_page_config(page_title="Sports Predictor", page_icon="⚽", layout="wide")

    st.title("⚽ Sports Predictor")
    st.caption("Find +EV bets by comparing Elo model probabilities against live sportsbook odds")

    with st.sidebar:
        st.header("Configuration")

        leagues = available_leagues()
        default_idx = leagues.index("NBA") if "NBA" in leagues else 0
        league_label = st.selectbox("League", options=leagues, index=default_idx)

        if is_separator(league_label):
            st.warning("Please select a league above or below the separator.")
            st.stop()

        def _apply_defaults(league: str):
            d = DEFAULTS.get(league)
            if not d:
                return
            st.session_state["min_edge"] = d["min_edge"]
            st.session_state["odds_range"] = (d["min_odds"], d["max_odds"])
            st.session_state["history_days"] = d["history_days"]
            st.session_state["lookahead_days"] = d["lookahead_days"]
            st.session_state["top_n"] = d["top_n"]

        prev = st.session_state.get("_prev_league")
        if prev != league_label:
            _apply_defaults(league_label)
            st.session_state["_prev_league"] = league_label

        d_fb = DEFAULTS.get(league_label, {})
        if "min_edge" not in st.session_state:
            st.session_state["min_edge"] = d_fb.get("min_edge", 5)
        if "odds_range" not in st.session_state:
            st.session_state["odds_range"] = (d_fb.get("min_odds", 1.50), d_fb.get("max_odds", 5.00))
        if "top_n" not in st.session_state:
            st.session_state["top_n"] = d_fb.get("top_n", 10)
        if "history_days" not in st.session_state:
            st.session_state["history_days"] = d_fb.get("history_days", 90)
        if "lookahead_days" not in st.session_state:
            st.session_state["lookahead_days"] = d_fb.get("lookahead_days", 3)

        st.subheader("Value Filters")
        min_edge_pct = st.slider("Min Edge %", 1, 30, step=1, key="min_edge")
        min_edge_val = min_edge_pct / 100.0
        odds_range = st.slider("Odds Range", 1.10, 10.0, step=0.05, key="odds_range")
        top_n_options = [5, 10, 15, 20, 25]
        top_n_val = st.session_state["top_n"]
        top_n_idx = top_n_options.index(top_n_val) if top_n_val in top_n_options else 1
        top_n = st.selectbox("Max Results", options=top_n_options, index=top_n_idx)

        st.subheader("History Window")
        history_days = st.slider("Elo History (days)", 30, 365, step=10, key="history_days")
        lookahead_days = st.slider("Lookahead (days)", 1, 14, step=1, key="lookahead_days")

        if st.button("Reset filters to league defaults"):
            _apply_defaults(league_label)
            st.rerun()

    tab_value, tab_history = st.tabs(["Value Bets", "Saved Picks"])

    with tab_value:
        render_value_bets(league_label, min_edge_val, odds_range, top_n, history_days, lookahead_days)

    with tab_history:
        render_saved_picks()


def _count_games_with_odds(harvest_games):
    count = 0
    for g in harvest_games:
        snaps = extract_moneylines(g)
        if snaps:
            count += 1
    return count


def _harvest_date_range(harvest_games):
    from mapper import _parse_iso
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


def _show_harvest_games(harvest_games):
    with st.expander("Scheduled games from sportsbooks"):
        for g in harvest_games:
            h = g.get("homeTeam", {}).get("mediumName", "?")
            a = g.get("awayTeam", {}).get("mediumName", "?")
            t = g.get("scheduledTime", "?")
            snaps = extract_moneylines(g)
            home_odds = consensus_decimal(snaps, "home")
            away_odds = consensus_decimal(snaps, "away")
            odds_str = f"H={home_odds:.2f} / A={away_odds:.2f}" if home_odds and away_odds else "No odds posted"
            st.write(f"**{h}** vs **{a}** — {t} — {odds_str}")


def _dedup_best_side(value_bets):
    best = {}
    for vb in value_bets:
        key = (vb["home_team"], vb["away_team"])
        if key not in best or vb["edge"] > best[key]["edge"]:
            best[key] = vb
    deduped = list(best.values())
    deduped.sort(key=lambda x: x["edge"], reverse=True)
    return deduped


def _show_diagnostics(odds_fetched, odds_with_lines, fixtures_fetched=None, matched=None, value_bets=None):
    with st.expander("Pipeline diagnostics", expanded=False):
        cols = st.columns(5)
        cols[0].metric("Odds games fetched", odds_fetched)
        cols[1].metric("With posted lines", odds_with_lines if odds_with_lines is not None else "—")
        cols[2].metric("Fixtures fetched", fixtures_fetched if fixtures_fetched is not None else "—")
        cols[3].metric("Matched", matched if matched is not None else "—")
        cols[4].metric("Value bets", value_bets if value_bets is not None else "—")


def render_value_bets(league_label, min_edge, odds_range, top_n, history_days, lookahead_days=3):
    st.subheader("Value Bets")
    st.caption("Compare Elo model predictions against live sportsbook consensus odds")

    has_token = bool(os.getenv("APIFY_TOKEN", ""))
    if not has_token:
        st.warning(
            "Apify API token not found. Add APIFY_TOKEN to your Secrets "
            "(Tools → Secrets) to enable live odds fetching."
        )
        return

    harvest_key = sofascore_to_harvest(league_label)
    if not harvest_key:
        st.error(f"No odds source mapping for league: {league_label}")
        return

    sofascore_filter = SOFASCORE_LEAGUE_FILTERS.get(league_label, league_label)

    three_outcome = not is_two_outcome(league_label)
    if three_outcome:
        st.warning(
            "This league is a 3-outcome market (home / draw / away). "
            "The current model is 2-outcome Elo, so edges may be overstated. "
            "Only the best side per match is shown to avoid double-counting."
        )

    if st.button("Find Value Bets", type="primary", use_container_width=True):
        _run_pipeline(league_label, harvest_key, sofascore_filter,
                      min_edge, odds_range, top_n, history_days,
                      dedup_per_match=three_outcome)


def _run_pipeline(league_label, harvest_key, sofascore_filter,
                  min_edge, odds_range, top_n, history_days,
                  dedup_per_match=False):
    progress = st.progress(0, text="Starting pipeline...")

    try:
        progress.progress(10, text="Fetching historical results for Elo ratings...")
        results = get_results_history(league=sofascore_filter, since_days=history_days)
        if not results:
            st.warning(f"No historical results found for {league_label}. Elo ratings will use defaults (1500).")

        progress.progress(25, text=f"Building Elo ratings from {len(results)} games...")
        elo_dicts = [_game_to_elo_dict(g) for g in results]
        elo_ratings = build_elo_ratings(elo_dicts) if elo_dicts else {}

        progress.progress(40, text=f"Fetching live odds from sportsbooks ({harvest_key})...")
        try:
            harvest_games = run_actor_get_items(
                actor_id="harvest~sportsbook-odds-scraper",
                actor_input={"league": harvest_key},
                timeout=120,
            )
        except Exception as e:
            st.error(f"Failed to fetch odds: {e}")
            progress.empty()
            return

        if not harvest_games:
            st.info(f"No odds data available for {league_label} right now.")
            progress.empty()
            return

        games_with_odds = _count_games_with_odds(harvest_games)
        earliest_dt, latest_dt = _harvest_date_range(harvest_games)

        if games_with_odds == 0:
            progress.empty()
            st.warning(
                f"Found {len(harvest_games)} scheduled {league_label} game(s) but "
                f"0 have posted moneylines yet. Lines typically appear 1–2 days before game time."
            )
            if earliest_dt:
                st.info(f"Earliest scheduled game: **{earliest_dt.strftime('%b %d, %Y %H:%M UTC')}** — try again within 48 hours of that time.")
            _show_diagnostics(
                odds_fetched=len(harvest_games),
                odds_with_lines=0,
                fixtures_fetched=None,
                matched=None,
            )
            _show_harvest_games(harvest_games)
            return

        latest_game_date = latest_dt.date() if latest_dt else None
        fixture_end = max(date.today() + timedelta(days=lookahead_days), latest_game_date) if latest_game_date else date.today() + timedelta(days=lookahead_days)

        progress.progress(55, text="Fetching upcoming fixtures...")
        upcoming = get_upcoming_games(league=sofascore_filter, date_to=fixture_end)
        if not upcoming:
            progress.empty()
            st.info(f"No upcoming fixtures found for {league_label} through {fixture_end}.")
            _show_diagnostics(
                odds_fetched=len(harvest_games),
                odds_with_lines=games_with_odds,
                fixtures_fetched=0,
                matched=None,
            )
            return

        progress.progress(70, text=f"Matching {len(upcoming)} fixtures to {games_with_odds} odds events...")
        matched = match_games_to_odds(upcoming, harvest_games)

        if not matched:
            progress.empty()
            st.warning(
                f"Could not match any of {len(upcoming)} fixtures to "
                f"{len(harvest_games)} odds events. Team names may differ between sources."
            )
            _show_diagnostics(
                odds_fetched=len(harvest_games),
                odds_with_lines=games_with_odds,
                fixtures_fetched=len(upcoming),
                matched=0,
            )
            _show_harvest_games(harvest_games)
            return

        progress.progress(85, text="Computing value bets...")
        value_bets = _compute_values(matched, elo_ratings, min_edge, odds_range)

        if dedup_per_match and value_bets:
            value_bets = _dedup_best_side(value_bets)

        progress.progress(100, text="Done!")
        progress.empty()

        _show_diagnostics(
            odds_fetched=len(harvest_games),
            odds_with_lines=games_with_odds,
            fixtures_fetched=len(upcoming),
            matched=len(matched),
            value_bets=len(value_bets),
        )

        if not value_bets:
            st.info(
                f"No value bets found with edge >= {min_edge:.0%} in odds range "
                f"{odds_range[0]:.2f} – {odds_range[1]:.2f}. Try adjusting filters."
            )
            _show_matched_summary(matched, elo_ratings)
            return

        value_bets = value_bets[:top_n]
        _display_value_bets(value_bets, league_label)

    except Exception as e:
        progress.empty()
        st.error(f"Pipeline error: {e}")


def _compute_values(matched, elo_ratings, min_edge, odds_range):
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

            value_bets.append({
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
            })

    value_bets.sort(key=lambda x: x["edge"], reverse=True)
    return value_bets


def _display_value_bets(value_bets, league_label):
    st.success(f"Found {len(value_bets)} value bet(s) for {league_label}")

    experimental = not is_two_outcome(league_label)

    rows = []
    for vb in value_bets:
        rows.append({
            "Match": f"{vb['home_team']} vs {vb['away_team']}",
            "Date": vb["date"],
            "Time": vb["time"],
            "Pick": vb["selection"],
            "Odds": f"{vb['odds_decimal']:.2f}",
            "Model %": f"{vb['model_prob']:.1%}",
            "Implied %": f"{vb['implied_prob']:.1%}",
            "Edge": f"{vb['edge']:.1%}",
            "EV/unit": f"{vb['ev_per_unit']:.3f}",
            "Elo H": f"{vb['home_elo']:.0f}",
            "Elo A": f"{vb['away_elo']:.0f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if experimental:
        st.info("Saving picks is disabled for 3-outcome leagues unless explicitly enabled.")
        allow_save = st.checkbox(
            "I understand this is a 3-outcome market and want to save these picks anyway"
        )
    else:
        allow_save = True

    col1, col2 = st.columns([1, 3])
    with col1:
        save_disabled = not allow_save
        if st.button("Save All Picks", type="secondary", disabled=save_disabled):
            try:
                init_db()
                for vb in value_bets:
                    vb["is_experimental"] = 1 if experimental else 0
                count = save_picks(value_bets)
                st.success(f"Saved {count} pick(s)")
            except Exception as e:
                st.error(f"Failed to save: {e}")


def _show_matched_summary(matched, elo_ratings):
    with st.expander("Matched games (no value bets found)"):
        for m in matched:
            fx = m["fixture"]
            og = m["odds_game"]
            snaps = extract_moneylines(og)
            home_odds = consensus_decimal(snaps, "home")
            away_odds = consensus_decimal(snaps, "away")
            r_home = float(elo_ratings.get(fx.home, 1500.0))
            r_away = float(elo_ratings.get(fx.away, 1500.0))
            p_home = elo_win_prob(r_home, r_away)

            st.write(
                f"**{fx.home}** vs **{fx.away}** — "
                f"Elo: {r_home:.0f} / {r_away:.0f} — "
                f"Model: {p_home:.1%} / {1-p_home:.1%} — "
                f"Odds: {home_odds or 'N/A'} / {away_odds or 'N/A'} — "
                f"Confidence: {m['match_confidence']:.0f}%"
            )


def render_saved_picks():
    st.subheader("Saved Picks History")
    try:
        history = get_recent_picks(limit=25)
        if history:
            hist_rows = []
            for h in history:
                exp_flag = "Experimental" if h.get("is_experimental") else ""
                hist_rows.append({
                    "Saved": h.get("created_at", ""),
                    "Match": f"{h['home_team']} vs {h['away_team']}",
                    "League": h.get("league", ""),
                    "Selection": h.get("selection", ""),
                    "Odds": f"{h.get('odds_decimal', 0):.2f}" if h.get("odds_decimal") else "",
                    "Edge": f"{h.get('edge', 0):.1%}" if h.get("edge") else "",
                    "Result": h.get("result", "Pending"),
                    "P/L": f"{h.get('profit_loss', 0):.2f}" if h.get("profit_loss") is not None else "",
                    "Type": exp_flag,
                })
            st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No saved picks yet.")
    except Exception:
        st.info("No saved picks yet.")


if __name__ == "__main__":
    main()
