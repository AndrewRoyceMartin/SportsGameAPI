from __future__ import annotations

import streamlit as st
from datetime import date, timedelta

from config_env import get_env_report
from league_map import (
    sofascore_to_harvest,
    available_leagues,
    is_two_outcome,
    is_separator,
)
from league_defaults import DEFAULTS
from stats_provider import clear_events_cache
from pipeline import (
    fetch_elo_ratings,
    fetch_harvest_odds,
    count_games_with_odds,
    harvest_date_range,
    fetch_fixtures,
    match_fixtures_to_odds,
    compute_values,
    dedup_best_side,
)
from ui_helpers import (
    show_diagnostics,
    show_harvest_games,
    show_matched_summary,
    display_value_bets_table,
    render_save_controls,
    render_saved_picks,
)


def main():
    st.set_page_config(page_title="Sports Predictor", page_icon="\u26bd", layout="wide")

    st.title("\u26bd Sports Predictor")
    st.caption(
        "Find +EV bets by comparing Elo model probabilities against live sportsbook odds"
    )

    with st.sidebar:
        st.header("Configuration")

        leagues = available_leagues()
        default_idx = leagues.index("NBA") if "NBA" in leagues else 0
        league_label = st.selectbox(
            "League",
            options=leagues,
            index=default_idx,
            help="Choose the sport/league to scan. Production leagues are 2-outcome markets. Experimental leagues may overstate edge (e.g., soccer draw risk).",
        )

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
            st.session_state["odds_range"] = (
                d_fb.get("min_odds", 1.50),
                d_fb.get("max_odds", 5.00),
            )
        if "top_n" not in st.session_state:
            st.session_state["top_n"] = d_fb.get("top_n", 10)
        if "history_days" not in st.session_state:
            st.session_state["history_days"] = d_fb.get("history_days", 90)
        if "lookahead_days" not in st.session_state:
            st.session_state["lookahead_days"] = d_fb.get("lookahead_days", 3)

        st.subheader("Value Filters")
        min_edge_pct = st.slider(
            "Min Edge %",
            1,
            30,
            step=1,
            key="min_edge",
            help="Only show bets where the model's win probability is at least this much higher than the market-implied probability. Example: 5% means model says 55% vs market 50%.",
        )
        min_edge_val = min_edge_pct / 100.0
        odds_range = st.slider(
            "Odds Range",
            1.10,
            10.0,
            step=0.05,
            key="odds_range",
            help="Only consider bets with decimal odds inside this range. Lower odds = favourites; higher odds = underdogs. Narrowing this can reduce volatility.",
        )
        top_n_options = [5, 10, 15, 20, 25]
        top_n_val = st.session_state["top_n"]
        top_n_idx = top_n_options.index(top_n_val) if top_n_val in top_n_options else 1
        top_n = st.selectbox(
            "Max Results",
            options=top_n_options,
            index=top_n_idx,
            help="Show at most this many top-ranked bets after filtering. Ranked by expected value (EV/unit).",
        )

        st.subheader("History Window")
        history_days = st.slider(
            "Elo History (days)",
            30,
            365,
            step=10,
            key="history_days",
            help="How far back to build team ratings from past results. Longer = more stable; shorter = more responsive to recent form.",
        )
        lookahead_days = st.slider(
            "Lookahead (days)",
            1,
            14,
            step=1,
            key="lookahead_days",
            help="How many days ahead to search for upcoming fixtures and odds. If odds aren't posted yet, try a shorter window or wait until closer to game time.",
        )

        if st.button("Reset filters to league defaults"):
            _apply_defaults(league_label)
            st.rerun()

    tab_value, tab_history = st.tabs(["Value Bets", "Saved Picks"])

    with tab_value:
        _render_value_bets(
            league_label, min_edge_val, odds_range, top_n, history_days, lookahead_days
        )

    with tab_history:
        render_saved_picks()


def _render_value_bets(
    league_label, min_edge, odds_range, top_n, history_days, lookahead_days=3
):
    st.subheader("Value Bets")
    st.caption("Compare Elo model predictions against live sportsbook consensus odds")

    missing, stale = get_env_report()
    if missing:
        st.warning(
            "Missing required secrets: **" + ", ".join(missing) + "**. "
            "Add them to your Secrets (Tools \u2192 Secrets) to enable live odds fetching."
        )
        return

    harvest_key = sofascore_to_harvest(league_label)
    if not harvest_key:
        st.error(f"No odds source mapping for league: {league_label}")
        return

    three_outcome = not is_two_outcome(league_label)
    if three_outcome:
        st.warning(
            "This league is a 3-outcome market (home / draw / away). "
            "The current model is 2-outcome Elo, so edges may be overstated. "
            "Only the best side per match is shown to avoid double-counting."
        )

    if st.button("Find Value Bets", type="primary", use_container_width=True):
        _run_pipeline(
            league_label,
            harvest_key,
            min_edge,
            odds_range,
            top_n,
            history_days,
            lookahead_days=lookahead_days,
            dedup_per_match=three_outcome,
        )


def _run_pipeline(
    league_label,
    harvest_key,
    min_edge,
    odds_range,
    top_n,
    history_days,
    lookahead_days=3,
    dedup_per_match=False,
):
    progress = st.progress(0, text="Starting pipeline...")

    clear_events_cache()

    try:
        progress.progress(10, text="Fetching historical results for Elo ratings...")
        elo_ratings, result_count = fetch_elo_ratings(league_label, history_days)
        if result_count == 0:
            st.warning(
                f"No historical results found for {league_label}. Elo ratings will use defaults (1500)."
            )

        progress.progress(25, text=f"Building Elo ratings from {result_count} games...")

        progress.progress(
            40,
            text=f"Fetching live odds from sportsbooks ({harvest_key}, {lookahead_days} day window)...",
        )
        try:
            harvest_games = fetch_harvest_odds(harvest_key, lookahead_days)
        except Exception as e:
            st.error(f"Failed to fetch odds: {e}")
            progress.empty()
            return

        if not harvest_games:
            st.info(f"No odds data available for {league_label} right now.")
            progress.empty()
            return

        games_with_odds = count_games_with_odds(harvest_games)
        earliest_dt, latest_dt = harvest_date_range(harvest_games)

        if games_with_odds == 0:
            progress.empty()
            st.warning(
                f"Found {len(harvest_games)} scheduled {league_label} game(s) but "
                f"0 have posted moneylines yet. Lines typically appear 1\u20132 days before game time."
            )
            if earliest_dt:
                st.info(
                    f"Earliest scheduled game: **{earliest_dt.strftime('%b %d, %Y %H:%M UTC')}** \u2014 try again within 48 hours of that time."
                )
            show_diagnostics(
                odds_fetched=len(harvest_games),
                odds_with_lines=0,
                fixtures_fetched=None,
                matched=None,
            )
            show_harvest_games(harvest_games)
            return

        latest_game_date = latest_dt.date() if latest_dt else None

        progress.progress(55, text="Fetching upcoming fixtures...")
        upcoming, fixture_end = fetch_fixtures(
            league_label, lookahead_days, latest_game_date
        )
        if not upcoming:
            progress.empty()
            st.info(
                f"No upcoming fixtures found for {league_label} through {fixture_end}."
            )
            show_diagnostics(
                odds_fetched=len(harvest_games),
                odds_with_lines=games_with_odds,
                fixtures_fetched=0,
                matched=None,
            )
            return

        progress.progress(
            70,
            text=f"Matching {len(upcoming)} fixtures to {games_with_odds} odds events...",
        )
        matched = match_fixtures_to_odds(upcoming, harvest_games)

        if not matched:
            progress.empty()
            st.warning(
                f"Could not match any of {len(upcoming)} fixtures to "
                f"{len(harvest_games)} odds events. Team names may differ between sources."
            )
            show_diagnostics(
                odds_fetched=len(harvest_games),
                odds_with_lines=games_with_odds,
                fixtures_fetched=len(upcoming),
                matched=0,
            )
            show_harvest_games(harvest_games)
            return

        progress.progress(85, text="Computing value bets...")
        value_bets = compute_values(matched, elo_ratings, min_edge, odds_range)

        if dedup_per_match and value_bets:
            value_bets = dedup_best_side(value_bets)

        progress.progress(100, text="Done!")
        progress.empty()

        show_diagnostics(
            odds_fetched=len(harvest_games),
            odds_with_lines=games_with_odds,
            fixtures_fetched=len(upcoming),
            matched=len(matched),
            value_bets=len(value_bets),
        )

        if not value_bets:
            st.info(
                f"No value bets found with edge >= {min_edge:.0%} in odds range "
                f"{odds_range[0]:.2f} \u2013 {odds_range[1]:.2f}. Try adjusting filters."
            )
            show_matched_summary(matched, elo_ratings)
            return

        value_bets = value_bets[:top_n]
        display_value_bets_table(value_bets, league_label)
        render_save_controls(value_bets, league_label)

    except Exception as e:
        progress.empty()
        st.error(f"Pipeline error: {e}")


if __name__ == "__main__":
    main()
