from __future__ import annotations

import streamlit as st
from datetime import date, timedelta

from config_env import get_env_report
from league_map import (
    sofascore_to_harvest,
    available_leagues,
    is_two_outcome,
    is_separator,
    ALL_SPORTS,
    LEAGUE_SPORT,
    SPORT_DISPLAY_TO_KEY,
)
from league_defaults import DEFAULTS, RUN_PROFILES, apply_profile
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
    get_unmatched,
    run_backtest,
)
from ui_helpers import (
    show_diagnostics,
    show_harvest_games,
    show_matched_summary,
    display_value_bets_table,
    render_save_controls,
    render_saved_picks,
    explain_empty_run,
    show_unmatched_samples,
    render_pick_cards,
    attach_quality,
    render_backtest_results,
)


def main():
    st.set_page_config(page_title="Sports Predictor", page_icon="\u26bd", layout="wide")

    st.title("\u26bd Sports Predictor")
    st.caption(
        "Find +EV bets by comparing Elo model probabilities against live sportsbook odds"
    )

    with st.sidebar:
        st.header("Settings")

        sport_display = st.selectbox(
            "Sport",
            options=ALL_SPORTS,
            index=0,
            help="Filter leagues by sport category.",
            key="sport_select",
        )
        sport_filter = SPORT_DISPLAY_TO_KEY.get(sport_display, "All")

        if "prev_sport" not in st.session_state:
            st.session_state.prev_sport = sport_filter
        if st.session_state.prev_sport != sport_filter:
            st.session_state.prev_sport = sport_filter
            st.session_state.pop("league_select", None)
            st.session_state.pop("league_search", None)
            st.rerun()

        league_search = st.text_input(
            "Search leagues",
            value="",
            placeholder="Type to filter...",
            help="Filter leagues by name.",
            key="league_search",
        )

        leagues = available_leagues(sport_filter=sport_filter, search=league_search)
        if not leagues:
            st.warning("No leagues match your filter.")
            st.stop()

        league_options = ["All"] + leagues
        default_idx = league_options.index("NBA") if "NBA" in league_options else 0
        league_label = st.selectbox(
            "League",
            options=league_options,
            index=default_idx,
            key="league_select",
            help="Choose a league to scan, or 'All' for multi-league scan.",
        )

        if is_separator(league_label):
            st.warning("Please select a league.")
            st.stop()

        run_all = league_label == "All"

        profile = st.selectbox(
            "Run Profile",
            options=list(RUN_PROFILES.keys()),
            index=1,
            help="Conservative = higher thresholds. Balanced = defaults. Aggressive = lower thresholds.",
        )

        best_bets_mode = st.toggle(
            "Best Bets Mode",
            value=True,
            help="Focus on near-term, high-quality picks. Locks defaults and sorts by quality score.",
        )

        with st.expander("Advanced Settings"):
            lock_defaults = best_bets_mode or st.toggle(
                "Lock defaults",
                value=False,
                help="Prevent accidental filter changes.",
                disabled=best_bets_mode,
            )

            def _apply_defaults(league: str):
                d = DEFAULTS.get(league)
                if not d:
                    return
                d = apply_profile(d, profile)
                st.session_state["min_edge"] = d["min_edge"]
                st.session_state["odds_range"] = (d["min_odds"], d["max_odds"])
                st.session_state["history_days"] = d["history_days"]
                st.session_state["lookahead_days"] = d["lookahead_days"]
                st.session_state["top_n"] = d["top_n"]

            prev = st.session_state.get("_prev_league")
            prev_profile = st.session_state.get("_prev_profile")
            if prev != league_label or prev_profile != profile:
                _apply_defaults(league_label)
                st.session_state["_prev_league"] = league_label
                st.session_state["_prev_profile"] = profile

            d_fb = DEFAULTS.get(league_label, {})
            d_fb = apply_profile(d_fb, profile) if d_fb else d_fb
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

            min_edge_pct = st.slider(
                "Min Edge %", 1, 30, step=1,
                key="min_edge", disabled=lock_defaults,
            )
            odds_range = st.slider(
                "Odds Range", 1.10, 10.0, step=0.05,
                key="odds_range", disabled=lock_defaults,
            )
            top_n_options = [5, 10, 15, 20, 25]
            top_n_val = st.session_state["top_n"]
            top_n_idx = top_n_options.index(top_n_val) if top_n_val in top_n_options else 1
            top_n = st.selectbox(
                "Max Results", options=top_n_options,
                index=top_n_idx, disabled=lock_defaults,
            )
            history_days = st.slider(
                "Elo History (days)", 30, 365, step=10,
                key="history_days", disabled=lock_defaults,
            )
            lookahead_days = st.slider(
                "Lookahead (days)", 1, 28, step=1,
                key="lookahead_days", disabled=lock_defaults,
            )

            if st.button("Reset to defaults"):
                _apply_defaults(league_label)
                st.rerun()

        min_edge_val = st.session_state["min_edge"] / 100.0

        if run_all:
            st.caption(f"Scanning **All** leagues ({sport_filter}) \u2014 {profile}")
        else:
            sport_tag = LEAGUE_SPORT.get(league_label, "")
            mode_tag = "Best Bets" if best_bets_mode else "Full Scan"
            st.caption(f"**{league_label}** ({sport_tag}) \u2014 {profile} \u2014 {mode_tag}")

    all_real_leagues = [l for l in leagues if not is_separator(l)]

    if best_bets_mode:
        effective_top_n = min(top_n, 10)
    else:
        effective_top_n = top_n

    tab_picks, tab_explore, tab_backtest, tab_diagnostics, tab_saved = st.tabs(
        ["Picks", "Explore", "Backtest", "Diagnostics", "Saved Picks"]
    )

    if "last_run_data" not in st.session_state:
        st.session_state["last_run_data"] = None

    with tab_picks:
        if run_all:
            _render_all_leagues_picks(
                all_real_leagues, min_edge_val, odds_range, effective_top_n,
                history_days, lookahead_days, profile, best_bets_mode,
            )
        else:
            _render_picks(
                league_label, min_edge_val, odds_range, effective_top_n,
                history_days, lookahead_days, best_bets_mode,
            )

    with tab_explore:
        run_data = st.session_state.get("last_run_data")
        if run_data and run_data.get("value_bets"):
            vb = run_data["value_bets"]
            league = run_data.get("league_label", "")
            st.subheader(f"All Results \u2014 {league}")
            st.caption(f"{len(vb)} value bet(s) found")
            display_value_bets_table(vb, league)
            render_save_controls(vb, league, key_prefix="explore")
        else:
            st.info("Run the pipeline from the Picks tab to see detailed results here.")

    with tab_backtest:
        st.subheader("Model Backtest")
        st.caption(
            "Test the Elo model against past results. Trains on older games, "
            "then predicts recent games to measure accuracy."
        )

        if run_all:
            bt_league = st.selectbox(
                "League to backtest",
                options=all_real_leagues,
                key="bt_league",
                help="Pick one league to backtest.",
            )
        else:
            bt_league = league_label

        bt_c1, bt_c2 = st.columns(2)
        with bt_c1:
            bt_history = st.slider(
                "Training period (days)",
                30, 365, value=90, step=10,
                key="bt_history",
                help="How many days of older games to use for building Elo ratings.",
            )
        with bt_c2:
            bt_test = st.slider(
                "Test period (days)",
                7, 90, value=30, step=7,
                key="bt_test",
                help="How many days of recent games to test predictions against.",
            )

        if st.button("Run Backtest", type="primary", use_container_width=True):
            with st.spinner(f"Backtesting {bt_league}... fetching {bt_history + bt_test} days of results"):
                clear_events_cache()
                bt_result = run_backtest(bt_league, history_days=bt_history, test_days=bt_test)
            render_backtest_results(bt_result)

    with tab_diagnostics:
        run_data = st.session_state.get("last_run_data")
        if run_data:
            st.subheader("Pipeline Diagnostics")
            show_diagnostics(
                odds_fetched=run_data.get("odds_fetched", 0),
                odds_with_lines=run_data.get("odds_with_lines"),
                fixtures_fetched=run_data.get("fixtures_fetched"),
                matched=run_data.get("matched"),
                value_bets=run_data.get("value_bets_count"),
            )
            if run_data.get("unmatched_fx") or run_data.get("unmatched_odds"):
                show_unmatched_samples(
                    run_data.get("unmatched_fx", []),
                    run_data.get("unmatched_odds", []),
                )
            if run_data.get("harvest_games"):
                show_harvest_games(run_data["harvest_games"])
            if run_data.get("matched_list") and not run_data.get("value_bets"):
                show_matched_summary(
                    run_data["matched_list"],
                    run_data.get("elo_ratings", {}),
                )
        else:
            st.info("Run the pipeline to see diagnostics.")

    with tab_saved:
        render_saved_picks()


def _render_picks(
    league_label, min_edge, odds_range, top_n,
    history_days, lookahead_days, best_bets_mode,
):
    three_outcome = not is_two_outcome(league_label)

    if best_bets_mode:
        st.subheader("Top Picks")
        st.caption("Showing the highest-quality value bets, sorted by composite quality score")
    else:
        st.subheader("Value Bets")
        st.caption("Compare Elo model predictions against live sportsbook consensus odds")

    missing, stale = get_env_report()
    if missing:
        st.warning(
            "Missing required secrets: **" + ", ".join(missing) + "**. "
            "Add them in Settings to enable live odds."
        )
        return

    harvest_key = sofascore_to_harvest(league_label)
    if not harvest_key:
        st.error(f"No odds source for: {league_label}")
        return

    if three_outcome:
        st.warning(
            "3-outcome market \u2014 edges may be overstated. Best side per match shown."
        )

    if st.button("Find Value Bets", type="primary", use_container_width=True):
        value_bets, run_data = _run_pipeline(
            league_label, harvest_key, min_edge, odds_range, top_n,
            history_days, lookahead_days=lookahead_days,
            dedup_per_match=three_outcome, best_bets_mode=best_bets_mode,
        )

        run_data["league_label"] = league_label
        st.session_state["last_run_data"] = run_data

        if value_bets:
            render_pick_cards(value_bets, league_label)
            render_save_controls(value_bets, league_label)

            if run_data.get("unmatched_fx") or run_data.get("unmatched_odds"):
                st.caption("Some games couldn't be matched \u2014 check the Diagnostics tab for details.")


def _render_all_leagues_picks(
    league_list, min_edge, odds_range, top_n,
    history_days, lookahead_days, profile, best_bets_mode,
):
    st.subheader("Multi-League Scan")
    st.caption("Scanning all leagues for the best value bets")

    missing, stale = get_env_report()
    if missing:
        st.warning(
            "Missing required secrets: **" + ", ".join(missing) + "**"
        )
        return

    if st.button("Scan All Leagues", type="primary", use_container_width=True):
        all_value_bets = []
        all_odds = 0
        all_fixtures = 0
        all_matched = 0
        progress = st.progress(0, text="Starting multi-league scan...")
        total = len(league_list)

        for i, league in enumerate(league_list):
            pct = int((i / total) * 100)
            progress.progress(pct, text=f"Scanning {league} ({i+1}/{total})...")

            harvest_key = sofascore_to_harvest(league)
            if not harvest_key:
                continue

            d = DEFAULTS.get(league, {})
            d = apply_profile(d, profile) if d else {}
            league_history = d.get("history_days", history_days)
            league_lookahead = d.get("lookahead_days", lookahead_days)

            three_outcome = not is_two_outcome(league)

            try:
                clear_events_cache()
                elo_ratings, _ = fetch_elo_ratings(league, league_history)
                harvest_games = fetch_harvest_odds(harvest_key, league_lookahead, league_label=league)
                if not harvest_games:
                    continue
                games_with_odds = count_games_with_odds(harvest_games)
                all_odds += games_with_odds
                if games_with_odds == 0:
                    continue
                _, latest_dt = harvest_date_range(harvest_games)
                latest_game_date = latest_dt.date() if latest_dt else None
                upcoming, _ = fetch_fixtures(league, league_lookahead, latest_game_date)
                all_fixtures += len(upcoming)
                if not upcoming:
                    continue
                matched = match_fixtures_to_odds(upcoming, harvest_games, league=league)
                all_matched += len(matched)
                if not matched:
                    continue
                bets = compute_values(matched, elo_ratings, min_edge, odds_range)
                if three_outcome and bets:
                    bets = dedup_best_side(bets)
                for b in bets:
                    b["league"] = league
                all_value_bets.extend(bets)
            except Exception as e:
                st.warning(f"Error scanning {league}: {e}")
                continue

        progress.progress(100, text="Done!")
        progress.empty()

        if not all_value_bets:
            st.info("No value bets found across any league with current filters.")
            st.session_state["last_run_data"] = {
                "odds_fetched": all_odds,
                "odds_with_lines": all_odds,
                "fixtures_fetched": all_fixtures,
                "matched": all_matched,
                "value_bets_count": 0,
                "value_bets": [],
                "league_label": "All Leagues",
            }
            return

        all_value_bets = attach_quality(all_value_bets)

        if best_bets_mode:
            all_value_bets.sort(key=lambda x: x.get("quality", 0), reverse=True)
        else:
            all_value_bets.sort(key=lambda x: x.get("edge", 0), reverse=True)

        all_value_bets = all_value_bets[:top_n]

        st.session_state["last_run_data"] = {
            "odds_fetched": all_odds,
            "odds_with_lines": all_odds,
            "fixtures_fetched": all_fixtures,
            "matched": all_matched,
            "value_bets_count": len(all_value_bets),
            "value_bets": all_value_bets,
            "league_label": "All Leagues",
        }

        render_pick_cards(all_value_bets, "All Leagues")
        render_save_controls(all_value_bets, "All Leagues")


def _run_pipeline(
    league_label, harvest_key, min_edge, odds_range, top_n,
    history_days, lookahead_days=3, dedup_per_match=False, best_bets_mode=False,
):
    run_data = {
        "odds_fetched": 0,
        "odds_with_lines": None,
        "fixtures_fetched": None,
        "matched": None,
        "value_bets_count": 0,
        "value_bets": [],
        "harvest_games": [],
        "matched_list": [],
        "elo_ratings": {},
        "unmatched_fx": [],
        "unmatched_odds": [],
    }

    progress = st.progress(0, text="Starting pipeline...")

    clear_events_cache()

    try:
        progress.progress(10, text="Fetching historical results for Elo ratings...")
        elo_ratings, result_count = fetch_elo_ratings(league_label, history_days)
        run_data["elo_ratings"] = elo_ratings
        if result_count == 0:
            st.warning(
                f"No historical results for {league_label}. Using default ratings."
            )

        progress.progress(25, text=f"Built Elo from {result_count} games...")

        progress.progress(40, text="Fetching live odds...")
        try:
            harvest_games = fetch_harvest_odds(harvest_key, lookahead_days, league_label=league_label)
        except Exception as e:
            st.error(f"Failed to fetch odds: {e}")
            progress.empty()
            return [], run_data

        run_data["harvest_games"] = harvest_games
        run_data["odds_fetched"] = len(harvest_games)

        if not harvest_games:
            progress.empty()
            explain_empty_run(0, None, None, None, min_edge, odds_range, league_label)
            return [], run_data

        games_with_odds = count_games_with_odds(harvest_games)
        run_data["odds_with_lines"] = games_with_odds
        earliest_dt, latest_dt = harvest_date_range(harvest_games)

        if games_with_odds == 0:
            progress.empty()
            explain_empty_run(
                len(harvest_games), 0, None, None, min_edge, odds_range, league_label
            )
            return [], run_data

        latest_game_date = latest_dt.date() if latest_dt else None

        progress.progress(55, text="Fetching upcoming fixtures...")
        upcoming, fixture_end = fetch_fixtures(
            league_label, lookahead_days, latest_game_date
        )
        run_data["fixtures_fetched"] = len(upcoming)

        if not upcoming:
            progress.empty()
            explain_empty_run(
                len(harvest_games), games_with_odds, 0, None,
                min_edge, odds_range, league_label,
            )
            return [], run_data

        progress.progress(70, text=f"Matching {len(upcoming)} fixtures to {games_with_odds} odds events...")
        matched = match_fixtures_to_odds(upcoming, harvest_games, league=league_label)
        run_data["matched"] = len(matched)
        run_data["matched_list"] = matched

        unmatched_fx, unmatched_odds = get_unmatched(upcoming, harvest_games, matched or [])
        run_data["unmatched_fx"] = unmatched_fx
        run_data["unmatched_odds"] = unmatched_odds

        if not matched:
            progress.empty()
            explain_empty_run(
                len(harvest_games), games_with_odds, len(upcoming), 0,
                min_edge, odds_range, league_label,
                unmatched_fixtures=unmatched_fx, unmatched_odds=unmatched_odds,
            )
            return [], run_data

        progress.progress(85, text="Computing value bets...")
        value_bets = compute_values(matched, elo_ratings, min_edge, odds_range)

        if dedup_per_match and value_bets:
            value_bets = dedup_best_side(value_bets)

        progress.progress(100, text="Done!")
        progress.empty()

        if not value_bets:
            explain_empty_run(
                len(harvest_games), games_with_odds, len(upcoming), len(matched),
                min_edge, odds_range, league_label,
            )
            run_data["value_bets_count"] = 0
            return [], run_data

        value_bets = attach_quality(value_bets)

        if best_bets_mode:
            value_bets.sort(key=lambda x: x.get("quality", 0), reverse=True)

        value_bets = value_bets[:top_n]

        run_data["value_bets_count"] = len(value_bets)
        run_data["value_bets"] = value_bets

        return value_bets, run_data

    except Exception as e:
        progress.empty()
        st.error(f"Pipeline error: {e}")
        return [], run_data


if __name__ == "__main__":
    main()
