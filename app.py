from __future__ import annotations

import streamlit as st
from datetime import date, timedelta

from store import init_db, save_elo_override, load_elo_override, load_all_elo_overrides
from config_env import get_env_report
from connectivity import check_all
from league_map import (
    sofascore_to_harvest,
    available_leagues,
    is_two_outcome,
    is_separator,
    ALL_SPORTS,
    LEAGUE_SPORT,
    SPORT_DISPLAY_TO_KEY,
    EXPERIMENTAL_LEAGUES,
    _SEPARATOR,
)
from league_defaults import DEFAULTS, RUN_PROFILES, apply_profile, get_elo_params
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
    run_walkforward_backtest,
    tune_elo_params,
    preflight_availability,
    preflight_scan,
    preflight_with_odds_probe,
    summarize_match_time_deltas,
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
    render_walkforward_results,
    render_hero_card,
    render_action_strip,
    render_bet_builder,
    render_backtest_settings_button,
    render_funnel_stepper,
    render_shortlist_summary,
    render_quick_filters,
    render_au_season_banner,
    inject_action_styles,
    render_tuning_results,
    render_availability_table,
)


_CUSTOM_CSS = """
<style>
/* --- Global spacing tweaks --- */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* --- Card / panel feel --- */
div[data-testid="stMetric"],
div[data-testid="stDataFrame"],
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stExpander"],
div[data-testid="stForm"] {
  background: #FFFFFF;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 16px;
  box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
}

/* Reduce "double borders" on nested blocks */
div[data-testid="stVerticalBlockBorderWrapper"] > div {
  border-radius: 16px;
}

/* --- Buttons --- */
.stButton > button {
  border-radius: 12px !important;
  padding: 0.55rem 0.95rem !important;
  border: 1px solid rgba(37, 99, 235, 0.25) !important;
  background: linear-gradient(180deg, rgba(37,99,235,0.95), rgba(37,99,235,0.85)) !important;
  color: #FFFFFF !important;
  font-weight: 650 !important;
  transition: all 0.2s ease;
}
.stButton > button:hover {
  filter: brightness(1.03);
  transform: translateY(-1px);
}

/* --- Tags / pills (Streamlit chips) --- */
span[data-baseweb="tag"]{
  border-radius: 999px !important;
  border: 1px solid rgba(15,23,42,0.12) !important;
  background: rgba(37,99,235,0.08) !important;
}

/* --- Sidebar polish --- */
section[data-testid="stSidebar"] {
  background: #FFFFFF;
  border-right: 1px solid rgba(15, 23, 42, 0.08);
}

/* --- Tabs --- */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  border-radius: 10px;
  padding: 4px;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 8px;
  font-weight: 500;
  padding: 8px 20px;
  transition: all 0.2s ease;
}

/* --- Metrics --- */
[data-testid="stMetricLabel"] {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  opacity: 0.7;
}
[data-testid="stMetricValue"] {
  font-weight: 700;
  font-size: 1.3rem;
}

/* --- Inputs --- */
.stSelectbox [data-baseweb="select"] > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
  border-radius: 8px;
  transition: border-color 0.2s ease;
}
.stSelectbox [data-baseweb="select"] > div:focus-within,
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
  border-color: #2563EB;
  box-shadow: 0 0 0 2px rgba(37,99,235,0.15);
}

/* --- Toggle --- */
[data-testid="stToggle"] span[data-baseweb="toggle"] {
  border-radius: 20px;
}

/* --- Divider --- */
hr {
  border-color: rgba(15,23,42,0.08);
}

/* --- Download button --- */
.stDownloadButton > button {
  border-radius: 12px;
  font-weight: 600;
}

/* --- Alert boxes --- */
.stAlert {
  border-radius: 10px;
}

/* --- Progress bar --- */
.stProgress > div > div {
  background: linear-gradient(90deg, #2563EB, #60A5FA);
  border-radius: 8px;
}

/* --- Scrollbar --- */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}
::-webkit-scrollbar-track {
  background: transparent;
}
::-webkit-scrollbar-thumb {
  background: rgba(37,99,235,0.25);
  border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
  background: rgba(37,99,235,0.4);
}
</style>
"""


def main():
    st.set_page_config(page_title="Sports Predictor", page_icon="\u26bd", layout="wide")
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
    inject_action_styles()
    init_db()

    st.title("\u26bd Sports Predictor")
    st.caption(
        "Find +EV bets by comparing Elo model probabilities against live sportsbook odds"
    )

    with st.sidebar:
        _CONN_CACHE_KEY = "_conn_status"
        _CONN_TTL = 300
        import time as _time
        from datetime import datetime as _dt

        def _get_conn_status():
            cached = st.session_state.get(_CONN_CACHE_KEY)
            if cached and (_time.time() - cached["ts"]) < _CONN_TTL:
                return cached["results"], cached["ts"]
            results = check_all()
            now = _time.time()
            st.session_state[_CONN_CACHE_KEY] = {"ts": now, "results": results}
            return results, now

        conn_results, checked_ts = _get_conn_status()
        all_ok = all(c.ok for c in conn_results)
        checked_dt = _dt.utcfromtimestamp(checked_ts).strftime("%H:%M:%S UTC")

        with st.container():
            st.markdown(
                f"**Data Sources** {'🟢' if all_ok else '🔴'}",
            )
            for c in conn_results:
                icon = "✅" if c.ok else "❌"
                latency = f" ({c.latency_ms}ms)" if c.latency_ms is not None else ""
                detail = f" — {c.detail}" if c.detail and c.detail != "OK" else ""
                st.caption(f"{icon} **{c.name}**{latency}{detail}")
            st.caption(f"Checked at {checked_dt}")
            if st.button("Refresh", key="conn_refresh", use_container_width=True):
                st.session_state.pop(_CONN_CACHE_KEY, None)
                st.rerun()

        st.divider()
        st.header("Settings")

        sport_display = st.selectbox(
            "Sport",
            options=ALL_SPORTS,
            index=0,
            help="Filter leagues by sport category.",
            key="sport_select",
        )
        sport_filter = SPORT_DISPLAY_TO_KEY.get(sport_display, "All")

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

        all_real_leagues = [l for l in leagues if not is_separator(l)]

        preflight_info = st.session_state.get("preflight_info", {})

        def _league_format(lg: str) -> str:
            if is_separator(lg):
                return lg
            info = preflight_info.get(lg)
            if info is not None:
                n = info.get("fixtures_count", 0)
                return f"{lg} ({n} upcoming)" if n > 0 else f"{lg} (0 upcoming)"
            return lg

        if preflight_info:
            prod_leagues = [l for l in leagues if not is_separator(l) and l not in EXPERIMENTAL_LEAGUES]
            exp_leagues = [l for l in leagues if not is_separator(l) and l in EXPERIMENTAL_LEAGUES]
            prod_leagues.sort(key=lambda l: (-preflight_info.get(l, {}).get("fixtures_count", 0), l))
            exp_leagues.sort(key=lambda l: (-preflight_info.get(l, {}).get("fixtures_count", 0), l))
            if prod_leagues and exp_leagues:
                sorted_leagues = prod_leagues + [_SEPARATOR] + exp_leagues
            else:
                sorted_leagues = prod_leagues + exp_leagues
            league_options = ["All"] + sorted_leagues
        else:
            league_options = ["All"] + leagues

        stored_league = st.session_state.get("league_select")
        if stored_league and stored_league not in league_options:
            st.session_state.pop("league_select", None)

        default_idx = 0
        league_label = st.selectbox(
            "League",
            options=league_options,
            index=default_idx,
            format_func=_league_format,
            key="league_select",
            help="Choose a league to scan, or 'All' for multi-league scan.",
        )

        if is_separator(league_label):
            st.warning("Please select a league.")
            st.stop()

        run_all = league_label == "All"

        st.session_state.setdefault("elo_overrides", {})
        if not run_all and league_label not in st.session_state["elo_overrides"]:
            _stored = load_elo_override(league_label)
            if _stored:
                _stored.pop("_updated_at", None)
                st.session_state["elo_overrides"][league_label] = _stored
        if run_all:
            _all_stored = load_all_elo_overrides()
            for _lg, _params in _all_stored.items():
                if _lg not in st.session_state["elo_overrides"]:
                    _params.pop("_updated_at", None)
                    st.session_state["elo_overrides"][_lg] = _params

        avail_key = "preflight_results"
        scan_window = st.selectbox(
            "Scan window",
            options=[3, 7, 14, 21],
            index=1,
            help="How many days ahead to check for upcoming games.",
            key="scan_window",
        )
        check_cols = st.columns([1, 1])
        with check_cols[0]:
            check_single = st.button(
                "Check Availability",
                help="Quick check if fixtures exist for the selected league(s).",
                use_container_width=True,
                key="preflight_btn",
            )
        with check_cols[1]:
            scan_all_btn = st.button(
                "Scan All",
                help="Check fixture availability across all visible leagues.",
                use_container_width=True,
                key="preflight_scan_btn",
            )

        if check_single:
            if run_all:
                with st.spinner("Checking all leagues..."):
                    clear_events_cache()
                    results = preflight_scan(all_real_leagues, lookahead_days=scan_window)
                st.session_state[avail_key] = results
            else:
                with st.spinner(f"Checking {league_label} (fixtures + odds probe)..."):
                    clear_events_cache()
                    result = preflight_with_odds_probe(league_label, lookahead_override=scan_window)
                st.session_state[avail_key] = [result]

        if scan_all_btn:
            with st.spinner(f"Scanning {len(all_real_leagues)} leagues..."):
                clear_events_cache()
                results = preflight_scan(all_real_leagues, lookahead_days=scan_window)
            st.session_state[avail_key] = results

        if check_single or scan_all_btn:
            pf_data = st.session_state.get(avail_key, [])
            info_map = {r["league"]: r for r in pf_data}
            st.session_state["preflight_info"] = info_map
            st.rerun()

        preflight_data = st.session_state.get(avail_key)
        if preflight_data:
            with st.expander("Availability", expanded=True):
                render_availability_table(preflight_data)

            for _idx, _row in enumerate(preflight_data):
                _probe_key = f"probe_{_row['league']}"
                if st.session_state.get(_probe_key):
                    with st.spinner(f"Probing odds for {_row['league']}..."):
                        _probed = preflight_with_odds_probe(
                            _row["league"],
                            lookahead_override=_row.get("lookahead_days"),
                        )
                    preflight_data[_idx] = _probed
                    st.session_state[avail_key] = preflight_data
                    pf_data = st.session_state.get(avail_key, [])
                    st.session_state["preflight_info"] = {r["league"]: r for r in pf_data}
                    st.rerun()

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

        if "league_config" not in st.session_state:
            st.session_state["league_config"] = {}

        def _get_league_config(league: str) -> dict:
            configs = st.session_state["league_config"]
            if league not in configs:
                d = DEFAULTS.get(league, {})
                d = apply_profile(d, profile) if d else {}
                configs[league] = {
                    "confidence_target": "Any",
                    "min_edge": d.get("min_edge", 5),
                    "odds_range": (d.get("min_odds", 1.50), d.get("max_odds", 5.00)),
                    "history_days": d.get("history_days", 90),
                    "lookahead_days": d.get("lookahead_days", 3),
                    "top_n": d.get("top_n", 10),
                }
            return configs[league]

        def _save_league_config(league: str) -> None:
            st.session_state["league_config"][league] = {
                "confidence_target": st.session_state.get("conf_target_sel", "Any"),
                "min_edge": st.session_state.get("min_edge", 5),
                "odds_range": st.session_state.get("odds_range", (1.50, 5.00)),
                "history_days": st.session_state.get("history_days", 90),
                "lookahead_days": st.session_state.get("lookahead_days", 3),
                "top_n": st.session_state.get("top_n", 10),
            }

        def _load_league_config(league: str) -> None:
            cfg = _get_league_config(league)
            st.session_state["min_edge"] = cfg["min_edge"]
            st.session_state["odds_range"] = cfg["odds_range"]
            st.session_state["history_days"] = cfg["history_days"]
            st.session_state["lookahead_days"] = cfg["lookahead_days"]
            st.session_state["top_n"] = cfg["top_n"]
            st.session_state["conf_target_sel"] = cfg["confidence_target"]

        conf_options = ["Any", "55%+", "60%+", "65%+", "70%+"]
        bt_applied = st.session_state.get("backtest_settings_applied", False)
        bt_target = st.session_state.get("confidence_target_value", 0)
        if bt_applied and bt_target:
            target_label = f"{bt_target}%+"
            if target_label in conf_options and "conf_target_sel" not in st.session_state:
                st.session_state["conf_target_sel"] = target_label
                st.session_state["backtest_settings_applied"] = False

        prev = st.session_state.get("_prev_league")
        prev_profile = st.session_state.get("_prev_profile")
        if prev != league_label or prev_profile != profile:
            if prev and prev != league_label:
                _save_league_config(prev)
            if prev_profile != profile:
                d = DEFAULTS.get(league_label, {})
                d = apply_profile(d, profile) if d else {}
                if d:
                    st.session_state["league_config"].pop(league_label, None)
            _load_league_config(league_label)
            st.session_state["_prev_league"] = league_label
            st.session_state["_prev_profile"] = profile

        confidence_target = st.selectbox(
            "Confidence Target",
            options=conf_options,
            key="conf_target_sel",
            help="Filters picks by model win probability (Elo). Higher confidence reduces volume but tends to improve hit rate. Don't treat it as 'bet now' — use with Edge/EV.",
        )
        min_model_prob = {"Any": 0.0, "55%+": 0.55, "60%+": 0.60, "65%+": 0.65, "70%+": 0.70}.get(confidence_target, 0.0)

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
                st.session_state["league_config"].pop(league_label, None)
                st.rerun()

        min_edge_val = st.session_state["min_edge"] / 100.0

        if run_all:
            st.caption(f"Scanning **All** leagues ({sport_filter}) \u2014 {profile}")
        else:
            sport_tag = LEAGUE_SPORT.get(league_label, "")
            mode_tag = "Best Bets" if best_bets_mode else "Full Scan"
            st.caption(f"**{league_label}** ({sport_tag}) \u2014 {profile} \u2014 {mode_tag}")

    if best_bets_mode:
        effective_top_n = min(top_n, 10)
    else:
        effective_top_n = top_n

    tab_picks, tab_explore, tab_backtest, tab_diagnostics, tab_saved = st.tabs(
        ["Place Bets", "Browse All", "Trust & Tuning", "Fix Issues", "Track Results"]
    )

    if "last_run_data" not in st.session_state:
        st.session_state["last_run_data"] = None

    with tab_picks:
        if run_all:
            _render_all_leagues_picks(
                all_real_leagues, min_edge_val, odds_range, effective_top_n,
                history_days, lookahead_days, profile, best_bets_mode,
                min_model_prob=min_model_prob,
                confidence_target=confidence_target,
            )
        else:
            _render_picks(
                league_label, min_edge_val, odds_range, effective_top_n,
                history_days, lookahead_days, best_bets_mode,
                min_model_prob=min_model_prob,
                profile=profile, confidence_target=confidence_target,
            )

    with tab_explore:
        run_data = st.session_state.get("last_run_data")
        if run_data and run_data.get("value_bets"):
            vb = run_data["value_bets"]
            league = run_data.get("league_label", "")
            st.subheader(f"All Results \u2014 {league}")
            st.caption(f"{len(vb)} value bet(s) found")

            explore_edge = st.slider(
                "Show bets down to edge %",
                min_value=0, max_value=30, value=int(min_edge_val * 100),
                step=1, key="explore_edge_slider",
                help="Explore with a lower edge threshold without changing your saved defaults.",
            )
            show_near_misses = st.toggle(
                "Show near-misses (1-2% below threshold)",
                value=False, key="explore_near_misses",
            )

            explore_bets = vb
            if run_data.get("matched_list") and (explore_edge < int(min_edge_val * 100) or show_near_misses):
                from pipeline import compute_values
                elo_r = run_data.get("elo_ratings", {})
                matched_list = run_data.get("matched_list", [])
                game_counts = run_data.get("game_counts", {})
                near_edge = max(0, explore_edge - 2) / 100.0 if show_near_misses else explore_edge / 100.0
                wide_odds = (1.10, 10.0)
                explore_bets = compute_values(
                    matched_list, elo_r, near_edge, wide_odds,
                    league=league, game_counts=game_counts,
                    elo_overrides=st.session_state.get("elo_overrides"),
                )
                explore_bets = attach_quality(explore_bets)
                explore_bets.sort(key=lambda x: x.get("edge", 0), reverse=True)

            display_value_bets_table(explore_bets, league)
            render_save_controls(explore_bets, league, key_prefix="explore")
        else:
            st.info("Run the pipeline from the Place Bets tab to see detailed results here.")

    with tab_backtest:
        st.subheader("Trust & Tuning")

        with st.container():
            st.markdown(
                "Follow these three steps to build confidence in the model and "
                "find the best settings for each league."
            )
            step_cols = st.columns(3)
            with step_cols[0]:
                st.markdown(
                    "**Step 1 — Baseline Backtest**\n\n"
                    "Run a backtest with current settings to see how the model "
                    "performs on past games. This is your starting benchmark."
                )
            with step_cols[1]:
                st.markdown(
                    "**Step 2 — Tune Parameters**\n\n"
                    "Use the grid search to try different Elo settings. It finds "
                    "the combination that scores best on log loss + calibration."
                )
            with step_cols[2]:
                st.markdown(
                    "**Step 3 — Confirm Improvement**\n\n"
                    "Re-run the backtest with the new settings to verify the "
                    "improvement holds and isn't just overfitting."
                )

        st.divider()

        if run_all:
            bt_league = st.selectbox(
                "League to backtest",
                options=all_real_leagues,
                key="bt_league",
                help="Pick one league to backtest.",
            )
        else:
            bt_league = league_label

        st.subheader("Backtest")
        st.caption(
            "Test the Elo model against past results. Trains on older games, "
            "then predicts recent games to measure accuracy."
        )

        bt_mode = st.radio(
            "Backtest mode",
            ["Single split", "Walk-forward (multi-fold)"],
            horizontal=True,
            key="bt_mode",
            help="**Single split** tests one time window — quick but can be noisy. "
                 "**Walk-forward** runs multiple overlapping windows and averages results — "
                 "slower but much more reliable.",
        )

        if bt_mode == "Single split":
            bt_c1, bt_c2 = st.columns(2)
            with bt_c1:
                bt_history = st.slider(
                    "Training period (days)",
                    30, 365, value=90, step=10,
                    key="bt_history",
                    help="How many days of older games to use for building Elo ratings. "
                         "More data gives the model more to learn from, but very old games may be less relevant.",
                )
            with bt_c2:
                bt_test = st.slider(
                    "Test period (days)",
                    7, 90, value=30, step=7,
                    key="bt_test",
                    help="How many days of recent games to test predictions against. "
                         "The model never sees these results during training.",
                )

            if st.button("Run Backtest", type="primary", use_container_width=True, key="run_bt_single"):
                with st.spinner(f"Backtesting {bt_league}... fetching {bt_history + bt_test} days of results"):
                    clear_events_cache()
                    bt_result = run_backtest(bt_league, history_days=bt_history, test_days=bt_test)
                render_backtest_results(bt_result)
                render_backtest_settings_button(bt_result)
        else:
            wf_c1, wf_c2, wf_c3 = st.columns(3)
            with wf_c1:
                wf_train = st.slider(
                    "Training window (days)",
                    60, 365, value=180, step=30,
                    key="wf_train",
                    help="How many days of training data per fold. Larger windows give more stable Elo ratings.",
                )
            with wf_c2:
                wf_test = st.slider(
                    "Test window (days)",
                    14, 90, value=30, step=7,
                    key="wf_test",
                    help="How many days to test per fold. Each fold slides forward to test a fresh period.",
                )
            with wf_c3:
                wf_folds = st.slider(
                    "Number of folds",
                    2, 8, value=4, step=1,
                    key="wf_folds",
                    help="More folds = more reliable average, but requires more historical data. "
                         "4 folds is a good default.",
                )

            total_needed = wf_train + wf_test * wf_folds
            st.caption(f"Needs ~{total_needed} days of historical data")

            if st.button("Run Walk-Forward", type="primary", use_container_width=True, key="run_bt_wf"):
                with st.spinner(f"Walk-forward backtest on {bt_league}... {wf_folds} folds"):
                    clear_events_cache()
                    wf_result = run_walkforward_backtest(
                        bt_league,
                        total_days=total_needed,
                        train_days=wf_train,
                        test_days=wf_test,
                        folds=wf_folds,
                    )
                render_walkforward_results(wf_result)
                render_backtest_settings_button(wf_result)

        st.divider()
        st.subheader("Tune Elo Parameters")
        st.caption(
            "Grid search over K-factor and home advantage to find the best parameters "
            "for this league. Scored by log loss (rewards calibration, punishes overconfidence) "
            "rather than raw accuracy, which can be misleading for betting."
        )
        st.info(
            "**Why not just optimise for accuracy?** A model that always picks the favourite "
            "can be 60%+ accurate but useless for betting — the odds already price that in. "
            "Log loss rewards *well-calibrated* probabilities: if the model says 70%, "
            "it should win ~70% of the time. That's what makes edge calculations trustworthy.",
            icon="\u2139\ufe0f",
        )
        if st.button("Run Tuning Grid Search", type="primary", use_container_width=True, key="run_tune"):
            with st.spinner(f"Tuning parameters for {bt_league}... testing K and home advantage combinations"):
                clear_events_cache()
                tune_result = tune_elo_params(bt_league, total_days=270, train_days=210, test_days=30)
            render_tuning_results(tune_result)

            if not tune_result.get("error"):
                best = tune_result["best_params"]
                current = tune_result.get("current_params", {})
                changed = (best.get("k") != current.get("k") or best.get("home_adv") != current.get("home_adv"))
                if changed:
                    if st.button(
                        f"Apply tuned params (K={best['k']}, Home Adv={best['home_adv']})",
                        type="primary",
                        use_container_width=True,
                        key="apply_tuned",
                    ):
                        base_ep = get_elo_params(bt_league)
                        st.session_state.setdefault("elo_overrides", {})[bt_league] = {
                            "k": best["k"],
                            "home_adv": best["home_adv"],
                            "scale": base_ep["scale"],
                            "recency_half_life": base_ep["recency_half_life"],
                        }
                        save_elo_override(bt_league, st.session_state["elo_overrides"][bt_league])
                        st.success(
                            f"Applied! K={best['k']}, Home Adv={best['home_adv']} will be used "
                            f"for {bt_league} predictions. Re-run the backtest to confirm improvement."
                        )
                        st.rerun()

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

            td_info = run_data.get("time_deltas")
            if td_info and td_info.get("with_times"):
                with st.expander("Match Time Alignment", expanded=False):
                    tc1, tc2, tc3, tc4 = st.columns(4)
                    tc1.metric("Matched pairs", td_info["count"])
                    tc2.metric("Median gap", f"{td_info['median_h']:.1f}h" if td_info["median_h"] is not None else "N/A")
                    tc3.metric("Max gap", f"{td_info['max_h']:.1f}h" if td_info["max_h"] is not None else "N/A")
                    gt12 = td_info.get("gt_12h", 0)
                    gt24 = td_info.get("gt_24h", 0)
                    if gt24 > 0:
                        tc4.metric("Suspicious (>24h)", gt24)
                    elif gt12 > 0:
                        tc4.metric("Wide gap (>12h)", gt12)
                    else:
                        tc4.metric("All within 12h", "Yes")

                    suspicious = td_info.get("suspicious", [])
                    if suspicious:
                        st.markdown("**Suspicious matches (>12h time gap)**")
                        import pandas as _pd
                        sus_rows = []
                        for s in suspicious:
                            sus_rows.append({
                                "Game": s["fixture"],
                                "Fixture time": s["fx_time"],
                                "Odds time": s["odds_time"],
                                "Gap (h)": s["delta_h"],
                                "Name score": s["confidence"],
                            })
                        st.dataframe(_pd.DataFrame(sus_rows), use_container_width=True, hide_index=True)
                    elif td_info["max_h"] is not None and td_info["max_h"] < 2:
                        st.success("All matched pairs have times within 2 hours — alignment looks good.")
                    else:
                        st.info("No suspicious time gaps detected.")

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
            st.info("Run the pipeline from Place Bets to see diagnostics here.")

    with tab_saved:
        render_saved_picks()


def _render_picks(
    league_label, min_edge, odds_range, top_n,
    history_days, lookahead_days, best_bets_mode,
    min_model_prob=0.0,
    profile="Balanced", confidence_target="Any",
):
    three_outcome = not is_two_outcome(league_label)

    render_au_season_banner(league_label)

    render_funnel_stepper(
        league_label, profile, min_edge, odds_range,
        confidence_target, lookahead_days, best_bets_mode,
    )

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

    st.session_state.setdefault("elo_overrides", {})
    _elo_ov = st.session_state["elo_overrides"]
    if league_label in _elo_ov:
        _tuned = _elo_ov[league_label]
        st.success(
            f"Using tuned params: K={_tuned.get('k', '?')}, "
            f"Home Adv={_tuned.get('home_adv', '?')}, "
            f"Half-life={_tuned.get('recency_half_life', '?')}",
            icon="\u2705",
        )
    else:
        st.caption("\u26a0\ufe0f Using default Elo params (no tuned params yet)")

    cal_col1, cal_col2 = st.columns([1, 1])
    with cal_col1:
        if st.button("Find Value Bets", type="primary", use_container_width=True):
            _do_find = True
        else:
            _do_find = False
    with cal_col2:
        if st.button(
            "Quick Calibrate",
            use_container_width=True,
            help="Runs a fast grid search to find the best K-factor and home advantage "
                 "for this league, then applies the tuned params automatically.",
        ):
            with st.spinner(f"Calibrating {league_label}... testing parameter combinations"):
                clear_events_cache()
                tune_result = tune_elo_params(league_label, total_days=270, train_days=210, test_days=30)
            if tune_result.get("error"):
                st.warning(tune_result["error"])
            else:
                best = tune_result["best_params"]
                base_ep = get_elo_params(league_label)
                st.session_state["elo_overrides"][league_label] = {
                    "k": best["k"],
                    "home_adv": best["home_adv"],
                    "scale": base_ep["scale"],
                    "recency_half_life": base_ep["recency_half_life"],
                }
                save_elo_override(league_label, st.session_state["elo_overrides"][league_label])
                st.success(
                    f"Calibrated! Best params: K={best['k']}, Home Adv={best['home_adv']} "
                    f"(log loss: {tune_result['best_log_loss']:.4f}, "
                    f"accuracy: {tune_result['best_accuracy']:.1%}). "
                    f"These will be used for your next scan."
                )
                st.rerun()

    if _do_find:
        preflight_info = st.session_state.get("preflight_info", {})
        pf = preflight_info.get(league_label, {})
        pf_status = pf.get("status", "")
        if pf_status == "no_fixtures":
            st.warning(
                f"**{league_label}** has no fixtures in the current lookahead window. "
                f"Try extending the lookahead or switching to a league with upcoming games."
            )
            _do_find = False
        elif pf_status == "odds_empty":
            st.warning(
                f"**{league_label}** has fixtures but the odds provider returned no data. "
                f"The scraper may be temporarily empty — try again later or switch leagues."
            )
        elif pf_status == "odds_error":
            odds_err = pf.get("odds_error", "")
            st.warning(
                f"**{league_label}** odds probe had an error: {odds_err[:80]}. "
                f"You can still try running, but odds may not be available."
            )

    if _do_find:
        st.session_state.setdefault("league_config", {})[league_label] = {
            "confidence_target": st.session_state.get("conf_target_sel", "Any"),
            "min_edge": st.session_state.get("min_edge", 5),
            "odds_range": st.session_state.get("odds_range", (1.50, 5.00)),
            "history_days": st.session_state.get("history_days", 90),
            "lookahead_days": st.session_state.get("lookahead_days", 3),
            "top_n": st.session_state.get("top_n", 10),
        }
        value_bets, run_data = _run_pipeline(
            league_label, harvest_key, min_edge, odds_range, top_n,
            history_days, lookahead_days=lookahead_days,
            dedup_per_match=three_outcome, best_bets_mode=best_bets_mode,
            min_model_prob=min_model_prob,
            elo_overrides=st.session_state.get("elo_overrides"),
        )

        run_data["league_label"] = league_label
        st.session_state["last_run_data"] = run_data

        render_action_strip(run_data)

        if value_bets:
            render_hero_card(value_bets[0])

            if best_bets_mode:
                render_shortlist_summary(value_bets)

            filtered_bets = render_quick_filters(value_bets, key_prefix="picks_qf")

            col_left, col_right = st.columns([3, 1])
            with col_left:
                render_pick_cards(filtered_bets, league_label)
            with col_right:
                render_bet_builder(filtered_bets, league_label)

            if run_data.get("unmatched_fx") or run_data.get("unmatched_odds"):
                st.caption("Some games couldn't be matched \u2014 check the Fix Issues tab for details.")


def _render_all_leagues_picks(
    league_list, min_edge, odds_range, top_n,
    history_days, lookahead_days, profile, best_bets_mode,
    min_model_prob=0.0,
    confidence_target="Any",
):
    render_funnel_stepper(
        "All", profile, min_edge, odds_range,
        confidence_target, lookahead_days, best_bets_mode, run_all=True,
    )

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
                _league_ov = st.session_state.get("elo_overrides", {}).get(league)
                elo_ratings, league_gc, _ = fetch_elo_ratings(league, league_history, elo_overrides=_league_ov)
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
                bets = compute_values(matched, elo_ratings, min_edge, odds_range, league=league, game_counts=league_gc, elo_overrides=st.session_state.get("elo_overrides"))
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

        if min_model_prob > 0:
            all_value_bets = [vb for vb in all_value_bets if vb.get("model_prob", 0) >= min_model_prob]

        if best_bets_mode:
            all_value_bets.sort(key=lambda x: x.get("quality", 0), reverse=True)
        else:
            all_value_bets.sort(key=lambda x: x.get("edge", 0), reverse=True)

        all_value_bets = all_value_bets[:top_n]

        run_data = {
            "odds_fetched": all_odds,
            "odds_with_lines": all_odds,
            "fixtures_fetched": all_fixtures,
            "matched": all_matched,
            "value_bets_count": len(all_value_bets),
            "value_bets": all_value_bets,
            "league_label": "All Leagues",
        }
        st.session_state["last_run_data"] = run_data

        render_action_strip(run_data)

        if all_value_bets:
            render_hero_card(all_value_bets[0])

            if best_bets_mode:
                render_shortlist_summary(all_value_bets)

            filtered_bets = render_quick_filters(all_value_bets, key_prefix="all_qf")

            col_left, col_right = st.columns([3, 1])
            with col_left:
                render_pick_cards(filtered_bets, "All Leagues")
            with col_right:
                render_bet_builder(filtered_bets, "All Leagues")


def _run_pipeline(
    league_label, harvest_key, min_edge, odds_range, top_n,
    history_days, lookahead_days=3, dedup_per_match=False, best_bets_mode=False,
    min_model_prob=0.0, elo_overrides=None,
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
        elo_ratings, game_counts, result_count = fetch_elo_ratings(league_label, history_days, elo_overrides=elo_overrides)
        run_data["elo_ratings"] = elo_ratings
        run_data["game_counts"] = game_counts
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
        if matched:
            run_data["time_deltas"] = summarize_match_time_deltas(matched)

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
        value_bets = compute_values(matched, elo_ratings, min_edge, odds_range, league=league_label, game_counts=game_counts, elo_overrides=elo_overrides)

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

        if min_model_prob > 0:
            value_bets = [vb for vb in value_bets if vb.get("model_prob", 0) >= min_model_prob]

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
