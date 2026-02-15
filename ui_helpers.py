from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd

from odds_extract import extract_moneylines, consensus_decimal
from odds_fetch import get_fetch_errors, get_fatal_error, get_odds_source
from stats_provider import get_fetch_failure_count, get_http_429_count, get_http_5xx_count, get_http_404_count, get_last_status_code
from features import elo_win_prob
from league_map import is_two_outcome


def show_diagnostics(
    odds_fetched: int,
    odds_with_lines: Optional[int],
    fixtures_fetched: Optional[int] = None,
    matched: Optional[int] = None,
    value_bets: Optional[int] = None,
) -> None:
    with st.expander("Pipeline diagnostics", expanded=False):
        cols = st.columns(5)
        cols[0].metric("Odds games fetched", odds_fetched)
        cols[1].metric(
            "With posted lines", odds_with_lines if odds_with_lines is not None else "\u2014"
        )
        cols[2].metric(
            "Fixtures fetched",
            fixtures_fetched if fixtures_fetched is not None else "\u2014",
        )
        cols[3].metric("Matched", matched if matched is not None else "\u2014")
        cols[4].metric("Value bets", value_bets if value_bets is not None else "\u2014")
        st.caption(
            "**Odds games fetched** = events returned by the odds source. "
            "**With posted lines** = events with actual moneyline prices (not blank). "
            "**Fixtures fetched** = upcoming events from the stats source. "
            "**Matched** = fixtures paired to odds successfully. "
            "**Value bets** = picks that passed your filters."
        )
        odds_source = get_odds_source()
        if odds_source:
            st.caption(f"Odds source: **{odds_source}**")
        fatal = get_fatal_error()
        fetch_errors = get_fetch_errors()
        if fatal:
            st.error(
                f"**Odds fetch failed** (fatal config/auth error):\n\n"
                f"`{fatal[:300]}`"
            )
        elif fetch_errors:
            st.warning(f"Odds fetch failed ({len(fetch_errors)} error(s)):")
            for source, reason in fetch_errors:
                st.caption(f"  {reason[:200]}")
        stats_failures = get_fetch_failure_count()
        http_429 = get_http_429_count()
        http_5xx = get_http_5xx_count()
        http_404 = get_http_404_count()
        last_status = get_last_status_code()
        if stats_failures or http_429 or http_5xx or http_404:
            parts = []
            if stats_failures:
                parts.append(f"{stats_failures} day(s) failed")
            if http_404:
                parts.append(f"{http_404} not-found (404) — likely wrong sport slug")
            if http_429:
                parts.append(f"{http_429} rate-limit (429)")
            if http_5xx:
                parts.append(f"{http_5xx} server error (5xx)")
            st.warning("Stats provider issues: " + ", ".join(parts) + ".")
            if last_status and last_status != 200:
                st.caption(f"Last HTTP status from stats source: {last_status}")

        from config_env import get_env_report
        env_missing, env_stale = get_env_report()
        if not env_missing:
            st.caption("\u2705 APIFY_TOKEN present")
        else:
            st.caption("\u274c Missing: " + ", ".join(env_missing))
        if env_stale:
            st.caption(
                "\u26a0\ufe0f Unused secrets still set: " + ", ".join(env_stale)
                + " \u2014 consider removing them from Secrets."
            )


def explain_empty_run(
    odds_fetched: int,
    odds_with_lines: Optional[int],
    fixtures_fetched: Optional[int],
    matched: Optional[int],
    min_edge: float,
    odds_range: tuple,
    league_label: str,
) -> None:
    if odds_fetched == 0:
        st.info(
            f"No odds data available for **{league_label}** right now. "
            "The sportsbooks may not have posted lines yet. Try again closer to game time."
        )
    elif odds_with_lines is not None and odds_with_lines == 0:
        st.warning(
            f"Found {odds_fetched} scheduled game(s) but none have moneyline prices posted yet. "
            "Lines typically appear 1-2 days before game time."
        )
    elif fixtures_fetched is not None and fixtures_fetched == 0:
        st.info(
            f"Odds found ({odds_fetched} games) but no upcoming fixtures from the stats source. "
            "The schedule may not be published yet for this league window."
        )
    elif matched is not None and matched == 0:
        st.warning(
            f"Found {fixtures_fetched} fixture(s) and {odds_fetched} odds event(s), but could not "
            "match any of them together. Team/player names likely differ between sources. "
            "Check the **Unmatched Samples** section below for details."
        )
    else:
        edge_pct = min_edge * 100
        st.info(
            f"All {matched} matched game(s) were checked, but none passed filters: "
            f"min edge {edge_pct:.0f}%, odds range {odds_range[0]:.2f} - {odds_range[1]:.2f}. "
            "Try lowering the min edge or widening the odds range."
        )


def show_unmatched_samples(
    unmatched_fixtures: List[Dict[str, Any]],
    unmatched_odds: List[Dict[str, Any]],
) -> None:
    if not unmatched_fixtures and not unmatched_odds:
        return
    with st.expander("Unmatched samples (debug mapping issues)", expanded=False):
        if unmatched_fixtures:
            st.markdown("**Fixtures without matching odds** (top 10)")
            rows = []
            for f in unmatched_fixtures:
                rows.append({"Home": f["home"], "Away": f["away"], "Time": f["time"]})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if unmatched_odds:
            st.markdown("**Odds events without matching fixtures** (top 10)")
            rows = []
            for o in unmatched_odds:
                rows.append({"Home": o["home"], "Away": o["away"], "Time": o["time"]})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "If team names look similar but don't match, aliases may need to be added to mapper.py."
        )


def show_harvest_games(harvest_games: List[Dict[str, Any]]) -> None:
    with st.expander("Scheduled games from sportsbooks"):
        for g in harvest_games:
            h = g.get("homeTeam", {}).get("mediumName", "?")
            a = g.get("awayTeam", {}).get("mediumName", "?")
            t = g.get("scheduledTime", "?")
            snaps = extract_moneylines(g)
            home_odds = consensus_decimal(snaps, "home")
            away_odds = consensus_decimal(snaps, "away")
            odds_str = (
                f"H={home_odds:.2f} / A={away_odds:.2f}"
                if home_odds and away_odds
                else "No odds posted"
            )
            st.write(f"**{h}** vs **{a}** \u2014 {t} \u2014 {odds_str}")


def show_matched_summary(
    matched: List[Dict[str, Any]], elo_ratings: dict
) -> None:
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
                f"**{fx.home}** vs **{fx.away}** \u2014 "
                f"Elo: {r_home:.0f} / {r_away:.0f} \u2014 "
                f"Model: {p_home:.1%} / {1-p_home:.1%} \u2014 "
                f"Odds: {home_odds or 'N/A'} / {away_odds or 'N/A'} \u2014 "
                f"Confidence: {m['match_confidence']:.0f}%"
            )


def show_results_explainer() -> None:
    with st.expander("What do these results mean?"):
        st.markdown(
            "**Match / Date / Time**: The event and scheduled start time (UTC).\n\n"
            "**Pick**: The side the model rates as the best value (highest EV) for this match.\n\n"
            "**Odds**: Consensus decimal odds from the books (median across sportsbooks).\n\n"
            "**Model %**: The model's estimated chance this pick wins.\n\n"
            "**Implied %**: The market's implied chance from the odds (\u2248 1 / odds).\n\n"
            "**Edge**: Model % \u2212 Implied %. Positive edge means the model thinks the market underprices this outcome.\n\n"
            "**EV/unit**: Expected profit per 1 unit staked (EV = ModelProb \u00d7 Odds \u2212 1). "
            "Example: EV/unit = 0.10 means +0.10 units expected return per 1 unit bet (before commissions/limits).\n\n"
            "**Elo H / Elo A**: The model strength ratings for the two teams/participants used to generate Model %."
        )


AVAILABLE_COLUMNS = [
    "Match", "Date", "Time", "Pick", "Odds", "Model %",
    "Implied %", "Edge", "EV/unit", "Elo H", "Elo A",
    "Confidence",
]

DEFAULT_COLUMNS = [
    "Match", "Date", "Time", "Pick", "Odds", "Model %",
    "Implied %", "Edge", "EV/unit",
]

SORT_PRESETS = {
    "Best EV": "EV/unit",
    "Highest Edge": "Edge",
    "Soonest Start": "sort_time",
    "Best Confidence": "Confidence",
}


def display_value_bets_table(
    value_bets: List[Dict[str, Any]], league_label: str
) -> bool:
    st.success(f"Found {len(value_bets)} value bet(s) for {league_label}")
    show_results_explainer()

    experimental = not is_two_outcome(league_label)

    c1, c2 = st.columns([1, 1])
    with c1:
        sort_choice = st.selectbox(
            "Sort by",
            options=list(SORT_PRESETS.keys()),
            index=0,
            key="sort_preset",
        )
    with c2:
        visible_cols = st.multiselect(
            "Columns",
            options=AVAILABLE_COLUMNS,
            default=DEFAULT_COLUMNS,
            key="visible_cols",
        )

    has_league_col = any("league" in vb for vb in value_bets)

    rows = []
    for vb in value_bets:
        row = {
                "Match": f"{vb['home_team']} vs {vb['away_team']}",
                "Date": vb["date"],
                "Time": vb["time"],
                "Pick": vb["selection"],
                "_odds": vb["odds_decimal"],
                "_model_p": vb["model_prob"],
                "_implied_p": vb["implied_prob"],
                "_edge": vb["edge"],
                "_ev": vb["ev_per_unit"],
                "_elo_h": vb["home_elo"],
                "_elo_a": vb["away_elo"],
                "_conf": vb["match_confidence"],
                "sort_time": vb["date"] + " " + vb["time"],
            }
        if has_league_col:
            row["League"] = vb.get("league", "")
        rows.append(row)

    df = pd.DataFrame(rows)

    sort_map = {
        "EV/unit": "_ev",
        "Edge": "_edge",
        "Confidence": "_conf",
        "sort_time": "sort_time",
    }
    sort_col = sort_map.get(SORT_PRESETS.get(sort_choice, "EV/unit"), "_ev")
    if sort_col == "sort_time":
        df = df.sort_values("sort_time", ascending=True)
    else:
        df = df.sort_values(sort_col, ascending=False)

    df["Odds"] = df["_odds"].apply(lambda x: f"{x:.2f}")
    df["Model %"] = df["_model_p"].apply(lambda x: f"{x:.1%}")
    df["Implied %"] = df["_implied_p"].apply(lambda x: f"{x:.1%}")
    df["Edge"] = df["_edge"].apply(lambda x: f"{x:.1%}")
    df["EV/unit"] = df["_ev"].apply(lambda x: f"{x:.3f}")
    df["Elo H"] = df["_elo_h"].apply(lambda x: f"{x:.0f}")
    df["Elo A"] = df["_elo_a"].apply(lambda x: f"{x:.0f}")
    df["Confidence"] = df["_conf"].apply(lambda x: f"{x:.0f}%")

    display_cols = [c for c in visible_cols if c in df.columns]
    if has_league_col and "League" not in display_cols:
        display_cols = ["League"] + display_cols
    if not display_cols:
        display_cols = DEFAULT_COLUMNS

    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
    return experimental


def render_save_controls(
    value_bets: List[Dict[str, Any]], league_label: str
) -> None:
    from store import save_picks, init_db

    experimental = not is_two_outcome(league_label)

    if experimental:
        st.info(
            "Saving picks is disabled for 3-outcome leagues unless explicitly enabled."
        )
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


def render_saved_picks() -> None:
    from store import get_recent_picks

    st.subheader("Saved Picks History")
    try:
        history = get_recent_picks(limit=25)
        if history:
            hist_rows = []
            for h in history:
                exp_flag = "Experimental" if h.get("is_experimental") else ""
                hist_rows.append(
                    {
                        "Saved": h.get("created_at", ""),
                        "Match": f"{h['home_team']} vs {h['away_team']}",
                        "League": h.get("league", ""),
                        "Selection": h.get("selection", ""),
                        "Odds": (
                            f"{h.get('odds_decimal', 0):.2f}"
                            if h.get("odds_decimal")
                            else ""
                        ),
                        "Edge": f"{h.get('edge', 0):.1%}" if h.get("edge") else "",
                        "Result": h.get("result", "Pending"),
                        "P/L": (
                            f"{h['profit_loss']:.2f}"
                            if h.get("profit_loss") is not None
                            else ""
                        ),
                        "Type": exp_flag,
                    }
                )
            st.dataframe(
                pd.DataFrame(hist_rows), use_container_width=True, hide_index=True
            )
        else:
            st.info("No saved picks yet.")
    except Exception:
        st.info("No saved picks yet.")
