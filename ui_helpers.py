from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd

from odds_extract import extract_moneylines, consensus_decimal
from odds_fetch import get_fetch_errors
from stats_provider import get_fetch_failure_count
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
        fetch_errors = get_fetch_errors()
        if fetch_errors:
            st.warning(f"Odds fetch skipped {len(fetch_errors)} day(s) due to errors:")
            for date_str, reason in fetch_errors:
                st.caption(f"  {date_str}: {reason[:120]}")
        stats_failures = get_fetch_failure_count()
        if stats_failures:
            st.warning(
                f"Stats provider skipped {stats_failures} day(s) due to API errors."
            )

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


def display_value_bets_table(
    value_bets: List[Dict[str, Any]], league_label: str
) -> bool:
    st.success(f"Found {len(value_bets)} value bet(s) for {league_label}")
    show_results_explainer()

    experimental = not is_two_outcome(league_label)

    rows = []
    for vb in value_bets:
        rows.append(
            {
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
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
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
