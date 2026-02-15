from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd

from odds_extract import extract_moneylines, consensus_decimal
from odds_fetch import get_fetch_errors, get_fatal_error, get_odds_source
from stats_provider import get_fetch_failure_count, get_http_429_count, get_http_5xx_count, get_http_404_count, get_last_status_code
from features import elo_win_prob
from league_map import is_two_outcome
from mapper import _name_score, _parse_iso


def compute_bet_quality(vb: Dict[str, Any]) -> int:
    edge = vb.get("edge", 0)
    ev = vb.get("ev_per_unit", 0)
    conf = vb.get("match_confidence", 0)
    odds = vb.get("odds_decimal", 0)

    edge_score = min(edge / 0.20, 1.0) * 35

    ev_score = min(max(ev, 0) / 0.30, 1.0) * 25

    conf_score = min(conf / 100.0, 1.0) * 20

    if 1.5 <= odds <= 4.0:
        odds_score = 20
    elif 1.3 <= odds < 1.5 or 4.0 < odds <= 6.0:
        odds_score = 12
    elif 1.1 <= odds < 1.3 or 6.0 < odds <= 8.0:
        odds_score = 6
    else:
        odds_score = 2

    total = edge_score + ev_score + conf_score + odds_score
    return max(0, min(100, int(round(total))))


def quality_label(score: int) -> str:
    if score >= 80:
        return "Strong"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Fair"
    return "Weak"


def quality_tier(score: int) -> str:
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    return "C"


def attach_quality(value_bets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for vb in value_bets:
        vb["quality"] = compute_bet_quality(vb)
    return value_bets


def show_diagnostics(
    odds_fetched: int,
    odds_with_lines: Optional[int],
    fixtures_fetched: Optional[int] = None,
    matched: Optional[int] = None,
    value_bets: Optional[int] = None,
) -> None:
    cols = st.columns(5)
    cols[0].metric("Odds games", odds_fetched)
    cols[1].metric(
        "With lines", odds_with_lines if odds_with_lines is not None else "\u2014"
    )
    cols[2].metric(
        "Fixtures",
        fixtures_fetched if fixtures_fetched is not None else "\u2014",
    )
    cols[3].metric("Matched", matched if matched is not None else "\u2014")
    cols[4].metric("Value bets", value_bets if value_bets is not None else "\u2014")
    st.caption(
        "**Odds games** = events from odds source. "
        "**With lines** = events with moneyline prices. "
        "**Fixtures** = upcoming events from stats source. "
        "**Matched** = fixtures paired to odds. "
        "**Value bets** = picks that passed filters."
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
            parts.append(f"{http_404} not-found (404)")
        if http_429:
            parts.append(f"{http_429} rate-limit (429)")
        if http_5xx:
            parts.append(f"{http_5xx} server error (5xx)")
        st.warning("Stats provider issues: " + ", ".join(parts))
        if last_status and last_status != 200:
            st.caption(f"Last HTTP status: {last_status}")

    from config_env import get_env_report
    env_missing, env_stale = get_env_report()
    if not env_missing:
        st.caption("\u2705 APIFY_TOKEN present")
    else:
        st.caption("\u274c Missing: " + ", ".join(env_missing))
    if env_stale:
        st.caption(
            "\u26a0\ufe0f Unused secrets: " + ", ".join(env_stale)
        )


def _get_home(x: Dict[str, Any]) -> str:
    return (
        x.get("home")
        or (x.get("homeTeam") or {}).get("mediumName")
        or (x.get("homeTeam") or {}).get("name")
        or ""
    )


def _get_away(x: Dict[str, Any]) -> str:
    return (
        x.get("away")
        or (x.get("awayTeam") or {}).get("mediumName")
        or (x.get("awayTeam") or {}).get("name")
        or ""
    )


def _diagnose_unmatched(
    unmatched_fixtures: List[Dict[str, Any]],
    unmatched_odds: List[Dict[str, Any]],
) -> str:
    if not unmatched_fixtures or not unmatched_odds:
        return "coverage_gap"
    best_min = 0
    best_time_diff_hours = None
    for uf in unmatched_fixtures[:10]:
        uf_h, uf_a = _get_home(uf), _get_away(uf)
        uf_dt = _parse_iso(uf.get("time", ""))
        for uo in unmatched_odds[:10]:
            uo_h, uo_a = _get_home(uo), _get_away(uo)
            uo_dt = _parse_iso(uo.get("time", ""))

            h_score = _name_score(uf_h, uo_h)
            a_score = _name_score(uf_a, uo_a)
            pair_min = min(h_score, a_score)

            h_flip = _name_score(uf_h, uo_a)
            a_flip = _name_score(uf_a, uo_h)
            pair_min_flip = min(h_flip, a_flip)

            candidate = max(pair_min, pair_min_flip)
            if candidate > best_min:
                best_min = candidate
                if uf_dt and uo_dt:
                    best_time_diff_hours = abs((uf_dt - uo_dt).total_seconds()) / 3600
                else:
                    best_time_diff_hours = None

    if best_min >= 55 and (best_time_diff_hours is None or best_time_diff_hours <= 36):
        return "naming"
    return "coverage_gap"


def explain_empty_run(
    odds_fetched: int,
    odds_with_lines: Optional[int],
    fixtures_fetched: Optional[int],
    matched: Optional[int],
    min_edge: float,
    odds_range: tuple,
    league_label: str,
    unmatched_fixtures: Optional[List[Dict[str, Any]]] = None,
    unmatched_odds: Optional[List[Dict[str, Any]]] = None,
) -> None:
    if odds_fetched == 0:
        st.info(
            f"No odds data available for **{league_label}** right now. "
            "The sportsbooks may not have posted lines yet."
        )
        st.markdown("**What to try:**")
        st.markdown("- Come back closer to game time when lines are posted")
        st.markdown("- Try a different league that has active games today")
    elif odds_with_lines is not None and odds_with_lines == 0:
        st.warning(
            f"Found {odds_fetched} scheduled game(s) but none have moneyline prices yet."
        )
        st.markdown("**What to try:**")
        st.markdown("- Lines typically appear 1\u20132 days before game time")
        st.markdown("- Check back later or try another league")
    elif fixtures_fetched is not None and fixtures_fetched == 0:
        st.info(
            f"Odds found ({odds_fetched} games) but no fixtures from the stats source."
        )
        st.markdown("**What to try:**")
        st.markdown("- Extend the lookahead window in Advanced settings")
        st.markdown("- The schedule may not be published yet for this window")
    elif matched is not None and matched == 0:
        reason = _diagnose_unmatched(
            unmatched_fixtures or [], unmatched_odds or []
        )
        if reason == "naming":
            st.warning(
                f"Found {fixtures_fetched} fixture(s) and {odds_fetched} odds event(s), "
                "but names don't match between sources."
            )
            st.markdown("**What to try:**")
            st.markdown("- Check the Diagnostics tab for unmatched name details")
            st.markdown("- Team name aliases may need updating")
        else:
            st.info(
                f"Found {fixtures_fetched} fixture(s) and {odds_fetched} odds event(s), "
                "but they cover different rounds/dates."
            )
            st.markdown("**What to try:**")
            st.markdown("- The fixture source hasn't published these matches yet \u2014 common for far-ahead rounds")
            st.markdown("- Try again closer to game day or extend the lookahead window")
    else:
        edge_pct = min_edge * 100
        st.info(
            f"All {matched} matched game(s) checked, but none passed filters: "
            f"min edge {edge_pct:.0f}%, odds {odds_range[0]:.2f}\u2013{odds_range[1]:.2f}."
        )
        st.markdown("**What to try:**")
        st.markdown("- Lower the minimum edge threshold")
        st.markdown("- Widen the odds range")
        st.markdown("- Switch to the Aggressive run profile")


def show_unmatched_samples(
    unmatched_fixtures: List[Dict[str, Any]],
    unmatched_odds: List[Dict[str, Any]],
) -> None:
    if not unmatched_fixtures and not unmatched_odds:
        return
    reason = _diagnose_unmatched(unmatched_fixtures, unmatched_odds)
    title = (
        "Unmatched samples (naming issues)"
        if reason == "naming"
        else "Unmatched samples (different rounds/dates)"
    )
    with st.expander(title, expanded=False):
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
        if reason == "naming":
            st.caption(
                "Team names look similar but don't match \u2014 aliases may need updating."
            )
        else:
            st.caption(
                "These appear to be different rounds/dates \u2014 "
                "normal for far-ahead rounds."
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


def render_pick_cards(
    value_bets: List[Dict[str, Any]], league_label: str
) -> None:
    if not value_bets:
        return

    avg_edge = sum(vb["edge"] for vb in value_bets) / len(value_bets)
    avg_ev = sum(vb["ev_per_unit"] for vb in value_bets) / len(value_bets)
    avg_quality = sum(vb.get("quality", 0) for vb in value_bets) / len(value_bets)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Picks Found", len(value_bets))
    mc2.metric("Avg Edge", f"{avg_edge:.1%}")
    mc3.metric("Avg EV/unit", f"{avg_ev:.3f}")
    mc4.metric("Avg Quality", f"{avg_quality:.0f}/100")

    st.divider()

    for i, vb in enumerate(value_bets):
        q = vb.get("quality", 0)
        q_label = quality_label(q)
        tier = quality_tier(q)

        with st.container(border=True):
            top_left, top_right = st.columns([3, 1])
            with top_left:
                st.markdown(f"### {vb['home_team']} vs {vb['away_team']}")
                league_display = vb.get("league", league_label)
                st.caption(f"{league_display} \u2022 {vb['date']} \u2022 {vb['time']}")
            with top_right:
                st.metric("Quality", f"{q}/100", help="Composite score (0-100) combining edge strength, EV, match confidence, and odds range. Higher = more bettable.")
                st.caption(f"Tier {tier} \u2022 {q_label}")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Pick", vb["selection"], help="The team or player the model says has the best value for this match.")
            c2.metric("Odds", f"{vb['odds_decimal']:.2f}", help="Consensus decimal odds from sportsbooks (median across all available books).")
            c3.metric("Edge", f"{vb['edge']:.1%}", help="Model probability minus implied probability. Higher edge = bigger disagreement with the market in your favour.")
            c4.metric("EV/unit", f"{vb['ev_per_unit']:.3f}", help="Expected profit per $1 staked. E.g. 0.10 means you expect to make $0.10 profit for every $1 bet over time.")
            c5.metric("Model %", f"{vb['model_prob']:.1%}", help="The Elo model's estimated chance this pick wins the match.")

            det1, det2, det3 = st.columns(3)
            det1.caption(f"Implied: {vb['implied_prob']:.1%}")
            det2.caption(f"Elo: {vb['home_elo']:.0f} / {vb['away_elo']:.0f}")
            det3.caption(f"Match confidence: {vb['match_confidence']:.0f}%")


def render_save_controls(
    value_bets: List[Dict[str, Any]], league_label: str, key_prefix: str = "picks"
) -> None:
    from store import save_picks, init_db

    experimental = not is_two_outcome(league_label)

    if experimental:
        st.info(
            "This is a 3-outcome market \u2014 edges may be overstated."
        )
        allow_save = st.checkbox(
            "I understand and want to save these picks anyway",
            key=f"exp_save_{key_prefix}_{league_label}",
        )
    else:
        allow_save = True

    col1, col2 = st.columns([1, 3])
    with col1:
        save_disabled = not allow_save
        if st.button("Save All Picks", type="secondary", disabled=save_disabled, key=f"save_{key_prefix}_{league_label}"):
            try:
                init_db()
                for vb in value_bets:
                    vb["is_experimental"] = 1 if experimental else 0
                count = save_picks(value_bets)
                st.success(f"Saved {count} pick(s)")
            except Exception as e:
                st.error(f"Failed to save: {e}")


def show_results_explainer() -> None:
    with st.expander("What do these columns mean?"):
        st.markdown(
            "**Match**: The event matchup.\n\n"
            "**Pick**: The side with the best value.\n\n"
            "**Odds**: Consensus decimal odds (median across sportsbooks).\n\n"
            "**Model %**: The model's estimated win probability.\n\n"
            "**Implied %**: The market's implied probability (\u2248 1/odds).\n\n"
            "**Edge**: Model % \u2212 Implied %. Positive = underpriced by the market.\n\n"
            "**EV/unit**: Expected profit per 1 unit staked.\n\n"
            "**Quality**: Composite score (0\u2013100) combining edge, EV, confidence, and odds range.\n\n"
            "**Tier**: A (80+) = Strong, B (60\u201379) = Good, C (<60) = Fair/Weak."
        )


AVAILABLE_COLUMNS = [
    "Match", "Date", "Time", "Pick", "Odds", "Model %",
    "Implied %", "Edge", "EV/unit", "Quality", "Tier",
    "Elo H", "Elo A", "Confidence",
]

DEFAULT_COLUMNS = [
    "Match", "Date", "Pick", "Odds", "Edge", "EV/unit", "Quality", "Tier",
]

SORT_PRESETS = {
    "Best Quality": "quality",
    "Best EV": "EV/unit",
    "Highest Edge": "Edge",
    "Soonest Start": "sort_time",
    "Best Confidence": "Confidence",
}


def display_value_bets_table(
    value_bets: List[Dict[str, Any]], league_label: str
) -> bool:
    show_results_explainer()

    experimental = not is_two_outcome(league_label)

    c1, c2 = st.columns([1, 1])
    with c1:
        sort_choice = st.selectbox(
            "Sort by",
            options=list(SORT_PRESETS.keys()),
            index=0,
            key="sort_preset_explore",
        )
    with c2:
        visible_cols = st.multiselect(
            "Columns",
            options=AVAILABLE_COLUMNS,
            default=DEFAULT_COLUMNS,
            key="visible_cols_explore",
        )

    has_league_col = any("league" in vb for vb in value_bets)

    rows = []
    for vb in value_bets:
        q = vb.get("quality", 0)
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
            "_quality": q,
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
        "quality": "_quality",
    }
    sort_col = sort_map.get(SORT_PRESETS.get(sort_choice, "quality"), "_quality")
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
    df["Quality"] = df["_quality"].apply(lambda x: f"{x}")
    df["Tier"] = df["_quality"].apply(lambda x: quality_tier(x))

    display_cols = [c for c in visible_cols if c in df.columns]
    if has_league_col and "League" not in display_cols:
        display_cols = ["League"] + display_cols
    if not display_cols:
        display_cols = DEFAULT_COLUMNS

    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
    return experimental


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
            st.info("No saved picks yet. Run the pipeline and save your best bets!")
    except Exception:
        st.info("No saved picks yet.")
