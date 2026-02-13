from __future__ import annotations

import os
from datetime import date, datetime, timedelta
import streamlit as st
import pandas as pd
from api_client import SportsAPIClient
from features import build_elo_ratings, elo_win_prob


def format_date(d: str) -> str:
    if not d:
        return ""
    parts = d.split("-")
    if len(parts) == 3:
        return f"{parts[2]}/{parts[1]}/{parts[0]}"
    return d


def parse_date_input(text: str) -> date | None:
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    return None


def date_to_display(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def date_to_api(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise RuntimeError(f"Missing environment variable: {name}")
    return v


def get_client() -> SportsAPIClient:
    return SportsAPIClient(
        base_url=env("SPORTS_API_BASE_URL"),
        api_key=env("SPORTS_API_KEY"),
    )


@st.cache_data(ttl=600, show_spinner=False)
def fetch_leagues():
    client = get_client()
    return client.list_leagues()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_completed(league_id: str, from_date: str, to_date: str):
    client = get_client()
    return client.list_completed_games(league_id, from_date, to_date)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_upcoming(league_id: str, from_date: str, to_date: str):
    client = get_client()
    return client.list_upcoming_games(league_id, from_date, to_date)


def render_sidebar():
    with st.sidebar:
        st.header("Configuration")

        try:
            leagues_raw = fetch_leagues()
        except Exception as e:
            st.error(f"Could not load leagues: {e}")
            leagues_raw = []

        league_options = {}
        for lg in leagues_raw:
            label = f"{lg.get('league_name', '')} ({lg.get('country_name', '')})"
            league_options[label] = lg.get("league_key", "")

        if league_options:
            selected_label = st.selectbox("League", options=sorted(league_options.keys()))
            league_id = league_options[selected_label]
        else:
            league_id = st.text_input("League ID", value="152")

        st.divider()
        st.subheader("Date Ranges")

        today = date.today()

        st.caption("Historical games (for Elo ratings)")
        hist_start_str = st.text_input(
            "From (dd/mm/yyyy)",
            value=date_to_display(today - timedelta(days=180)),
        )
        hist_end_str = st.text_input(
            "To (dd/mm/yyyy)",
            value=date_to_display(today - timedelta(days=1)),
        )

        hist_start_d = parse_date_input(hist_start_str)
        hist_end_d = parse_date_input(hist_end_str)
        if hist_start_d is None or hist_end_d is None:
            st.error("Invalid historical date format. Use dd/mm/yyyy.")

        st.caption("Upcoming games (for predictions)")
        upcoming_start_str = st.text_input(
            "From  (dd/mm/yyyy)",
            value=date_to_display(today),
        )
        upcoming_end_str = st.text_input(
            "To  (dd/mm/yyyy)",
            value=date_to_display(today + timedelta(days=14)),
        )

        upcoming_start_d = parse_date_input(upcoming_start_str)
        upcoming_end_d = parse_date_input(upcoming_end_str)
        if upcoming_start_d is None or upcoming_end_d is None:
            st.error("Invalid upcoming date format. Use dd/mm/yyyy.")

        st.divider()
        st.subheader("Elo Settings")
        home_adv = st.slider("Home Advantage (Elo pts)", 0, 150, 65, step=5)
        k_factor = st.slider("K-Factor", 5, 50, 20, step=1)
        num_predictions = st.slider("Predictions to show", 5, 50, 25, step=5)

        st.divider()
        refresh = st.button("Refresh Data", use_container_width=True)

    return (
        str(league_id),
        date_to_api(hist_start_d) if hist_start_d else "",
        date_to_api(hist_end_d) if hist_end_d else "",
        date_to_api(upcoming_start_d) if upcoming_start_d else "",
        date_to_api(upcoming_end_d) if upcoming_end_d else "",
        home_adv, k_factor, num_predictions, refresh,
        hist_start_d is None or hist_end_d is None or upcoming_start_d is None or upcoming_end_d is None,
    )


def render_metrics(completed, upcoming, elo):
    cols = st.columns(4)
    cols[0].metric("Completed Games", len(completed))
    cols[1].metric("Upcoming Games", len(upcoming))
    cols[2].metric("Teams Tracked", len(elo))
    if elo:
        top_team = max(elo, key=elo.get)
        cols[3].metric("Top Rated Team", top_team, f"{elo[top_team]:.0f}")


def render_predictions(upcoming, elo, home_adv, num_predictions):
    st.subheader("Upcoming Match Predictions")

    if not upcoming:
        st.info("No upcoming games found for the selected league and date range.")
        return

    rows = []
    for fx in upcoming[:num_predictions]:
        h = fx["home_team"]
        a = fx["away_team"]
        r_h = float(elo.get(h, 1500.0))
        r_a = float(elo.get(a, 1500.0))
        p_home = elo_win_prob(r_h, r_a, home_adv=home_adv)
        p_away = 1.0 - p_home

        if p_home >= 0.5:
            pick = h
            confidence = p_home
        else:
            pick = a
            confidence = p_away

        rows.append({
            "Date": format_date(fx.get("date_utc", "")),
            "Time": fx.get("time", ""),
            "Home": h,
            "Away": a,
            "Home Elo": round(r_h, 1),
            "Away Elo": round(r_a, 1),
            "P(Home)": f"{p_home:.1%}",
            "P(Away)": f"{p_away:.1%}",
            "Pick": pick,
            "Confidence": f"{confidence:.1%}",
            "Round": fx.get("league_round", ""),
        })

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Home Elo": st.column_config.NumberColumn(format="%.1f"),
            "Away Elo": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def render_rankings(elo):
    st.subheader("Current Elo Rankings")

    if not elo:
        st.info("No ratings available yet.")
        return

    sorted_teams = sorted(elo.items(), key=lambda x: x[1], reverse=True)
    rows = []
    for rank, (team, rating) in enumerate(sorted_teams, 1):
        rows.append({
            "Rank": rank,
            "Team": team,
            "Elo Rating": round(rating, 1),
            "vs Avg (1500)": round(rating - 1500.0, 1),
        })

    df = pd.DataFrame(rows)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Elo Rating": st.column_config.NumberColumn(format="%.1f"),
                "vs Avg (1500)": st.column_config.NumberColumn(format="%+.1f"),
            },
        )

    with col2:
        top_n = min(10, len(sorted_teams))
        chart_data = pd.DataFrame(
            {"Team": [t for t, _ in sorted_teams[:top_n]],
             "Rating": [r for _, r in sorted_teams[:top_n]]}
        )
        chart_data = chart_data.set_index("Team")
        st.bar_chart(chart_data, horizontal=True)


def render_recent_results(completed):
    st.subheader("Recent Results")

    if not completed:
        st.info("No completed games found.")
        return

    recent = list(completed[-15:])
    recent.reverse()

    rows = []
    for g in recent:
        hs = g.get("home_score", 0)
        as_ = g.get("away_score", 0)
        if hs > as_:
            result = f"{g['home_team']} Win"
        elif as_ > hs:
            result = f"{g['away_team']} Win"
        else:
            result = "Draw"

        rows.append({
            "Date": format_date(g.get("date_utc", "")),
            "Round": g.get("league_round", ""),
            "Home": g["home_team"],
            "Away": g["away_team"],
            "Score": f"{hs} - {as_}",
            "Result": result,
            "Stadium": g.get("stadium", ""),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title="Sports Predictor", page_icon="⚽", layout="wide")

    st.title("⚽ Sports Predictor")
    st.caption("Elo-based match predictions powered by AllSportsAPI")

    (
        league_id,
        hist_start, hist_end,
        upcoming_start, upcoming_end,
        home_adv, k_factor, num_predictions, refresh,
        has_date_error,
    ) = render_sidebar()

    if refresh:
        st.cache_data.clear()

    if has_date_error:
        st.stop()

    try:
        with st.spinner("Fetching completed games..."):
            completed = fetch_completed(league_id, hist_start, hist_end)
    except Exception as e:
        st.error(f"Failed to fetch completed games: {e}")
        st.info("Check that your API URL and key are configured correctly in Secrets.")
        return

    try:
        with st.spinner("Fetching upcoming games..."):
            upcoming = fetch_upcoming(league_id, upcoming_start, upcoming_end)
    except Exception as e:
        st.warning(f"Could not fetch upcoming games: {e}")
        upcoming = []

    if not completed:
        st.warning(
            "No completed games found for this league and date range. "
            "Try expanding the historical date range or selecting a different league."
        )
        if upcoming:
            st.info(f"Found {len(upcoming)} upcoming game(s), but no history to build Elo ratings from.")
        return

    elo = build_elo_ratings(completed, k=k_factor, home_adv=home_adv)

    render_metrics(completed, upcoming, elo)
    st.divider()

    tab_predict, tab_rankings, tab_results = st.tabs(
        ["Predictions", "Rankings", "Recent Results"]
    )

    with tab_predict:
        render_predictions(upcoming, elo, home_adv, num_predictions)

    with tab_rankings:
        render_rankings(elo)

    with tab_results:
        render_recent_results(completed)


if __name__ == "__main__":
    main()
