from __future__ import annotations

import os
import streamlit as st
import pandas as pd
from api_client import SportsAPIClient
from features import build_elo_ratings, elo_win_prob


def env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise RuntimeError(f"Missing environment variable: {name}")
    return v


@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(sport: str, league: str, season: str):
    base_url = env("SPORTS_API_BASE_URL")
    api_key = env("SPORTS_API_KEY")
    client = SportsAPIClient(base_url=base_url, api_key=api_key)

    completed = client.list_completed_games(sport=sport, league=league, season=season)
    upcoming = client.list_upcoming_games(sport=sport, league=league, season=season)
    return completed, upcoming


def render_sidebar():
    with st.sidebar:
        st.header("Configuration")

        sport = st.selectbox(
            "Sport",
            options=["soccer", "basketball", "baseball", "hockey", "football"],
            index=0,
        )

        league = st.text_input("League", value=os.getenv("LEAGUE", "EPL"))
        season = st.text_input("Season", value=os.getenv("SEASON", "2025-2026"))

        home_adv = st.slider("Home Advantage (Elo pts)", 0, 150, 65, step=5)
        k_factor = st.slider("K-Factor", 5, 50, 20, step=1)
        num_predictions = st.slider("Predictions to show", 5, 50, 25, step=5)

        st.divider()
        refresh = st.button("Refresh Data", use_container_width=True)

    return sport, league, season, home_adv, k_factor, num_predictions, refresh


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
        st.info("No upcoming games found for this league and season.")
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
            "Date": fx.get("date_utc", "TBD"),
            "Home": h,
            "Away": a,
            "Home Elo": round(r_h, 1),
            "Away Elo": round(r_a, 1),
            "P(Home Win)": f"{p_home:.1%}",
            "P(Away Win)": f"{p_away:.1%}",
            "Pick": pick,
            "Confidence": f"{confidence:.1%}",
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


def render_rankings(elo, home_adv):
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

    recent = completed[-10:]
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
            "Date": g.get("date_utc", ""),
            "Home": g["home_team"],
            "Away": g["away_team"],
            "Score": f"{hs} - {as_}",
            "Result": result,
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title="Sports Predictor", page_icon="🏆", layout="wide")

    st.title("🏆 Sports Predictor")
    st.caption("Elo-based match predictions powered by historical data")

    sport, league, season, home_adv, k_factor, num_predictions, refresh = render_sidebar()

    if refresh:
        st.cache_data.clear()

    try:
        with st.spinner("Fetching data from API..."):
            completed, upcoming = fetch_data(sport, league, season)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        st.info("Check that your API URL and key are configured correctly in Secrets.")
        return

    if not completed:
        st.warning("No historical games returned. Check your API endpoints, league, and season settings.")
        return

    elo = build_elo_ratings(completed, k=k_factor, home_adv=home_adv)

    render_metrics(completed, upcoming, elo)
    st.divider()

    tab_predict, tab_rankings, tab_results = st.tabs(["Predictions", "Rankings", "Recent Results"])

    with tab_predict:
        render_predictions(upcoming, elo, home_adv, num_predictions)

    with tab_rankings:
        render_rankings(elo, home_adv)

    with tab_results:
        render_recent_results(completed)


if __name__ == "__main__":
    main()
