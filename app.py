from __future__ import annotations

import os
from datetime import date, datetime, timedelta
import streamlit as st
import altair as alt
import pandas as pd
from api_client import SportsAPIClient
from features import build_elo_ratings, elo_win_prob
from sportsbet_odds import fetch_sportsbet_odds
from mapper import match_fixtures_to_odds
from value_engine import compute_value_bets
from store import save_picks, get_recent_picks, init_db


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
def fetch_countries():
    client = get_client()
    return client.list_countries()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_leagues(country_id: str = ""):
    client = get_client()
    return client.list_leagues(country_id=country_id if country_id else None)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_completed(league_id: str, from_date: str, to_date: str):
    client = get_client()
    return client.list_completed_games(league_id, from_date, to_date)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_upcoming(league_id: str, from_date: str, to_date: str):
    client = get_client()
    return client.list_upcoming_games(league_id, from_date, to_date)


def _date_chunks(from_date: str, to_date: str, max_days: int = 15):
    start = datetime.strptime(from_date, "%Y-%m-%d").date()
    end = datetime.strptime(to_date, "%Y-%m-%d").date()
    while start <= end:
        chunk_end = min(start + timedelta(days=max_days - 1), end)
        yield str(start), str(chunk_end)
        start = chunk_end + timedelta(days=1)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_completed(from_date: str, to_date: str):
    client = get_client()
    all_games = []
    for chunk_start, chunk_end in _date_chunks(from_date, to_date):
        all_games.extend(client.list_completed_games("", chunk_start, chunk_end))
    return all_games


@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_upcoming(from_date: str, to_date: str):
    client = get_client()
    all_fixtures = []
    for chunk_start, chunk_end in _date_chunks(from_date, to_date):
        all_fixtures.extend(client.list_upcoming_games("", chunk_start, chunk_end))
    return all_fixtures


@st.cache_data(ttl=300, show_spinner=False)
def fetch_country_completed(league_ids: tuple, from_date: str, to_date: str):
    all_games = []
    for lid in league_ids:
        all_games.extend(fetch_completed(str(lid), from_date, to_date))
    return all_games


@st.cache_data(ttl=300, show_spinner=False)
def fetch_country_upcoming(league_ids: tuple, from_date: str, to_date: str):
    all_fixtures = []
    for lid in league_ids:
        all_fixtures.extend(fetch_upcoming(str(lid), from_date, to_date))
    return all_fixtures


def render_sidebar():
    with st.sidebar:
        st.header("Configuration")

        try:
            countries_raw = fetch_countries()
        except Exception as e:
            st.error(f"Could not load countries: {e}")
            countries_raw = []

        country_options = {"All countries": ""}
        for c in countries_raw:
            name = c.get("country_name", "")
            if name:
                country_options[name] = str(c.get("country_key", ""))

        selected_country = st.selectbox("Country", options=sorted(country_options.keys()))
        country_id = country_options[selected_country]

        try:
            leagues_raw = fetch_leagues(country_id)
        except Exception as e:
            st.error(f"Could not load leagues: {e}")
            leagues_raw = []

        league_options = {}
        for lg in leagues_raw:
            label = lg.get("league_name", "")
            if not country_id:
                label = f"{label} ({lg.get('country_name', '')})"
            league_options[label] = lg.get("league_key", "")

        country_league_ids = [str(lg.get("league_key", "")) for lg in leagues_raw]

        if league_options:
            all_league_options = {"All leagues": ""}
            all_league_options.update({k: v for k, v in sorted(league_options.items())})
            selected_label = st.selectbox("League", options=list(all_league_options.keys()))
            league_id = all_league_options[selected_label]
        else:
            league_id = st.text_input("League ID", value="152")
            country_league_ids = []

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
        st.subheader("Value Bets")
        min_edge = st.slider("Min Edge %", 1, 30, 5, step=1) / 100.0
        odds_range = st.slider("Odds Range", 1.10, 10.0, (1.80, 3.50), step=0.05)
        top_n = st.selectbox("Top N picks", options=[5, 10, 15, 20, 25], index=1)

        st.divider()
        refresh = st.button("Refresh Data", width="stretch")

    return (
        str(league_id),
        country_id,
        country_league_ids,
        date_to_api(hist_start_d) if hist_start_d else "",
        date_to_api(hist_end_d) if hist_end_d else "",
        date_to_api(upcoming_start_d) if upcoming_start_d else "",
        date_to_api(upcoming_end_d) if upcoming_end_d else "",
        home_adv, k_factor, num_predictions,
        min_edge, odds_range, top_n,
        refresh,
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
        width="stretch",
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
            width="stretch",
            hide_index=True,
            column_config={
                "Elo Rating": st.column_config.NumberColumn(format="%.1f"),
                "vs Avg (1500)": st.column_config.NumberColumn(format="%+.1f"),
            },
        )

    with col2:
        top_n = min(10, len(sorted_teams))
        if top_n > 0:
            chart_teams = [t for t, _ in sorted_teams[:top_n]]
            chart_ratings = [r for _, r in sorted_teams[:top_n]]
            if all(isinstance(r, (int, float)) and r == r for r in chart_ratings):
                chart_data = pd.DataFrame(
                    {"Team": chart_teams, "Rating": chart_ratings}
                )
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X("Rating:Q", title="Elo Rating"),
                    y=alt.Y("Team:N", sort="-x", title=None),
                    tooltip=["Team", "Rating"],
                ).properties(height=max(top_n * 30, 200))
                st.altair_chart(chart, width="stretch")
            else:
                st.info("Chart unavailable — ratings contain invalid values.")


def render_best_bets(upcoming, elo, home_adv):
    st.subheader("Top Confidence Picks")

    if not upcoming:
        st.info("No upcoming games found for the selected league and date range.")
        return

    rows = []
    for fx in upcoming:
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
            "date": format_date(fx.get("date_utc", "")),
            "time": fx.get("time", ""),
            "home": h,
            "away": a,
            "home_elo": round(r_h, 1),
            "away_elo": round(r_a, 1),
            "p_home": p_home,
            "p_away": p_away,
            "pick": pick,
            "confidence": confidence,
            "round": fx.get("league_round", ""),
            "league": fx.get("league_name", ""),
            "country": fx.get("country_name", ""),
            "elo_diff": abs(r_h - r_a),
        })

    rows.sort(key=lambda r: r["confidence"], reverse=True)

    st.caption(f"All {len(rows)} upcoming game(s) ranked by prediction confidence")

    for i, r in enumerate(rows[:3]):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**{r['home']}** vs **{r['away']}**")
            league_info = r['league']
            if r['country']:
                league_info = f"{r['league']} · {r['country']}"
            st.caption(f"{r['date']} {r['time']}  ·  {league_info}  ·  {r['round']}")
        with col2:
            st.metric("Pick", r["pick"])
        with col3:
            st.metric("Confidence", f"{r['confidence']:.1%}")

    if len(rows) > 3:
        st.divider()

    display_rows = []
    for r in rows:
        display_rows.append({
            "Date": r["date"],
            "Time": r["time"],
            "League": r["league"],
            "Country": r["country"],
            "Home": r["home"],
            "Away": r["away"],
            "Home Elo": r["home_elo"],
            "Away Elo": r["away_elo"],
            "Elo Gap": round(r["elo_diff"], 1),
            "P(Home)": f"{r['p_home']:.1%}",
            "P(Away)": f"{r['p_away']:.1%}",
            "Pick": r["pick"],
            "Confidence": f"{r['confidence']:.1%}",
            "Round": r["round"],
        })

    df = pd.DataFrame(display_rows)
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Home Elo": st.column_config.NumberColumn(format="%.1f"),
            "Away Elo": st.column_config.NumberColumn(format="%.1f"),
            "Elo Gap": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def render_team_search(completed, upcoming, elo, home_adv, hist_start, hist_end, upcoming_start, upcoming_end):
    st.subheader("Team Search")

    search_all = st.toggle("Search all leagues", value=False)

    if search_all:
        try:
            with st.spinner("Fetching data across all leagues..."):
                all_completed = fetch_all_completed(hist_start, hist_end)
                all_upcoming = fetch_all_upcoming(upcoming_start, upcoming_end)
            all_elo = build_elo_ratings(all_completed, home_adv=home_adv)
        except Exception as e:
            st.error(f"Could not fetch all-league data: {e}")
            all_completed, all_upcoming, all_elo = completed, upcoming, elo
    else:
        all_completed, all_upcoming, all_elo = completed, upcoming, elo

    if not all_elo:
        st.info("No team data available yet.")
        return

    team_names = sorted(all_elo.keys())

    search_text = st.text_input("Type a team name to filter")
    if search_text:
        filtered = [t for t in team_names if search_text.lower() in t.lower()]
    else:
        filtered = team_names

    if not filtered:
        st.warning("No teams match your search.")
        return

    selected_team = st.selectbox("Select a team", options=filtered)

    if not selected_team:
        return

    rating = all_elo.get(selected_team, 1500.0)
    sorted_teams = sorted(all_elo.items(), key=lambda x: x[1], reverse=True)
    rank = next((i for i, (t, _) in enumerate(sorted_teams, 1) if t == selected_team), len(sorted_teams))

    col1, col2, col3 = st.columns(3)
    col1.metric("Elo Rating", f"{rating:.1f}")
    col2.metric("Rank", f"{rank} / {len(sorted_teams)}")
    col3.metric("vs Average", f"{rating - 1500.0:+.1f}")

    team_completed = [g for g in all_completed if g["home_team"] == selected_team or g["away_team"] == selected_team]
    team_upcoming = [g for g in all_upcoming if g["home_team"] == selected_team or g["away_team"] == selected_team]

    if team_completed:
        wins = sum(1 for g in team_completed if (g["home_score"] > g["away_score"] and g["home_team"] == selected_team) or (g["away_score"] > g["home_score"] and g["away_team"] == selected_team))
        draws = sum(1 for g in team_completed if g["home_score"] == g["away_score"])
        losses = len(team_completed) - wins - draws

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Played", len(team_completed))
        c2.metric("Wins", wins)
        c3.metric("Draws", draws)
        c4.metric("Losses", losses)

    if team_upcoming:
        st.markdown("---")
        st.markdown("**Upcoming Matches**")
        rows = []
        for fx in team_upcoming:
            h = fx["home_team"]
            a = fx["away_team"]
            r_h = float(all_elo.get(h, 1500.0))
            r_a = float(all_elo.get(a, 1500.0))
            p_home = elo_win_prob(r_h, r_a, home_adv=home_adv)
            p_away = 1.0 - p_home
            pick = h if p_home >= 0.5 else a
            confidence = p_home if p_home >= 0.5 else p_away

            rows.append({
                "Date": format_date(fx.get("date_utc", "")),
                "Time": fx.get("time", ""),
                "League": fx.get("league_name", ""),
                "Home": h,
                "Away": a,
                "P(Home)": f"{p_home:.1%}",
                "P(Away)": f"{p_away:.1%}",
                "Pick": pick,
                "Confidence": f"{confidence:.1%}",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if team_completed:
        st.markdown("---")
        st.markdown("**Recent Results**")
        recent = list(team_completed[-10:])
        recent.reverse()
        rows = []
        for g in recent:
            hs = g.get("home_score", 0)
            as_ = g.get("away_score", 0)
            if g["home_team"] == selected_team:
                opponent = g["away_team"]
                venue = "Home"
            else:
                opponent = g["home_team"]
                venue = "Away"

            if (hs > as_ and g["home_team"] == selected_team) or (as_ > hs and g["away_team"] == selected_team):
                result = "Win"
            elif hs == as_:
                result = "Draw"
            else:
                result = "Loss"

            rows.append({
                "Date": format_date(g.get("date_utc", "")),
                "League": g.get("league_name", ""),
                "Opponent": opponent,
                "Venue": venue,
                "Score": f"{hs} - {as_}",
                "Result": result,
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


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
    st.dataframe(df, width="stretch", hide_index=True)


def render_value_bets(upcoming, elo, home_adv, min_edge, odds_range, top_n):
    st.subheader("Value Bets")
    st.caption("Matches where the Elo model finds an edge over Sportsbet odds")

    has_token = bool(os.getenv("APIFY_TOKEN", ""))
    if not has_token:
        st.warning(
            "Apify API token not found. Add APIFY_TOKEN to your Secrets "
            "(Tools -> Secrets) to enable live odds fetching."
        )

    if has_token:
        _render_value_bets_fetcher(upcoming, elo, home_adv, min_edge, odds_range, top_n)

    st.divider()
    st.caption("Saved Picks History")
    try:
        history = get_recent_picks(limit=25)
        if history:
            hist_rows = []
            for h in history:
                hist_rows.append({
                    "Saved": h.get("created_at", ""),
                    "Match": f"{h['home_team']} vs {h['away_team']}",
                    "Selection": h.get("selection", ""),
                    "Odds": f"{h.get('odds_decimal', 0):.2f}" if h.get("odds_decimal") else "",
                    "Edge": f"{h.get('edge', 0):.1%}" if h.get("edge") else "",
                    "Result": h.get("result", "Pending"),
                    "P/L": f"{h.get('profit_loss', 0):.2f}" if h.get("profit_loss") is not None else "",
                })
            st.dataframe(pd.DataFrame(hist_rows), width="stretch", hide_index=True)
        else:
            st.info("No saved picks yet. Find value bets and save them above.")
    except Exception:
        st.info("No saved picks yet.")


def _render_value_bets_fetcher(upcoming, elo, home_adv, min_edge, odds_range, top_n):
    sportsbet_url = st.text_input(
        "Sportsbet URL to scrape",
        value="https://www.sportsbet.com.au/betting/soccer",
    )

    fetch_odds_btn = st.button("Fetch Odds & Find Value", width="stretch")

    if fetch_odds_btn:
        try:
            with st.spinner("Fetching odds from Sportsbet (this may take a few minutes)..."):
                odds_data = fetch_sportsbet_odds(
                    start_urls=[sportsbet_url],
                    timeout_secs=300,
                )
            st.success(f"Fetched {len(odds_data)} odds entries from Sportsbet")

            if not odds_data:
                st.warning("No odds data returned. Try a different Sportsbet URL.")
                return

            with st.spinner("Matching fixtures to odds..."):
                matched = match_fixtures_to_odds(upcoming, odds_data)

            st.info(f"Matched {len(matched)} fixture-odds pairs")

            if not matched:
                st.warning(
                    "Could not match any upcoming fixtures to Sportsbet events. "
                    "This can happen if team names differ significantly or no events overlap."
                )
                return

            odds_min, odds_max = odds_range
            value_picks = compute_value_bets(
                matched, elo,
                home_adv=home_adv,
                min_edge=min_edge,
                odds_min=odds_min,
                odds_max=odds_max,
                top_n=top_n,
            )

            if not value_picks:
                st.info(
                    f"No value bets found with edge >= {min_edge:.0%} "
                    f"and odds between {odds_min:.2f} - {odds_max:.2f}. "
                    "Try lowering the minimum edge or widening the odds range."
                )
                return

            st.success(f"Found {len(value_picks)} value bet(s)")

            display_rows = []
            for vb in value_picks:
                display_rows.append({
                    "Match": f"{vb['home_team']} vs {vb['away_team']}",
                    "League": vb.get("league", ""),
                    "Market": vb.get("market", ""),
                    "Selection": vb["selection"],
                    "Odds": f"{vb['odds_decimal']:.2f}",
                    "Implied %": f"{vb['implied_prob']:.1%}",
                    "Model %": f"{vb['model_prob']:.1%}",
                    "Edge": f"{vb['edge']:.1%}",
                    "EV/unit": f"{vb['ev_per_unit']:.3f}",
                    "Pick": vb["selection"],
                })

            st.dataframe(
                pd.DataFrame(display_rows),
                width="stretch",
                hide_index=True,
            )

            save_col1, save_col2 = st.columns([1, 3])
            with save_col1:
                if st.button("Save Picks"):
                    count = save_picks(value_picks)
                    st.success(f"Saved {count} pick(s) to database")

        except Exception as e:
            st.error(f"Error fetching odds: {e}")


def main():
    st.set_page_config(page_title="Sports Predictor", page_icon="⚽", layout="wide")

    st.title("⚽ Sports Predictor")
    st.caption("Elo-based match predictions powered by AllSportsAPI")

    (
        league_id,
        country_id, country_league_ids,
        hist_start, hist_end,
        upcoming_start, upcoming_end,
        home_adv, k_factor, num_predictions,
        min_edge, odds_range, top_n,
        refresh,
        has_date_error,
    ) = render_sidebar()

    if refresh:
        st.cache_data.clear()

    if has_date_error:
        st.stop()

    try:
        with st.spinner("Fetching completed games..."):
            if league_id:
                completed = fetch_completed(league_id, hist_start, hist_end)
            elif country_id and country_league_ids:
                completed = fetch_country_completed(tuple(country_league_ids), hist_start, hist_end)
            else:
                completed = fetch_all_completed(hist_start, hist_end)
    except Exception as e:
        st.error(f"Failed to fetch completed games: {e}")
        st.info("Check that your API URL and key are configured correctly in Secrets.")
        return

    try:
        with st.spinner("Fetching upcoming games..."):
            if league_id:
                upcoming = fetch_upcoming(league_id, upcoming_start, upcoming_end)
            elif country_id and country_league_ids:
                upcoming = fetch_country_upcoming(tuple(country_league_ids), upcoming_start, upcoming_end)
            else:
                upcoming = fetch_all_upcoming(upcoming_start, upcoming_end)
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

    tab_best, tab_value, tab_predict, tab_team, tab_rankings, tab_results = st.tabs(
        ["Best Bets", "Value Bets", "All Predictions", "Team Search", "Rankings", "Recent Results"]
    )

    with tab_best:
        render_best_bets(upcoming, elo, home_adv)

    with tab_value:
        render_value_bets(upcoming, elo, home_adv, min_edge, odds_range, top_n)

    with tab_predict:
        render_predictions(upcoming, elo, home_adv, num_predictions)

    with tab_team:
        render_team_search(completed, upcoming, elo, home_adv, hist_start, hist_end, upcoming_start, upcoming_end)

    with tab_rankings:
        render_rankings(elo)

    with tab_results:
        render_recent_results(completed)


if __name__ == "__main__":
    main()
