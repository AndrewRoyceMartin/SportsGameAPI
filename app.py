from __future__ import annotations

import os
import streamlit as st
import pandas as pd
from store import save_picks, get_recent_picks, init_db


def main():
    st.set_page_config(page_title="Sports Predictor", page_icon="⚽", layout="wide")

    st.title("⚽ Sports Predictor")
    st.caption("Odds-based sports predictions powered by Apify")

    with st.sidebar:
        st.header("Configuration")

        st.subheader("Value Bets")
        min_edge = st.slider("Min Edge %", 1, 30, 5, step=1) / 100.0
        odds_range = st.slider("Odds Range", 1.10, 10.0, (1.80, 3.50), step=0.05)
        top_n = st.selectbox("Top N picks", options=[5, 10, 15, 20, 25], index=1)

    tab_value, tab_history = st.tabs(["Value Bets", "Saved Picks"])

    with tab_value:
        render_value_bets(min_edge, odds_range, top_n)

    with tab_history:
        render_saved_picks()


def render_value_bets(min_edge, odds_range, top_n):
    st.subheader("Value Bets")
    st.caption("Find edges by comparing model predictions against sportsbook odds")

    has_token = bool(os.getenv("APIFY_TOKEN", ""))
    if not has_token:
        st.warning(
            "Apify API token not found. Add APIFY_TOKEN to your Secrets "
            "(Tools → Secrets) to enable live odds fetching."
        )
        return

    st.info("Odds pipeline ready — harvest actor integration coming next.")


def render_saved_picks():
    st.subheader("Saved Picks History")
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
            st.info("No saved picks yet.")
    except Exception:
        st.info("No saved picks yet.")


if __name__ == "__main__":
    main()
