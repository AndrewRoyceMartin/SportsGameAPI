from __future__ import annotations

import os
from datetime import date, timedelta
from api_client import SportsAPIClient
from features import build_elo_ratings, elo_win_prob


def env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise RuntimeError(f"Missing environment variable: {name}")
    return v


def main() -> None:
    base_url = env("SPORTS_API_BASE_URL")
    api_key = env("SPORTS_API_KEY")
    league_id = os.getenv("LEAGUE", "152")

    client = SportsAPIClient(base_url=base_url, api_key=api_key)

    today = date.today()
    hist_start = str(today - timedelta(days=180))
    hist_end = str(today - timedelta(days=1))
    upcoming_start = str(today)
    upcoming_end = str(today + timedelta(days=14))

    games = client.list_completed_games(league_id, hist_start, hist_end)
    upcoming = client.list_upcoming_games(league_id, upcoming_start, upcoming_end)

    if not games:
        raise RuntimeError("No historical games returned. Check your API key and league ID.")

    elo = build_elo_ratings(games)

    print(f"\nPredictions (League {league_id})")
    print("-" * 70)
    for fx in upcoming[:25]:
        h, a = fx["home_team"], fx["away_team"]
        r_h = float(elo.get(h, 1500.0))
        r_a = float(elo.get(a, 1500.0))
        p_home = elo_win_prob(r_h, r_a)

        pick = h if p_home >= 0.5 else a
        print(f'{fx["date_utc"]} | {h} vs {a} | P(home win)={p_home:.3f} | pick={pick}')


if __name__ == "__main__":
    main()
