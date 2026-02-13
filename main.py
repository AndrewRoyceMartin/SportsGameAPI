from __future__ import annotations

import os
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

    sport = os.getenv("SPORT", "soccer")
    league = os.getenv("LEAGUE", "EPL")
    season = os.getenv("SEASON", "2025-2026")

    client = SportsAPIClient(base_url=base_url, api_key=api_key)

    # Pull data
    games = client.list_completed_games(sport=sport, league=league, season=season)
    upcoming = client.list_upcoming_games(sport=sport, league=league, season=season)

    if not games:
        raise RuntimeError("No historical games returned. Check your API mapping/endpoints.")

    # Build Elo
    elo = build_elo_ratings(games)

    # Predict
    print(f"\nPredictions for {league} ({season})")
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
