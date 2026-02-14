"""Integration test for stats_provider.py (SofaScore public API).

Run: python test_stats_provider.py
"""
from __future__ import annotations

from datetime import date, timedelta
from stats_provider import get_upcoming_games, get_results_history


def main():
    print("=== Testing get_results_history (EPL, last 7 days) ===")
    results = get_results_history(league="Premier League (England)", since_days=7)
    print(f"Found {len(results)} completed games")
    for g in results[:5]:
        assert g.status == "completed"
        assert g.home_score is not None
        assert g.away_score is not None
        print(f"  {g.start_time_utc:%Y-%m-%d %H:%M} | {g.home} {g.home_score}-{g.away_score} {g.away} | {g.league}")

    print()
    print("=== Testing get_upcoming_games (EPL, next 14 days) ===")
    upcoming = get_upcoming_games(
        league="Premier League (England)",
        date_from=date.today(),
        date_to=date.today() + timedelta(days=14),
    )
    print(f"Found {len(upcoming)} upcoming games")
    for g in upcoming[:5]:
        assert g.status == "upcoming"
        assert g.home_score is None
        assert g.away_score is None
        print(f"  {g.start_time_utc:%Y-%m-%d %H:%M} | {g.home} vs {g.away} | {g.league}")

    print()
    print("All assertions passed.")


if __name__ == "__main__":
    main()
