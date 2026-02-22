"""Debug script to inspect time deltas between fixtures and odds for AU leagues."""
from datetime import date, timedelta
from time_utils import to_naive_utc
from stats_provider import get_upcoming_games
from odds_fetch import fetch_odds_for_window
from mapper import _parse_iso, _name_score

LEAGUES = ["AFL", "NRL", "NBL"]

def main():
    for LEAGUE in LEAGUES:
        print(f"\n{'='*60}")
        print(f"League: {LEAGUE}")
        print(f"{'='*60}")

        try:
            fixtures = get_upcoming_games(
                LEAGUE,
                date_from=date.today(),
                date_to=date.today() + timedelta(days=21),
            )
        except Exception as e:
            print(f"  Fixtures error: {e}")
            fixtures = []

        try:
            odds_games = fetch_odds_for_window(LEAGUE, lookahead_days=21, league_label=LEAGUE)
        except Exception as e:
            print(f"  Odds error: {e}")
            odds_games = []

        print(f"  Fixtures: {len(fixtures)}")
        print(f"  Odds games: {len(odds_games)}")

        if not fixtures:
            print("  (no fixtures)")
            continue
        if not odds_games:
            print("  (no odds games)")
            continue

        print(f"\n  === FIXTURES (first 10) ===")
        for fx in fixtures[:10]:
            fx_dt = to_naive_utc(fx.start_time_utc)
            print(f"  {fx.home:25s} vs {fx.away:25s} | utc={fx.start_time_utc} | naive_utc={fx_dt} | tzinfo={getattr(fx.start_time_utc, 'tzinfo', 'N/A')}")

        print(f"\n  === ODDS (first 10) ===")
        for og in odds_games[:10]:
            h = (og.get("homeTeam") or {}).get("mediumName", "")
            a = (og.get("awayTeam") or {}).get("mediumName", "")
            t = og.get("scheduledTime", "")
            parsed = _parse_iso(t)
            print(f"  {h:25s} vs {a:25s} | raw={t} | parsed={parsed}")

        print(f"\n  === CLOSEST MATCH DELTAS ===")
        for fx in fixtures[:10]:
            fx_dt = to_naive_utc(fx.start_time_utc)
            if fx_dt is None:
                print(f"  {fx.home} vs {fx.away} | NO TIME")
                continue

            best = None
            for og in odds_games:
                h = (og.get("homeTeam") or {}).get("mediumName", "")
                a = (og.get("awayTeam") or {}).get("mediumName", "")
                og_dt = _parse_iso(og.get("scheduledTime", ""))
                if og_dt is None:
                    continue

                home_sc = _name_score(fx.home, h)
                away_sc = _name_score(fx.away, a)
                avg = (home_sc + away_sc) / 2
                home_flip = _name_score(fx.home, a)
                away_flip = _name_score(fx.away, h)
                avg_flip = (home_flip + away_flip) / 2
                name_sc = max(avg, avg_flip)

                diff_h = (og_dt - fx_dt).total_seconds() / 3600.0

                if name_sc >= 50:
                    cand = (abs(diff_h), name_sc, diff_h, h, a, og.get("scheduledTime", ""))
                    if best is None or name_sc > best[1] or (name_sc == best[1] and abs(diff_h) < best[0]):
                        best = cand

            if best:
                print(
                    f"  {fx.home:25s} vs {fx.away:25s} | "
                    f"delta={best[2]:+.2f}h | name_sc={best[1]:.0f} | "
                    f"odds: {best[3]} vs {best[4]} | {best[5]}"
                )
            else:
                print(f"  {fx.home:25s} vs {fx.away:25s} | NO MATCH (name score < 50 for all odds)")


if __name__ == "__main__":
    main()
