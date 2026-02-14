# Sports Predictor

## Overview
A Streamlit-based sports prediction application that finds value bets by comparing Elo model predictions against live sportsbook odds. Odds data sourced from Apify (harvest/sportsbook-odds-scraper), stats/fixtures/results from SofaScore public API. Uses median consensus pricing across multiple sportsbooks to identify betting edges.

## Project Architecture
- `app.py` - Main Streamlit application (Value Bets + Saved Picks tabs), full pipeline orchestration
- `apify_client.py` - Unified Apify REST client (run actor, get dataset items)
- `stats_provider.py` - SofaScore API client with Game dataclass, multi-sport support (football, basketball, ice-hockey, american-football, mma)
- `league_map.py` - Maps league labels to SofaScore filters and Harvest actor league keys; groups leagues into Production vs Experimental
- `league_defaults.py` - Per-league default filter values (min edge, odds range, history/lookahead days, top N)
- `odds_fetch.py` - Multi-day odds fetching with deduplication across lookahead window
- `odds_math.py` - American-to-decimal conversion, implied probability calculation
- `odds_extract.py` - MoneylineSnapshot dataclass, moneyline extraction, consensus median pricing
- `mapper.py` - Match Game fixtures to harvest odds events (time window + fuzzy team names + alias expansion)
- `value_engine.py` - Core math: implied probability, edge, expected value calculations
- `store.py` - SQLite store for picks, odds-at-time, and results tracking
- `features.py` - Elo rating system with home advantage
- `.streamlit/config.toml` - Streamlit server configuration (port 5000)

## Supported Leagues
Harvest actor supports: NFL, NBA, NHL, UCL, UFC, College-Football, College-Basketball
SofaScore provides fixtures/results for all of these via sport-specific endpoints.

## Tech Stack
- Python 3.11
- Streamlit (UI framework)
- pandas / numpy (data processing)
- requests (API calls to Apify REST API + SofaScore API)
- thefuzz / rapidfuzz (fuzzy team name matching)
- SQLite (picks storage)

## Data Sources

### Odds: Apify harvest/sportsbook-odds-scraper
- Client: `apify_client.py` → `run_actor_get_items()`
- Returns games with homeTeam/awayTeam/scheduledTime/odds[] per sportsbook
- Each odds entry has moneyLine (currentHomeOdds/currentAwayOdds in American format)
- American odds converted to decimal via `odds_math.py`
- Consensus pricing uses median across all sportsbooks via `odds_extract.py`

### Stats/Fixtures/Results: SofaScore Public API
- Direct REST API at `https://api.sofascore.com/api/v1`
- Sport-specific endpoints: `/sport/{sport}/scheduled-events/{date}`
- Client: `stats_provider.py` with automatic sport routing per league
- `get_upcoming_games(league, date_from, date_to)` → List[Game]
- `get_results_history(league, since_days)` → List[Game]
- Game dataclass: league, start_time_utc, home, away, home_score, away_score, status

## Value Engine Pipeline
1. User selects league from dropdown (defaults to NBA, production leagues first)
2. Per-league defaults auto-applied (min edge, odds range, history/lookahead days)
3. Fetch historical results via SofaScore → build Elo ratings
4. Fetch upcoming fixtures via SofaScore (lookahead window)
5. Fetch live odds from Apify harvest actor across lookahead window (multi-day, deduped)
6. Match fixtures to odds games via fuzzy team names + time window (mapper.py)
7. For each matched game: compute consensus median moneyline odds (home/away)
8. Compute model probability from Elo, implied probability from odds
9. Calculate edge (model_p - implied_p) and EV per unit stake for both sides
10. Filter by min edge, odds range; rank by edge; display top N
11. 3-outcome leagues: dedup best side per match, require explicit save opt-in, tag as experimental

## Production Safety
- 2-outcome leagues (NBA, NFL, NHL, College FB/BB): full pipeline, normal save
- Experimental leagues (Champions League, UFC): save gated behind checkbox, tagged in DB
- Duplicate pick protection via unique index on (match_date, home_team, away_team, selection, odds_decimal)

## Required Secrets
- `APIFY_TOKEN` - Apify API token (used for harvest odds actor)

## UI Tabs
1. **Value Bets** - League selection, pipeline execution, value bet table with save functionality
2. **Saved Picks** - History of saved value bet picks with P/L tracking

## Running
```
streamlit run app.py --server.port 5000
```
