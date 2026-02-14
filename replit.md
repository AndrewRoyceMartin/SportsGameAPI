# Sports Predictor

## Overview
A Streamlit-based sports prediction application that finds value bets by comparing model predictions against live sportsbook odds. Odds data sourced from Apify (harvest/sportsbook-odds-scraper), stats/fixtures/results from SofaScore public API. Uses median consensus pricing across multiple sportsbooks.

## Project Architecture
- `app.py` - Main Streamlit application (Value Bets + Saved Picks tabs)
- `main.py` - CLI entry point stub
- `apify_client.py` - Unified Apify REST client (run actor, get dataset items)
- `stats_provider.py` - SofaScore API client with Game dataclass, get_upcoming_games(), get_results_history()
- `odds_math.py` - American-to-decimal conversion, implied probability calculation
- `odds_extract.py` - MoneylineSnapshot dataclass, moneyline extraction, consensus median pricing
- `mapper.py` - Match stats fixtures to odds events (time window + fuzzy team names)
- `value_engine.py` - Compute implied probability, edge, EV; filter and rank value bets
- `store.py` - SQLite store for picks, odds-at-time, and results tracking
- `features.py` - Elo rating system with home advantage
- `test_apify_odds.py` - Integration test for harvest odds actor
- `test_stats_provider.py` - Integration test for SofaScore stats
- `.streamlit/config.toml` - Streamlit server configuration (port 5000)

### Legacy (being replaced)
- `apify_runner.py` - Old Apify client wrapper (uses apify-client SDK)
- `sportsbet_odds.py` - Old Sportsbet scraper actor caller

## Tech Stack
- Python 3.11
- Streamlit (UI framework)
- pandas / numpy (data processing)
- requests (API calls to Apify REST API + SofaScore API)
- thefuzz / rapidfuzz (fuzzy team name matching)
- SQLite (picks storage)

## Data Sources

### Odds: Apify harvest/sportsbook-odds-scraper
- Unified client: `apify_client.py` → `run_actor_get_items()`
- Returns games with sportsbook moneyline/spread/totals odds
- American odds converted to decimal via `odds_math.py`
- Consensus pricing uses median across all sportsbooks via `odds_extract.py`

### Stats/Fixtures/Results: SofaScore Public API
- Direct REST API at `https://api.sofascore.com/api/v1`
- Client: `stats_provider.py`
- `get_upcoming_games(league, date_from, date_to)` → List[Game]
- `get_results_history(league, since_days)` → List[Game]
- Game dataclass: league, start_time_utc, home, away, home_score, away_score, status
- League filter matches substring (e.g., "Premier League (England)")
- Deduplicates events by SofaScore event ID

## Value Engine Pipeline
1. Fetch upcoming fixtures via `stats_provider.get_upcoming_games()`
2. Fetch historical results via `stats_provider.get_results_history()`
3. Build Elo ratings from historical results
4. Fetch live odds from Apify odds actor (harvest/sportsbook-odds-scraper)
5. Extract moneyline snapshots, compute consensus median decimal odds
6. Match fixtures to odds events (time window + fuzzy team names)
7. Compute implied probability from odds, model probability from Elo
8. Calculate edge (model_p - implied_p) and EV per unit stake
9. Filter by min edge, odds range; rank by edge; return top N

## Required Secrets
- `APIFY_TOKEN` - Apify API token (used for odds actor)
- `ODDS_ACTOR_ID` - harvest~sportsbook-odds-scraper

## UI Tabs
1. **Value Bets** - Odds comparison with edge/EV calculation
2. **Saved Picks** - History of saved value bet picks

## Running
```
streamlit run app.py --server.port 5000
```
