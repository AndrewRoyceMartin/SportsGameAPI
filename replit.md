# Sports Predictor

## Overview
A Streamlit-based sports prediction application that finds value bets by comparing model predictions against live sportsbook odds. All data is sourced from Apify actors — odds from harvest/sportsbook-odds-scraper and stats/fixtures from a sports stats actor (TBD). Uses median consensus pricing across multiple sportsbooks.

## Project Architecture
- `app.py` - Main Streamlit application (Value Bets + Saved Picks tabs)
- `main.py` - CLI entry point stub
- `apify_client.py` - Unified Apify REST client (run actor, get dataset items)
- `apify_runner.py` - Legacy Apify client wrapper (uses apify-client SDK; being replaced)
- `odds_math.py` - American-to-decimal conversion, implied probability calculation
- `odds_extract.py` - MoneylineSnapshot dataclass, moneyline extraction, consensus median pricing
- `sportsbet_odds.py` - Legacy Sportsbet scraper actor caller (being replaced)
- `mapper.py` - Match stats fixtures to odds events (time window + fuzzy team names)
- `value_engine.py` - Compute implied probability, edge, EV; filter and rank value bets
- `store.py` - SQLite store for picks, odds-at-time, and results tracking
- `features.py` - Elo rating system with home advantage
- `test_apify_odds.py` - Integration test for harvest odds actor
- `.streamlit/config.toml` - Streamlit server configuration (port 5000)

## Tech Stack
- Python 3.11
- Streamlit (UI framework)
- pandas / numpy (data processing)
- requests (API calls to Apify REST API)
- thefuzz / rapidfuzz (fuzzy team name matching)
- SQLite (picks storage)

## Apify Integration
### Unified Client (`apify_client.py`)
- Uses Apify REST API directly (no SDK dependency)
- `run_actor_get_items(actor_id, actor_input, limit, timeout)` → list of dicts
- Endpoint: `POST /v2/acts/{actor_id}/run-sync-get-dataset-items`

### Odds Actor (harvest/sportsbook-odds-scraper)
- Returns games with sportsbook moneyline/spread/totals odds
- American odds converted to decimal via `odds_math.py`
- Consensus pricing uses median across all sportsbooks via `odds_extract.py`

### Stats/Results Actor (TBD)
- Will provide historical match results, fixtures, team data
- Needed for Elo ratings and backtesting

## Value Engine Pipeline
1. Fetch upcoming fixtures + historical results from stats actor
2. Build Elo ratings from historical results
3. Fetch live odds from odds actor (harvest/sportsbook-odds-scraper)
4. Extract moneyline snapshots, compute consensus median decimal odds
5. Match fixtures to odds events (time window + fuzzy team names)
6. Compute implied probability from odds, model probability from Elo
7. Calculate edge (model_p - implied_p) and EV per unit stake
8. Filter by min edge, odds range; rank by edge; return top N

## Required Secrets
- `APIFY_TOKEN` - Apify API token (used for all actors)
- `ODDS_ACTOR_ID` - harvest~sportsbook-odds-scraper
- `STATS_ACTOR_ID` - TBD (sports stats actor)

## UI Tabs
1. **Value Bets** - Odds comparison with edge/EV calculation
2. **Saved Picks** - History of saved value bet picks

## Running
```
streamlit run app.py --server.port 5000
```
