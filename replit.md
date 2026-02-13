# Sports Predictor

## Overview
A Streamlit-based sports prediction application that uses Elo ratings to predict game outcomes and find value bets by comparing model predictions against live Sportsbet odds. Data is sourced from AllSportsAPI v2 and odds from Sportsbet via Apify.

## Project Architecture
- `app.py` - Main Streamlit application entry point (frontend + all tabs)
- `main.py` - CLI entry point for predictions
- `api_client.py` - AllSportsAPI v2 client (Countries, Leagues, Fixtures, H2H)
- `features.py` - Elo rating system with home advantage
- `apify_runner.py` - Generic Apify actor runner and dataset fetcher
- `sportsbet_odds.py` - Sportsbet scraper actor caller with odds normalisation
- `mapper.py` - Match stats fixtures to Sportsbet events (time window + fuzzy team names)
- `value_engine.py` - Compute implied probability, edge, EV; filter and rank value bets
- `store.py` - SQLite store for picks, odds-at-time, and results tracking
- `.streamlit/config.toml` - Streamlit server configuration (port 5000)

## Tech Stack
- Python 3.11
- Streamlit (UI framework)
- scikit-learn (ML predictions)
- pandas / numpy (data processing)
- requests (API calls)
- apify-client (Apify actor integration)
- thefuzz / rapidfuzz (fuzzy team name matching)
- SQLite (picks storage)

## API Details (AllSportsAPI v2)
- Base URL: `https://apiv2.allsportsapi.com/football/`
- Auth: `APIkey` query parameter
- Key endpoints:
  - `met=Countries` - list countries
  - `met=Leagues` - list leagues (optional `countryId`)
  - `met=Fixtures` - list fixtures by date range (`from`, `to`, optional `leagueId`)
  - `met=H2H` - head to head (`firstTeamId`, `secondTeamId`)
- Response format: `{"success": 1, "result": [...]}`
- Game fields: `event_home_team`, `event_away_team`, `event_final_result` ("X - Y"), `event_date`, `event_status`
- Date range limit: 15 days max when no leagueId specified (handled via chunking)

## Apify / Sportsbet Integration
- Actor: `lexis-solutions/sportsbet-com-au-scraper`
- Returns: events with `eventTime`, `competition`, `eventName`, `outcomes`
- Outcomes contain: selection name, odds/price, market
- Normalised to: event, home_team, away_team, start_time, market, selection, odds_decimal

## Value Engine Pipeline
1. Fetch upcoming fixtures from AllSportsAPI
2. Build Elo ratings from historical results
3. Fetch live odds from Sportsbet via Apify
4. Match fixtures to odds events (time window ±2hrs, fuzzy team names)
5. Compute implied probability from odds, model probability from Elo
6. Calculate edge (model_p - implied_p) and EV per unit stake
7. Filter by min edge, odds range; rank by edge; return top N

## Required Secrets
- `SPORTS_API_BASE_URL` - Base URL for AllSportsAPI
- `SPORTS_API_KEY` - API key for AllSportsAPI
- `APIFY_TOKEN` - Apify API token for Sportsbet scraper

## UI Tabs
1. **Best Bets** - Top confidence picks ranked by Elo prediction strength
2. **Value Bets** - Odds comparison with edge/EV calculation (requires Apify token)
3. **All Predictions** - Full list of upcoming match predictions
4. **Team Search** - Team profile with Elo, record, upcoming/recent (supports all-league search)
5. **Rankings** - Elo rankings with bar chart
6. **Recent Results** - Completed match results

## Running
```
streamlit run app.py --server.port 5000
```
