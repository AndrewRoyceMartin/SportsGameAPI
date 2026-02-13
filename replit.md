# Sports Predictor

## Overview
A Streamlit-based sports prediction application that uses Elo ratings to predict game outcomes. Data is sourced from AllSportsAPI v2.

## Project Architecture
- `app.py` - Main Streamlit application entry point (frontend)
- `main.py` - CLI entry point for predictions
- `api_client.py` - AllSportsAPI v2 client (Countries, Leagues, Fixtures, H2H)
- `features.py` - Elo rating system with home advantage
- `.streamlit/config.toml` - Streamlit server configuration (port 5000)

## Tech Stack
- Python 3.11
- Streamlit (UI framework)
- scikit-learn (ML predictions)
- pandas / numpy (data processing)
- requests (API calls)

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

## Required Secrets
- `SPORTS_API_BASE_URL` - Base URL for AllSportsAPI (e.g. https://apiv2.allsportsapi.com/football/)
- `SPORTS_API_KEY` - API key for AllSportsAPI

## Running
```
streamlit run app.py --server.port 5000
```
