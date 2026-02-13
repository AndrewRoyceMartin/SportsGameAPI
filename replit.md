# Sports Predictor

## Overview
A Streamlit-based sports prediction application that uses machine learning to predict game outcomes.

## Project Architecture
- `app.py` - Main Streamlit application entry point
- `.streamlit/config.toml` - Streamlit server configuration (port 5000)

## Tech Stack
- Python 3.11
- Streamlit (UI framework)
- scikit-learn (ML predictions)
- pandas / numpy (data processing)
- requests (API calls)

## Required Secrets
- `SPORTS_API_BASE_URL` - Base URL for sports data API
- `SPORTS_API_KEY` - API key for sports data API
- `SPORT` (optional) - Sport type
- `LEAGUE` (optional) - League name
- `SEASON` (optional) - Season identifier

## Running
```
streamlit run app.py --server.port 5000
```
