# Sports Predictor

## Overview
A Streamlit-based sports prediction application that finds value bets by comparing Elo model predictions against live sportsbook odds. Odds data sourced from Apify (harvest/sportsbook-odds-scraper), stats/fixtures/results from SofaScore public API. Uses median consensus pricing across multiple sportsbooks to identify betting edges.

## Project Architecture
- `app.py` - Streamlit entry point: page config, sidebar controls (sport filter, search, run profiles, lock defaults), tab layout; delegates to pipeline.py and ui_helpers.py
- `pipeline.py` - Data pipeline orchestration: fetch odds/fixtures, build Elo, match, compute EV, dedup, get_unmatched
- `ui_helpers.py` - Reusable UI components: diagnostics panel, reason-coded empty states, unmatched samples, sorting presets, column toggles, harvest games display, results explainer, save controls, saved picks table
- `config_env.py` - Centralized environment audit: required/stale secret checks, startup validation
- `apify_client.py` - Unified Apify REST client (run actor, get dataset items)
- `stats_provider.py` - SofaScore API client with Game dataclass, multi-sport support (football, basketball, ice-hockey, american-football, baseball, tennis, mma)
- `league_map.py` - Maps league labels to SofaScore filters and Harvest actor league keys; groups leagues into Production vs Experimental; LEAGUE_SPORT mapping for sport filter UI
- `league_defaults.py` - Per-league default filter values (min edge, odds range, history/lookahead days, top N); RUN_PROFILES (Conservative/Balanced/Aggressive) with apply_profile(); ELO_PARAMS per-league Elo tuning (K, home_adv, scale, recency_half_life)
- `sportsbet_odds.py` - Sportsbet AU odds parser: extracts Head to Head markets, parses AU datetimes, transforms to Harvest-compatible format
- `odds_fetch.py` - Odds fetching with provider routing (Harvest for US leagues, Sportsbet for AU leagues)
- `odds_math.py` - American-to-decimal conversion, implied probability calculation
- `odds_extract.py` - MoneylineSnapshot dataclass, moneyline extraction, consensus median pricing
- `mapper.py` - Match Game fixtures to harvest odds events (time window + fuzzy team names + alias expansion); handles "Surname, Name" tennis format
- `value_engine.py` - Core math: implied probability, edge, expected value calculations
- `store.py` - SQLite store for picks, odds-at-time, and results tracking
- `features.py` - Elo rating system with per-league tuning (K, home advantage, scale), dynamic K by games played, margin-of-victory weighting, recency decay
- `.streamlit/config.toml` - Streamlit server configuration (port 5000)

## Supported Leagues
Harvest actor supports: NFL, NBA, NHL, UCL, UFC, College-Football, College-Basketball
Sportsbet actor (lexis-solutions~sportsbet-com-au-scraper) supports: AFL, NRL, NBL — odds via Match Betting markets
SofaScore provides fixtures/results via sport-specific endpoints (basketball, ice-hockey, american-football, football, mma, aussie-rules, rugby).

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
1. User selects sport filter + search to narrow leagues, then picks a league
2. Run profile (Conservative/Balanced/Aggressive) adjusts defaults automatically
3. Per-league defaults auto-applied with profile multipliers (min edge, odds range, history/lookahead days)
4. Lock defaults toggle prevents accidental manual edits
5. Fetch historical results via SofaScore → build Elo ratings
6. Fetch upcoming fixtures via SofaScore (lookahead window)
7. Fetch live odds from Apify harvest actor across lookahead window (multi-day, deduped)
8. Match fixtures to odds games via fuzzy team names + time window (mapper.py)
9. For each matched game: compute consensus median moneyline odds (home/away)
10. Compute model probability from Elo, implied probability from odds
11. Calculate edge (model_p - implied_p) and EV per unit stake for both sides
12. Filter by min edge, odds range; rank by edge; display top N
13. Reason-coded empty-state messages explain why no value bets were found
14. Unmatched samples expander shows fixtures/odds that failed to pair
15. 3-outcome leagues: dedup best side per match, require explicit save opt-in, tag as experimental

## UI Features
- **5-tab layout**: Picks (card-based shortlist), Explore (sortable table), Backtest (model accuracy testing), Diagnostics (pipeline stats), Saved Picks (history)
- **Best Bets Mode**: Toggle (default ON) that auto-sorts by quality, limits to top 10, locks defaults
- **Bet Quality scoring**: Composite 0-100 score from edge (35%), EV (25%), confidence (20%), odds sanity (20%)
- **Quality tiers**: A (80+) = Strong, B (60-79) = Good, C (<60) = Fair/Weak
- **Rating maturity penalty**: Quality score penalized for teams with few games (0.90x if <10, 0.95x if <20)
- **Card-based picks**: Match details, quality badge, maturity badge (Mature/Developing/Early), "Why" explainer text, games played counts per pick
- **Simplified sidebar**: Basic controls always visible (sport, league, profile, Best Bets Mode) + Advanced expander
- **Sport filter**: Filter leagues by sport category (All, Basketball, Football, Hockey, MMA, Soccer)
- **Search**: Live text filtering of league names
- **Run profiles**: Conservative (higher edge, tighter odds), Balanced (league defaults), Aggressive (lower edge, wider odds)
- **Sorting presets**: Sort by Best Quality, Best EV, Highest Edge, Soonest Start, Best Confidence
- **Column toggles**: Show/hide columns including Quality and Tier columns
- **Actionable empty states**: Specific guidance with "What to try" suggestions at each pipeline stage
- **Smart diagnostics**: Distinguishes naming mismatches from coverage gaps with targeted guidance

## Production Safety
- 2-outcome leagues (NBA, NFL, NHL, AFL, NRL, NBL, College FB/BB): full pipeline, normal save
- Experimental leagues (Champions League, UFC): save gated behind checkbox, tagged in DB
- Duplicate pick protection via unique index on (match_date, home_team, away_team, selection, odds_decimal)

## Required Secrets
- `APIFY_TOKEN` - Apify API token (used for harvest odds actor)
- Centralized in `config_env.py`; startup audit surfaces missing/stale secrets in diagnostics

## Elo Model Features
- **Per-league parameters**: K-factor, home advantage, Elo scale, recency half-life tuned per league via ELO_PARAMS
- **Dynamic K**: Teams with <10 games get K*1.8, <20 games get K*1.3, stabilizing as more data accumulates
- **Margin-of-victory weighting**: Blowouts update Elo more than close games using log(MOV+1) with auto-correction for rating gaps
- **Recency decay**: Older training games have reduced K via exponential decay with configurable half-life per league
- **Backtest scoring**: Brier score, log loss, bucket lift (confident vs overall accuracy delta)
- **Walk-forward validation**: Multi-fold backtesting with mean±std of all metrics, naive Elo baseline comparison, always-home baseline, per-fold detail table
- **Baseline comparisons**: Naive Elo (K=20, HomeAdv=65, Scale=400, no MOV/recency), always-pick-home, coin-flip Brier

## UI Tabs
1. **Picks** - Card-based shortlist of top value bets with quality badges and save controls
2. **Explore** - Full sortable table with all value bets, column toggles, sort presets
3. **Backtest** - Test Elo model against historical results with accuracy, Brier score, log loss, bucket lift metrics
4. **Diagnostics** - Pipeline stats, unmatched samples, odds source info, error details
5. **Saved Picks** - History of saved picks with P/L tracking

## Running
```
streamlit run app.py --server.port 5000
```
