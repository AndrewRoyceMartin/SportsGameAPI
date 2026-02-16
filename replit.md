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
- **5-tab layout**: Place Bets (card-based shortlist + bet builder), Browse All (discovery table), Trust & Tuning (backtesting), Fix Issues (diagnostics), Track Results (feedback loop)
- **3-step funnel stepper**: Top of Place Bets shows Scan → Risk → Review workflow with current rules pills (edge %, odds range, lookahead, confidence)
- **Best Bet Right Now**: Hero card with "Why this is #1" one-liner, quarter-Kelly stake suggestion, risk tag, confidence calibration hint
- **Quick Filters**: Above cards — Starts in (6h/24h/3d/All), Tier (A/B/C), Risk (Low/Medium/High) for browsing picks
- **Shortlist summary**: Portfolio view when Best Bets Mode ON — picks count, avg edge, avg confidence, avg quality, earliest start, estimated outlay
- **Bet Builder**: Right column layout with A-tier count, all picks list, stake input, total outlay, Copy as Text, CSV export with Quality/Tier columns
- **Pick card hierarchy**: Top row = team/time/badges/risk, Middle row = Model%/Market%/Edge/Odds (big metrics), Bottom = collapsed Details expander (quality breakdown, Elo, games, why explainer, quarter-Kelly stake)
- **Action strip with buttons**: Actionable session_state mutations — "Lower edge to X%", "Widen odds range", "Extend lookahead to 21 days", "Switch to Aggressive"
- **Browse All discovery**: Edge slider (without changing defaults), near-misses toggle (show bets 1-2% below threshold), recomputes values at lower thresholds
- **Track Results feedback loop**: Won/Lost/Void buttons on each pending pick, rolling stats panel (Record W-L-V, Hit Rate, ROI, Total P/L), hit rate by quality tier, hit rate by confidence bucket
- **AU season banner**: Preseason warning for AFL/NRL/NBL during Jan-Mar months
- **Confidence Target**: First-class sidebar control (Any/55%+/60%+/65%+/70%+) filtering by minimum model probability
- **Best Bets Mode**: Toggle (default ON) that auto-sorts by quality, limits to top 10, locks defaults
- **Bet Quality scoring**: Composite 0-100 score from edge (35%), EV (25%), confidence (20%), odds sanity (20%)
- **Quality tiers**: A (80+) = Strong, B (60-79) = Good, C (<60) = Fair/Weak
- **Rating maturity penalty**: Quality score penalized for teams with few games (0.90x if <10, 0.95x if <20)
- **Risk tags**: Low (edge>=10%, mature), Medium (edge>=5% or developing), High (early/low edge)
- **Sport filter**: Filter leagues by sport category (All, Basketball, Football, Hockey, MMA, Soccer)
- **Run profiles**: Conservative (higher edge, tighter odds), Balanced (league defaults), Aggressive (lower edge, wider odds)
- **Sorting presets**: Sort by Best Quality, Best EV, Highest Edge, Soonest Start, Best Confidence
- **Column toggles**: Show/hide columns including Quality and Tier columns
- **AU timezone display**: AFL/NRL/NBL games show local AEDT time with UTC secondary
- **Backtest suggested settings**: "Use backtest-optimised settings" button auto-applies confidence thresholds based on model accuracy
- **Clean light theme**: Light grey-blue background (#F6F8FC), white cards with subtle shadows, blue primary buttons (#2563EB), rounded 16px panels, soft blue pill/chip tints

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
1. **Place Bets** - Hero card + card-based shortlist of top value bets with quality badges, bet builder, and save controls
2. **Browse All** - Full sortable table with all value bets, column toggles, sort presets, near-miss toggle, edge slider
3. **Trust & Tuning** - Test Elo model against historical results with accuracy, Brier score, log loss, bucket lift metrics + apply optimised settings
4. **Fix Issues** - Pipeline stats, unmatched samples, odds source info, error details
5. **Track Results** - Won/Lost/Void outcome tracking, rolling stats (ROI, hit rate by tier/confidence), P/L history

## Running
```
streamlit run app.py --server.port 5000
```
