import os
import json
import requests

APIFY_TOKEN = os.getenv("APIFY_TOKEN")
ACTOR_ID = os.getenv("ODDS_ACTOR_ID", "harvest~sportsbook-odds-scraper")

if not APIFY_TOKEN:
    raise RuntimeError("Missing APIFY_TOKEN")

actor_input = {
    "league": "UFC",
    "sportsbook": "Bet365"
}

url = f"https://api.apify.com/v2/acts/{ACTOR_ID}/run-sync-get-dataset-items"
params = {
    "token": APIFY_TOKEN,
    "format": "json",
    "clean": "true",
    "limit": 3
}

r = requests.post(url, params=params, json=actor_input, timeout=600)
r.raise_for_status()
items = r.json()

print(f"Fetched {len(items)} games\n")

from odds_extract import extract_moneylines, consensus_decimal

snaps = extract_moneylines(items[0])
print(f"Books with moneyline prices: {len(snaps)}")
print("First 5 book snapshots:")
for s in snaps[:5]:
    print(f"  {s.sportsbook}: home={s.home_decimal} away={s.away_decimal}")

home_cons = consensus_decimal(snaps, "home")
away_cons = consensus_decimal(snaps, "away")
print(f"\nConsensus (median) decimal: home={home_cons} away={away_cons}")
