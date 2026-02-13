import os
import json
import requests

APIFY_TOKEN = os.getenv("APIFY_TOKEN")
ACTOR_ID = os.getenv("SPORTSBET_ACTOR_ID", "lexis-solutions~sportsbet-com-au-scraper")

if not APIFY_TOKEN:
    raise RuntimeError("Missing APIFY_TOKEN in Replit Secrets")

actor_input = {
    "startUrls": [
        {"url": "https://www.sportsbet.com.au/betting/basketball"}
    ]
}

url = f"https://api.apify.com/v2/acts/{ACTOR_ID}/run-sync-get-dataset-items"
params = {
    "token": APIFY_TOKEN,
    "format": "json",
    "clean": "true",
    "limit": 5
}

r = requests.post(url, params=params, json=actor_input, timeout=300)
r.raise_for_status()

items = r.json()
print(f"Fetched {len(items)} items (showing up to 5):\n")
print(json.dumps(items, indent=2)[:4000])
