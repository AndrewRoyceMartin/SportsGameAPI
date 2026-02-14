import os
import json
import requests

APIFY_TOKEN = os.getenv("APIFY_TOKEN")
ACTOR_ID = os.getenv("SPORTSBET_ACTOR_ID")

if not APIFY_TOKEN:
    raise RuntimeError("Missing APIFY_TOKEN")

if not ACTOR_ID:
    raise RuntimeError("Missing SPORTSBET_ACTOR_ID")

TEST_URL = "https://www.sportsbet.com.au/betting/basketball-us/all-basketball-us"

actor_input = {
    "startUrls": [
        {"url": TEST_URL}
    ]
}

url = f"https://api.apify.com/v2/acts/{ACTOR_ID}/run-sync-get-dataset-items"
params = {
    "token": APIFY_TOKEN,
    "format": "json",
    "clean": "true",
    "limit": 1
}

response = requests.post(
    url,
    params=params,
    json=actor_input,
    timeout=600
)
response.raise_for_status()

items = response.json()

print(f"\nTop-level items returned: {len(items)}\n")
print("Top-level keys:", list(items[0].keys()), "\n")

results = items[0].get("results", [])
print(f"Results count: {len(results)}\n")

if not results:
    print("No results found.")
    exit()

first = results[0]
print("First result keys:", list(first.keys()), "\n")

print("First result (truncated):")
print(json.dumps(first, indent=2)[:3000])

# EXTRA: explicitly inspect participants if present
participants = first.get("participants")
if participants:
    print("\nParticipants found:")
    for p in participants:
        print(f"- {p.get('name')}: {p.get('data')}")
else:
    print("\nNo participants field found in first result.")
