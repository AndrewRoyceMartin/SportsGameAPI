"""Debug: inspect raw Sportsbet eventTime values to identify timezone handling."""
from apify_runner import run_actor_get_items
from sportsbet_odds import SPORTSBET_LEAGUES, _parse_sportsbet_datetime
from time_utils import iso_utc

LEAGUE = "AFL"

def main():
    config = SPORTSBET_LEAGUES[LEAGUE]
    actor_input = {"startUrls": [{"url": config["url"]}]}
    items = run_actor_get_items("lexis-solutions~sportsbet-com-au-scraper", actor_input, timeout=120)
    
    if not items:
        print("No items returned")
        return
    
    print(f"Raw items: {len(items)}")
    
    for item in items:
        results_list = item.get("results", [])
        if not isinstance(results_list, list):
            results_list = [item]
        
        for row in results_list:
            event_time_raw = row.get("eventTime", "")
            participants = row.get("participants", [])
            names = []
            for p in participants:
                names.append(p.get("name", "?"))
            
            parsed_utc = _parse_sportsbet_datetime(event_time_raw)
            
            print(f"  Raw eventTime: '{event_time_raw}'")
            print(f"  Parsed (naive UTC): {parsed_utc}")
            print(f"  iso_utc: {iso_utc(parsed_utc)}")
            print(f"  Participants: {names}")
            print()

if __name__ == "__main__":
    main()
