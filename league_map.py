LEAGUE_MAP = {
    "Champions League": "UCL",
    "NBA": "NBA",
    "NHL": "NHL",
    "NFL": "NFL",
    "MLB": "MLB",
    "ATP Tennis": "ATP",
    "WTA Tennis": "WTA",
    "College Football": "College-Football",
    "College Basketball": "College-Basketball",
    "AFL": "AFL",
    "NRL": "NRL",
    "NBL": "NBL",
    "Super Rugby": "Super-Rugby",
    "A-League": "A-League",
    "UFC": "UFC",
}

TWO_OUTCOME_LEAGUES = [
    "NBA",
    "NFL",
    "NHL",
    "MLB",
    "ATP Tennis",
    "WTA Tennis",
    "AFL",
    "NRL",
    "NBL",
    "Super Rugby",
    "College Football",
    "College Basketball",
]

EXPERIMENTAL_LEAGUES = [
    "Champions League",
    "A-League",
    "UFC",
]

_SEPARATOR = "\u2014 Experimental \u2014"

LEAGUE_SPORT = {
    "NBA": "Basketball",
    "College Basketball": "Basketball",
    "NBL": "Basketball",
    "NFL": "Football",
    "College Football": "Football",
    "NHL": "Hockey",
    "MLB": "Baseball",
    "ATP Tennis": "Tennis",
    "WTA Tennis": "Tennis",
    "AFL": "Aussie Rules",
    "NRL": "Rugby League",
    "Super Rugby": "Rugby Union",
    "Champions League": "Soccer",
    "A-League": "Soccer",
    "UFC": "MMA",
}

ALL_SPORTS = ["All"] + sorted(set(LEAGUE_SPORT.values()))


def sofascore_to_harvest(league_label: str) -> str | None:
    for sofascore_substr, harvest_key in LEAGUE_MAP.items():
        if sofascore_substr.lower() in league_label.lower():
            return harvest_key
    return None


def available_leagues(sport_filter: str = "All", search: str = "") -> list[str]:
    search_lower = search.strip().lower()

    def _matches(league: str) -> bool:
        if sport_filter != "All" and LEAGUE_SPORT.get(league, "") != sport_filter:
            return False
        if search_lower and search_lower not in league.lower():
            return False
        return True

    prod = [l for l in TWO_OUTCOME_LEAGUES if _matches(l)]
    exp = [l for l in EXPERIMENTAL_LEAGUES if _matches(l)]

    if prod and exp:
        return prod + [_SEPARATOR] + exp
    return prod + exp


def is_two_outcome(league_label: str) -> bool:
    return league_label in TWO_OUTCOME_LEAGUES


def is_separator(league_label: str) -> bool:
    return league_label == _SEPARATOR
