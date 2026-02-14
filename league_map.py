LEAGUE_MAP = {
    "Champions League": "UCL",
    "NBA": "NBA",
    "NHL": "NHL",
    "NFL": "NFL",
    "College Football": "College-Football",
    "College Basketball": "College-Basketball",
    "UFC": "UFC",
}

TWO_OUTCOME_LEAGUES = {
    "NBA",
    "NHL",
    "NFL",
    "College Football",
    "College Basketball",
    "UFC",
}


def sofascore_to_harvest(league_label: str) -> str | None:
    for sofascore_substr, harvest_key in LEAGUE_MAP.items():
        if sofascore_substr.lower() in league_label.lower():
            return harvest_key
    return None


def available_leagues() -> list[str]:
    return list(LEAGUE_MAP.keys())


def is_two_outcome(league_label: str) -> bool:
    return league_label in TWO_OUTCOME_LEAGUES
