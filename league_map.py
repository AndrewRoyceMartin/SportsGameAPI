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
    "UFC": "UFC",
}

TWO_OUTCOME_LEAGUES = [
    "NBA",
    "NFL",
    "NHL",
    "MLB",
    "ATP Tennis",
    "WTA Tennis",
    "College Football",
    "College Basketball",
]

EXPERIMENTAL_LEAGUES = [
    "Champions League",
    "UFC",
]

_SEPARATOR = "\u2014 Experimental \u2014"


def sofascore_to_harvest(league_label: str) -> str | None:
    for sofascore_substr, harvest_key in LEAGUE_MAP.items():
        if sofascore_substr.lower() in league_label.lower():
            return harvest_key
    return None


def available_leagues() -> list[str]:
    return TWO_OUTCOME_LEAGUES + [_SEPARATOR] + EXPERIMENTAL_LEAGUES


def is_two_outcome(league_label: str) -> bool:
    return league_label in TWO_OUTCOME_LEAGUES


def is_separator(league_label: str) -> bool:
    return league_label == _SEPARATOR


