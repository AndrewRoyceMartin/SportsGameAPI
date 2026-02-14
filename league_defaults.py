RUN_PROFILES = {
    "Conservative": {
        "min_edge_mult": 1.5,
        "odds_min_add": 0.20,
        "odds_max_add": -0.50,
    },
    "Balanced": {
        "min_edge_mult": 1.0,
        "odds_min_add": 0.0,
        "odds_max_add": 0.0,
    },
    "Aggressive": {
        "min_edge_mult": 0.6,
        "odds_min_add": -0.10,
        "odds_max_add": 1.00,
    },
}


def apply_profile(defaults: dict, profile: str) -> dict:
    p = RUN_PROFILES.get(profile)
    if not p or not defaults:
        return defaults
    d = dict(defaults)
    d["min_edge"] = max(1, round(d["min_edge"] * p["min_edge_mult"]))
    d["min_odds"] = max(1.10, round(d["min_odds"] + p["odds_min_add"], 2))
    d["max_odds"] = max(d["min_odds"] + 0.50, round(d["max_odds"] + p["odds_max_add"], 2))
    return d


DEFAULTS = {
    "NBA": {
        "min_edge": 3,
        "min_odds": 1.60,
        "max_odds": 3.50,
        "history_days": 60,
        "lookahead_days": 3,
        "top_n": 10,
    },
    "NHL": {
        "min_edge": 3,
        "min_odds": 1.60,
        "max_odds": 3.75,
        "history_days": 60,
        "lookahead_days": 3,
        "top_n": 10,
    },
    "NFL": {
        "min_edge": 4,
        "min_odds": 1.60,
        "max_odds": 4.00,
        "history_days": 365,
        "lookahead_days": 7,
        "top_n": 10,
    },
    "College Basketball": {
        "min_edge": 4,
        "min_odds": 1.70,
        "max_odds": 4.50,
        "history_days": 60,
        "lookahead_days": 2,
        "top_n": 15,
    },
    "College Football": {
        "min_edge": 5,
        "min_odds": 1.70,
        "max_odds": 5.00,
        "history_days": 365,
        "lookahead_days": 7,
        "top_n": 15,
    },
    "Champions League": {
        "min_edge": 5,
        "min_odds": 1.80,
        "max_odds": 4.00,
        "history_days": 90,
        "lookahead_days": 7,
        "top_n": 10,
    },
    "UFC": {
        "min_edge": 4,
        "min_odds": 1.50,
        "max_odds": 4.00,
        "history_days": 180,
        "lookahead_days": 7,
        "top_n": 10,
    },
}
