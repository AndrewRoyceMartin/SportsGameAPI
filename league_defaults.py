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


ELO_PARAMS = {
    "NBA": {"k": 25, "home_adv": 70, "scale": 400, "recency_half_life": 45},
    "NHL": {"k": 20, "home_adv": 35, "scale": 400, "recency_half_life": 45},
    "NFL": {"k": 30, "home_adv": 50, "scale": 350, "recency_half_life": 0},
    "College Basketball": {"k": 25, "home_adv": 80, "scale": 400, "recency_half_life": 45},
    "College Football": {"k": 30, "home_adv": 60, "scale": 350, "recency_half_life": 0},
    "AFL": {"k": 30, "home_adv": 50, "scale": 400, "recency_half_life": 60},
    "NRL": {"k": 25, "home_adv": 40, "scale": 400, "recency_half_life": 45},
    "NBL": {"k": 25, "home_adv": 60, "scale": 400, "recency_half_life": 45},
    "A-League Men": {"k": 25, "home_adv": 55, "scale": 400, "recency_half_life": 45},
    "Super Rugby Pacific": {"k": 25, "home_adv": 45, "scale": 400, "recency_half_life": 45},
    "Champions League": {"k": 20, "home_adv": 45, "scale": 450, "recency_half_life": 60},
    "UFC": {"k": 30, "home_adv": 0, "scale": 400, "recency_half_life": 90},
}

DEFAULT_ELO = {"k": 20, "home_adv": 65, "scale": 400, "recency_half_life": 45}


def get_elo_params(league: str) -> dict:
    return ELO_PARAMS.get(league, DEFAULT_ELO)


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
    "AFL": {
        "min_edge": 3,
        "min_odds": 1.30,
        "max_odds": 4.50,
        "history_days": 120,
        "lookahead_days": 21,
        "top_n": 15,
    },
    "NRL": {
        "min_edge": 3,
        "min_odds": 1.30,
        "max_odds": 4.50,
        "history_days": 120,
        "lookahead_days": 21,
        "top_n": 15,
    },
    "NBL": {
        "min_edge": 3,
        "min_odds": 1.30,
        "max_odds": 4.00,
        "history_days": 120,
        "lookahead_days": 14,
        "top_n": 15,
    },
    "A-League Men": {
        "min_edge": 3,
        "min_odds": 1.40,
        "max_odds": 4.50,
        "history_days": 120,
        "lookahead_days": 14,
        "top_n": 15,
    },
    "Super Rugby Pacific": {
        "min_edge": 3,
        "min_odds": 1.30,
        "max_odds": 4.50,
        "history_days": 120,
        "lookahead_days": 14,
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
