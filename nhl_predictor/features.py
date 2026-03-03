"""
nhl_predictor/features.py
==========================
Feature engineering for the prediction model.
Handles goalie SV%, back-to-back fatigue, and injury filtering.
"""

from datetime import datetime, timedelta
from typing import Optional
from . import api


# ─── League average constants ─────────────────────────────────────────────────

LEAGUE_AVG_SV_PCT = 0.906   # 2024-25 NHL average
LEAGUE_AVG_GAA = 2.98
ELITE_SV_PCT = 0.925   # top-10 goalie tier
POOR_SV_PCT = 0.890   # struggling goalie tier


# ─── Goalie adjustment ────────────────────────────────────────────────────────


def get_goalie_factor(opposing_team_abbr: str, date: str) -> float:
    """
    Returns a multiplier (0.75–1.25) representing how much a player"s scoring
    probability is adjusted based on the opposing goalie"s quality.

    Logic:
        - Elite goalie (SV% >= ELITE_SV_PCT)  → factor < 1.0  (harder to score)
        - Average goalie                        → factor ≈ 1.0
        - Poor goalie (SV% <= POOR_SV_PCT)     → factor > 1.0  (easier to score)

    The adjustment is linear between [0.75, 1.25] mapped across
    [POOR_SV_PCT, ELITE_SV_PCT].
    """
    goalie = api.get_expected_starting_goalie(opposing_team_abbr, date)
    if not goalie:
        return 1.0  # no data → neutral

    sv_pct = goalie.get("save_pct", LEAGUE_AVG_SV_PCT)
    gp = goalie.get("games_played", 0)

    # Require at least 5 games for meaningful SV% (avoid small-sample noise)
    if gp < 5:
        return 1.0

    # Linear mapping: ELITE_SV_PCT → 0.75, POOR_SV_PCT → 1.25
    # At league average → 1.0
    spread = ELITE_SV_PCT - POOR_SV_PCT  # 0.035
    deviation = sv_pct - LEAGUE_AVG_SV_PCT
    factor = 1.0 - (deviation / spread) * 0.25

    return round(min(max(factor, 0.75), 1.25), 4)


def get_goalie_info(opposing_team_abbr: str, date: str) -> dict:
    """
    Returns goalie metadata for display in output.
    """
    goalie = api.get_expected_starting_goalie(opposing_team_abbr, date)
    if not goalie:
        return {"name": "Unknown", "save_pct": LEAGUE_AVG_SV_PCT, "gaa": LEAGUE_AVG_GAA, "factor": 1.0}

    factor = get_goalie_factor(opposing_team_abbr, date)
    return {
        "name":     goalie.get("name", "Unknown"),
        "save_pct": goalie.get("save_pct", LEAGUE_AVG_SV_PCT),
        "gaa":      goalie.get("gaa", LEAGUE_AVG_GAA),
        "factor":   factor,
    }


# ─── Back-to-back detection ──────────────────────────────────────────────────


def is_back_to_back(team_abbr: str, date: str) -> bool:
    """
    Returns True if the team played a game the day before `date`.
    """
    yesterday = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        schedule = api.get_team_schedule(team_abbr)
    except Exception:
        return False

    for game in schedule:
        if game["date"] == yesterday and game["game_state"] in ("OFF", "FINAL", "7"):
            return True

    return False


def get_b2b_factor(team_abbr: str, date: str) -> float:
    """
    Returns a penalty multiplier for back-to-back games.

    Research shows teams on the second night of a B2B score ~8-12% fewer goals,
    and their individual players face more fatigue — especially forwards.
    We apply a 0.88 multiplier (12% penalty) for B2B situations.
    """
    return 0.88 if is_back_to_back(team_abbr, date) else 1.0


# ─── Injury filtering ────────────────────────────────────────────────────────


def filter_healthy_forwards(team_abbr: str) -> list[dict]:
    """
    Returns only forwards who are not on the injury report.
    Falls back to full roster if the injury endpoint is unavailable.
    """
    try:
        forwards = api.get_roster_with_status(team_abbr)
        healthy = [f for f in forwards if f.get("injury_status", "active") == "active"]
        # If we somehow filtered everyone, fall back
        return healthy if healthy else api.get_roster(team_abbr)
    except Exception:
        return api.get_roster(team_abbr)


# ─── Combined game-level context ─────────────────────────────────────────────


def get_game_context(game: dict, date: str) -> dict:
    """
    Computes all game-level contextual factors for both teams.
    Returns a dict with factors for home and away sides.
    """
    home = game["home_team_abbr"]
    away = game["away_team_abbr"]

    return {
        "home": {
            "b2b_factor":     get_b2b_factor(home, date),
            "is_b2b":         is_back_to_back(home, date),
            "goalie_factor":  get_goalie_factor(away, date),   # home player faces away goalie
            "goalie_info":    get_goalie_info(away, date),
        },
        "away": {
            "b2b_factor":     get_b2b_factor(away, date),
            "is_b2b":         is_back_to_back(away, date),
            "goalie_factor":  get_goalie_factor(home, date),   # away player faces home goalie
            "goalie_info":    get_goalie_info(home, date),
        },
    }
