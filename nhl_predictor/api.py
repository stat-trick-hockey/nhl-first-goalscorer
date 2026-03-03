"""
nhl_predictor/api.py
=====================
All NHL API calls in one place. Uses api-web.nhle.com -- no key required.
"""

import time
from datetime import datetime, timedelta
from typing import Optional

import requests

NHL_BASE = "https://api-web.nhle.com/v1"
SEASON = "20252026"
PREV_SEASON = "20242025"

_session = requests.Session()
_session.headers.update({"User-Agent": "NHL-FirstGoal-Predictor/2.0"})


def _get(path: str, retries: int = 3, base: str = NHL_BASE) -> dict:
    """GET with exponential-backoff retry. Returns {} on 404."""
    url = f"{base}{path}"
    for attempt in range(retries):
        try:
            r = _session.get(url, timeout=12)
            if r.status_code == 404:
                return {}
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(1.5**attempt)
    return {}


def _player_name(p: dict) -> str:
    """Extract 'First Last' from an NHL API player dict."""
    first = p.get("firstName", {}).get("default", "")
    last = p.get("lastName", {}).get("default", "")
    return f"{first} {last}".strip()


# --- Schedule -----------------------------------------------------------------


def get_games_for_date(date: str) -> list[dict]:
    """
    Returns games scheduled for a specific date (YYYY-MM-DD).
    Each dict: game_id, home/away team abbr + id, venue, start_time_utc.
    """
    data = _get(f"/schedule/{date}")
    games = []
    for day in data.get("gameWeek", []):
        if day.get("date") != date:
            continue
        for g in day.get("games", []):
            if g.get("gameType", 2) not in (2, 3):
                continue
            games.append(
                {
                    "game_id": g["id"],
                    "home_team_abbr": g["homeTeam"]["abbrev"],
                    "away_team_abbr": g["awayTeam"]["abbrev"],
                    "home_team_id": g["homeTeam"]["id"],
                    "away_team_id": g["awayTeam"]["id"],
                    "venue": g.get("venue", {}).get("default", ""),
                    "start_time_utc": g.get("startTimeUTC", ""),
                }
            )
    return games


def get_team_schedule(team_abbr: str, season: str = SEASON) -> list[dict]:
    """
    Returns completed + upcoming games for a team this season.
    Used to detect back-to-back situations.
    Each dict: game_id, date, home_or_away, opponent_abbr, game_state.
    """
    data = _get(f"/club-schedule-season/{team_abbr}/{season}")
    games = []
    for g in data.get("games", []):
        is_home = g.get("homeTeam", {}).get("abbrev") == team_abbr
        opp = g.get("awayTeam", {}) if is_home else g.get("homeTeam", {})
        games.append(
            {
                "game_id": g["id"],
                "date": g.get("gameDate", ""),
                "home_or_away": "home" if is_home else "away",
                "opponent": opp.get("abbrev", ""),
                "game_state": g.get("gameState", ""),
            }
        )
    return sorted(games, key=lambda x: x["date"])


# --- Roster -------------------------------------------------------------------


def get_roster(team_abbr: str, season: str = SEASON) -> list[dict]:
    """
    Returns forwards from a team's active roster.
    Each dict: player_id, name, position, sweater_number.
    """
    data = _get(f"/club-roster/{team_abbr}/{season}")
    forwards = []
    for p in data.get("forwards", []):
        forwards.append(
            {
                "player_id": p["id"],
                "name": _player_name(p),
                "position": p.get("positionCode", "F"),
                "sweater_number": p.get("sweaterNumber", 0),
            }
        )
    return forwards


def get_roster_with_status(team_abbr: str) -> list[dict]:
    """
    Returns roster including injury status from the injuries endpoint.
    Status values: 'active', 'injured', 'ir', 'day-to-day', 'ltir'
    """
    injury_data = _get(f"/roster/{team_abbr}/current")
    injured_ids = set()
    for p in injury_data.get("injured", []):
        injured_ids.add(p.get("playerId") or p.get("id"))

    forwards = get_roster(team_abbr)
    for f in forwards:
        f["injury_status"] = "injured" if f["player_id"] in injured_ids else "active"
    return forwards


# --- Player stats -------------------------------------------------------------


def get_player_season_stats(player_id: int, season: str = SEASON) -> dict:
    """
    Season-level stats for a skater.
    Returns goals, shots, pp_goals, games_played and derived per-game rates.
    """
    data = _get(f"/player/{player_id}/landing")

    stats = {}
    featured = data.get("featuredStats", {})
    reg = featured.get("regularSeason", {}).get("subSeason", {})
    if reg.get("gamesPlayed", 0) > 0:
        stats = reg

    if not stats:
        for s in data.get("seasonTotals", []):
            if s.get("season") == int(season) and s.get("gameTypeId") == 2:
                stats = s
                break

    if not stats:
        return {}

    gp = max(stats.get("gamesPlayed", 1), 1)
    goals = stats.get("goals", 0)
    shots = stats.get("shots", 0)
    pp_goals = stats.get("powerPlayGoals", 0)
    toi = stats.get("avgToi", "0:00")

    try:
        m, s2 = toi.split(":")
        avg_toi_min = int(m) + int(s2) / 60
    except Exception:
        avg_toi_min = 0.0

    return {
        "goals": goals,
        "games_played": gp,
        "shots": shots,
        "pp_goals": pp_goals,
        "goals_per_game": goals / gp,
        "shots_per_game": shots / gp,
        "pp_goals_per_game": pp_goals / gp,
        "avg_toi_min": avg_toi_min,
    }


def get_recent_form(player_id: int, n: int = 5, season: str = SEASON) -> dict:
    """
    Goal-scoring form over the last N completed games.
    """
    data = _get(f"/player/{player_id}/game-log/{season}/2")
    logs = data.get("gameLog", [])[:n]

    if not logs:
        return {
            "recent_goals": 0,
            "recent_games": 0,
            "recent_goals_per_game": 0.0,
            "recent_shots": 0,
            "recent_shots_per_game": 0.0,
        }

    goals = sum(g.get("goals", 0) for g in logs)
    shots = sum(g.get("shots", 0) for g in logs)
    n_actual = len(logs)
    return {
        "recent_goals": goals,
        "recent_games": n_actual,
        "recent_goals_per_game": goals / n_actual,
        "recent_shots": shots,
        "recent_shots_per_game": shots / n_actual,
    }


def get_full_game_log(player_id: int, season: str = SEASON) -> list[dict]:
    """
    Full season game-by-game log for a player.
    Used by the historical scraper and ML training pipeline.
    """
    data = _get(f"/player/{player_id}/game-log/{season}/2")
    return data.get("gameLog", [])


# --- Goalie stats -------------------------------------------------------------


def get_team_goalies(team_abbr: str, season: str = SEASON) -> list[dict]:
    """
    Returns goalies on the team roster.
    """
    data = _get(f"/club-roster/{team_abbr}/{season}")
    goalies = []
    for g in data.get("goalies", []):
        goalies.append(
            {
                "player_id": g["id"],
                "name": _player_name(g),
            }
        )
    return goalies


def get_goalie_stats(player_id: int, season: str = SEASON) -> dict:
    """
    Season SV% and GAA for a goalie.
    """
    data = _get(f"/player/{player_id}/landing")

    stats = {}
    featured = data.get("featuredStats", {})
    reg = featured.get("regularSeason", {}).get("subSeason", {})
    if reg.get("gamesPlayed", 0) > 0:
        stats = reg

    if not stats:
        for s in data.get("seasonTotals", []):
            if s.get("season") == int(season) and s.get("gameTypeId") == 2:
                stats = s
                break

    if not stats:
        return {"save_pct": 0.910, "gaa": 3.00, "games_played": 0}

    return {
        "save_pct": stats.get("savePctg", 0.910),
        "gaa": stats.get("goalsAgainstAvg", 3.00),
        "games_played": stats.get("gamesPlayed", 0),
    }


def get_expected_starting_goalie(team_abbr: str, date: str) -> Optional[dict]:
    """
    Returns the goalie with the most games played this season (likely starter).
    Returns: { player_id, name, save_pct, gaa } or None.
    """
    goalies = get_team_goalies(team_abbr)
    if not goalies:
        return None

    best = None
    best_gp = -1
    for g in goalies:
        stats = get_goalie_stats(g["player_id"])
        gp = stats.get("games_played", 0)
        if gp > best_gp:
            best_gp = gp
            best = {**g, **stats}

    return best


# --- Game result scraping -----------------------------------------------------


def get_game_boxscore(game_id: int) -> dict:
    """Full boxscore for a completed game."""
    return _get(f"/gamecenter/{game_id}/boxscore")


def get_game_play_by_play(game_id: int) -> dict:
    """Play-by-play for a completed game."""
    return _get(f"/gamecenter/{game_id}/play-by-play")


def get_first_goalscorer(game_id: int) -> Optional[dict]:
    """
    Parses play-by-play to find the first goal scorer of the game.
    Returns: { player_id, name, team_abbr, period, time, goal_type } or None.
    """
    data = get_game_play_by_play(game_id)
    plays = data.get("plays", [])

    for play in plays:
        if play.get("typeDescKey") != "goal":
            continue
        details = play.get("details", {})
        scorer_id = details.get("scoringPlayerId")
        if not scorer_id:
            continue

        roster = {p["playerId"]: p for p in data.get("rosterSpots", [])}
        player = roster.get(scorer_id, {})
        name = _player_name(player)

        home_team = data.get("homeTeam", {}).get("abbrev", "")
        away_team = data.get("awayTeam", {}).get("abbrev", "")
        team_id = details.get("eventOwnerTeamId")
        team_abbr = home_team if team_id == data.get("homeTeam", {}).get("id") else away_team

        return {
            "player_id": scorer_id,
            "name": name,
            "team_abbr": team_abbr,
            "period": play.get("periodDescriptor", {}).get("number", 1),
            "time": play.get("timeInPeriod", ""),
            "goal_type": details.get("shotType", ""),
            "game_id": game_id,
        }

    return None
