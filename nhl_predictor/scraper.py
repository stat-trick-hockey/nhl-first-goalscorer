"""
nhl_predictor/scraper.py
=========================
Scrapes historical NHL games to build the ML training dataset.

For each completed game, it:
  1. Identifies the first goalscorer from play-by-play
  2. Pulls pre-game stats for every forward in both rosters
  3. Labels each player as first_goal=1 (scorer) or 0 (everyone else)
  4. Appends the labeled rows to data/training_data.csv

Run once to backfill a full season, then incrementally daily.

Usage:
    python -m nhl_predictor.scraper --season 20242025
    python -m nhl_predictor.scraper --season 20232024 --season 20242025
    python -m nhl_predictor.scraper --days-back 30      # recent games only
"""

import argparse
import csv
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from . import api
from .model import FEATURE_NAMES, estimate_line_ranks, build_feature_vector

logger = logging.getLogger(__name__)

# Ensure scraper logs are visible in CI even without explicit config
if not logger.handlers and not logging.getLogger().handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

DATA_DIR = Path(__file__).parent.parent / "data"
TRAINING_FILE = DATA_DIR / "training_data.csv"

# All columns in the CSV
CSV_COLUMNS = FEATURE_NAMES + [
    "player_id",
    "player_name",
    "team",
    "opponent",
    "game_id",
    "date",
    "first_goal",   # target label: 1 if this player scored the first goal
]

# NHL regular seasons to scrape (start/end dates)
SEASONS = {
    "20222023": ("2022-10-07", "2023-04-13"),
    "20232024": ("2023-10-10", "2024-04-18"),
    "20242025": ("2024-10-08", "2025-04-17"),
    "20252026": ("2025-10-07", "2026-04-16"),
}


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_scraped_game_ids() -> set[int]:
    """Returns set of game IDs already in the training CSV (for incremental updates)."""
    if not TRAINING_FILE.exists():
        return set()
    ids = set()
    with open(TRAINING_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ids.add(int(row["game_id"]))
            except (KeyError, ValueError):
                pass
    return ids


def get_dates_in_range(start: str, end: str) -> list[str]:
    """Returns all dates YYYY-MM-DD between start and end inclusive."""
    d = datetime.strptime(start, "%Y-%m-%d")
    end_d = datetime.strptime(end, "%Y-%m-%d")
    dates = []
    while d <= end_d:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return dates


def scrape_game(game: dict, date: str, scraped_ids: set[int]) -> list[dict]:
    """
    Scrapes one game and returns a list of labeled player rows.
    Returns [] if game already scraped or no first goal found.
    """
    game_id = game["game_id"]
    if game_id in scraped_ids:
        return []

    # Find first goalscorer
    first_goal = api.get_first_goalscorer(game_id)
    if first_goal is None:
        logger.debug("No first goal found for game %d (may be incomplete)", game_id)
        return []

    first_scorer_id = first_goal["player_id"]
    rows = []

    for side in ("home", "away"):
        team_abbr = game[f"{side}_team_abbr"]
        is_home = (side == "home")
        opp_abbr = game["away_team_abbr"] if is_home else game["home_team_abbr"]

        forwards = api.get_roster(team_abbr)
        if not forwards:
            continue

        # Season stats
        season_stats = {}
        for p in forwards:
            season_stats[p["player_id"]] = api.get_player_season_stats(p["player_id"])
            time.sleep(0.1)

        # Recent form
        recent_stats = {}
        line_ranks = estimate_line_ranks(forwards, season_stats)
        top_ids = {pid for pid, rank in line_ranks.items() if rank <= 2}
        for pid in top_ids:
            recent_stats[pid] = api.get_recent_form(pid)
            time.sleep(0.1)

        # Opposing goalie
        opp_goalie = api.get_expected_starting_goalie(opp_abbr, date)
        goalie_sv = opp_goalie.get("save_pct", 0.906) if opp_goalie else 0.906

        # B2B detection
        from .features import is_back_to_back
        b2b = is_back_to_back(team_abbr, date)

        for player in forwards:
            pid = player["player_id"]
            season = season_stats.get(pid, {})
            recent = recent_stats.get(pid, {})
            line = line_ranks.get(pid, 4)

            if season.get("games_played", 0) == 0:
                continue

            fv = build_feature_vector(season, recent, line, is_home, goalie_sv, b2b)

            row = dict(zip(FEATURE_NAMES, fv.tolist()))
            row.update({
                "player_id":   pid,
                "player_name": player["name"],
                "team":        team_abbr,
                "opponent":    opp_abbr,
                "game_id":     game_id,
                "date":        date,
                "first_goal":  1 if pid == first_scorer_id else 0,
            })
            rows.append(row)

    return rows


def append_rows(rows: list[dict]):
    """Appends rows to the training CSV, creating it with headers if needed."""
    if not rows:
        return

    file_exists = TRAINING_FILE.exists()
    with open(TRAINING_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    logger.info("Appended %d rows to %s", len(rows), TRAINING_FILE)


def scrape_season(season_key: str, delay: float = 0.5):
    """Scrapes all completed games from a season and appends to training CSV."""
    if season_key not in SEASONS:
        raise ValueError(f"Unknown season: {season_key}. Options: {list(SEASONS.keys())}")

    start, end = SEASONS[season_key]
    end_date = min(end, datetime.today().strftime("%Y-%m-%d"))
    dates = get_dates_in_range(start, end_date)
    scraped_ids = load_scraped_game_ids()

    total_games = 0
    total_rows = 0
    skipped_games = 0

    logger.info("Scraping season %s (%s → %s) — %d dates", season_key, start, end_date, len(dates))

    for date in dates:
        try:
            games = api.get_games_for_date(date)
        except Exception as e:
            logger.warning("Failed to get schedule for %s: %s", date, e)
            time.sleep(delay)
            continue

        for game in games:
            game_id = game["game_id"]
            if game_id in scraped_ids:
                skipped_games += 1
                continue

            logger.info("  Scraping game %d  (%s @ %s  —  %s)",
                        game_id, game["away_team_abbr"], game["home_team_abbr"], date)
            try:
                rows = scrape_game(game, date, scraped_ids)
                if rows:
                    append_rows(rows)
                    scraped_ids.add(game_id)
                    total_games += 1
                    total_rows  += len(rows)
                    logger.info("  → %d rows written (total: %d games, %d rows)", len(rows), total_games, total_rows)
                else:
                    logger.info("  → skipped (no first goal or already scraped)")
            except Exception as e:
                logger.error("  Failed game %d: %s", game_id, e, exc_info=True)

            time.sleep(delay)

    logger.info("Done. %d games scraped, %d rows written, %d skipped.",
                total_games, total_rows, skipped_games)
    return total_rows


def scrape_recent(days_back: int = 30, delay: float = 0.5):
    """Scrapes the last N days — good for daily incremental updates."""
    today = datetime.today()
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days_back, 0, -1)]
    scraped_ids = load_scraped_game_ids()

    total_rows = 0
    for date in dates:
        try:
            games = api.get_games_for_date(date)
        except Exception:
            continue
        for game in games:
            if game["game_id"] in scraped_ids:
                continue
            try:
                rows = scrape_game(game, date, scraped_ids)
                if rows:
                    append_rows(rows)
                    scraped_ids.add(game["game_id"])
                    total_rows += len(rows)
            except Exception as e:
                logger.error("Failed game %d: %s", game["game_id"], e)
            time.sleep(delay)

    return total_rows


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Scrape historical NHL first-goalscorer data")
    parser.add_argument("--season",    action="append", default=[],
                        help="Season key (e.g. 20242025). Repeat for multiple.")
    parser.add_argument("--days-back", type=int, default=None,
                        help="Scrape last N days instead of full season(s).")
    parser.add_argument("--delay",     type=float, default=0.5,
                        help="Seconds between API calls (default: 0.5)")
    args = parser.parse_args()

    ensure_data_dir()

    if args.days_back:
        n = scrape_recent(days_back=args.days_back, delay=args.delay)
        print(f"✅  Scraped {n} rows for the last {args.days_back} days.")
    else:
        seasons = args.season or list(SEASONS.keys())
        total = 0
        for s in seasons:
            total += scrape_season(s, delay=args.delay)
        print(f"✅  Total rows scraped: {total}")


if __name__ == "__main__":
    main()
