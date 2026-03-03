"""
nhl_predictor/tracker.py
=========================
Tracks historical prediction accuracy:
  - Did the #1 pick score first?
  - Was the first goalscorer in the top 5?
  - Logs per-date results to data/accuracy_log.jsonl

Usage:
    python -m nhl_predictor.tracker --date 2025-03-01
    python -m nhl_predictor.tracker --report
    python -m nhl_predictor.tracker --report --last 30
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from . import api

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
PICKS_DIR = Path(__file__).parent.parent / "picks"
ACCURACY_LOG = DATA_DIR / "accuracy_log.jsonl"


# --- Log helpers --------------------------------------------------------------


def load_accuracy_log() -> list[dict]:
    if not ACCURACY_LOG.exists():
        return []
    records = []
    with open(ACCURACY_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def append_accuracy_record(record: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(ACCURACY_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def already_tracked(date: str) -> bool:
    for record in load_accuracy_log():
        if record.get("date") == date:
            return True
    return False


# --- Core evaluation ----------------------------------------------------------


def load_picks_for_date(date: str) -> Optional[dict]:
    """Loads the picks JSON file for a given date."""
    picks_file = PICKS_DIR / f"{date}.json"
    if not picks_file.exists():
        logger.warning("No picks file found for %s at %s", date, picks_file)
        return None
    with open(picks_file) as f:
        return json.load(f)


def evaluate_date(date: str) -> Optional[dict]:
    """
    Evaluates prediction accuracy for a given date.
    Compares our picks against actual first goalscorers from completed games.
    Returns an accuracy record dict, or None if games not yet complete.
    """
    if already_tracked(date):
        logger.info("Date %s already tracked -- skipping", date)
        return None

    picks_data = load_picks_for_date(date)
    if not picks_data:
        return None

    picks = picks_data.get("picks", [])
    if not picks:
        logger.info("No picks for %s", date)
        return None

    try:
        games = api.get_games_for_date(date)
    except Exception as e:
        logger.error("Failed to fetch games for %s: %s", date, e)
        return None

    if not games:
        return None

    actual_first_scorers = {}
    games_complete = 0

    for game in games:
        game_id = game["game_id"]
        try:
            result = api.get_first_goalscorer(game_id)
            if result:
                actual_first_scorers[game_id] = result
                games_complete += 1
        except Exception as e:
            logger.warning("Could not get first goalscorer for game %d: %s", game_id, e)

    if games_complete == 0:
        logger.info("No completed games found for %s -- skipping", date)
        return None

    pick_ids = [p["player_id"] for p in picks]

    first_scorer_ids = {v["player_id"] for v in actual_first_scorers.values()}

    top1_correct = (pick_ids[0] in first_scorer_ids) if pick_ids else False
    top3_correct = any(pid in first_scorer_ids for pid in pick_ids[:3]) if len(pick_ids) >= 3 else False
    top5_correct = any(pid in first_scorer_ids for pid in pick_ids[:5]) if pick_ids else False

    actual_ranks = []
    for fid in first_scorer_ids:
        if fid in pick_ids:
            actual_ranks.append(pick_ids.index(fid) + 1)

    best_rank = min(actual_ranks) if actual_ranks else None

    game_results = []
    for game in games:
        gid = game["game_id"]
        actual = actual_first_scorers.get(gid)
        if not actual:
            continue
        aid = actual["player_id"]
        in_picks = aid in pick_ids
        rank_in_picks = (pick_ids.index(aid) + 1) if in_picks else None
        matchup = game["away_team_abbr"] + " @ " + game["home_team_abbr"]
        game_results.append(
            {
                "game": matchup,
                "actual_scorer": actual["name"],
                "actual_team": actual["team_abbr"],
                "in_our_picks": in_picks,
                "rank_in_picks": rank_in_picks,
            }
        )

    record = {
        "date": date,
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "n_games": games_complete,
        "n_picks": len(picks),
        "top1_correct": top1_correct,
        "top3_correct": top3_correct,
        "top5_correct": top5_correct,
        "best_rank": best_rank,
        "our_top1": picks[0]["name"] if picks else None,
        "actual_first_scorers": [v["name"] for v in actual_first_scorers.values()],
        "game_results": game_results,
    }

    append_accuracy_record(record)
    logger.info(
        "Tracked accuracy for %s: top1=%s, top5=%s, best_rank=%s",
        date,
        top1_correct,
        top5_correct,
        best_rank,
    )
    return record


# --- Reporting ----------------------------------------------------------------


def compute_report(records: list[dict]) -> dict:
    """Computes summary statistics from a list of accuracy records."""
    if not records:
        return {}

    n = len(records)
    top1_hits = sum(1 for r in records if r.get("top1_correct"))
    top3_hits = sum(1 for r in records if r.get("top3_correct"))
    top5_hits = sum(1 for r in records if r.get("top5_correct"))

    ranks = [r["best_rank"] for r in records if r.get("best_rank") is not None]
    avg_rank = round(sum(ranks) / len(ranks), 2) if ranks else None
    found_rate = round(len(ranks) / n, 3)

    return {
        "total_days": n,
        "top1_accuracy": round(top1_hits / n, 3),
        "top3_accuracy": round(top3_hits / n, 3),
        "top5_accuracy": round(top5_hits / n, 3),
        "avg_rank_when_found": avg_rank,
        "found_in_list_rate": found_rate,
        "top1_hits": top1_hits,
        "top3_hits": top3_hits,
        "top5_hits": top5_hits,
    }


def print_report(last_n: Optional[int] = None):
    """Prints a formatted accuracy report to stdout."""
    records = load_accuracy_log()
    if not records:
        print("No accuracy data yet. Run: python -m nhl_predictor.tracker --date YYYY-MM-DD")
        return

    records.sort(key=lambda r: r.get("date", ""))

    if last_n:
        records = records[-last_n:]

    report = compute_report(records)
    if not report:
        return

    first_date = records[0]["date"]
    last_date = records[-1]["date"]
    total = report["total_days"]
    top1_acc = report["top1_accuracy"]
    top1_hits = report["top1_hits"]
    top3_acc = report["top3_accuracy"]
    top3_hits = report["top3_hits"]
    top5_acc = report["top5_accuracy"]
    top5_hits = report["top5_hits"]
    found_rate = report["found_in_list_rate"]

    print("\nPREDICTION ACCURACY REPORT")
    if last_n:
        print(f"  Last {last_n} days  ({first_date} to {last_date})")
    else:
        print(f"  All time  ({first_date} to {last_date})")
    print("-" * 50)
    print(f"  Days tracked:         {total}")
    print(f"  Top-1 accuracy:       {top1_acc:.1%}  ({top1_hits}/{total} days)")
    print(f"  Top-3 accuracy:       {top3_acc:.1%}  ({top3_hits}/{total} days)")
    print(f"  Top-5 accuracy:       {top5_acc:.1%}  ({top5_hits}/{total} days)")
    print(f"  Scorer found in list: {found_rate:.1%}")
    if report.get("avg_rank_when_found"):
        avg = report["avg_rank_when_found"]
        print(f"  Avg rank (when found): {avg}")
    print()

    print("  Recent results:")
    print(f"  {'Date':<12} {'#1 Pick':<22} {'Actual First Scorer':<22} {'Hit'}")
    print("  " + "-" * 12 + " " + "-" * 22 + " " + "-" * 22 + " " + "-" * 4)
    for r in records[-15:]:
        if r.get("top1_correct"):
            tick = "[1]"
        elif r.get("top5_correct"):
            tick = "[5]"
        else:
            tick = "[ ]"
        actual = ", ".join(r.get("actual_first_scorers", [])[:2])
        our_top1 = str(r.get("our_top1", ""))
        row_date = r["date"]
        print(f"  {row_date:<12} {our_top1:<22} {actual:<22} {tick}")
    print()


# --- CLI ----------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Track NHL first-goalscorer prediction accuracy")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Evaluate accuracy for this date (YYYY-MM-DD). Defaults to yesterday.",
    )
    parser.add_argument("--report", action="store_true", help="Print accuracy summary report")
    parser.add_argument("--last", type=int, default=None, help="With --report: limit to last N days")
    args = parser.parse_args()

    if args.report:
        print_report(last_n=args.last)
        return

    target = args.date or (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    result = evaluate_date(target)

    if result:
        our_top1 = result["our_top1"]
        top1_icon = "YES" if result["top1_correct"] else "NO"
        top5_icon = "YES" if result["top5_correct"] else "NO"
        scorers = ", ".join(result["actual_first_scorers"])
        print(f"\nAccuracy for {target}:")
        print(f"  #1 pick ({our_top1}) scored first: {top1_icon}")
        print(f"  First scorer in top 5:  {top5_icon}")
        print(f"  Actual first scorers:   {scorers}")
        if result["best_rank"]:
            best = result["best_rank"]
            print(f"  Best rank in our list:  #{best}")
        print()
        for g in result["game_results"]:
            matchup = g["game"]
            scorer = g["actual_scorer"]
            if g["in_our_picks"]:
                rank = g["rank_in_picks"]
                status = f"Rank #{rank}"
            else:
                status = "Not in picks"
            print(f"  {matchup}: {scorer} scored first -- {status}")
    else:
        print(f"Could not evaluate {target} -- games may not be complete yet.")


if __name__ == "__main__":
    main()
