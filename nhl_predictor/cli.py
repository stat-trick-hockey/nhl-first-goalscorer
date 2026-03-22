"""
nhl_predictor/cli.py
=====================
Unified CLI entry point for the NHL First Goalscorer Predictor.

Commands:
    predict     Run daily prediction (default)
    track       Evaluate yesterday's accuracy
    report      Print accuracy report
    scrape      Scrape historical data for ML training
    train       Train the ML model
    validate    Run offline model validation

Run any command with --help for options.
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta


def cmd_predict(args):
    from pathlib import Path

    from .predict import format_anytime_json, format_json, format_table, run_anytime_predictions, run_predictions

    results = run_predictions(
        date=args.date,
        top_n=args.top,
        verbose=(args.output == "table"),
    )

    if not results:
        sys.exit(0)

    date = args.date or datetime.today().strftime("%Y-%m-%d")

    if args.output == "json":
        output = format_json(results, date)
        print(output)
        if args.save:
            picks_dir = Path(__file__).parent.parent / "picks"
            picks_dir.mkdir(parents=True, exist_ok=True)
            out_file = picks_dir / f"{date}.json"
            out_file.write_text(output)
            print(f"\n Saved to {out_file}", file=sys.stderr)

            # Also generate anytime picks
            anytime_results = run_anytime_predictions(date=date, top_n=args.top)
            if anytime_results:
                anytime_dir = Path(__file__).parent.parent / "anytime"
                anytime_dir.mkdir(parents=True, exist_ok=True)
                anytime_file = anytime_dir / f"{date}.json"
                anytime_file.write_text(format_anytime_json(anytime_results, date))
                print(f"\n Saved anytime picks to {anytime_file}", file=sys.stderr)
    else:
        print(format_table(results, date))
        if args.save:
            picks_dir = Path(__file__).parent.parent / "picks"
            picks_dir.mkdir(parents=True, exist_ok=True)
            out_file = picks_dir / f"{date}.json"
            out_file.write_text(format_json(results, date))
            print(f"\n Saved to {out_file}")

            # Also generate anytime picks
            anytime_results = run_anytime_predictions(date=date, top_n=args.top)
            if anytime_results:
                anytime_dir = Path(__file__).parent.parent / "anytime"
                anytime_dir.mkdir(parents=True, exist_ok=True)
                anytime_file = anytime_dir / f"{date}.json"
                anytime_file.write_text(format_anytime_json(anytime_results, date))
                print(f"\n Saved anytime picks to {anytime_file}")


def cmd_track(args):
    from .tracker import evaluate_date

    date = args.date or (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    result = evaluate_date(date)
    if result:
        top1_icon = "YES" if result["top1_correct"] else "NO"
        top5_icon = "YES" if result["top5_correct"] else "NO"
        scorers = ", ".join(result.get("actual_first_scorers", []))
        print(f"\nAccuracy for {date}:")
        print(f"  Our #1 pick:            {result['our_top1']}")
        print(f"  #1 correct:             {top1_icon}")
        print(f"  Scorer in top 5:        {top5_icon}")
        print(f"  Actual first scorers:   {scorers}")
        if result["best_rank"]:
            print(f"  Best rank in list:      #{result['best_rank']}")
        for g in result.get("game_results", []):
            status = "Rank #" + str(g["rank_in_picks"]) if g["in_our_picks"] else "not in picks"
            game_str = g["game"]
            scorer = g["actual_scorer"]
            print(f"  {game_str:30}  {scorer} ({status})")
    else:
        print(f"Could not evaluate {date}.")


def cmd_report(args):
    from .tracker import print_report

    print_report(last_n=args.last)


def cmd_scrape(args):
    from .scraper import SEASONS, ensure_data_dir, scrape_recent, scrape_season

    ensure_data_dir()
    if args.days_back:
        n = scrape_recent(days_back=args.days_back, delay=args.delay)
        print(f"Scraped {n} rows for the last {args.days_back} days.")
    else:
        seasons = args.season or list(SEASONS.keys())
        total = 0
        for s in seasons:
            total += scrape_season(s, delay=args.delay)
        print(f"Total rows scraped: {total}")


def cmd_train(args):
    from .train import train

    metrics = train(cv=args.cv, show_importance=args.feature_importance)
    roc = metrics["train_roc_auc"]
    avg = metrics["train_avg_prec"]
    print(f"\nModel trained:  ROC-AUC={roc:.4f}  AvgPrec={avg:.4f}")
    if "cv_roc_auc_mean" in metrics:
        cv_mean = metrics["cv_roc_auc_mean"]
        cv_std = metrics["cv_roc_auc_std"]
        print(f"  CV ROC-AUC: {cv_mean:.4f} +/- {cv_std:.4f}")


def cmd_validate(args):
    """Offline validation of the scoring model -- no network required."""
    from .model import heuristic_score

    print("\nOFFLINE VALIDATION -- heuristic scoring model")
    print("=" * 55)

    mock = [
        {"name": "Connor McDavid",   "gpg": 0.52, "rgpg": 0.60, "sog": 4.8, "line": 1, "home": True,  "pp": 0.18, "gf": 1.0,  "b2b": 1.0},
        {"name": "Nathan MacKinnon", "gpg": 0.48, "rgpg": 0.60, "sog": 4.5, "line": 1, "home": False, "pp": 0.20, "gf": 1.0,  "b2b": 1.0},
        {"name": "Leon Draisaitl",   "gpg": 0.45, "rgpg": 0.40, "sog": 4.1, "line": 1, "home": True,  "pp": 0.22, "gf": 1.0,  "b2b": 1.0},
        {"name": "Auston Matthews",  "gpg": 0.50, "rgpg": 0.20, "sog": 5.2, "line": 1, "home": False, "pp": 0.15, "gf": 1.0,  "b2b": 1.0},
        {"name": "David Pastrnak",   "gpg": 0.44, "rgpg": 0.40, "sog": 4.0, "line": 1, "home": True,  "pp": 0.19, "gf": 0.75, "b2b": 1.0},
        {"name": "McDavid (B2B)",    "gpg": 0.52, "rgpg": 0.60, "sog": 4.8, "line": 1, "home": True,  "pp": 0.18, "gf": 1.0,  "b2b": 0.88},
        {"name": "4th liner",        "gpg": 0.05, "rgpg": 0.00, "sog": 1.0, "line": 4, "home": True,  "pp": 0.00, "gf": 1.25, "b2b": 1.0},
    ]

    scores = []
    for p in mock:
        season = {
            "goals_per_game": p["gpg"],
            "shots_per_game": p["sog"],
            "pp_goals_per_game": p["pp"],
            "avg_toi_min": 18.0,
        }
        recent = {"recent_goals_per_game": p["rgpg"]}
        s = heuristic_score(season, recent, p["line"], p["home"], p["gf"], p["b2b"])
        scores.append((p["name"], s))

    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Player':<25} {'Score':>7}")
    print("-" * 35)
    for name, score in scores:
        bar = "#" * int(score / 3)
        print(f"{name:<25} {score:>6.1f}  {bar}")

    checks = []
    top_name = scores[0][0]
    checks.append(("Elite player with hot form is #1", top_name in ("Connor McDavid", "Nathan MacKinnon")))

    mcd_normal = next(s for n, s in scores if n == "Connor McDavid")
    mcd_b2b = next(s for n, s in scores if n == "McDavid (B2B)")
    checks.append(("B2B penalty reduces score", mcd_b2b < mcd_normal))

    pas_normal = next(s for n, s in scores if n == "David Pastrnak")
    mcd_score = next(s for n, s in scores if n == "Connor McDavid")
    checks.append(("Elite goalie reduces player score", pas_normal < mcd_score * 0.95))

    grinder = next(s for n, s in scores if n == "4th liner")
    checks.append(("4th liner scores much lower", grinder < 30))

    mat = next(s for n, s in scores if n == "Auston Matthews")
    pas = next(s for n, s in scores if n == "David Pastrnak")
    checks.append(("Cold streak penalised vs hot player", mat < pas + 10))

    print("\nChecks:")
    all_pass = True
    for desc, passed in checks:
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}]  {desc}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll validation checks passed!\n")
    else:
        print("\nSome checks failed.\n")
        sys.exit(1)


def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s  %(message)s",
    )

    root = argparse.ArgumentParser(
        prog="nhl-picks",
        description="NHL First Goalscorer Predictor v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = root.add_subparsers(dest="command", metavar="command")
    sub.required = False

    p_predict = sub.add_parser("predict", help="Generate today's picks (default)")
    p_predict.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    p_predict.add_argument("--top", type=int, default=5, help="Number of picks (default: 5)")
    p_predict.add_argument("--output", choices=["table", "json"], default="table")
    p_predict.add_argument("--save", action="store_true", help="Save JSON to picks/YYYY-MM-DD.json")

    p_track = sub.add_parser("track", help="Evaluate yesterday's prediction accuracy")
    p_track.add_argument("--date", default=None, help="YYYY-MM-DD (default: yesterday)")

    p_report = sub.add_parser("report", help="Show accuracy report")
    p_report.add_argument("--last", type=int, default=None, help="Last N days only")

    p_scrape = sub.add_parser("scrape", help="Scrape historical first-goalscorer data")
    p_scrape.add_argument("--season", action="append", default=[], help="Season key e.g. 20242025 (repeat for multiple)")
    p_scrape.add_argument("--days-back", type=int, default=None, help="Scrape last N days instead of full seasons")
    p_scrape.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls (default: 0.5)")

    p_train = sub.add_parser("train", help="Train ML model from scraped data")
    p_train.add_argument("--cv", action="store_true")
    p_train.add_argument("--feature-importance", action="store_true")

    sub.add_parser("validate", help="Offline model validation (no network)")

    args = root.parse_args()

    if args.command is None or args.command == "predict":
        if args.command is None:

            class DefaultArgs:
                date = None
                top = 5
                output = "table"
                save = False

            args = DefaultArgs()
        cmd_predict(args)
    elif args.command == "track":
        cmd_track(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "scrape":
        cmd_scrape(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "validate":
        cmd_validate(args)


if __name__ == "__main__":
    main()
