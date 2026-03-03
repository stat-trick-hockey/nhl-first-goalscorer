"""
nhl_predictor/predict.py
=========================
Main daily prediction pipeline.
Orchestrates: schedule → roster → stats → features → model → ranked output.
"""

import json
import logging
import time
from datetime import datetime
from typing import Optional

from . import api
from .features import filter_healthy_forwards, get_game_context
from .model import estimate_line_ranks, combined_score, WEIGHTS

logger = logging.getLogger(__name__)

LEAGUE_AVG_SV_PCT = 0.906


def run_predictions(date: Optional[str] = None, top_n: int = 5, verbose: bool = True) -> list[dict]:
    """
    Full pipeline: schedule → rosters → stats → features → score → rank.
    Returns top_n candidates sorted by final score descending.
    """
    target_date = date or datetime.today().strftime("%Y-%m-%d")

    if verbose:
        print(f"\n🏒  NHL First Goalscorer Predictor v2.0 — {target_date}")
        print("=" * 60)

    # ── 1. Schedule ──────────────────────────────────────────────
    if verbose: print("📅  Fetching schedule...", end=" ", flush=True)
    games = api.get_games_for_date(target_date)
    if not games:
        if verbose: print(f"\n❌  No NHL games found for {target_date}.")
        return []
    if verbose:
        print(f"{len(games)} game(s)")
        for g in games:
            print(f"    {g["away_team_abbr"]} @ {g["home_team_abbr"]}")

    all_candidates = []

    for game in games:
        home = game["home_team_abbr"]
        away = game["away_team_abbr"]

        if verbose: print(f"\n🔍  {away} @ {home}")

        # ── 2. Game context (goalie, B2B) ────────────────────────
        if verbose: print(f"    Fetching game context (goalie SV%, B2B)...", end=" ", flush=True)
        try:
            context = get_game_context(game, target_date)
        except Exception as e:
            logger.warning("Context fetch failed for %s @ %s: %s", away, home, e)
            context = {
                "home": {"b2b_factor": 1.0, "is_b2b": False, "goalie_factor": 1.0,
                         "goalie_info": {"name": "?", "save_pct": LEAGUE_AVG_SV_PCT, "gaa": 2.98, "factor": 1.0}},
                "away": {"b2b_factor": 1.0, "is_b2b": False, "goalie_factor": 1.0,
                         "goalie_info": {"name": "?", "save_pct": LEAGUE_AVG_SV_PCT, "gaa": 2.98, "factor": 1.0}},
            }
        if verbose:
            for side in ("home", "away"):
                gi = context[side]["goalie_info"]
                b2b_str = " [B2B⚠️]" if context[side]["is_b2b"] else ""
                team = home if side == "home" else away
                opp = away if side == "home" else home
                print(f"\n    {team}{b2b_str}  |  Opposing goalie: {gi["name"]} "
                      f"SV%={gi["save_pct"]:.3f}  factor={gi["factor"]:.2f}")

        for side in ("home", "away"):
            team_abbr = home if side == "home" else away
            opp_abbr = away if side == "home" else home
            is_home = (side == "home")
            ctx = context[side]
            goalie_factor= ctx["goalie_factor"]
            goalie_sv = ctx["goalie_info"]["save_pct"]
            b2b_factor = ctx["b2b_factor"]
            is_b2b = ctx["is_b2b"]
            goalie_name = ctx["goalie_info"]["name"]

            # ── 3. Healthy roster ─────────────────────────────────
            if verbose: print(f"\n    [{team_abbr}] Loading healthy forwards...", end=" ", flush=True)
            try:
                forwards = filter_healthy_forwards(team_abbr)
            except Exception:
                forwards = api.get_roster(team_abbr)
            if not forwards:
                if verbose: print("0 — skipping")
                continue
            if verbose: print(f"{len(forwards)}")

            # ── 4. Season stats ───────────────────────────────────
            if verbose: print(f"    [{team_abbr}] Season stats...", end=" ", flush=True)
            season_stats = {}
            for i, player in enumerate(forwards):
                pid = player["player_id"]
                season_stats[pid] = api.get_player_season_stats(pid)
                if i % 5 == 4:
                    time.sleep(0.25)
            if verbose: print("done")

            # ── 5. Line ranks ──────────────────────────────────────
            line_ranks = estimate_line_ranks(forwards, season_stats)

            # ── 6. Recent form (Lines 1-2 only) ───────────────────
            if verbose: print(f"    [{team_abbr}] Recent form...", end=" ", flush=True)
            top_ids = {pid for pid, rank in line_ranks.items() if rank <= 2}
            recent_stats = {}
            for i, pid in enumerate(top_ids):
                recent_stats[pid] = api.get_recent_form(pid)
                if i % 5 == 4:
                    time.sleep(0.25)
            if verbose: print("done")

            # ── 7. Score each forward ──────────────────────────────
            for player in forwards:
                pid = player["player_id"]
                season = season_stats.get(pid, {})
                recent = recent_stats.get(pid, {})
                line = line_ranks.get(pid, 4)

                if season.get("games_played", 0) == 0:
                    continue

                scores = combined_score(
                    season = season,
                    recent = recent,
                    line_rank = line,
                    is_home = is_home,
                    goalie_factor= goalie_factor,
                    goalie_sv_pct= goalie_sv,
                    b2b_factor = b2b_factor,
                    is_b2b = is_b2b,
                )

                all_candidates.append({
                    "player_id":             pid,
                    "name":                  player["name"],
                    "team":                  team_abbr,
                    "opponent":              opp_abbr,
                    "home_away":             "Home" if is_home else "Away",
                    "position":              player.get("position", "F"),
                    "line":                  line,
                    "score":                 scores["final"],
                    "heuristic_score":       scores["heuristic"],
                    "ml_score":              scores["ml"],
                    "model_used":            scores["model_used"],
                    # Stats
                    "goals_per_game":        round(season.get("goals_per_game",    0), 3),
                    "recent_goals_per_game": round(recent.get("recent_goals_per_game", 0), 3),
                    "shots_per_game":        round(season.get("shots_per_game",    0), 2),
                    "pp_goals_per_game":     round(season.get("pp_goals_per_game", 0), 3),
                    "avg_toi_min":           round(season.get("avg_toi_min",       0), 1),
                    "games_played":          season.get("games_played", 0),
                    "goals":                 season.get("goals", 0),
                    # Context
                    "is_b2b":                is_b2b,
                    "opposing_goalie":       goalie_name,
                    "goalie_sv_pct":         goalie_sv,
                    "goalie_factor":         goalie_factor,
                    "matchup":               f"{away} @ {home}",
                    "game_id":               game["game_id"],
                })

    all_candidates.sort(key=lambda x: x["score"], reverse=True)
    return all_candidates[:top_n]


# ─── Output formatters ────────────────────────────────────────────────────────


def format_table(results: list[dict], date: str) -> str:
    medals = ["🥇", "🥈", "🥉", "4️⃣ ", "5️⃣ ", "6️⃣ ", "7️⃣ ", "8️⃣ ", "9️⃣ ", "🔟"]
    lines = []
    lines.append(f"\n\n🥇  TOP {len(results)} FIRST GOALSCORER PICKS — {date}")
    lines.append("─" * 95)
    lines.append(
        f"{"Rank":<5} {"Player":<22} {"Team":<5} {"vs":<5} {"H/A":<5} "
        f"{"Ln":<4} {"Score":<7} {"G/GP":<6} {"L5G":<5} {"SOG":<5} "
        f"{"Goalie SV%":<11} {"B2B":<4} {"Model"}"
    )
    lines.append("─" * 95)

    for i, p in enumerate(results):
        medal = medals[i] if i < len(medals) else f"{i+1}."
        b2b_flag = "⚠️" if p.get("is_b2b") else "  "
        model = "🤖" if p.get("model_used") == "ensemble" else "📐"
        lines.append(
            f"{medal:<5} "
            f"{p["name"]:<22} "
            f"{p["team"]:<5} "
            f"{p["opponent"]:<5} "
            f"{p["home_away"]:<5} "
            f"L{p["line"]:<3} "
            f"{p["score"]:<7} "
            f"{p["goals_per_game"]:<6} "
            f"{p["recent_goals_per_game"]:<5} "
            f"{p["shots_per_game"]:<5} "
            f"{p["goalie_sv_pct"]:.3f} ({p["goalie_factor"]:.2f}x) "
            f"{b2b_flag}   "
            f"{model}"
        )

    lines.append("─" * 95)
    lines.append(f"\n📐 = heuristic model   🤖 = ensemble (heuristic + ML)")
    lines.append(f"⚠️  = team on back-to-back   Goalie (factor) = SV% adjustment multiplier")
    lines.append(f"\n📊  Heuristic weights:")
    for k, v in WEIGHTS.items():
        lines.append(f"    {k:<32} {int(v*100)}%")

    return "\n".join(lines)


def format_json(results: list[dict], date: str) -> str:
    output = {
        "date":         date,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model":        "NHL First Goalscorer Predictor v2.0",
        "weights":      WEIGHTS,
        "picks":        results,
    }
    return json.dumps(output, indent=2)
