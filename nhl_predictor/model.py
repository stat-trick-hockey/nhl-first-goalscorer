"""
nhl_predictor/model.py
=======================
Two scoring modes:

1. HEURISTIC (default, always available)
   Weighted combination of normalized features.
   Now includes goalie SV%, back-to-back penalty, and injury filtering.

2. ML (when a trained model file exists at data/ml_model.pkl)
   GradientBoostingClassifier trained on historical first-goal outcomes.
   Falls back gracefully to heuristic if the model file is missing.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─── Heuristic model config ──────────────────────────────────────────────────

WEIGHTS = {
    "season_goals_per_game":  0.20,
    "recent_goals_per_game":  0.25,
    "shots_per_game":         0.12,
    "line_bonus":             0.12,
    "pp_goals_per_game":      0.08,
    "home_bonus":             0.04,
    "goalie_factor":          0.10,   # NEW — opposing goalie quality
    "b2b_penalty":            0.05,   # NEW — back-to-back fatigue
    "toi_factor":             0.04,   # NEW — average ice time
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

NORM = {
    "season_goals_per_game":  0.55,
    "recent_goals_per_game":  0.60,
    "shots_per_game":         5.5,
    "pp_goals_per_game":      0.25,
    "avg_toi_min":            22.0,   # ~elite forward TOI ceiling
}

LINE_BONUS = {1: 1.0, 2: 0.65, 3: 0.30, 4: 0.10}

# ─── Feature vector definition (must match ML training) ──────────────────────

FEATURE_NAMES = [
    "season_goals_per_game",
    "recent_goals_per_game",
    "shots_per_game",
    "recent_shots_per_game",
    "pp_goals_per_game",
    "avg_toi_min",
    "line_rank",           # 1-4
    "is_home",             # 0 or 1
    "goalie_sv_pct",       # opponent goalie SV%
    "b2b",                 # 0 or 1 — team on B2B
    "games_played",
]

MODEL_PATH = Path(__file__).parent.parent / "data" / "ml_model.pkl"
SCALER_PATH = Path(__file__).parent.parent / "data" / "ml_scaler.pkl"


# ─── Line rank estimation ─────────────────────────────────────────────────────


def estimate_line_ranks(forwards: list[dict], season_stats: dict[int, dict]) -> dict[int, int]:
    """
    Estimates forward line assignment (1-4) based on season goals + shots.
    Top 3 scorers = Line 1, next 3 = Line 2, etc.
    """
    scored = []
    for f in forwards:
        pid = f["player_id"]
        s = season_stats.get(pid, {})
        scored.append((pid, s.get("goals", 0), s.get("shots", 0)))

    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)

    ranks = {}
    for i, (pid, _, _) in enumerate(scored):
        ranks[pid] = min((i // 3) + 1, 4)
    return ranks


# ─── Feature builder ─────────────────────────────────────────────────────────


def build_feature_vector(
    season: dict,
    recent: dict,
    line_rank: int,
    is_home: bool,
    goalie_sv_pct: float,
    is_b2b: bool,
) -> np.ndarray:
    """
    Builds a feature vector consistent with FEATURE_NAMES.
    Used for both ML inference and heuristic scoring.
    """
    return np.array([
        season.get("goals_per_game",     0.0),
        recent.get("recent_goals_per_game", 0.0),
        season.get("shots_per_game",     0.0),
        recent.get("recent_shots_per_game", 0.0),
        season.get("pp_goals_per_game",  0.0),
        season.get("avg_toi_min",        0.0),
        float(line_rank),
        1.0 if is_home else 0.0,
        goalie_sv_pct,
        1.0 if is_b2b else 0.0,
        float(season.get("games_played", 0)),
    ], dtype=float)


# ─── Heuristic scorer ─────────────────────────────────────────────────────────


def heuristic_score(
    season: dict,
    recent: dict,
    line_rank: int,
    is_home: bool,
    goalie_factor: float  = 1.0,
    b2b_factor: float     = 1.0,
) -> float:
    """
    Returns a 0–100 heuristic score.
    Includes goalie SV% adjustment and back-to-back fatigue penalty.
    """
    def norm(v, ceil):
        return min(v / ceil, 1.0) if ceil > 0 else 0.0

    # Individual factor scores (0–1)
    s_season = norm(season.get("goals_per_game",     0), NORM["season_goals_per_game"])
    s_recent = norm(recent.get("recent_goals_per_game", 0), NORM["recent_goals_per_game"])
    s_shots = norm(season.get("shots_per_game",     0), NORM["shots_per_game"])
    s_line = LINE_BONUS.get(line_rank, 0.10)
    s_pp = norm(season.get("pp_goals_per_game",  0), NORM["pp_goals_per_game"])
    s_home = 1.0 if is_home else 0.0
    s_toi = norm(season.get("avg_toi_min",        0), NORM["avg_toi_min"])

    # Goalie factor is already a multiplier (0.75–1.25); map to 0–1 for weighting
    # Neutral (1.0) maps to 0.5; elite goalie (0.75) → 0.0; weak goalie (1.25) → 1.0
    s_goalie = (goalie_factor - 0.75) / 0.50

    # B2B: b2b_factor is 0.88 if B2B, else 1.0 → map to 0/1 penalty score
    # We treat "no B2B" = 1.0 and "B2B" = 0.0 as a binary flag for weighting
    s_b2b = 1.0 if b2b_factor == 1.0 else 0.0

    raw = (
        WEIGHTS["season_goals_per_game"] * s_season  +
        WEIGHTS["recent_goals_per_game"] * s_recent  +
        WEIGHTS["shots_per_game"]        * s_shots   +
        WEIGHTS["line_bonus"]            * s_line    +
        WEIGHTS["pp_goals_per_game"]     * s_pp      +
        WEIGHTS["home_bonus"]            * s_home    +
        WEIGHTS["goalie_factor"]         * s_goalie  +
        WEIGHTS["b2b_penalty"]           * s_b2b     +
        WEIGHTS["toi_factor"]            * s_toi
    )

    return round(raw * 100, 2)


# ─── ML model ─────────────────────────────────────────────────────────────────


class MLModel:
    """
    Wrapper around a trained sklearn GradientBoostingClassifier.
    Loads from disk; falls back to None if not found.
    """

    def __init__(self):
        self.model  = None
        self.scaler = None
        self.loaded = False
        self._load()

    def _load(self):
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            try:
                with open(MODEL_PATH,  "rb") as f: self.model  = pickle.load(f)
                with open(SCALER_PATH, "rb") as f: self.scaler = pickle.load(f)
                self.loaded = True
                logger.info("ML model loaded from %s", MODEL_PATH)
            except Exception as e:
                logger.warning("Failed to load ML model: %s — using heuristic only", e)
        else:
            logger.info("No ML model found at %s — using heuristic only", MODEL_PATH)

    def predict_proba(self, feature_vector: np.ndarray) -> float:
        """
        Returns probability (0–1) that this player scores first.
        Returns None if model not loaded.
        """
        if not self.loaded:
            return None
        try:
            X = self.scaler.transform(feature_vector.reshape(1, -1))
            prob = self.model.predict_proba(X)[0][1]  # P(first_goal=True)
            return float(prob)
        except Exception as e:
            logger.warning("ML inference failed: %s", e)
            return None


# Singleton ML model — loaded once at import time
_ml_model = MLModel()


def ml_score(
    season: dict,
    recent: dict,
    line_rank: int,
    is_home: bool,
    goalie_sv_pct: float,
    is_b2b: bool,
) -> Optional[float]:
    """
    Returns ML probability * 100 (0–100 scale), or None if model unavailable.
    """
    fv = build_feature_vector(season, recent, line_rank, is_home, goalie_sv_pct, is_b2b)
    prob = _ml_model.predict_proba(fv)
    return round(prob * 100, 2) if prob is not None else None


def combined_score(
    season: dict,
    recent: dict,
    line_rank: int,
    is_home: bool,
    goalie_factor: float,
    goalie_sv_pct: float,
    b2b_factor: float,
    is_b2b: bool,
    ml_weight: float = 0.40,
) -> dict:
    """
    Computes both heuristic and ML scores, then blends them.

    If ML model is unavailable, returns heuristic score only.
    Returns dict with: heuristic, ml, final, model_used
    """
    h = heuristic_score(season, recent, line_rank, is_home, goalie_factor, b2b_factor)
    m = ml_score(season, recent, line_rank, is_home, goalie_sv_pct, is_b2b)

    if m is not None:
        final = round((1 - ml_weight) * h + ml_weight * m, 2)
        model_used = "ensemble"
    else:
        final = h
        model_used = "heuristic"

    return {
        "heuristic":  h,
        "ml":         m,
        "final":      final,
        "model_used": model_used,
    }
