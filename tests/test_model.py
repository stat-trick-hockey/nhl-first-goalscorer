"""
tests/test_model.py
====================
Unit tests for the scoring model, features, and tracker.
All tests run offline — no network calls.

Run: python -m pytest tests/ -v
"""

import json
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from nhl_predictor.model import (
    heuristic_score, combined_score, estimate_line_ranks,
    build_feature_vector, WEIGHTS, NORM, LINE_BONUS, FEATURE_NAMES,
)
from nhl_predictor.features import (
    get_goalie_factor, get_b2b_factor,
    LEAGUE_AVG_SV_PCT, ELITE_SV_PCT, POOR_SV_PCT,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────


def make_season(gpg=0.30, sog=3.0, pp=0.08, gp=40, g=12, toi=17.0):
    return {
        "goals_per_game":    gpg,
        "shots_per_game":    sog,
        "pp_goals_per_game": pp,
        "games_played":      gp,
        "goals":             g,
        "avg_toi_min":       toi,
    }


def make_recent(rgpg=0.30, rsog=2.5):
    return {
        "recent_goals_per_game": rgpg,
        "recent_shots_per_game": rsog,
    }


# ─── heuristic_score ─────────────────────────────────────────────────────────


class TestHeuristicScore:
    def test_returns_float(self):
        s = heuristic_score(make_season(), make_recent(), 1, True)
        assert isinstance(s, float)

    def test_score_in_range(self):
        s = heuristic_score(make_season(), make_recent(), 1, True)
        assert 0 <= s <= 100

    def test_elite_player_near_ceiling(self):
        s = heuristic_score(
            make_season(gpg=NORM["season_goals_per_game"], sog=NORM["shots_per_game"], pp=NORM["pp_goals_per_game"]),
            make_recent(rgpg=NORM["recent_goals_per_game"]),
            line_rank=1, is_home=True,
            goalie_factor=1.25,   # weak goalie bonus
            b2b_factor=1.0,
        )
        assert s > 90, f"Elite player vs weak goalie should score >90, got {s}"

    def test_zero_stats_low_score(self):
        s = heuristic_score(make_season(0,0,0), make_recent(0,0), 4, False, 0.75, 0.88)
        assert s < 5, f"Zero-stat B2B grinder vs elite goalie should score <5, got {s}"

    def test_home_advantage(self):
        home = heuristic_score(make_season(), make_recent(), 1, True)
        away = heuristic_score(make_season(), make_recent(), 1, False)
        assert home > away

    def test_home_advantage_magnitude(self):
        home = heuristic_score(make_season(), make_recent(), 1, True)
        away = heuristic_score(make_season(), make_recent(), 1, False)
        expected_diff = WEIGHTS["home_bonus"] * 100
        assert abs((home - away) - expected_diff) < 0.01

    def test_line1_beats_line4(self):
        line1 = heuristic_score(make_season(), make_recent(), 1, False)
        line4 = heuristic_score(make_season(), make_recent(), 4, False)
        assert line1 > line4

    def test_hot_streak_boosts_score(self):
        hot = heuristic_score(make_season(), make_recent(0.60), 1, False)
        cold = heuristic_score(make_season(), make_recent(0.00), 1, False)
        assert hot > cold

    def test_b2b_penalty_reduces_score(self):
        fresh = heuristic_score(make_season(), make_recent(), 1, True, b2b_factor=1.00)
        tired = heuristic_score(make_season(), make_recent(), 1, True, b2b_factor=0.88)
        assert tired < fresh, "B2B should reduce score"
        b2b_contribution = WEIGHTS["b2b_penalty"] * 100
        assert (fresh - tired) <= b2b_contribution + 0.5

    def test_elite_goalie_reduces_score(self):
        vs_weak = heuristic_score(make_season(), make_recent(), 1, False, goalie_factor=1.25)
        vs_elite = heuristic_score(make_season(), make_recent(), 1, False, goalie_factor=0.75)
        assert vs_elite < vs_weak, "Elite goalie should reduce scorer's score"

    def test_reproducible(self):
        s1 = heuristic_score(make_season(), make_recent(), 2, True)
        s2 = heuristic_score(make_season(), make_recent(), 2, True)
        assert s1 == s2

    def test_weights_sum_to_one(self):
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_above_ceiling_clamped(self):
        s_at = heuristic_score(make_season(gpg=NORM["season_goals_per_game"]), make_recent(), 1, False)
        s_way = heuristic_score(make_season(gpg=NORM["season_goals_per_game"] * 10), make_recent(), 1, False)
        assert s_at == s_way, "Values above ceiling must be clamped"


# ─── Goalie factor ────────────────────────────────────────────────────────────


class TestGoalieFactor:
    def _mock_goalie(self, sv_pct, gp=30):
        return {"name": "Test Goalie", "save_pct": sv_pct, "gaa": 2.80, "games_played": gp}

    def test_league_average_returns_one(self):
        with patch("nhl_predictor.features.api.get_expected_starting_goalie",
                   return_value=self._mock_goalie(LEAGUE_AVG_SV_PCT)):
            f = get_goalie_factor("TOR", "2025-03-02")
        assert abs(f - 1.0) < 0.01

    def test_elite_goalie_below_one(self):
        with patch("nhl_predictor.features.api.get_expected_starting_goalie",
                   return_value=self._mock_goalie(ELITE_SV_PCT)):
            f = get_goalie_factor("TOR", "2025-03-02")
        assert f < 1.0, "Elite goalie should reduce scoring probability"
        assert f >= 0.75

    def test_poor_goalie_above_one(self):
        with patch("nhl_predictor.features.api.get_expected_starting_goalie",
                   return_value=self._mock_goalie(POOR_SV_PCT)):
            f = get_goalie_factor("TOR", "2025-03-02")
        assert f > 1.0, "Poor goalie should increase scoring probability"
        assert f <= 1.25

    def test_low_games_played_returns_one(self):
        with patch("nhl_predictor.features.api.get_expected_starting_goalie",
                   return_value=self._mock_goalie(ELITE_SV_PCT, gp=3)):
            f = get_goalie_factor("TOR", "2025-03-02")
        assert f == 1.0, "Fewer than 5 games played — should return neutral factor"

    def test_no_goalie_data_returns_one(self):
        with patch("nhl_predictor.features.api.get_expected_starting_goalie", return_value=None):
            f = get_goalie_factor("TOR", "2025-03-02")
        assert f == 1.0

    def test_factor_in_bounds(self):
        for sv in [0.870, 0.880, 0.890, 0.906, 0.915, 0.920, 0.930]:
            with patch("nhl_predictor.features.api.get_expected_starting_goalie",
                       return_value=self._mock_goalie(sv)):
                f = get_goalie_factor("TOR", "2025-03-02")
            assert 0.75 <= f <= 1.25, f"Factor {f} out of bounds for SV% {sv}"


# ─── Back-to-back ─────────────────────────────────────────────────────────────


class TestBackToBack:
    def _make_schedule(self, yesterday_game=True):
        from datetime import datetime, timedelta
        yesterday = (datetime.strptime("2025-03-02", "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        sched = []
        if yesterday_game:
            sched.append({"game_id": 1, "date": yesterday, "game_state": "OFF",
                          "home_or_away": "home", "opponent": "EDM"})
        sched.append({"game_id": 2, "date": "2025-03-02", "game_state": "FUT",
                      "home_or_away": "home", "opponent": "TOR"})
        return sched

    def test_b2b_detected(self):
        with patch("nhl_predictor.features.api.get_team_schedule",
                   return_value=self._make_schedule(True)):
            f = get_b2b_factor("BOS", "2025-03-02")
        assert f == 0.88

    def test_no_b2b_returns_one(self):
        with patch("nhl_predictor.features.api.get_team_schedule",
                   return_value=self._make_schedule(False)):
            f = get_b2b_factor("BOS", "2025-03-02")
        assert f == 1.0

    def test_api_failure_returns_one(self):
        with patch("nhl_predictor.features.api.get_team_schedule", side_effect=Exception("API error")):
            f = get_b2b_factor("BOS", "2025-03-02")
        assert f == 1.0


# ─── estimate_line_ranks ─────────────────────────────────────────────────────


class TestEstimateLineRanks:
    def _roster(self, n=12):
        return [{"player_id": i, "name": f"P{i}"} for i in range(n)]

    def _stats(self, roster, goals_list):
        return {
            p["player_id"]: make_season(gpg=g/40, g=g, gp=40)
            for p, g in zip(roster, goals_list)
        }

    def test_top_scorer_line1(self):
        roster = self._roster(12)
        stats = self._stats(roster, [40,30,25,20,18,15,12,10,8,6,3,1])
        ranks = estimate_line_ranks(roster, stats)
        assert ranks[0] == 1

    def test_last_scorer_line4(self):
        roster = self._roster(12)
        stats = self._stats(roster, [40,35,30,25,20,18,15,12,10,8,4,1])
        ranks = estimate_line_ranks(roster, stats)
        assert ranks[11] == 4

    def test_all_lines_present(self):
        roster = self._roster(12)
        stats = self._stats(roster, list(range(12, 0, -1)))
        ranks = estimate_line_ranks(roster, stats)
        assert set(ranks.values()) == {1, 2, 3, 4}

    def test_empty_roster(self):
        assert estimate_line_ranks([], {}) == {}

    def test_small_roster_no_crash(self):
        roster = self._roster(5)
        stats = self._stats(roster, [20,15,10,5,1])
        ranks = estimate_line_ranks(roster, stats)
        assert len(ranks) == 5
        assert all(1 <= v <= 4 for v in ranks.values())


# ─── build_feature_vector ────────────────────────────────────────────────────


class TestBuildFeatureVector:
    def test_correct_length(self):
        fv = build_feature_vector(make_season(), make_recent(), 1, True, 0.910, False)
        assert len(fv) == len(FEATURE_NAMES)

    def test_home_flag(self):
        home = build_feature_vector(make_season(), make_recent(), 1, True,  0.910, False)
        away = build_feature_vector(make_season(), make_recent(), 1, False, 0.910, False)
        idx = FEATURE_NAMES.index("is_home")
        assert home[idx] == 1.0
        assert away[idx] == 0.0

    def test_b2b_flag(self):
        b2b = build_feature_vector(make_season(), make_recent(), 1, True, 0.910, True)
        no_b2b = build_feature_vector(make_season(), make_recent(), 1, True, 0.910, False)
        idx = FEATURE_NAMES.index("b2b")
        assert b2b[idx]    == 1.0
        assert no_b2b[idx] == 0.0

    def test_goalie_sv_captured(self):
        fv = build_feature_vector(make_season(), make_recent(), 1, True, 0.925, False)
        idx = FEATURE_NAMES.index("goalie_sv_pct")
        assert fv[idx] == 0.925


# ─── combined_score ──────────────────────────────────────────────────────────


class TestCombinedScore:
    def test_returns_dict_with_expected_keys(self):
        result = combined_score(make_season(), make_recent(), 1, True, 1.0, 0.910, 1.0, False)
        assert "heuristic" in result
        assert "ml" in result
        assert "final" in result
        assert "model_used" in result

    def test_heuristic_only_when_no_ml(self):
        with patch("nhl_predictor.model._ml_model.loaded", False):
            with patch("nhl_predictor.model._ml_model.predict_proba", return_value=None):
                result = combined_score(make_season(), make_recent(), 1, True, 1.0, 0.910, 1.0, False)
        assert result["model_used"] == "heuristic"
        assert result["ml"] is None
        assert result["final"] == result["heuristic"]

    def test_ensemble_blends_scores(self):
        mock_model = MagicMock()
        mock_model.loaded = True
        mock_model.predict_proba.return_value = 0.30
        with patch("nhl_predictor.model._ml_model", mock_model):
            result = combined_score(make_season(), make_recent(), 1, True, 1.0, 0.910, 1.0, False,
                                    ml_weight=0.40)
        assert result["model_used"] == "ensemble"
        expected = 0.60 * result["heuristic"] + 0.40 * result["ml"]
        assert abs(result["final"] - round(expected, 2)) < 0.01


# ─── tracker accuracy logic ──────────────────────────────────────────────────


class TestTrackerReport:
    def test_compute_report_all_correct(self):
        from nhl_predictor.tracker import compute_report
        records = [
            {"top1_correct": True,  "top3_correct": True,  "top5_correct": True,  "best_rank": 1},
            {"top1_correct": True,  "top3_correct": True,  "top5_correct": True,  "best_rank": 1},
            {"top1_correct": False, "top3_correct": True,  "top5_correct": True,  "best_rank": 2},
            {"top1_correct": False, "top3_correct": False, "top5_correct": True,  "best_rank": 4},
            {"top1_correct": False, "top3_correct": False, "top5_correct": False, "best_rank": None},
        ]
        r = compute_report(records)
        assert r["total_days"]    == 5
        assert r["top1_accuracy"] == 0.4
        assert r["top3_accuracy"] == 0.6
        assert r["top5_accuracy"] == 0.8
        assert r["found_in_list_rate"] == 0.8

    def test_compute_report_empty(self):
        from nhl_predictor.tracker import compute_report
        assert compute_report([]) == {}

    def test_avg_rank_calculation(self):
        from nhl_predictor.tracker import compute_report
        records = [
            {"top1_correct": True,  "top3_correct": True,  "top5_correct": True,  "best_rank": 1},
            {"top1_correct": False, "top3_correct": True,  "top5_correct": True,  "best_rank": 3},
            {"top1_correct": False, "top3_correct": False, "top5_correct": False, "best_rank": None},
        ]
        r = compute_report(records)
        assert r["avg_rank_when_found"] == 2.0  # (1+3)/2


# ─── Integration scenario ────────────────────────────────────────────────────


class TestFullRankingScenario:
    def test_superstar_beats_grinder(self):
        elite = heuristic_score(make_season(0.50,4.5,0.20,toi=21.0), make_recent(0.50), 1, True,  1.15, 1.0)
        grinder = heuristic_score(make_season(0.04,1.0,0.00,toi=10.0), make_recent(0.00), 4, False, 0.80, 0.88)
        assert elite > grinder * 3

    def test_b2b_star_vs_fresh_second_liner(self):
        star_b2b = heuristic_score(make_season(0.48,4.5,0.18), make_recent(0.40), 1, True,  1.0, 0.88)
        fresh_2 = heuristic_score(make_season(0.30,3.0,0.10), make_recent(0.60), 2, True,  1.0, 1.00)
        # Both are valid floats — verify neither crashes
        assert isinstance(star_b2b, float)
        assert isinstance(fresh_2,  float)

    def test_five_tiered_players_correct_order(self):
        players = [
            ("Superstar",  0.52, 0.60, 5.0, 1, True,  0.20, 1.15, 1.00),
            ("Star",       0.44, 0.40, 4.0, 1, False, 0.16, 1.00, 1.00),
            ("Middler",    0.25, 0.20, 2.5, 2, True,  0.06, 1.00, 1.00),
            ("Checker",    0.10, 0.10, 1.5, 3, False, 0.01, 1.00, 1.00),
            ("Grinder",    0.04, 0.00, 1.0, 4, False, 0.00, 1.00, 0.88),
        ]
        scored = []
        for name, gpg, rgpg, sog, line, home, pp, gf, b2b in players:
            s = heuristic_score(make_season(gpg, sog, pp), make_recent(rgpg), line, home, gf, b2b)
            scored.append((name, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        assert scored[0][0]  == "Superstar"
        assert scored[-1][0] == "Grinder"
