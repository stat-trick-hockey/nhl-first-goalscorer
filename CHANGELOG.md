# Changelog

## [2.0.0] — 2025-03-02

### Added
- **Goalie SV% factor** — opposing goalie's save percentage now adjusts each player's score
  via a linear multiplier (0.75–1.25x). Elite goalies reduce scores; poor goalies increase them.
  Requires ≥5 games played to avoid small-sample noise.
- **Back-to-back fatigue penalty** — 12% score reduction when a team played the previous night.
  Detected from the team schedule endpoint.
- **Injury filtering** — forwards on the injury report are automatically excluded from picks.
  Falls back to full roster if the injury endpoint is unavailable.
- **Historical accuracy tracker** (`tracker.py`) — evaluates whether yesterday's picks were
  correct by parsing completed game play-by-play data. Logs per-date results to
  `data/accuracy_log.jsonl`.
- **ML ensemble model** (`train.py`, `scraper.py`) — GradientBoostingClassifier trained on
  historical first-goal outcomes scraped from the NHL API. When trained, blends with the
  heuristic at 40% ML / 60% heuristic. Falls back to heuristic-only if model file absent.
- **Historical scraper** (`scraper.py`) — scrapes every completed game for any season range,
  labels first goalscorers, and builds `data/training_data.csv` for ML training.
- **Weekly ML retrain workflow** (`.github/workflows/retrain.yml`) — runs every Monday to scrape
  the past 14 days and retrain the model. Commits updated model files back to the repo.
- **Accuracy tracking in daily workflow** — automatically evaluates yesterday after today's picks.
- Restructured as a proper Python package (`nhl_predictor/`) with separate modules for each concern.
- Unified CLI (`nhl-picks`) with subcommands: `predict`, `track`, `report`, `scrape`, `train`, `validate`.
- Added `avg_toi_min` (average ice time) as a new model feature.
- Output table now shows goalie SV% factor, B2B flag, and model mode (heuristic vs ensemble).
- 40 unit tests covering all new features.

### Changed
- `compute_score()` renamed to `heuristic_score()` — now includes goalie and B2B parameters.
- `run_predictions()` now calls `filter_healthy_forwards()` instead of raw roster.
- Model weights rebalanced to accommodate 3 new features (goalie, B2B, TOI).
- Output JSON now includes `heuristic_score`, `ml_score`, `model_used`, `opposing_goalie`,
  `goalie_sv_pct`, `goalie_factor`, `is_b2b`, `avg_toi_min` per pick.

---

## [1.0.0] — 2025-03-02

Initial release — 6-factor weighted heuristic model.
