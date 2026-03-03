"""
NHL First Goalscorer Predictor
================================
Predicts the top N most likely first goalscorers for any given NHL game day.

Modules:
    api         — NHL API client (schedule, rosters, stats, game logs)
    features    — Feature extraction (goalie SV%, back-to-back, injury flags)
    model       — Weighted heuristic scorer + ML model (GradientBoosting)
    tracker     — Historical accuracy tracking (did #1 pick score first?)
    scraper     — Historical first-goalscorer data scraper for ML training
    predict     — Main daily prediction pipeline
    cli         — CLI entry point
"""

__version__ = "2.0.0"
