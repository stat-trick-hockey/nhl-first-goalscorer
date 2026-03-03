# 🏒 NHL First Goalscorer Predictor

![Tests](https://github.com/stat-trick-hockey/nhl-first-goalscorer/actions/workflows/tests.yml/badge.svg)
![Daily Picks](https://github.com/stat-trick-hockey/nhl-first-goalscorer/actions/workflows/daily_picks.yml/badge.svg)
![ML Retrain](https://github.com/stat-trick-hockey/nhl-first-goalscorer/actions/workflows/retrain.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Daily morning script that predicts the **top N most likely first goalscorers** across all NHL games today.

Uses a **heuristic + ML ensemble**: a weighted scoring model augmented by a GradientBoosting classifier trained on historical first-goal outcomes — with goalie quality adjustments, back-to-back fatigue penalties, injury filtering, and automatic accuracy tracking.

---

## Quick Start

```bash
git clone https://github.com/stat-trick-hockey/nhl-first-goalscorer.git
cd nhl-first-goalscorer
pip install -r requirements.txt

python -m nhl_predictor.cli predict
```

**Sample output:**
```
🥇  TOP 5 FIRST GOALSCORER PICKS — 2025-03-02
───────────────────────────────────────────────────────────────────────────────────────────────
Rank  Player                 Team  vs    H/A   Ln  Score   G/GP   L5G   SOG   Goalie SV%    B2B  Model
───────────────────────────────────────────────────────────────────────────────────────────────
🥇    Connor McDavid         EDM   CGY   Home  L1  84.3    0.512  0.600  4.8   .908 (1.08x)       🤖
🥈    Nathan MacKinnon       COL   WPG   Away  L1  80.1    0.481  0.600  4.5   .901 (1.12x)       🤖
🥉    Leon Draisaitl         EDM   CGY   Home  L1  73.8    0.445  0.400  4.1   .908 (1.08x)  ⚠️  📐
4️⃣    David Pastrnak         BOS   TOR   Home  L1  71.4    0.436  0.400  4.0   .895 (1.19x)       🤖
5️⃣    Auston Matthews        TOR   BOS   Away  L1  62.1    0.497  0.200  5.2   .924 (0.80x)       🤖

📐 = heuristic model   🤖 = ensemble (heuristic + ML)
⚠️  = team on back-to-back
```

---

## Commands

```bash
# Run today's prediction
python -m nhl_predictor.cli predict
python -m nhl_predictor.cli predict --date 2025-03-10 --top 10 --output json --save

# Track yesterday's accuracy
python -m nhl_predictor.cli track
python -m nhl_predictor.cli track --date 2025-03-01

# View accuracy report
python -m nhl_predictor.cli report
python -m nhl_predictor.cli report --last 30

# Offline model validation (no internet)
python -m nhl_predictor.cli validate

# Scrape historical data for ML training
python -m nhl_predictor.cli scrape --season 20242025
python -m nhl_predictor.cli scrape --season 20232024 --season 20242025
python -m nhl_predictor.cli scrape --days-back 30

# Train the ML model
python -m nhl_predictor.cli train
python -m nhl_predictor.cli train --cv --feature-importance
```

---

## Scoring Model

### Heuristic (always active)

Each forward playing today is scored 0–100 using 9 weighted factors:

| Factor | Weight | Notes |
|--------|--------|-------|
| Recent goals/game (last 5) | **25%** | Hot/cold form — strongest near-term signal |
| Season goals/game | **20%** | Baseline production this season |
| Shots on goal/game | **12%** | Volume shooters get more first-goal chances |
| Forward line rank | **12%** | 1st liners: most ice time, offensive zone starts, PP reps |
| **Goalie SV% factor** | **10%** | Opposing goalie quality — elite SV% penalises, poor SV% rewards |
| PP goals/game | **8%** | Power play involvement = high-danger early chances |
| **Back-to-back penalty** | **5%** | 12% penalty when team played yesterday |
| Home/away | **4%** | Home teams score first ~52% of the time |
| Avg ice time | **4%** | More TOI = more scoring opportunities |

### ML Model (when trained)

A `GradientBoostingClassifier` trained on historical first-goalscorer data scraped from the NHL API. Predicts `P(first_goal=True)` for each player.

When the model file exists at `data/ml_model.pkl`, the final score blends:
```
final = 0.60 × heuristic + 0.40 × ML_probability × 100
```

Falls back silently to heuristic-only if the model is not yet trained.

### Goalie SV% factor

The opposing goalie's season SV% is mapped to a multiplier (0.75–1.25×):
- **Elite goalie** (SV% ≥ 0.925) → 0.75× (harder to score first)
- **League average** (SV% ≈ 0.906) → 1.00× (neutral)
- **Struggling goalie** (SV% ≤ 0.890) → 1.25× (easier to score first)

Requires ≥5 games played to avoid small-sample noise.

---

## ML Pipeline

### 1. Scrape training data

```bash
# Full backfill — scrapes 2022-23, 2023-24, 2024-25 seasons
python -m nhl_predictor.cli scrape

# Or a single season
python -m nhl_predictor.cli scrape --season 20242025
```

This reads every completed game's play-by-play, identifies the first goalscorer, and labels all forwards in both lineups as `first_goal=1` (scorer) or `0` (everyone else). Saves to `data/training_data.csv`.

Expect ~30-40 labeled rows per game × ~1,300 games/season = ~40,000–50,000 rows per season.

### 2. Train the model

```bash
python -m nhl_predictor.cli train --cv --feature-importance
```

Output:
```
✅  Model trained successfully
    Samples:          127,453
    First-goal rate:  1.8%
    Train ROC-AUC:    0.7842
    Train AvgPrec:    0.1204
    CV ROC-AUC:       0.7631 ± 0.0089

📊  Feature Importances:
  recent_goals_per_game          0.2841  ████████████████████████████
  season_goals_per_game          0.2214  ██████████████████████
  shots_per_game                 0.1533  ███████████████
  ...
```

### 3. Picks now use the ensemble automatically

The model is loaded at startup. The output table shows 🤖 next to ensemble picks and 📐 for heuristic-only.

### Weekly automatic retraining

The `retrain.yml` GitHub Actions workflow runs every Monday — scrapes the last 14 days and retrains the model, committing updated `.pkl` files back to the repo.

---

## Accuracy Tracking

Each day after games finish, the tracker compares your picks against actual first goalscorers:

```bash
python -m nhl_predictor.cli track          # evaluate yesterday
python -m nhl_predictor.cli report         # all-time summary
python -m nhl_predictor.cli report --last 30
```

```
📊  PREDICTION ACCURACY REPORT
     Last 30 days  (2025-02-01 → 2025-03-02)
──────────────────────────────────────────────────
  Days tracked:         30
  Top-1 accuracy:       16.7%  (5/30 days)
  Top-3 accuracy:       43.3%  (13/30 days)
  Top-5 accuracy:       66.7%  (20/30 days)
  Scorer found in list: 73.3%
  Avg rank (when found): 3.2

  Recent results:
  Date         #1 Pick                Actual First Scorer     ✓
  ─────────    ──────────────────     ──────────────────     ────
  2025-03-01   Connor McDavid         Leon Draisaitl          📋
  2025-02-28   Auston Matthews        Auston Matthews         ✅
  ...
```

Results are stored in `data/accuracy_log.jsonl` and committed automatically by the daily workflow.

---

## GitHub Actions Workflows

| Workflow | Schedule | What it does |
|----------|----------|-------------|
| `daily_picks.yml` | 7 AM ET daily | Generates picks, saves JSON to `picks/`, tracks yesterday's accuracy |
| `tests.yml` | Push / PR | Runs 40 unit tests across Python 3.11 + 3.12 |
| `retrain.yml` | Every Monday | Scrapes 14 days, retrains ML model, commits updated `.pkl` files |

### Enable on your fork

1. Fork the repo
2. **Actions → Enable workflows**
3. That's it — picks run automatically every morning

---

## Development

```bash
pip install -r requirements-dev.txt

python -m pytest tests/ -v
python -m nhl_predictor.cli validate     # offline check, no API calls
black nhl_predictor/ tests/
```

---

## Project Structure

```
nhl-first-goalscorer/
├── nhl_predictor/
│   ├── __init__.py
│   ├── api.py          — All NHL API calls (schedule, roster, stats, play-by-play)
│   ├── features.py     — Goalie SV% factor, B2B detection, injury filtering
│   ├── model.py        — Heuristic scorer + ML wrapper + ensemble blend
│   ├── scraper.py      — Historical first-goalscorer data scraper
│   ├── train.py        — ML model training (GradientBoosting + calibration)
│   ├── tracker.py      — Daily accuracy tracking & reporting
│   ├── predict.py      — Main prediction pipeline (orchestrates everything)
│   └── cli.py          — Unified CLI (predict/track/report/scrape/train/validate)
├── tests/
│   ├── conftest.py
│   └── test_model.py   — 40 unit tests (all offline)
├── data/               — ML model, training data, accuracy log
├── picks/              — Daily pick JSON files (auto-committed)
├── .github/workflows/
│   ├── daily_picks.yml
│   ├── tests.yml
│   └── retrain.yml
├── requirements.txt
├── requirements-ml.txt
├── requirements-dev.txt
├── pyproject.toml
├── CHANGELOG.md
└── LICENSE
```

---

## Data Source

All data from `api-web.nhle.com` — the NHL's official (undocumented) API. No key required.

---

## License

MIT — see [LICENSE](LICENSE)
