"""
nhl_predictor/train.py
=======================
Trains a GradientBoostingClassifier on the historical first-goalscorer dataset
produced by scraper.py, then saves the model + scaler to data/.

Usage:
    python -m nhl_predictor.train
    python -m nhl_predictor.train --cv            # show cross-validation scores
    python -m nhl_predictor.train --feature-importance
"""

import argparse
import csv
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
TRAINING_FILE = DATA_DIR / "training_data.csv"
MODEL_PATH = DATA_DIR / "ml_model.pkl"
SCALER_PATH = DATA_DIR / "ml_scaler.pkl"
METRICS_PATH = DATA_DIR / "model_metrics.json"

from .model import FEATURE_NAMES


def load_training_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Loads training_data.csv and returns (X, y).
    X: feature matrix, y: binary labels (1 = first goalscorer).
    Skips rows with missing or invalid feature values.
    """
    if not TRAINING_FILE.exists():
        raise FileNotFoundError(
            f"Training data not found at {TRAINING_FILE}\n"
            f"Run: python -m nhl_predictor.scraper --season 20242025"
        )

    X_rows, y_rows = [], []
    skipped = 0

    with open(TRAINING_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                features = [float(row[col]) for col in FEATURE_NAMES]
                label = int(row["first_goal"])
                X_rows.append(features)
                y_rows.append(label)
            except (KeyError, ValueError):
                skipped += 1

    if skipped:
        logger.warning("Skipped %d malformed rows in training data", skipped)

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int)

    logger.info("Loaded %d rows (%d first-goal positives, %.3f%% positive rate)",
                len(y), y.sum(), 100 * y.mean())
    return X, y


def train(cv: bool = False, show_importance: bool = False) -> dict:
    """
    Trains the GradientBoosting model and saves to disk.
    Returns a metrics dict.
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.metrics import (
            roc_auc_score, average_precision_score,
            precision_score, recall_score, classification_report
        )
        from sklearn.calibration import CalibratedClassifierCV
    except ImportError:
        raise ImportError(
            "scikit-learn is required for ML training.\n"
            "Install: pip install scikit-learn"
        )

    X, y = load_training_data()

    if len(y) < 200:
        logger.warning(
            "Only %d training samples — model may be unreliable. "
            "Scrape more seasons for better results.", len(y)
        )

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model — tuned for imbalanced dataset (most players don't score first)
    # class_weight is not directly supported by GBC; use sample_weight via scale_pos_weight trick
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    model = GradientBoostingClassifier(
        n_estimators = 300,
        max_depth = 4,
        learning_rate = 0.05,
        subsample = 0.8,
        min_samples_leaf = 20,
        random_state = 42,
    )

    # Sample weights to handle class imbalance
    sample_weights = np.where(y == 1, pos_weight, 1.0)

    logger.info("Training GradientBoostingClassifier on %d samples (pos_weight=%.1f)...",
                len(y), pos_weight)
    model.fit(X_scaled, y, sample_weight=sample_weights)

    # Calibrate probabilities (Platt scaling) — important for meaningful P(first_goal)
    calibrated = CalibratedClassifierCV(model, cv="prefit", method="sigmoid")
    calibrated.fit(X_scaled, y)

    # Metrics on training set
    y_proba = calibrated.predict_proba(X_scaled)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "trained_at":     datetime.utcnow().isoformat() + "Z",
        "n_samples":      int(len(y)),
        "n_positives":    int(y.sum()),
        "positive_rate":  float(y.mean()),
        "train_roc_auc":  round(float(roc_auc_score(y, y_proba)), 4),
        "train_avg_prec": round(float(average_precision_score(y, y_proba)), 4),
        "features":       FEATURE_NAMES,
    }

    # Cross-validation (5-fold stratified)
    if cv:
        logger.info("Running 5-fold cross-validation...")
        cv_scores = cross_val_score(
            GradientBoostingClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=20, random_state=42
            ),
            X_scaled, y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="roc_auc",
        )
        metrics["cv_roc_auc_mean"] = round(float(cv_scores.mean()), 4)
        metrics["cv_roc_auc_std"]  = round(float(cv_scores.std()), 4)
        logger.info("CV ROC-AUC: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    # Feature importance
    if show_importance:
        importances = model.feature_importances_
        importance_pairs = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
        print("\n📊  Feature Importances:")
        for name, imp in importance_pairs:
            bar = "█" * int(imp * 100)
            print(f"  {name:<30} {imp:.4f}  {bar}")
        metrics["feature_importances"] = {n: round(float(v), 4) for n, v in importance_pairs}

    # Save model + scaler
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH,  "wb") as f: pickle.dump(calibrated, f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    with open(METRICS_PATH,"w")  as f: json.dump(metrics, f, indent=2)

    logger.info("Model saved to %s", MODEL_PATH)
    logger.info("Metrics: ROC-AUC=%.4f  AvgPrecision=%.4f",
                metrics["train_roc_auc"], metrics["train_avg_prec"])

    return metrics


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Train the NHL first-goalscorer ML model")
    parser.add_argument("--cv",                 action="store_true", help="Run 5-fold cross-validation")
    parser.add_argument("--feature-importance", action="store_true", help="Show feature importances")
    args = parser.parse_args()

    metrics = train(cv=args.cv, show_importance=args.feature_importance)

    print(f"\n✅  Model trained successfully")
    print(f"    Samples:          {metrics["n_samples"]:,}")
    print(f"    First-goal rate:  {metrics["positive_rate"]:.3%}")
    print(f"    Train ROC-AUC:    {metrics["train_roc_auc"]:.4f}")
    print(f"    Train AvgPrec:    {metrics["train_avg_prec"]:.4f}")
    if "cv_roc_auc_mean" in metrics:
        print(f"    CV ROC-AUC:       {metrics["cv_roc_auc_mean"]:.4f} ± {metrics["cv_roc_auc_std"]:.4f}")
    print(f"\n    Model: {MODEL_PATH}")
    print(f"    Metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
