"""Train a regression model to predict total runs scored in a game.

Uses the same features as the win-probability classifier but with
`total_runs` as the target.  Saves a GradientBoostingRegressor (or
XGBRegressor if available) plus metadata to models/runs_model/.

Usage:
    python -m mlb.models.train_runs
    python -m mlb.models.train_runs --data data/training_data.parquet --out models/runs_model
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Columns that must NOT be used as features
META_AND_TARGET_COLS = {
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_win",
    "home_score",
    "away_score",
    "total_runs",
    "season",
}


def train_runs_model(
    data_path: str | Path = "data/training_data.parquet",
    output_dir: str | Path = "models/runs_model",
    test_size: float = 0.20,
) -> dict:
    """Train and save a total-runs regression model.

    Returns a dict of evaluation metrics.
    """
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────
    df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows from %s", len(df), data_path)

    if "total_runs" not in df.columns:
        raise ValueError(
            "Column 'total_runs' not found in training data.  "
            "Re-run build_training_data.py to add regression targets."
        )

    # ── Separate features / target ───────────────────────────
    feature_cols = [c for c in df.columns if c not in META_AND_TARGET_COLS]
    X = df[feature_cols].copy()
    y = df["total_runs"].values.astype(float)

    logger.info("Features: %d columns, target mean=%.2f std=%.2f", len(feature_cols), y.mean(), y.std())

    # ── Time-aware split (train on older games, test on newer) ─
    if "game_date" in df.columns:
        dates = pd.to_datetime(df["game_date"])
        sorted_idx = dates.argsort().values
        X = X.iloc[sorted_idx].reset_index(drop=True)
        y = y[sorted_idx]
        split = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y[:split], y[split:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    logger.info("Split: %d train, %d test", len(X_train), len(X_test))

    # ── Scale features ───────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train.values.astype(float))
    X_test_sc = scaler.transform(X_test.values.astype(float))

    # ── Replace NaN with 0 after scaling ─────────────────────
    X_train_sc = np.nan_to_num(X_train_sc, nan=0.0)
    X_test_sc = np.nan_to_num(X_test_sc, nan=0.0)

    # ── Try XGBoost first, fall back to sklearn GBR ──────────
    # Baseball run totals are inherently noisy (std ~4.5), so heavy
    # regularization prevents overfitting while still capturing
    # signal from park factors, SP quality, lineup strength, etc.
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.02,
            subsample=0.7,
            colsample_bytree=0.5,
            min_child_weight=30,
            reg_alpha=3.0,
            reg_lambda=8.0,
            gamma=0.5,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
        )
        model_name = "XGBRegressor"
        logger.info("Using XGBRegressor")
    except ImportError:
        model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.7,
            min_samples_leaf=50,
            max_features="sqrt",
            random_state=42,
        )
        model_name = "GradientBoostingRegressor"
        logger.info("Using sklearn GradientBoostingRegressor (xgboost not available)")

    # ── Train ────────────────────────────────────────────────
    logger.info("Training %s ...", model_name)
    model.fit(X_train_sc, y_train)

    # ── Evaluate ─────────────────────────────────────────────
    y_pred_train = model.predict(X_train_sc)
    y_pred_test = model.predict(X_test_sc)

    metrics = {
        "model": model_name,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_features": len(feature_cols),
        "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "test_r2": float(r2_score(y_test, y_pred_test)),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "pred_mean_test": float(y_pred_test.mean()),
        "pred_std_test": float(y_pred_test.std()),
    }

    logger.info(
        "Train — MAE: %.3f  RMSE: %.3f  R2: %.3f",
        metrics["train_mae"], metrics["train_rmse"], metrics["train_r2"],
    )
    logger.info(
        "Test  — MAE: %.3f  RMSE: %.3f  R2: %.3f",
        metrics["test_mae"], metrics["test_rmse"], metrics["test_r2"],
    )
    logger.info(
        "Target  mean=%.2f std=%.2f | Pred mean=%.2f std=%.2f",
        metrics["y_mean"], metrics["y_std"],
        metrics["pred_mean_test"], metrics["pred_std_test"],
    )

    # ── Save ─────────────────────────────────────────────────
    joblib.dump(model, output_dir / "runs_regressor.joblib")
    joblib.dump(scaler, output_dir / "runs_scaler.joblib")
    joblib.dump(feature_cols, output_dir / "runs_feature_names.joblib")
    joblib.dump(metrics, output_dir / "runs_metrics.joblib")

    logger.info("Saved runs model to %s", output_dir)

    return metrics


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train total-runs regression model")
    parser.add_argument("--data", type=str, default="data/training_data.parquet")
    parser.add_argument("--out", type=str, default="models/runs_model")
    args = parser.parse_args()

    metrics = train_runs_model(args.data, args.out)
    print(f"\nRuns model trained: {metrics['model']}")
    print(f"  Test MAE:  {metrics['test_mae']:.3f} runs")
    print(f"  Test RMSE: {metrics['test_rmse']:.3f} runs")
    print(f"  Test R2:   {metrics['test_r2']:.3f}")
    print(f"  Pred std:  {metrics['pred_std_test']:.3f} (target std: {metrics['y_std']:.3f})")


if __name__ == "__main__":
    main()
