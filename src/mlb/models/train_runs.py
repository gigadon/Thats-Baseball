"""Train a regression model to predict total runs scored in a game.

Predicts the *deviation* from the market total line (over/under),
then adds it back at inference time.  This lets the model focus on
learning adjustments rather than predicting raw totals from scratch.

Uses an Optuna-tuned ensemble of LightGBM + XGBoost with feature
selection.  Saves models and metadata to models/runs_model/.

Usage:
    PYTHONPATH=src python -m mlb.models.train_runs
    PYTHONPATH=src python -m mlb.models.train_runs --data data/training_data.parquet --out models/runs_model
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

# RunsEnsemble lives in its own module so the pickled model always references a
# stable, importable class path (running this file as ``python -m`` makes it
# ``__main__``, which would pickle as ``__main__.RunsEnsemble`` and fail to load
# in the prediction service).
from mlb.models.runs_ensemble import RunsEnsemble

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

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


def _select_features(
    X: np.ndarray, y: np.ndarray, feature_names: list[str],
    threshold: float = 0.001, max_features: int = 150,
) -> tuple[list[int], list[str]]:
    """Select features using LightGBM importance."""
    selector = lgb.LGBMRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        verbose=-1, random_state=42, n_jobs=-1,
    )
    selector.fit(X, y)
    importances = selector.feature_importances_
    total = importances.sum() or 1.0
    normalized = importances / total

    keep_mask = normalized >= threshold
    keep_indices = set(np.where(keep_mask)[0].tolist())

    # Force critical runs features
    forced = ["market_total", "park_runs_factor", "park_hr_factor"]
    for fname in forced:
        if fname in feature_names:
            keep_indices.add(feature_names.index(fname))

    keep_indices = sorted(keep_indices)

    if len(keep_indices) > max_features:
        forced_idx = {feature_names.index(f) for f in forced if f in feature_names}
        top_idx = set(np.argsort(normalized)[::-1][:max_features].tolist())
        keep_indices = sorted(top_idx | forced_idx)

    if len(keep_indices) < 20:
        keep_indices = sorted(np.argsort(normalized)[::-1][:20].tolist())

    selected_names = [feature_names[i] for i in keep_indices]
    logger.info("Selected %d / %d features for runs model", len(selected_names), len(feature_names))
    return keep_indices, selected_names


def _make_lgbm(params: dict) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.7),
        min_child_samples=params.get("min_child_samples", 20),
        reg_alpha=params.get("reg_alpha", 0.1),
        reg_lambda=params.get("reg_lambda", 1.0),
        verbose=-1,
        random_state=42,
        n_jobs=-1,
    )


def _make_xgb(params: dict) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.7),
        min_child_weight=params.get("min_child_weight", 5),
        reg_alpha=params.get("reg_alpha", 0.1),
        reg_lambda=params.get("reg_lambda", 1.0),
        gamma=params.get("gamma", 0.0),
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )


def _optuna_objective(trial, X_train, y_train, model_type: str):
    """Optuna objective for hyperparameter tuning."""
    if model_type == "lgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        model = _make_lgbm(params)
    else:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        }
        model = _make_xgb(params)

    # Time-series CV: X_train is sorted chronologically before tuning, so
    # shuffled folds would leak future games into the validation score.
    kf = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_t, X_v = X_train[train_idx], X_train[val_idx]
        y_t, y_v = y_train[train_idx], y_train[val_idx]
        model.fit(X_t, y_t)
        y_pred = model.predict(X_v)
        scores.append(mean_squared_error(y_v, y_pred))

    return np.mean(scores)


def train_runs_model(
    data_path: str | Path = "data/training_data.parquet",
    output_dir: str | Path = "models/runs_model",
    test_size: float = 0.20,
    n_trials: int = 30,
) -> dict:
    """Train and save an ensemble total-runs regression model."""
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────
    df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows from %s", len(df), data_path)

    if "total_runs" not in df.columns:
        raise ValueError("Column 'total_runs' not found in training data.")


    # ── Separate features / target ───────────────────────────
    feature_cols = [c for c in df.columns if c not in META_AND_TARGET_COLS]
    X = df[feature_cols].values.astype(float)
    y_raw = df["total_runs"].values.astype(float)

    # Target = deviation from market total (how much actual runs differ from the line)
    market_total_col = feature_cols.index("market_total") if "market_total" in feature_cols else None
    if market_total_col is not None:
        market_totals = X[:, market_total_col]
        mt_filled = np.where(np.isnan(market_totals), 8.5, market_totals)
        y = y_raw - mt_filled
        logger.info("Target: deviation from market_total, mean=%.2f std=%.2f", y.mean(), y.std())
    else:
        y = y_raw
        logger.warning("market_total not found in features, predicting raw total_runs")

    logger.info("Features: %d columns, raw total mean=%.2f std=%.2f", len(feature_cols), y_raw.mean(), y_raw.std())

    # ── Time-aware split ─────────────────────────────────────
    dates = pd.to_datetime(df["game_date"])
    sorted_idx = dates.argsort().values
    X = X[sorted_idx]
    y = y[sorted_idx]
    feature_cols_ordered = feature_cols  # column order unchanged

    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    logger.info("Split: %d train, %d test", len(X_train), len(X_test))

    # ── Feature selection ────────────────────────────────────
    X_train_nan = np.nan_to_num(X_train, nan=0.0)
    keep_indices, selected_names = _select_features(
        X_train_nan, y_train, feature_cols_ordered,
    )

    X_train_sel = X_train[:, keep_indices]
    X_test_sel = X_test[:, keep_indices]

    # LightGBM handles NaN natively; XGBoost needs nan_to_num
    X_train_xgb = np.nan_to_num(X_train_sel, nan=0.0)
    X_test_xgb = np.nan_to_num(X_test_sel, nan=0.0)

    # ── Optuna tune LightGBM ────────────────────────────────
    logger.info("Tuning LightGBM (%d trials)...", n_trials)
    study_lgbm = optuna.create_study(direction="minimize")
    study_lgbm.optimize(
        lambda trial: _optuna_objective(trial, X_train_sel, y_train, "lgbm"),
        n_trials=n_trials,
    )
    best_lgbm_params = study_lgbm.best_params
    logger.info("Best LightGBM MSE: %.4f", study_lgbm.best_value)

    # ── Optuna tune XGBoost ──────────────────────────────────
    logger.info("Tuning XGBoost (%d trials)...", n_trials)
    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(
        lambda trial: _optuna_objective(trial, X_train_xgb, y_train, "xgb"),
        n_trials=n_trials,
    )
    best_xgb_params = study_xgb.best_params
    logger.info("Best XGBoost MSE: %.4f", study_xgb.best_value)

    # ── Train final models ───────────────────────────────────
    lgbm_model = _make_lgbm(best_lgbm_params)
    lgbm_model.fit(X_train_sel, y_train)

    xgb_model = _make_xgb(best_xgb_params)
    xgb_model.fit(X_train_xgb, y_train)

    # ── Calibrate ensemble weights via CV ──────────────────────
    # Find optimal alpha (LGBM weight) and shrinkage using cross-validation
    logger.info("Calibrating ensemble weights...")

    def _calibration_objective(trial):
        alpha = trial.suggest_float("alpha", 0.0, 1.0)
        shrinkage = trial.suggest_float("shrinkage", 0.0, 1.5)

        kf = TimeSeriesSplit(n_splits=5)
        scores = []
        for tr_idx, val_idx in kf.split(X_train_sel):
            Xt, Xv = X_train_sel[tr_idx], X_train_sel[val_idx]
            Xt_xgb = np.nan_to_num(Xt, nan=0.0)
            Xv_xgb = np.nan_to_num(Xv, nan=0.0)
            yt, yv = y_train[tr_idx], y_train[val_idx]

            m_lgbm = _make_lgbm(best_lgbm_params)
            m_lgbm.fit(Xt, yt)
            m_xgb = _make_xgb(best_xgb_params)
            m_xgb.fit(Xt_xgb, yt)

            dev = alpha * m_lgbm.predict(Xv) + (1 - alpha) * m_xgb.predict(Xv_xgb)
            dev = dev * shrinkage
            scores.append(mean_squared_error(yv, dev))

        return np.mean(scores)

    study_cal = optuna.create_study(direction="minimize")
    study_cal.optimize(_calibration_objective, n_trials=40)
    best_alpha = study_cal.best_params["alpha"]
    best_shrinkage = study_cal.best_params["shrinkage"]
    logger.info("Calibrated: alpha=%.3f (LGBM weight), shrinkage=%.3f", best_alpha, best_shrinkage)

    # ── Deviation predictions with calibrated weights ────────
    y_dev_lgbm = lgbm_model.predict(X_test_sel)
    y_dev_xgb = xgb_model.predict(X_test_xgb)
    y_dev_test = (best_alpha * y_dev_lgbm + (1 - best_alpha) * y_dev_xgb) * best_shrinkage

    y_dev_lgbm_train = lgbm_model.predict(X_train_sel)
    y_dev_xgb_train = xgb_model.predict(X_train_xgb)
    y_dev_train = (best_alpha * y_dev_lgbm_train + (1 - best_alpha) * y_dev_xgb_train) * best_shrinkage

    # Convert back to absolute total runs for evaluation
    if market_total_col is not None:
        mt_idx_sel = selected_names.index("market_total") if "market_total" in selected_names else None
    else:
        mt_idx_sel = None

    if mt_idx_sel is not None:
        mt_test = np.nan_to_num(X_test_sel[:, mt_idx_sel], nan=8.5)
        mt_train = np.nan_to_num(X_train_sel[:, mt_idx_sel], nan=8.5)
    else:
        mt_test = np.full(len(y_test), 8.5)
        mt_train = np.full(len(y_train), 8.5)

    y_pred_test = y_dev_test + mt_test
    y_pred_train = y_dev_train + mt_train
    y_total_test = y_test + mt_test   # actual total runs
    y_total_train = y_train + mt_train

    # Held-out residuals (actual - predicted total). This is the runs model's
    # true predictive uncertainty; it is persisted so the betting engine can turn
    # a predicted total into a *calibrated* over/under probability (empirical
    # residual CDF, or its std) instead of a hardcoded sigma.
    test_residuals = np.sort((y_total_test - y_pred_test).astype(float))
    residual_std = float(np.sqrt(mean_squared_error(y_total_test, y_pred_test)))

    # ── Evaluate (in terms of actual total runs) ─────────────
    metrics = {
        "model": "LightGBM+XGBoost calibrated deviation ensemble",
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_features": len(selected_names),
        "alpha": best_alpha,
        "shrinkage": best_shrinkage,
        "train_mae": float(mean_absolute_error(y_total_train, y_pred_train)),
        "test_mae": float(mean_absolute_error(y_total_test, y_pred_test)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_total_train, y_pred_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_total_test, y_pred_test))),
        "train_r2": float(r2_score(y_total_train, y_pred_train)),
        "test_r2": float(r2_score(y_total_test, y_pred_test)),
        "lgbm_test_r2": float(r2_score(y_total_test, y_dev_lgbm + mt_test)),
        "xgb_test_r2": float(r2_score(y_total_test, y_dev_xgb + mt_test)),
        "dev_mae": float(mean_absolute_error(y_test, y_dev_test)),
        "dev_r2": float(r2_score(y_test, y_dev_test)),
        "y_mean": float(y_raw.mean()),
        "y_std": float(y_raw.std()),
        "dev_mean": float(y.mean()),
        "dev_std": float(y.std()),
        "pred_mean_test": float(y_pred_test.mean()),
        "pred_std_test": float(y_pred_test.std()),
        # Calibration inputs for the over/under probability (see betting engine).
        "residual_std": residual_std,
        "residuals": test_residuals.tolist(),
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
        "  LightGBM R2: %.3f  |  XGBoost R2: %.3f",
        metrics["lgbm_test_r2"], metrics["xgb_test_r2"],
    )
    logger.info(
        "Deviation — MAE: %.3f  R2: %.3f  (mean=%.2f std=%.2f)",
        metrics["dev_mae"], metrics["dev_r2"],
        metrics["dev_mean"], metrics["dev_std"],
    )
    logger.info(
        "Total runs — mean=%.2f std=%.2f | Pred mean=%.2f std=%.2f",
        metrics["y_mean"], metrics["y_std"],
        metrics["pred_mean_test"], metrics["pred_std_test"],
    )

    # ── Save ─────────────────────────────────────────────────
    # Find market_total index within selected features for the ensemble wrapper
    mt_sel_idx = selected_names.index("market_total") if "market_total" in selected_names else None
    ensemble = RunsEnsemble(
        lgbm_model, xgb_model,
        market_total_idx=mt_sel_idx,
        alpha=best_alpha,
        shrinkage=best_shrinkage,
        residual_std=residual_std,
        residuals=test_residuals.tolist(),
    )

    joblib.dump(ensemble, output_dir / "runs_regressor.joblib")
    joblib.dump(None, output_dir / "runs_scaler.joblib")  # No scaler needed
    joblib.dump(selected_names, output_dir / "runs_feature_names.joblib")
    joblib.dump(metrics, output_dir / "runs_metrics.joblib")

    logger.info("Saved runs model to %s", output_dir)

    # Print feature importances
    lgbm_imp = dict(zip(selected_names, lgbm_model.feature_importances_))
    sorted_imp = sorted(lgbm_imp.items(), key=lambda x: -x[1])
    logger.info("Top 15 features:")
    for i, (name, val) in enumerate(sorted_imp[:15]):
        logger.info("  %2d. %s: %.0f", i + 1, name, val)

    return metrics


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train total-runs regression model")
    parser.add_argument("--data", type=str, default="data/training_data.parquet")
    parser.add_argument("--out", type=str, default="models/runs_model")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    metrics = train_runs_model(args.data, args.out, n_trials=args.trials)
    print(f"\nRuns model trained: {metrics['model']}")
    print(f"  Test MAE:  {metrics['test_mae']:.3f} runs")
    print(f"  Test RMSE: {metrics['test_rmse']:.3f} runs")
    print(f"  Test R2:   {metrics['test_r2']:.3f}")
    print(f"  Dev MAE:   {metrics['dev_mae']:.3f}  Dev R2: {metrics['dev_r2']:.3f}")
    print(f"  Pred std:  {metrics['pred_std_test']:.3f} (target std: {metrics['y_std']:.3f})")
    print(f"  LightGBM R2: {metrics['lgbm_test_r2']:.3f}  |  XGBoost R2: {metrics['xgb_test_r2']:.3f}")


if __name__ == "__main__":
    main()
