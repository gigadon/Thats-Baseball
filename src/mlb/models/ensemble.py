"""Stacking ensemble with logistic regression meta-learner.

Architecture:
    Layer 1: XGBoost (0.25), GBM (0.20), LightGBM (0.25), CatBoost (0.20), RF (0.10)
    Layer 2: Logistic Regression meta-model trained on OOF predictions

Supports both weighted average and stacking modes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from mlb.models.base import (
    BaseModel,
    CatBoostModel,
    GradientBoostingModel,
    LightGBMModel,
    ModelMetrics,
    RandomForestModel,
    XGBoostModel,
    _expected_calibration_error,
)

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble."""

    mode: str = "stacking"  # "stacking" or "weighted_average"
    weights: dict[str, float] = field(default_factory=lambda: {
        "XGBoost": 0.25,
        "GradientBoosting": 0.20,
        "LightGBM": 0.25,
        "CatBoost": 0.20,
        "RandomForest": 0.10,
    })
    cv_folds: int = 5
    random_state: int = 42


class EnsembleModel:
    """Stacking ensemble that combines multiple base models."""

    def __init__(self, config: EnsembleConfig | None = None):
        self.config = config or EnsembleConfig()
        self.base_models: list[BaseModel] = [
            XGBoostModel(),
            GradientBoostingModel(),
            LightGBMModel(),
            CatBoostModel(),
            RandomForestModel(),
        ]
        self.meta_model: LogisticRegression | None = None
        self.calibrator: CalibratedClassifierCV | None = None
        self.feature_names: list[str] = []
        self.is_fitted = False
        self._oof_metrics: dict[str, ModelMetrics] = {}

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None):
        """Train the full ensemble.

        1. Generate out-of-fold predictions from each base model
        2. Train meta-learner on stacked OOF predictions
        3. Retrain each base model on the full dataset
        """
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        n_samples = X.shape[0]
        n_models = len(self.base_models)

        logger.info(
            "Training ensemble: %d base models, %d samples, %d features",
            n_models, n_samples, X.shape[1],
        )

        if self.config.mode == "stacking":
            self._train_stacking(X, y)
        else:
            self._train_weighted(X, y)

        self.is_fitted = True

    def _train_stacking(self, X: np.ndarray, y: np.ndarray):
        """Train with stacking: OOF predictions → meta-learner."""
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        oof_preds = np.zeros((n_samples, n_models))

        kf = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        # Generate OOF predictions
        for model_idx, model in enumerate(self.base_models):
            logger.info("Generating OOF predictions for %s", model.name)
            fold_preds = np.zeros(n_samples)

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]

                # Clone model for this fold
                fold_model = model.__class__()
                fold_model.train(X_train, y_train, self.feature_names)
                fold_preds[val_idx] = fold_model.predict_proba(X_val)

            oof_preds[:, model_idx] = fold_preds

            # Evaluate OOF performance
            from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

            preds_binary = (fold_preds >= 0.5).astype(int)
            self._oof_metrics[model.name] = ModelMetrics(
                accuracy=float(np.mean(preds_binary == y)),
                brier_score=float(brier_score_loss(y, fold_preds)),
                log_loss=float(log_loss(y, np.clip(fold_preds, 1e-7, 1 - 1e-7))),
                auc_roc=float(roc_auc_score(y, fold_preds)),
                calibration_error=_expected_calibration_error(y, fold_preds),
            )
            logger.info(
                "%s OOF — Acc: %.3f, Brier: %.4f, AUC: %.3f",
                model.name,
                self._oof_metrics[model.name].accuracy,
                self._oof_metrics[model.name].brier_score,
                self._oof_metrics[model.name].auc_roc,
            )

        # Train meta-learner on OOF predictions
        self.meta_model = LogisticRegression(
            C=1.0, max_iter=1000, random_state=self.config.random_state
        )
        self.meta_model.fit(oof_preds, y)
        logger.info("Meta-model trained on %d OOF predictions", n_samples)

        # Calibrate: fit isotonic regression on meta-model OOF outputs
        meta_probs = self.meta_model.predict_proba(oof_preds)[:, 1]
        self._fit_calibrator(meta_probs, y)

        # Retrain base models on full dataset
        for model in self.base_models:
            model.train(X, y, self.feature_names)
            logger.info("%s retrained on full dataset", model.name)

    def _train_weighted(self, X: np.ndarray, y: np.ndarray):
        """Train with simple weighted average (no meta-learner)."""
        for model in self.base_models:
            model.train(X, y, self.feature_names)

    def _fit_calibrator(self, probs: np.ndarray, y: np.ndarray):
        """Fit isotonic calibration on OOF probabilities."""
        from sklearn.isotonic import IsotonicRegression

        self.calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        self.calibrator.fit(probs, y)
        cal_probs = self.calibrator.predict(probs)

        from sklearn.metrics import brier_score_loss
        raw_brier = brier_score_loss(y, probs)
        cal_brier = brier_score_loss(y, cal_probs)
        raw_ece = _expected_calibration_error(y, probs)
        cal_ece = _expected_calibration_error(y, cal_probs)
        logger.info(
            "Calibration: Brier %.4f → %.4f, ECE %.4f → %.4f",
            raw_brier, cal_brier, raw_ece, cal_ece,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ensemble win probability for each sample."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble is not fitted")

        base_preds = self._get_base_predictions(X)

        if self.config.mode == "stacking" and self.meta_model is not None:
            probs = self.meta_model.predict_proba(base_preds)[:, 1]
        else:
            probs = self._weighted_average(base_preds)

        if self.calibrator is not None:
            probs = self.calibrator.predict(probs)

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    def predict_detailed(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return predictions from each model plus the ensemble."""
        base_preds = self._get_base_predictions(X)
        result: dict[str, np.ndarray] = {}

        for i, model in enumerate(self.base_models):
            result[model.name] = base_preds[:, i]

        result["ensemble"] = self.predict_proba(X)
        result["std"] = np.std(base_preds, axis=1)  # Disagreement measure
        return result

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, ModelMetrics]:
        """Evaluate ensemble and each base model."""
        results: dict[str, ModelMetrics] = {}

        # Individual models
        for model in self.base_models:
            results[model.name] = model.evaluate(X, y)

        # Ensemble
        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

        results["Ensemble"] = ModelMetrics(
            accuracy=float(np.mean(preds == y)),
            brier_score=float(brier_score_loss(y, probs)),
            log_loss=float(log_loss(y, probs)),
            auc_roc=float(roc_auc_score(y, probs)),
            calibration_error=_expected_calibration_error(y, probs),
        )

        return results

    def feature_importance(self) -> dict[str, float]:
        """Weighted average feature importance across base models."""
        weights = self.config.weights
        combined: dict[str, float] = {}

        for model in self.base_models:
            w = weights.get(model.name, 0.25)
            imp = model.feature_importance()
            for feat, val in imp.items():
                combined[feat] = combined.get(feat, 0.0) + val * w

        # Normalize
        total = sum(combined.values()) or 1.0
        return {k: v / total for k, v in sorted(combined.items(), key=lambda x: -x[1])}

    def save(self, directory: str | Path):
        """Save all models and meta-learner to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        for model in self.base_models:
            model.save(directory / f"{model.name}.joblib")

        if self.meta_model is not None:
            joblib.dump(self.meta_model, directory / "meta_model.joblib")

        if self.calibrator is not None:
            joblib.dump(self.calibrator, directory / "calibrator.joblib")

        joblib.dump(
            {
                "config": self.config,
                "feature_names": self.feature_names,
                "oof_metrics": self._oof_metrics,
            },
            directory / "ensemble_meta.joblib",
        )
        logger.info("Ensemble saved to %s", directory)

    def load(self, directory: str | Path):
        """Load all models from a directory."""
        directory = Path(directory)

        meta = joblib.load(directory / "ensemble_meta.joblib")
        self.config = meta["config"]
        self.feature_names = meta["feature_names"]
        self._oof_metrics = meta.get("oof_metrics", {})

        for model in self.base_models:
            model.load(directory / f"{model.name}.joblib")

        meta_path = directory / "meta_model.joblib"
        if meta_path.exists():
            self.meta_model = joblib.load(meta_path)

        cal_path = directory / "calibrator.joblib"
        if cal_path.exists():
            self.calibrator = joblib.load(cal_path)

        self.is_fitted = True
        logger.info("Ensemble loaded from %s", directory)

    @property
    def oof_metrics(self) -> dict[str, ModelMetrics]:
        return self._oof_metrics

    def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        n_models = len(self.base_models)
        base_preds = np.zeros((X.shape[0], n_models))
        for i, model in enumerate(self.base_models):
            base_preds[:, i] = model.predict_proba(X)
        return base_preds

    def _weighted_average(self, base_preds: np.ndarray) -> np.ndarray:
        weights = np.array([
            self.config.weights.get(m.name, 0.25) for m in self.base_models
        ])
        weights = weights / weights.sum()
        return base_preds @ weights
