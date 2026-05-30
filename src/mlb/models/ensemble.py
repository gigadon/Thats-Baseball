"""Stacking ensemble with logistic regression meta-learner.

Architecture:
    Layer 1: XGBoost (0.28), LightGBM (0.28), CatBoost (0.22), GBM (0.22)
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
    LogisticRegressionModel,
    ModelMetrics,
    NeuralNetModel,
    RandomForestModel,
    XGBoostModel,
    _expected_calibration_error,
)

# NeuralNetModel and GradientBoostingModel still imported for load() compatibility

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble."""

    mode: str = "stacking"  # "stacking" or "weighted_average"
    weights: dict[str, float] = field(default_factory=lambda: {
        "XGBoost": 0.25,
        "LightGBM": 0.25,
        "CatBoost": 0.20,
        "RandomForest": 0.15,
        "LogisticRegression": 0.15,
    })
    cv_folds: int = 5
    random_state: int = 42


class EnsembleModel:
    """Stacking ensemble that combines multiple base models."""

    def __init__(self, config: EnsembleConfig | None = None):
        self.config = config or EnsembleConfig()
        self.base_models: list[BaseModel] = [
            XGBoostModel(),
            LightGBMModel(),
            CatBoostModel(),
            RandomForestModel(),
            LogisticRegressionModel(),
        ]
        self.meta_model: LogisticRegression | None = None
        self.calibrator: CalibratedClassifierCV | None = None
        self.feature_names: list[str] = []
        self.is_fitted = False
        self._oof_metrics: dict[str, ModelMetrics] = {}
        # Fold-trained models: fold_models[model_idx][fold_idx] = BaseModel
        # Used at inference to average predictions (bagging), avoiding
        # the leakage of retraining on full data after OOF stacking.
        self.fold_models: list[list[BaseModel]] = []

    def train(
        self, X: np.ndarray, y: np.ndarray,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ):
        """Train the full ensemble.

        1. Generate out-of-fold predictions from each base model
        2. Train meta-learner on stacked OOF predictions
        3. Retrain each base model on the full dataset
        """
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self._sample_weight = sample_weight
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
        """Train with stacking: OOF predictions → meta-learner.

        Fold-trained models are kept for inference (bagged predictions)
        instead of retraining on full data, which would cause leakage
        between the meta-learner's calibration and the base model outputs.
        """
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        oof_preds = np.zeros((n_samples, n_models))

        kf = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        # Generate OOF predictions and store fold-trained models
        sw = self._sample_weight
        self.fold_models = [[] for _ in range(n_models)]

        for model_idx, model in enumerate(self.base_models):
            logger.info("Generating OOF predictions for %s", model.name)
            fold_preds = np.zeros(n_samples)

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]
                sw_train = sw[train_idx] if sw is not None else None

                # Clone model for this fold and keep it
                fold_model = model.__class__()
                fold_model.train(X_train, y_train, self.feature_names, sample_weight=sw_train)
                fold_preds[val_idx] = fold_model.predict_proba(X_val)
                self.fold_models[model_idx].append(fold_model)

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

        # Tune meta-learner: compare LogisticRegression vs MLP
        from sklearn.metrics import brier_score_loss
        from sklearn.neural_network import MLPClassifier

        candidates = {}

        # 1) LogisticRegression with C tuning
        for c_val in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
            cv_briers = []
            for tr_idx, val_idx in StratifiedKFold(
                n_splits=5, shuffle=True, random_state=42
            ).split(oof_preds, y):
                m = LogisticRegression(
                    C=c_val, max_iter=1000, random_state=self.config.random_state
                )
                sw_tr = sw[tr_idx] if sw is not None else None
                m.fit(oof_preds[tr_idx], y[tr_idx], sample_weight=sw_tr)
                probs = m.predict_proba(oof_preds[val_idx])[:, 1]
                cv_briers.append(brier_score_loss(y[val_idx], probs))
            candidates[f"LR_C={c_val}"] = (np.mean(cv_briers), "lr", c_val)

        # 2) MLP with different architectures
        for hidden in [(8,), (16,), (8, 4), (16, 8)]:
            for alpha in [0.01, 0.1, 1.0]:
                cv_briers = []
                for tr_idx, val_idx in StratifiedKFold(
                    n_splits=5, shuffle=True, random_state=42
                ).split(oof_preds, y):
                    m = MLPClassifier(
                        hidden_layer_sizes=hidden, alpha=alpha,
                        max_iter=500, random_state=self.config.random_state,
                        early_stopping=True, validation_fraction=0.15,
                    )
                    m.fit(oof_preds[tr_idx], y[tr_idx])
                    probs = m.predict_proba(oof_preds[val_idx])[:, 1]
                    cv_briers.append(brier_score_loss(y[val_idx], probs))
                candidates[f"MLP_{hidden}_a={alpha}"] = (np.mean(cv_briers), "mlp", (hidden, alpha))

        # Pick best
        best_name = min(candidates, key=lambda k: candidates[k][0])
        best_brier, best_type, best_param = candidates[best_name]
        logger.info("Meta-learner comparison (top 5):")
        for name, (brier, _, _) in sorted(candidates.items(), key=lambda x: x[1][0])[:5]:
            logger.info("  %s: Brier=%.4f%s", name, brier, " <-- best" if name == best_name else "")

        # Train final meta-learner
        if best_type == "mlp":
            hidden, alpha = best_param
            self.meta_model = MLPClassifier(
                hidden_layer_sizes=hidden, alpha=alpha,
                max_iter=500, random_state=self.config.random_state,
                early_stopping=True, validation_fraction=0.15,
            )
            self.meta_model.fit(oof_preds, y)
            logger.info("Meta-model: MLP %s alpha=%.2f (Brier=%.4f)", hidden, alpha, best_brier)
        else:
            self.meta_model = LogisticRegression(
                C=best_param, max_iter=1000, random_state=self.config.random_state
            )
            self.meta_model.fit(oof_preds, y, sample_weight=sw)
            logger.info("Meta-model: LogisticRegression C=%.4f (Brier=%.4f)", best_param, best_brier)

        # Calibrate: fit isotonic regression on meta-model OOF outputs
        meta_probs = self.meta_model.predict_proba(oof_preds)[:, 1]
        self._fit_calibrator(meta_probs, y)

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
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

        results: dict[str, ModelMetrics] = {}

        # Individual models (use bagged fold predictions)
        base_preds = self._get_base_predictions(X)
        for i, model in enumerate(self.base_models):
            probs = base_preds[:, i]
            preds = (probs >= 0.5).astype(int)
            results[model.name] = ModelMetrics(
                accuracy=float(np.mean(preds == y)),
                brier_score=float(brier_score_loss(y, probs)),
                log_loss=float(log_loss(y, np.clip(probs, 1e-7, 1 - 1e-7))),
                auc_roc=float(roc_auc_score(y, probs)),
                calibration_error=_expected_calibration_error(y, probs),
            )

        # Ensemble
        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)
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

        for model_idx, model in enumerate(self.base_models):
            w = weights.get(model.name, 0.25)
            if self.fold_models and model_idx < len(self.fold_models):
                # Average importance across fold models
                fold_imps: list[dict[str, float]] = [
                    fm.feature_importance() for fm in self.fold_models[model_idx]
                ]
                merged: dict[str, float] = {}
                for fi in fold_imps:
                    for feat, val in fi.items():
                        merged[feat] = merged.get(feat, 0.0) + val / len(fold_imps)
                imp = merged
            else:
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

        # Save fold-trained models (preferred) or single models (fallback)
        if self.fold_models:
            for model_idx, model in enumerate(self.base_models):
                for fold_idx, fold_model in enumerate(self.fold_models[model_idx]):
                    fold_model.save(directory / f"{model.name}_fold{fold_idx}.joblib")
        else:
            for model in self.base_models:
                model.save(directory / f"{model.name}.joblib")

        if self.meta_model is not None:
            joblib.dump(self.meta_model, directory / "meta_model.joblib")

        if self.calibrator is not None:
            joblib.dump(self.calibrator, directory / "calibrator.joblib")

        n_folds = len(self.fold_models[0]) if self.fold_models else 0
        joblib.dump(
            {
                "config": self.config,
                "feature_names": self.feature_names,
                "oof_metrics": self._oof_metrics,
                "n_folds": n_folds,
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
        n_folds = meta.get("n_folds", 0)

        # Try loading fold-trained models first, fall back to single models
        if n_folds > 0:
            self.fold_models = []
            for model in self.base_models:
                folds = []
                for k in range(n_folds):
                    fold_model = model.__class__()
                    fold_model.load(directory / f"{model.name}_fold{k}.joblib")
                    folds.append(fold_model)
                self.fold_models.append(folds)
        else:
            # Backward compat: load single models
            self.fold_models = []
            for model in self.base_models:
                path = directory / f"{model.name}.joblib"
                if path.exists():
                    model.load(path)

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

        if self.fold_models:
            # Average predictions from all K fold-trained models (bagging)
            for i in range(n_models):
                fold_preds = np.array([m.predict_proba(X) for m in self.fold_models[i]])
                base_preds[:, i] = fold_preds.mean(axis=0)
        else:
            # Backward compat: single model per slot (old format)
            for i, model in enumerate(self.base_models):
                base_preds[:, i] = model.predict_proba(X)

        return base_preds

    def _weighted_average(self, base_preds: np.ndarray) -> np.ndarray:
        weights = np.array([
            self.config.weights.get(m.name, 0.25) for m in self.base_models
        ])
        weights = weights / weights.sum()
        return base_preds @ weights
