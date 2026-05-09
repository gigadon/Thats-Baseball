"""Base model interface and individual model wrappers.

Each wrapper provides a unified train/predict/evaluate interface
around XGBoost, LightGBM, Gradient Boosting, and Random Forest.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Evaluation metrics for a trained model."""

    accuracy: float
    brier_score: float
    log_loss: float
    auc_roc: float
    calibration_error: float  # Expected Calibration Error


@dataclass
class ModelConfig:
    """Hyperparameters for model training."""

    params: dict[str, Any] = field(default_factory=dict)
    name: str = ""


class BaseModel(ABC):
    """Abstract base for all prediction models."""

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.model: Any = None
        self.feature_names: list[str] = []
        self.is_fitted = False

    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying model instance."""

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None):
        """Train the model on features X and binary labels y."""
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.model = self._create_model()
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("%s trained on %d samples, %d features", self.name, X.shape[0], X.shape[1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return win probability for each sample (home team wins)."""
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted")
        probs = self.model.predict_proba(X)
        return probs[:, 1]  # Probability of class 1 (home win)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (1 = home win)."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Evaluate model on test data."""
        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)

        return ModelMetrics(
            accuracy=float(np.mean(preds == y)),
            brier_score=float(brier_score_loss(y, probs)),
            log_loss=float(log_loss(y, probs)),
            auc_roc=float(roc_auc_score(y, probs)),
            calibration_error=_expected_calibration_error(y, probs),
        )

    def feature_importance(self) -> dict[str, float]:
        """Return feature importances as {name: importance}."""
        if not self.is_fitted or not hasattr(self.model, "feature_importances_"):
            return {}
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))

    def save(self, path: str | Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_names": self.feature_names, "config": self.config},
            path,
        )
        logger.info("%s saved to %s", self.name, path)

    def load(self, path: str | Path):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.config = data["config"]
        self.is_fitted = True
        logger.info("%s loaded from %s", self.name, path)

    @property
    def name(self) -> str:
        return self.config.name or self.__class__.__name__


# ─── Model Implementations ────────────────────────────────────


class XGBoostModel(BaseModel):
    """XGBoost gradient boosting model. Weight: 0.35 in ensemble."""

    def __init__(self, config: ModelConfig | None = None):
        default = ModelConfig(
            name="XGBoost",
            params={
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": -1,
            },
        )
        if config:
            default.params.update(config.params)
            default.name = config.name or default.name
        super().__init__(default)

    def _create_model(self):
        import xgboost as xgb

        return xgb.XGBClassifier(**self.config.params)


class LightGBMModel(BaseModel):
    """LightGBM gradient boosting model. Weight: 0.25 in ensemble."""

    def __init__(self, config: ModelConfig | None = None):
        default = ModelConfig(
            name="LightGBM",
            params={
                "n_estimators": 500,
                "max_depth": 7,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_samples": 20,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "objective": "binary",
                "metric": "binary_logloss",
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            },
        )
        if config:
            default.params.update(config.params)
            default.name = config.name or default.name
        super().__init__(default)

    def _create_model(self):
        import lightgbm as lgb

        return lgb.LGBMClassifier(**self.config.params)


class GradientBoostingModel(BaseModel):
    """Sklearn Gradient Boosting model. Weight: 0.30 in ensemble."""

    def __init__(self, config: ModelConfig | None = None):
        default = ModelConfig(
            name="GradientBoosting",
            params={
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "min_samples_leaf": 10,
                "max_features": "sqrt",
                "random_state": 42,
            },
        )
        if config:
            default.params.update(config.params)
            default.name = config.name or default.name
        super().__init__(default)

    def _create_model(self):
        return GradientBoostingClassifier(**self.config.params)


class RandomForestModel(BaseModel):
    """Random Forest model. Weight: 0.10 in ensemble."""

    def __init__(self, config: ModelConfig | None = None):
        default = ModelConfig(
            name="RandomForest",
            params={
                "n_estimators": 500,
                "max_depth": 10,
                "min_samples_leaf": 10,
                "max_features": "sqrt",
                "random_state": 42,
                "n_jobs": -1,
            },
        )
        if config:
            default.params.update(config.params)
            default.name = config.name or default.name
        super().__init__(default)

    def _create_model(self):
        return RandomForestClassifier(**self.config.params)


# ─── Helpers ──────────────────────────────────────────────────


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error — measures how well probabilities match outcomes."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return float(ece)
