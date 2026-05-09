"""ML training pipeline.

End-to-end pipeline: load data → preprocess → select features → train → evaluate → save.
Supports incremental retraining every N games.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from mlb.models.ensemble import EnsembleConfig, EnsembleModel

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")


@dataclass
class PipelineConfig:
    """Training pipeline configuration."""

    test_size: float = 0.20
    val_size: float = 0.10
    feature_selection_threshold: float = 0.001  # Min importance to keep
    max_features: int = 200  # Cap features after selection
    impute_strategy: str = "knn"  # "knn" or "median"
    knn_neighbors: int = 5
    retrain_interval: int = 50  # Retrain every N new games
    model_dir: str = "models"
    ensemble_config: EnsembleConfig = field(default_factory=EnsembleConfig)


class TrainingPipeline:
    """End-to-end ML training pipeline."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.scaler: StandardScaler | None = None
        self.imputer: KNNImputer | None = None
        self.selected_features: list[str] | None = None
        self.ensemble: EnsembleModel | None = None
        self.run_metrics: dict[str, Any] = {}

    def train(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        game_dates: pd.Series | None = None,
    ) -> EnsembleModel:
        """Run the full training pipeline.

        Args:
            features_df: DataFrame where each row is a game, columns are feature names.
            target: Binary series (1 = home win, 0 = away win).
            game_dates: Optional dates for time-series aware splitting.

        Returns:
            Trained EnsembleModel.
        """
        logger.info(
            "Starting training pipeline: %d samples, %d features",
            len(features_df), features_df.shape[1],
        )

        # 1. Split
        X_train, X_test, y_train, y_test = self._split(features_df, target, game_dates)
        logger.info("Split: %d train, %d test", len(X_train), len(X_test))

        # 2. Impute missing values
        X_train, X_test = self._impute(X_train, X_test)

        # 3. Scale
        X_train_scaled, X_test_scaled = self._scale(X_train, X_test)

        # 4. Feature selection (train a quick model, keep important features)
        feature_names = list(features_df.columns)
        X_train_selected, X_test_selected, selected_names = self._select_features(
            X_train_scaled, X_test_scaled, y_train, feature_names
        )
        logger.info("Selected %d / %d features", len(selected_names), len(feature_names))

        # 5. Train ensemble
        self.ensemble = EnsembleModel(self.config.ensemble_config)
        self.ensemble.train(X_train_selected, y_train, selected_names)

        # 6. Evaluate
        metrics = self.ensemble.evaluate(X_test_selected, y_test)
        self.run_metrics = {name: vars(m) for name, m in metrics.items()}

        for name, m in metrics.items():
            logger.info(
                "%s — Acc: %.3f, Brier: %.4f, AUC: %.3f, ECE: %.4f",
                name, m.accuracy, m.brier_score, m.auc_roc, m.calibration_error,
            )

        # 7. Save
        self._save()

        return self.ensemble

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the trained pipeline."""
        if self.ensemble is None or not self.ensemble.is_fitted:
            raise RuntimeError("Pipeline not trained — call train() or load() first")

        X = features_df.values.astype(float)

        # Impute
        if self.imputer is not None:
            X = self.imputer.transform(X)

        # Scale
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Select features
        if self.selected_features is not None:
            col_idx = [
                list(features_df.columns).index(f)
                for f in self.selected_features
                if f in features_df.columns
            ]
            X = X[:, col_idx]

        return self.ensemble.predict_proba(X)

    def predict_detailed(self, features_df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Get detailed predictions from each model + ensemble."""
        if self.ensemble is None or not self.ensemble.is_fitted:
            raise RuntimeError("Pipeline not trained")

        X = self._preprocess_for_predict(features_df)
        return self.ensemble.predict_detailed(X)

    def load(self, model_dir: str | Path | None = None):
        """Load a trained pipeline from disk."""
        import joblib

        model_dir = Path(model_dir or self.config.model_dir)

        pipeline_meta = joblib.load(model_dir / "pipeline_meta.joblib")
        self.scaler = pipeline_meta["scaler"]
        self.imputer = pipeline_meta["imputer"]
        self.selected_features = pipeline_meta["selected_features"]
        self.run_metrics = pipeline_meta.get("run_metrics", {})

        self.ensemble = EnsembleModel()
        self.ensemble.load(model_dir / "ensemble")

        logger.info("Pipeline loaded from %s", model_dir)

    # ── Internal Steps ─────────────────────────────────────

    def _split(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        game_dates: pd.Series | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data. Uses time-based split if dates provided, else random."""
        X = features_df.values.astype(float)
        y = target.values.astype(int)

        if game_dates is not None:
            # Time-based split: train on older games, test on newer
            sorted_idx = game_dates.argsort()
            X, y = X[sorted_idx], y[sorted_idx]
            split_point = int(len(X) * (1 - self.config.test_size))
            return X[:split_point], X[split_point:], y[:split_point], y[split_point:]
        else:
            return train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=42,
                stratify=y,
            )

    def _impute(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Impute missing values."""
        if not np.any(np.isnan(X_train)):
            return X_train, X_test

        if self.config.impute_strategy == "knn":
            self.imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
        else:
            from sklearn.impute import SimpleImputer
            self.imputer = SimpleImputer(strategy="median")

        X_train = self.imputer.fit_transform(X_train)
        X_test = self.imputer.transform(X_test)
        return X_train, X_test

    def _scale(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Standardize features."""
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test

    def _select_features(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Select top features using a quick LightGBM importance scan."""
        import lightgbm as lgb

        selector = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            verbose=-1,
            random_state=42,
            n_jobs=-1,
        )
        selector.fit(X_train, y_train)
        importances = selector.feature_importances_

        # Normalize
        total = importances.sum() or 1.0
        normalized = importances / total

        # Keep features above threshold, up to max
        keep_mask = normalized >= self.config.feature_selection_threshold
        keep_indices = np.where(keep_mask)[0]

        # If too many, take top N by importance
        if len(keep_indices) > self.config.max_features:
            top_idx = np.argsort(normalized)[::-1][: self.config.max_features]
            keep_indices = np.sort(top_idx)

        # Always keep at least 20 features
        if len(keep_indices) < 20:
            keep_indices = np.argsort(normalized)[::-1][:20]

        self.selected_features = [feature_names[i] for i in keep_indices]

        return X_train[:, keep_indices], X_test[:, keep_indices], self.selected_features

    def _save(self):
        """Save pipeline artifacts."""
        import joblib

        model_dir = Path(self.config.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save pipeline metadata
        joblib.dump(
            {
                "scaler": self.scaler,
                "imputer": self.imputer,
                "selected_features": self.selected_features,
                "run_metrics": self.run_metrics,
            },
            model_dir / "pipeline_meta.joblib",
        )

        # Save ensemble
        if self.ensemble:
            self.ensemble.save(model_dir / "ensemble")

        logger.info("Pipeline saved to %s", model_dir)

    def _preprocess_for_predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Apply imputation, scaling, feature selection for prediction."""
        X = features_df.values.astype(float)

        if self.imputer is not None and np.any(np.isnan(X)):
            X = self.imputer.transform(X)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        if self.selected_features is not None:
            col_idx = [
                list(features_df.columns).index(f)
                for f in self.selected_features
                if f in features_df.columns
            ]
            X = X[:, col_idx]

        return X
