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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
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

    def tune_and_train(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        game_dates: pd.Series | None = None,
        n_trials: int = 50,
    ) -> EnsembleModel:
        """Run Optuna hyperparameter tuning, then train with best params."""
        import optuna
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        logger.info("Starting hyperparameter tuning (%d trials per model)...", n_trials)

        # Preprocess once
        X_train, X_test, y_train, y_test = self._split(features_df, target, game_dates)
        X_train, X_test = self._impute(X_train, X_test)
        X_train_scaled, X_test_scaled = self._scale(X_train, X_test)
        feature_names = list(features_df.columns)
        X_train_sel, X_test_sel, selected_names = self._select_features(
            X_train_scaled, X_test_scaled, y_train, feature_names
        )
        logger.info("Selected %d / %d features", len(selected_names), len(feature_names))

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        def _cv_auc(model_cls, params):
            """Manual CV to avoid sklearn tags compatibility issues."""
            aucs = []
            for tr_idx, va_idx in cv.split(X_train_sel, y_train):
                m = model_cls(**params)
                m.fit(X_train_sel[tr_idx], y_train[tr_idx])
                preds = m.predict_proba(X_train_sel[va_idx])[:, 1]
                aucs.append(roc_auc_score(y_train[va_idx], preds))
            return np.mean(aucs)

        # Tune XGBoost
        def xgb_objective(trial):
            import xgboost as xgb
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state": 42, "n_jobs": -1,
                "objective": "binary:logistic", "eval_metric": "logloss",
            }
            return _cv_auc(xgb.XGBClassifier, params)

        study_xgb = optuna.create_study(direction="maximize")
        study_xgb.optimize(xgb_objective, n_trials=n_trials)
        logger.info("XGBoost best AUC: %.4f", study_xgb.best_value)

        # Tune LightGBM
        def lgbm_objective(trial):
            import lightgbm as lgb
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state": 42, "n_jobs": -1,
                "objective": "binary", "metric": "binary_logloss", "verbose": -1,
            }
            return _cv_auc(lgb.LGBMClassifier, params)

        study_lgbm = optuna.create_study(direction="maximize")
        study_lgbm.optimize(lgbm_objective, n_trials=n_trials)
        logger.info("LightGBM best AUC: %.4f", study_lgbm.best_value)

        # Tune GradientBoosting
        def gb_objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
                "max_features": "sqrt",
                "random_state": 42,
            }
            return _cv_auc(GradientBoostingClassifier, params)

        study_gb = optuna.create_study(direction="maximize")
        study_gb.optimize(gb_objective, n_trials=n_trials)
        logger.info("GradientBoosting best AUC: %.4f", study_gb.best_value)

        # Tune CatBoost
        def catboost_objective(trial):
            from catboost import CatBoostClassifier
            params = {
                "depth": trial.suggest_int("depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "iterations": trial.suggest_int("iterations", 300, 800),
                "random_strength": trial.suggest_float("random_strength", 0.5, 2.0),
                "random_seed": 42, "verbose": 0,
            }
            return _cv_auc(CatBoostClassifier, params)

        study_catboost = optuna.create_study(direction="maximize")
        study_catboost.optimize(catboost_objective, n_trials=n_trials)
        logger.info("CatBoost best AUC: %.4f", study_catboost.best_value)

        # Tune RandomForest
        def rf_objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 30),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "random_state": 42, "n_jobs": -1,
            }
            return _cv_auc(RandomForestClassifier, params)

        study_rf = optuna.create_study(direction="maximize")
        study_rf.optimize(rf_objective, n_trials=n_trials)
        logger.info("RandomForest best AUC: %.4f", study_rf.best_value)

        # Tune NeuralNet (MLP wrapped in StandardScaler pipeline)
        def nn_objective(trial):
            layer_config = trial.suggest_categorical(
                "layer_config",
                ["128_64_32", "256_128_64", "128_64", "256_128", "64_32"],
            )
            layer_map = {
                "128_64_32": (128, 64, 32),
                "256_128_64": (256, 128, 64),
                "128_64": (128, 64),
                "256_128": (256, 128),
                "64_32": (64, 32),
            }
            hidden_layers = layer_map[layer_config]
            params = {
                "hidden_layer_sizes": hidden_layers,
                "activation": "relu",
                "solver": "adam",
                "learning_rate": "adaptive",
                "learning_rate_init": trial.suggest_float(
                    "learning_rate_init", 0.0001, 0.01, log=True
                ),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
                "max_iter": 500,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "random_state": 42,
            }
            # MLP needs scaled features — use a pipeline for CV evaluation
            aucs = []
            for tr_idx, va_idx in cv.split(X_train_sel, y_train):
                pipe = SklearnPipeline([
                    ("scaler", StandardScaler()),
                    ("mlp", MLPClassifier(**params)),
                ])
                pipe.fit(X_train_sel[tr_idx], y_train[tr_idx])
                preds = pipe.predict_proba(X_train_sel[va_idx])[:, 1]
                aucs.append(roc_auc_score(y_train[va_idx], preds))
            return np.mean(aucs)

        study_nn = optuna.create_study(direction="maximize")
        study_nn.optimize(nn_objective, n_trials=n_trials)
        logger.info("NeuralNet best AUC: %.4f", study_nn.best_value)

        # Resolve best hidden_layer_sizes from categorical label
        nn_best = dict(study_nn.best_params)
        layer_map = {
            "128_64_32": (128, 64, 32),
            "256_128_64": (256, 128, 64),
            "128_64": (128, 64),
            "256_128": (256, 128),
            "64_32": (64, 32),
        }
        nn_best["hidden_layer_sizes"] = layer_map[nn_best.pop("layer_config")]

        # Apply best params to the base models
        from mlb.models.base import ModelConfig
        best_configs = [
            ModelConfig(name="XGBoost", params={
                **study_xgb.best_params,
                "random_state": 42, "n_jobs": -1,
                "objective": "binary:logistic", "eval_metric": "logloss",
            }),
            ModelConfig(name="GradientBoosting", params={
                **study_gb.best_params,
                "max_features": "sqrt", "random_state": 42,
            }),
            ModelConfig(name="LightGBM", params={
                **study_lgbm.best_params,
                "random_state": 42, "n_jobs": -1,
                "objective": "binary", "metric": "binary_logloss", "verbose": -1,
            }),
            ModelConfig(name="CatBoost", params={
                **study_catboost.best_params,
                "random_seed": 42, "verbose": 0,
            }),
            ModelConfig(name="RandomForest", params={
                **study_rf.best_params,
                "random_state": 42, "n_jobs": -1,
            }),
            ModelConfig(name="NeuralNet", params={
                **nn_best,
                "activation": "relu",
                "solver": "adam",
                "learning_rate": "adaptive",
                "max_iter": 500,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "random_state": 42,
            }),
        ]

        from mlb.models.base import (
            XGBoostModel, GradientBoostingModel, LightGBMModel, CatBoostModel,
            RandomForestModel, NeuralNetModel,
        )
        self.ensemble = EnsembleModel(self.config.ensemble_config)
        self.ensemble.base_models = [
            XGBoostModel(best_configs[0]),
            GradientBoostingModel(best_configs[1]),
            LightGBMModel(best_configs[2]),
            CatBoostModel(best_configs[3]),
            RandomForestModel(best_configs[4]),
            NeuralNetModel(best_configs[5]),
        ]

        # Train ensemble with tuned models
        self.ensemble.train(X_train_sel, y_train, selected_names)

        # Evaluate
        metrics = self.ensemble.evaluate(X_test_sel, y_test)
        self.run_metrics = {name: vars(m) for name, m in metrics.items()}

        for name, m in metrics.items():
            logger.info(
                "%s — Acc: %.3f, Brier: %.4f, AUC: %.3f, ECE: %.4f",
                name, m.accuracy, m.brier_score, m.auc_roc, m.calibration_error,
            )

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
