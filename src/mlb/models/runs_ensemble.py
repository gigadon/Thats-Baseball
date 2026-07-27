"""RunsEnsemble — total-runs prediction wrapper (LightGBM + XGBoost).

Kept in its own module (not in train_runs.py) so the pickled artifact always
references a stable, importable class path. When training is launched via
``python -m mlb.models.train_runs`` the train_runs module is ``__main__``, so a
class defined there would pickle as ``__main__.RunsEnsemble`` and fail to load
in any other process (e.g. the prediction service) — which silently fell back
to the heuristic run estimator.
"""

from __future__ import annotations

import numpy as np


class RunsEnsemble:
    """Ensemble wrapper combining LightGBM + XGBoost for runs prediction.

    The underlying models predict deviation from market_total; this wrapper
    adds market_total back to produce final total runs.

    Parameters calibrated via CV:
        alpha: weight for LightGBM (1-alpha for XGBoost)
        shrinkage: how much to trust the model's deviation
                   (0 = just use market_total, 1 = full model)

    Uncertainty (set at train time from held-out residuals so the model
    self-describes how noisy its total is — consumed by the betting engine to
    turn a predicted total into a calibrated over/under probability):
        residual_std: std of (actual_total - predicted_total) on the test set
        residuals:    the sorted held-out residual array (empirical CDF)
    """

    def __init__(self, lgbm, xgb, market_total_idx: int | None = None,
                 alpha: float = 0.5, shrinkage: float = 1.0,
                 residual_std: float | None = None,
                 residuals: list[float] | None = None):
        self.lgbm = lgbm
        self.xgb = xgb
        self.market_total_idx = market_total_idx
        self.alpha = alpha
        self.shrinkage = shrinkage
        self.residual_std = residual_std
        self.residuals = residuals

    def predict(self, X):
        # LightGBM handles NaN; XGBoost needs clean input
        X_clean = np.nan_to_num(X, nan=0.0)
        p_lgbm = self.lgbm.predict(X)
        p_xgb = self.xgb.predict(X_clean)
        deviation = self.alpha * p_lgbm + (1 - self.alpha) * p_xgb
        deviation = deviation * self.shrinkage

        # Add market_total back to get absolute total runs
        if self.market_total_idx is not None:
            market_total = np.nan_to_num(X[:, self.market_total_idx], nan=8.5)
            return deviation + market_total
        return deviation + 8.5  # fallback to league average
