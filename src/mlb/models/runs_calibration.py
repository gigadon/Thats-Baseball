"""Over/under probability calibration for the runs (totals) model.

Single source of truth for turning a *predicted total* into a probability of the
game going OVER / UNDER a line. Both the live betting engine
(``betting/engine.py``) and the totals backtester (``models/backtest_totals.py``)
call this so they always agree.

Two modes:
  * **Empirical residual CDF** (preferred) — uses the model's held-out residuals
    (actual_total − predicted_total, persisted at train time). This is honest
    about the runs model's real, skewed, discrete error distribution and handles
    integer-line push mass correctly.
  * **Normal fallback** — ``N(0, residual_std)`` when no residual array is
    available (e.g. an older model artifact). ``residual_std`` is the held-out
    RMSE (~4.45 runs for the current model), NOT the stale 4.1 that used to be
    hardcoded in the engine.
"""

from __future__ import annotations

import numpy as np

# Held-out RMSE of the current runs model, used only when a model artifact
# predates residual persistence. Kept in one place so it can't drift out of sync
# across call sites the way the old hardcoded 4.1 did.
DEFAULT_RESIDUAL_STD = 4.45


def over_under_probabilities(
    predicted_total: float,
    line: float,
    residuals: np.ndarray | list[float] | None = None,
    residual_std: float | None = None,
) -> tuple[float, float]:
    """Return ``(p_over, p_under)`` for ``predicted_total`` vs a totals ``line``.

    Model:  actual_total = predicted_total + residual.
        OVER  ⇔ residual > line − predicted_total  (= −diff)
        UNDER ⇔ residual < line − predicted_total

    With the empirical CDF, ``p_over + p_under`` can be < 1 — the remainder is the
    push mass at integer lines. With the Normal fallback they sum to 1 (no push
    mass in a continuous distribution).
    """
    diff = float(predicted_total) - float(line)

    resid = None if residuals is None else np.asarray(residuals, dtype=float)
    if resid is not None and resid.size > 0:
        p_over = float(np.mean(resid > -diff))
        p_under = float(np.mean(resid < -diff))
        return p_over, p_under

    from scipy.stats import norm

    sd = residual_std if (residual_std and residual_std > 0) else DEFAULT_RESIDUAL_STD
    p_over = float(norm.sf(-diff / sd))
    return p_over, 1.0 - p_over
