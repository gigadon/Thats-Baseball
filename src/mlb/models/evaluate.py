"""Model evaluation, calibration analysis, and backtesting.

Provides:
  - Rolling accuracy tracking
  - Brier score decomposition
  - Calibration curves
  - Historical backtest runner
  - ROI simulation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CalibrationBin:
    """One bin in a calibration curve."""

    bin_lower: float
    bin_upper: float
    predicted_avg: float
    observed_avg: float
    count: int


@dataclass
class EvaluationReport:
    """Comprehensive evaluation metrics."""

    # Overall
    accuracy: float
    brier_score: float
    log_loss_val: float
    auc_roc: float

    # Calibration
    calibration_error: float  # ECE
    calibration_bins: list[CalibrationBin]

    # By confidence tier
    high_confidence_acc: float  # Predictions with prob > 0.60 or < 0.40
    medium_confidence_acc: float  # 0.45-0.55
    low_confidence_acc: float  # 0.40-0.60

    # Rolling
    rolling_30d_accuracy: list[tuple[str, float]]  # (date, acc)
    rolling_30d_brier: list[tuple[str, float]]

    # Profit simulation (flat $100 bets on predicted winner)
    flat_bet_roi: float
    total_games: int
    correct_predictions: int


@dataclass
class BacktestResult:
    """Results from a historical backtest."""

    start_date: str
    end_date: str
    total_games: int
    total_bets: int

    # Performance
    win_rate: float
    roi_flat: float
    roi_kelly: float
    max_drawdown: float
    sharpe_ratio: float

    # Cumulative P&L
    cumulative_pnl: list[float]
    daily_pnl: list[tuple[str, float]]

    # By bet type
    moneyline_roi: float
    totals_roi: float

    # Monthly breakdown
    monthly_results: list[dict[str, Any]]


class ModelEvaluator:
    """Evaluate model predictions against actual results."""

    def evaluate(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        dates: np.ndarray | None = None,
    ) -> EvaluationReport:
        """Generate a comprehensive evaluation report.

        Args:
            predictions: Array of predicted home win probabilities.
            actuals: Array of actual outcomes (1 = home win).
            dates: Optional array of game dates for rolling metrics.
        """
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

        preds_binary = (predictions >= 0.5).astype(int)
        n = len(predictions)

        # Overall metrics
        accuracy = float(np.mean(preds_binary == actuals))
        brier = float(brier_score_loss(actuals, predictions))
        ll = float(log_loss(actuals, np.clip(predictions, 1e-7, 1 - 1e-7)))
        auc = float(roc_auc_score(actuals, predictions))

        # Calibration
        cal_bins = self._calibration_curve(predictions, actuals)
        ece = self._ece(predictions, actuals)

        # Confidence tiers
        high_mask = (predictions > 0.60) | (predictions < 0.40)
        med_mask = (predictions >= 0.45) & (predictions <= 0.55)
        low_mask = (predictions >= 0.40) & (predictions <= 0.60)

        high_acc = float(np.mean(preds_binary[high_mask] == actuals[high_mask])) if high_mask.any() else 0.0
        med_acc = float(np.mean(preds_binary[med_mask] == actuals[med_mask])) if med_mask.any() else 0.0
        low_acc = float(np.mean(preds_binary[low_mask] == actuals[low_mask])) if low_mask.any() else 0.0

        # Rolling metrics
        rolling_acc, rolling_brier = [], []
        if dates is not None:
            rolling_acc, rolling_brier = self._rolling_metrics(predictions, actuals, dates)

        # Flat bet ROI: bet $100 on predicted winner at -110 odds
        correct = int(np.sum(preds_binary == actuals))
        # At -110: win $90.91, lose $100
        profit = correct * 90.91 - (n - correct) * 100
        roi = profit / (n * 100) if n > 0 else 0.0

        return EvaluationReport(
            accuracy=accuracy,
            brier_score=brier,
            log_loss_val=ll,
            auc_roc=auc,
            calibration_error=ece,
            calibration_bins=cal_bins,
            high_confidence_acc=high_acc,
            medium_confidence_acc=med_acc,
            low_confidence_acc=low_acc,
            rolling_30d_accuracy=rolling_acc,
            rolling_30d_brier=rolling_brier,
            flat_bet_roi=roi,
            total_games=n,
            correct_predictions=correct,
        )

    def backtest(
        self,
        predictions_df: pd.DataFrame,
        strategy: str = "flat",
        bankroll: float = 10000.0,
        kelly_fraction: float = 0.25,
    ) -> BacktestResult:
        """Run a historical backtest on predictions.

        Args:
            predictions_df: DataFrame with columns:
                date, home_win_prob, actual_result, home_odds, away_odds,
                total_line, actual_total
            strategy: "flat" or "kelly"
            bankroll: Starting bankroll.
            kelly_fraction: Fraction of full Kelly to use.
        """
        df = predictions_df.sort_values("date").reset_index(drop=True)
        n = len(df)

        current_bankroll = bankroll
        cumulative_pnl = [0.0]
        daily_pnl: list[tuple[str, float]] = []
        monthly: dict[str, dict] = {}

        total_bets = 0
        wins = 0
        ml_bets = ml_wins = ml_profit = 0
        tot_bets_count = tot_wins = tot_profit = 0
        max_drawdown = 0.0
        peak = bankroll

        for _, row in df.iterrows():
            game_date = str(row["date"])
            prob = row["home_win_prob"]
            actual = row["actual_result"]
            home_odds = row.get("home_odds", -110)
            away_odds = row.get("away_odds", -110)

            # Determine bet side
            if prob >= 0.52:  # Only bet when edge exists
                bet_side = "home"
                odds = home_odds
                win = actual == 1
            elif prob <= 0.48:
                bet_side = "away"
                odds = away_odds
                win = actual == 0
            else:
                continue  # No edge, skip

            # Calculate stake
            if strategy == "kelly":
                implied_prob = _american_to_implied(odds)
                edge = (prob if bet_side == "home" else 1 - prob) - implied_prob
                if edge <= 0:
                    continue
                decimal_odds = _american_to_decimal(odds)
                kelly = edge / (decimal_odds - 1)
                stake = current_bankroll * kelly * kelly_fraction
                stake = min(stake, current_bankroll * 0.05)  # Cap at 5% of bankroll
            else:
                stake = 100.0  # Flat bet

            if stake <= 0 or current_bankroll < stake:
                continue

            # Settle bet
            total_bets += 1
            ml_bets += 1

            if win:
                payout = stake * (_american_to_decimal(odds) - 1)
                wins += 1
                ml_wins += 1
                ml_profit += payout
            else:
                payout = -stake
                ml_profit -= stake

            current_bankroll += payout
            cumulative_pnl.append(current_bankroll - bankroll)

            # Track drawdown
            peak = max(peak, current_bankroll)
            dd = (peak - current_bankroll) / peak
            max_drawdown = max(max_drawdown, dd)

            # Daily
            daily_pnl.append((game_date, payout))

            # Monthly
            month_key = game_date[:7]
            if month_key not in monthly:
                monthly[month_key] = {"month": month_key, "bets": 0, "wins": 0, "pnl": 0.0}
            monthly[month_key]["bets"] += 1
            monthly[month_key]["wins"] += 1 if win else 0
            monthly[month_key]["pnl"] += payout

        # Calculate Sharpe ratio
        daily_returns = [p for _, p in daily_pnl]
        if len(daily_returns) > 1:
            mean_ret = np.mean(daily_returns)
            std_ret = np.std(daily_returns)
            sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        total_wagered = total_bets * (100 if strategy == "flat" else 0)
        if strategy == "kelly":
            total_wagered = sum(abs(p) for _, p in daily_pnl if p < 0) + sum(
                p / (_american_to_decimal(-110) - 1) for _, p in daily_pnl if p > 0
            )

        roi_flat = ml_profit / (ml_bets * 100) if ml_bets > 0 else 0.0
        roi_kelly = (current_bankroll - bankroll) / bankroll if strategy == "kelly" else roi_flat

        return BacktestResult(
            start_date=str(df["date"].iloc[0]) if n > 0 else "",
            end_date=str(df["date"].iloc[-1]) if n > 0 else "",
            total_games=n,
            total_bets=total_bets,
            win_rate=wins / total_bets if total_bets > 0 else 0.0,
            roi_flat=roi_flat,
            roi_kelly=roi_kelly,
            max_drawdown=max_drawdown,
            sharpe_ratio=float(sharpe),
            cumulative_pnl=cumulative_pnl,
            daily_pnl=daily_pnl,
            moneyline_roi=ml_profit / (ml_bets * 100) if ml_bets > 0 else 0.0,
            totals_roi=0.0,  # Totals backtest not yet implemented
            monthly_results=list(monthly.values()),
        )

    # ── Helpers ────────────────────────────────────────────

    def _calibration_curve(
        self, predictions: np.ndarray, actuals: np.ndarray, n_bins: int = 10
    ) -> list[CalibrationBin]:
        bins = []
        edges = np.linspace(0, 1, n_bins + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (predictions >= lo) & (predictions < hi)
            count = int(mask.sum())
            if count == 0:
                bins.append(CalibrationBin(lo, hi, (lo + hi) / 2, 0.0, 0))
            else:
                bins.append(CalibrationBin(
                    bin_lower=lo,
                    bin_upper=hi,
                    predicted_avg=float(predictions[mask].mean()),
                    observed_avg=float(actuals[mask].mean()),
                    count=count,
                ))
        return bins

    def _ece(self, predictions: np.ndarray, actuals: np.ndarray, n_bins: int = 10) -> float:
        edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (predictions >= lo) & (predictions < hi)
            if mask.sum() == 0:
                continue
            bin_acc = actuals[mask].mean()
            bin_conf = predictions[mask].mean()
            ece += mask.sum() / len(predictions) * abs(bin_acc - bin_conf)
        return float(ece)

    def _rolling_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        dates: np.ndarray,
        window: int = 30,
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        df = pd.DataFrame({
            "date": dates,
            "pred": predictions,
            "actual": actuals,
            "correct": ((predictions >= 0.5).astype(int) == actuals).astype(int),
        }).sort_values("date")

        from sklearn.metrics import brier_score_loss

        rolling_acc = []
        rolling_brier = []

        for i in range(window, len(df)):
            window_df = df.iloc[i - window : i]
            d = str(window_df["date"].iloc[-1])
            acc = float(window_df["correct"].mean())
            brier = float(brier_score_loss(window_df["actual"], window_df["pred"]))
            rolling_acc.append((d, acc))
            rolling_brier.append((d, brier))

        return rolling_acc, rolling_brier


# ─── Odds Conversion Helpers ──────────────────────────────────


def _american_to_decimal(odds: float) -> float:
    if odds > 0:
        return 1 + odds / 100
    else:
        return 1 + 100 / abs(odds)


def _american_to_implied(odds: float) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)
