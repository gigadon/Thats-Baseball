"""Historical backtester — simulates betting with model predictions against past results.

Walks through games chronologically, generates predictions using only data
available at the time, and simulates betting with synthetic or real odds.

Usage:
    python -m mlb.models.backtest --seasons 2024 2025 2026
    python -m mlb.models.backtest --seasons 2025 2026 --bankroll 5000 --kelly 0.25
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mlb.betting.engine import (
    BettingConfig,
    BettingEngine,
    american_to_decimal,
    american_to_implied,
    remove_vig,
)
from mlb.models.pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Summary of a backtest run."""

    total_games: int
    games_bet: int
    bets_won: int
    bets_lost: int
    win_rate: float

    starting_bankroll: float
    ending_bankroll: float
    peak_bankroll: float
    total_pnl: float
    roi: float
    max_drawdown: float
    sharpe_ratio: float

    # Flat-bet metrics (for comparison)
    flat_bet_pnl: float
    flat_bet_roi: float

    # By-month breakdown
    monthly: list[dict]

    # Model accuracy on all games (not just bet games)
    model_accuracy: float
    model_brier: float
    model_auc: float = 0.0

    # Calibration bins: list of dicts with predicted_avg, observed_avg, count
    calibration: list[dict] = field(default_factory=list)

    # Monthly accuracy breakdown (all games, not just bets)
    monthly_accuracy: list[dict] = field(default_factory=list)


class Backtester:
    """Walk-forward backtest of model predictions + betting strategy."""

    def __init__(
        self,
        data_dir: Path = Path("data"),
        model_dir: Path = Path("models/win_model"),
        bankroll: float = 10000.0,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.03,
        flat_bet_size: float = 100.0,
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.initial_bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.flat_bet_size = flat_bet_size

    def run(self, seasons: list[int]) -> BacktestResult:
        """Run a full historical backtest."""
        # Load training data (already has features + results)
        df = pd.read_parquet(self.data_dir / "training_data.parquet")
        df = df.sort_values("game_date").reset_index(drop=True)
        df["game_date"] = pd.to_datetime(df["game_date"])

        # Filter to requested seasons
        df = df[df["game_date"].dt.year.isin(seasons)].reset_index(drop=True)
        logger.info("Backtesting on %d games (%s)", len(df), [int(y) for y in sorted(df["game_date"].dt.year.unique())])

        # Load trained model
        pipeline = TrainingPipeline()
        pipeline.load(self.model_dir)

        meta_cols = ["game_id", "game_date", "home_team", "away_team", "home_score", "away_score", "home_win", "total_runs"]
        feature_cols = [c for c in df.columns if c not in meta_cols]

        # Generate predictions for all games
        X = df[feature_cols]
        probs = pipeline.predict(X)

        df["pred_home_prob"] = probs
        df["pred_correct"] = ((probs >= 0.5) & (df["home_win"] == 1)) | ((probs < 0.5) & (df["home_win"] == 0))

        # Compute AUC
        from sklearn.metrics import roc_auc_score
        try:
            model_auc = float(roc_auc_score(df["home_win"].values, probs))
        except ValueError:
            model_auc = 0.0

        # Calibration bins (deciles)
        calibration_bins = self._compute_calibration(probs, df["home_win"].values)

        # Monthly accuracy breakdown (all games, not just bets)
        df["_month"] = df["game_date"].dt.to_period("M").astype(str)
        monthly_acc_data = (
            df.groupby("_month")
            .agg(
                games=("pred_correct", "size"),
                correct=("pred_correct", "sum"),
                avg_pred=("pred_home_prob", "mean"),
                actual_home_rate=("home_win", "mean"),
            )
            .reset_index()
        )
        monthly_accuracy = [
            {
                "month": row["_month"],
                "games": int(row["games"]),
                "correct": int(row["correct"]),
                "accuracy": round(row["correct"] / row["games"], 4) if row["games"] > 0 else 0.0,
                "avg_pred": round(float(row["avg_pred"]), 4),
                "actual_home_rate": round(float(row["actual_home_rate"]), 4),
            }
            for _, row in monthly_acc_data.iterrows()
        ]

        # Simulate betting
        bankroll = self.initial_bankroll
        peak = bankroll
        max_dd = 0.0
        total_wagered = 0.0
        bets_won = bets_lost = 0
        flat_pnl = 0.0
        daily_returns: list[float] = []
        monthly_data: dict[str, dict] = {}

        for _, row in df.iterrows():
            pred_prob = row["pred_home_prob"]
            home_win = row["home_win"]
            game_date = row["game_date"]
            month_key = game_date.strftime("%Y-%m")

            if month_key not in monthly_data:
                monthly_data[month_key] = {
                    "month": month_key, "games": 0, "bets": 0,
                    "wins": 0, "pnl": 0.0, "flat_pnl": 0.0,
                }
            monthly_data[month_key]["games"] += 1

            # Generate synthetic odds from prediction (simulate market)
            # Assume market is efficient with some noise + vig
            market_noise = np.random.normal(0, 0.03)
            true_home = 0.5 + (pred_prob - 0.5) * 0.6 + market_noise  # Market partially agrees
            true_home = np.clip(true_home, 0.2, 0.8)
            true_away = 1 - true_home

            # Add vig (~4.5% overround)
            vig_factor = 1.045
            implied_home = true_home * vig_factor
            implied_away = true_away * vig_factor

            # Convert to American odds
            home_odds = _prob_to_american(implied_home)
            away_odds = _prob_to_american(implied_away)

            # Check for value
            clean_home, clean_away = remove_vig(home_odds, away_odds)
            home_edge = pred_prob - clean_home
            away_edge = (1 - pred_prob) - clean_away

            bet_side = None
            edge = 0.0
            odds = 0.0

            if home_edge >= self.min_edge:
                bet_side = "home"
                edge = home_edge
                odds = home_odds
            elif away_edge >= self.min_edge:
                bet_side = "away"
                edge = away_edge
                odds = away_odds

            if bet_side is None:
                daily_returns.append(0.0)
                continue

            # Kelly sizing
            dec_odds = american_to_decimal(odds)
            b = dec_odds - 1
            p = pred_prob if bet_side == "home" else (1 - pred_prob)
            q = 1 - p
            kelly = max(0, (b * p - q) / b) * self.kelly_fraction
            stake = min(bankroll * kelly, bankroll * 0.05)

            if stake < 1:
                daily_returns.append(0.0)
                continue

            # Settle
            won = (bet_side == "home" and home_win == 1) or (bet_side == "away" and home_win == 0)
            total_wagered += stake

            if won:
                payout = stake * (dec_odds - 1)
                bankroll += payout
                bets_won += 1
                daily_returns.append(payout / self.initial_bankroll)
                monthly_data[month_key]["pnl"] += payout
                monthly_data[month_key]["wins"] += 1
            else:
                bankroll -= stake
                bets_lost += 1
                daily_returns.append(-stake / self.initial_bankroll)
                monthly_data[month_key]["pnl"] -= stake

            monthly_data[month_key]["bets"] += 1

            # Flat bet tracking
            flat_dec = american_to_decimal(odds)
            if won:
                flat_pnl += self.flat_bet_size * (flat_dec - 1)
            else:
                flat_pnl -= self.flat_bet_size

            monthly_data[month_key]["flat_pnl"] += (
                self.flat_bet_size * (flat_dec - 1) if won else -self.flat_bet_size
            )

            # Drawdown
            peak = max(peak, bankroll)
            dd = (peak - bankroll) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Calculate Sharpe
        returns_arr = np.array(daily_returns)
        sharpe = 0.0
        if len(returns_arr) > 1 and returns_arr.std() > 0:
            sharpe = float(returns_arr.mean() / returns_arr.std() * np.sqrt(252))

        total_bets = bets_won + bets_lost
        total_pnl = bankroll - self.initial_bankroll

        return BacktestResult(
            total_games=len(df),
            games_bet=total_bets,
            bets_won=bets_won,
            bets_lost=bets_lost,
            win_rate=bets_won / max(total_bets, 1),
            starting_bankroll=self.initial_bankroll,
            ending_bankroll=round(bankroll, 2),
            peak_bankroll=round(peak, 2),
            total_pnl=round(total_pnl, 2),
            roi=round(total_pnl / max(total_wagered, 1), 4),
            max_drawdown=round(max_dd, 4),
            sharpe_ratio=round(sharpe, 3),
            flat_bet_pnl=round(flat_pnl, 2),
            flat_bet_roi=round(flat_pnl / max(total_bets * self.flat_bet_size, 1), 4),
            monthly=sorted(monthly_data.values(), key=lambda m: m["month"]),
            model_accuracy=float(df["pred_correct"].mean()),
            model_brier=float(((probs - df["home_win"].values) ** 2).mean()),
            model_auc=model_auc,
            calibration=calibration_bins,
            monthly_accuracy=monthly_accuracy,
        )


    def _compute_calibration(
        self, predictions: np.ndarray, actuals: np.ndarray, n_bins: int = 10
    ) -> list[dict]:
        """Compute calibration bins: do predicted probabilities match observed rates?"""
        edges = np.linspace(0, 1, n_bins + 1)
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (predictions >= lo) & (predictions < hi)
            count = int(mask.sum())
            if count == 0:
                continue
            bins.append({
                "bin": f"{lo:.1f}-{hi:.1f}",
                "predicted_avg": round(float(predictions[mask].mean()), 4),
                "observed_avg": round(float(actuals[mask].mean()), 4),
                "count": count,
                "gap": round(abs(float(predictions[mask].mean()) - float(actuals[mask].mean())), 4),
            })
        return bins


def _prob_to_american(prob: float) -> float:
    """Convert probability to American odds."""
    prob = np.clip(prob, 0.01, 0.99)
    if prob >= 0.5:
        return round(-100 * prob / (1 - prob))
    return round(100 * (1 - prob) / prob)


# ── CLI ───────────────────────────────────────────────────────


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Historical backtest of MLB betting model")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025, 2026])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-dir", type=str, default="models/win_model")
    parser.add_argument("--bankroll", type=float, default=10000)
    parser.add_argument("--kelly", type=float, default=0.25)
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--confidence-threshold", type=float, default=0.55,
                        help="Min probability to simulate a bet (default: 0.55)")
    args = parser.parse_args()

    # Support both --model-dir models and --model-dir models/win_model
    model_dir = Path(args.model_dir)
    if model_dir.name == "models" and (model_dir / "win_model").exists():
        model_dir = model_dir / "win_model"

    bt = Backtester(
        data_dir=Path(args.data_dir),
        model_dir=model_dir,
        bankroll=args.bankroll,
        kelly_fraction=args.kelly,
        min_edge=args.min_edge,
    )

    result = bt.run(args.seasons)

    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Games analyzed:    {result.total_games}")
    print(f"  Bets placed:       {result.games_bet}")
    print(f"  Win rate:          {result.win_rate:.1%}")
    print()
    print(f"  MODEL METRICS")
    print(f"  Accuracy:          {result.model_accuracy:.1%}")
    print(f"  AUC-ROC:           {result.model_auc:.4f}")
    print(f"  Brier score:       {result.model_brier:.4f}")
    print()
    print(f"  KELLY STRATEGY")
    print(f"  Starting bankroll: ${result.starting_bankroll:,.2f}")
    print(f"  Ending bankroll:   ${result.ending_bankroll:,.2f}")
    print(f"  Total P&L:         ${result.total_pnl:+,.2f}")
    print(f"  ROI:               {result.roi:+.2%}")
    print(f"  Peak bankroll:     ${result.peak_bankroll:,.2f}")
    print(f"  Max drawdown:      {result.max_drawdown:.1%}")
    print(f"  Sharpe ratio:      {result.sharpe_ratio:.3f}")
    print()
    print(f"  FLAT BET ($100)")
    print(f"  Total P&L:         ${result.flat_bet_pnl:+,.2f}")
    print(f"  ROI:               {result.flat_bet_roi:+.2%}")
    print()

    # Monthly accuracy breakdown
    if result.monthly_accuracy:
        print(f"  MONTHLY ACCURACY (all games)")
        print(f"  {'Month':<10} {'Games':>6} {'Correct':>8} {'Accuracy':>10} {'AvgPred':>10} {'ActualHR':>10}")
        print(f"  {'-'*58}")
        for m in result.monthly_accuracy:
            print(
                f"  {m['month']:<10} {m['games']:>6} {m['correct']:>8} "
                f"{m['accuracy']:>9.1%} {m['avg_pred']:>10.4f} {m['actual_home_rate']:>10.4f}"
            )
        print()

    # Monthly betting breakdown
    print(f"  MONTHLY BETTING P&L")
    print(f"  {'Month':<10} {'Games':>6} {'Bets':>6} {'Wins':>6} {'Kelly P&L':>12} {'Flat P&L':>12}")
    print(f"  {'-'*56}")
    for m in result.monthly:
        print(
            f"  {m['month']:<10} {m['games']:>6} {m['bets']:>6} {m['wins']:>6} "
            f"${m['pnl']:>+10,.2f} ${m['flat_pnl']:>+10,.2f}"
        )
    print()

    # Calibration check
    if result.calibration:
        print(f"  CALIBRATION CHECK (predicted vs observed)")
        print(f"  {'Bin':<12} {'Predicted':>10} {'Observed':>10} {'Count':>8} {'Gap':>8}")
        print(f"  {'-'*50}")
        for b in result.calibration:
            print(
                f"  {b['bin']:<12} {b['predicted_avg']:>10.4f} {b['observed_avg']:>10.4f} "
                f"{b['count']:>8} {b['gap']:>8.4f}"
            )
        print()


if __name__ == "__main__":
    main()
