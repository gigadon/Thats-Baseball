"""Scheduled daily pipeline — runs backfill, predictions, and rankings automatically.

Uses asyncio scheduling (no external dependencies like APScheduler).
Designed to run as a long-lived process alongside the API server.

NOTHING RUNS THIS IN PRODUCTION. It was the Railway deployment, which died with
the trial; the live cadence is GitHub Actions:

    .github/workflows/daily-predictions.yml  — backfill, predictions, settlement
    .github/workflows/weekly-retrain.yml     — Monday retrain to staging

That is not a cosmetic distinction. The weekly retrain and the drift-triggered
retrain below silently stopped happening when Railway went away, and no model was
retrained on a schedule for weeks — the miss is invisible precisely because this
code still reads as if it were running. Change the workflows, not this file, to
change what actually happens; keep this in sync only if the process is revived.

Usage:
    python -m mlb.etl.scheduler                   # Run scheduler
    python -m mlb.etl.scheduler --run-now          # Run once immediately, then schedule
    python -m mlb.etl.scheduler --predict-hour 13  # Change prediction time (default 1 PM)
"""

from __future__ import annotations

import asyncio
import logging
import signal
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from mlb.etl.backfill import HistoricalBackfill
from mlb.etl.daily_runner import DailyRunner

logger = logging.getLogger(__name__)


class DailyScheduler:
    """Runs the daily MLB pipeline on a schedule."""

    def __init__(
        self,
        data_dir: Path = Path("data"),
        model_dir: Path = Path("models"),
        predict_hour: int = 13,   # 1 PM ET
        backfill_hour: int = 6,   # 6 AM ET
        settle_hour: int = 1,     # 1 AM ET
        retrain_day: int = 0,     # 0=Monday
        timezone: str = "America/New_York",
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.predict_hour = predict_hour
        self.backfill_hour = backfill_hour
        self.settle_hour = settle_hour
        self.retrain_day = retrain_day
        self.tz = ZoneInfo(timezone)
        self._running = True
        self._last_backfill: date | None = None
        self._last_predict: date | None = None
        self._last_settle: date | None = None
        self._last_retrain: date | None = None
        self._last_snapshot_hour: int | None = None

    async def start(self, run_now: bool = False):
        """Start the scheduler loop."""
        logger.info(
            "Scheduler started — backfill at %02d:00, predictions at %02d:00, settlement at %02d:00 (ET)",
            self.backfill_hour, self.predict_hour, self.settle_hour,
        )

        if run_now:
            await self._run_daily_cycle()

        while self._running:
            now = datetime.now(self.tz)
            today = now.date()

            # Run backfill if it's past backfill_hour and hasn't run today
            if now.hour >= self.backfill_hour and self._last_backfill != today:
                logger.info("Starting scheduled backfill...")
                await self._run_backfill(today)
                self._last_backfill = today

            # Run predictions if it's past predict_hour and hasn't run today
            if now.hour >= self.predict_hour and self._last_predict != today:
                logger.info("Starting scheduled predictions...")
                await self._run_predictions(today)
                self._last_predict = today

            # Run settlement if it's past settle_hour and hasn't run today
            # At 1 AM we settle yesterday's games (all games should be final by then)
            if now.hour >= self.settle_hour and self._last_settle != today:
                yesterday = today - timedelta(days=1)
                logger.info("Starting scheduled settlement for %s...", yesterday)
                await self._run_settlement(yesterday)
                self._last_settle = today

            # Capture line movement snapshot every 3 hours (10 AM, 1 PM, 4 PM, 7 PM ET)
            if now.hour >= 10 and now.hour != self._last_snapshot_hour:
                if now.hour in (10, 13, 16, 19):
                    await self._capture_line_snapshot()
                    self._last_snapshot_hour = now.hour

            # Weekly retrain: run on retrain_day after settlement
            if (
                now.weekday() == self.retrain_day
                and now.hour >= self.settle_hour
                and (self._last_retrain is None or (today - self._last_retrain).days >= 6)
            ):
                logger.info("Starting weekly model retrain...")
                await self._run_retrain()
                self._last_retrain = today

            # Sleep until the next check (every 15 minutes)
            await asyncio.sleep(900)

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        logger.info("Scheduler stopping...")

    async def _run_daily_cycle(self):
        """Run the full daily cycle: backfill yesterday + predict today + settle yesterday."""
        today = date.today()
        await self._run_backfill(today)
        await self._run_predictions(today)
        await self._run_settlement(today - timedelta(days=1))
        self._last_backfill = today
        self._last_predict = today

        # Retrain if today is retrain day
        if today.weekday() == self.retrain_day:
            await self._run_retrain()
            self._last_retrain = today

    async def _run_backfill(self, today: date):
        """Backfill yesterday's completed games."""
        yesterday = today - timedelta(days=1)
        try:
            backfill = HistoricalBackfill(delay=0.3, data_dir=self.data_dir)
            await backfill.backfill_range(yesterday, yesterday, season=today.year)
            logger.info("Backfill completed for %s", yesterday)
        except Exception:
            logger.exception("Backfill failed for %s", yesterday)

    async def _run_settlement(self, target: date):
        """Settle bets for the target date."""
        try:
            from mlb.betting.settlement import settle_day

            result = await settle_day(target, data_dir=self.data_dir)
            if result:
                s = result["summary"]
                logger.info(
                    "Settlement completed for %s: %dW-%dL, P&L: $%.2f",
                    target, s["bets_won"], s["bets_lost"], s["daily_pnl"],
                )
                # Send settlement alert
                try:
                    from mlb.alerts import AlertService

                    alerts = AlertService()
                    await alerts.send_settlement_alert(result)
                except Exception as e:
                    logger.warning("Settlement alert failed: %s", e)
            else:
                logger.info("No bets to settle for %s", target)

            # Also track prediction accuracy for this date
            try:
                from mlb.models.accuracy import track_accuracy
                await track_accuracy(target, data_dir=self.data_dir)
            except Exception as e:
                logger.warning("Accuracy tracking failed for %s: %s", target, e)

            # Check for model drift after accuracy is updated
            await self._check_drift(target)

        except Exception:
            logger.exception("Settlement failed for %s", target)

    async def _run_retrain(self):
        """Rebuild training data and retrain all models."""
        try:
            from mlb.etl.build_training_data import TrainingDataBuilder
            from mlb.models.pipeline import TrainingPipeline

            logger.info("Rebuilding training data...")
            builder = TrainingDataBuilder(data_dir=self.data_dir)
            df = builder.build()
            logger.info("Training data: %d rows, %d cols", len(df), len(df.columns))

            meta_cols = [
                "game_id", "game_date", "home_team_id", "away_team_id",
                "home_win", "home_score", "away_score", "total_runs",
                "home_team", "away_team",
            ]
            feature_cols = [c for c in df.columns if c not in meta_cols and df[c].dtype != "object"]
            X = df[feature_cols]
            y = df["home_win"]

            # Compute time-decay sample weights: recent games matter more.
            # Halflife of 365 days so older seasons still contribute but
            # recent data gets ~2x weight per year of recency.
            import numpy as np
            game_dates = pd.to_datetime(df["game_date"])
            max_date = game_dates.max()
            days_ago = (max_date - game_dates).dt.days.values.astype(float)
            halflife_days = 365
            sample_weight = np.exp(-np.log(2) * days_ago / halflife_days)

            logger.info("Training ensemble on %d features...", len(feature_cols))
            pipeline = TrainingPipeline()
            ensemble = pipeline.train(X, y, sample_weight=sample_weight)
            # Pipeline auto-saves to models/ensemble during training.
            # Copy to win_model dir so PredictionService picks them up.
            import shutil
            ens_src = self.model_dir / "ensemble"
            ens_dst = self.model_dir / "win_model" / "ensemble"
            ens_dst.mkdir(parents=True, exist_ok=True)
            if ens_src.exists():
                for f in ens_src.glob("*.joblib"):
                    shutil.copy2(f, ens_dst / f.name)
            # Also copy pipeline meta
            meta_src = self.model_dir / "pipeline_meta.joblib"
            if meta_src.exists():
                shutil.copy2(meta_src, self.model_dir / "win_model" / "pipeline_meta.joblib")

            logger.info("Retrain complete")

            # Send Slack alert
            try:
                from mlb.alerts import AlertService
                ens_metrics = pipeline.run_metrics.get("Ensemble", {})
                auc = ens_metrics.get("auc_roc", 0.0)
                acc = ens_metrics.get("accuracy", 0.0)
                alerts = AlertService()
                await alerts.send_alert(
                    f"Weekly Retrain Complete\n"
                    f"AUC: {auc:.4f} | Accuracy: {acc:.1%}\n"
                    f"Training rows: {len(df):,} | Features: {len(feature_cols)}"
                )
            except Exception as e:
                logger.warning("Retrain alert failed: %s", e)

        except Exception:
            logger.exception("Weekly retrain failed")

    async def _capture_line_snapshot(self):
        """Capture a line movement snapshot (current odds)."""
        try:
            from mlb.data.line_movement import capture_snapshot

            n = await capture_snapshot(data_dir=self.data_dir)
            if n > 0:
                logger.info("Line snapshot captured: %d games", n)
        except Exception:
            logger.exception("Line snapshot capture failed")

    async def _check_drift(self, today: date):
        """Check for model drift and alert if accuracy drops below threshold."""
        try:
            from mlb.models.accuracy import load_all_accuracy

            records = load_all_accuracy(self.data_dir)
            if not records:
                return

            # Look at last 7 days of accuracy
            cutoff = today - timedelta(days=7)
            recent = [
                r for r in records
                if date.fromisoformat(r["date"]) >= cutoff
            ]
            if not recent:
                return

            all_results = [r for rec in recent for r in rec.get("results", [])]
            total = len(all_results)
            if total < 10:
                return  # Not enough data to judge

            correct = sum(1 for r in all_results if r.get("correct"))
            accuracy = correct / total

            if accuracy < 0.52:
                logger.warning(
                    "MODEL DRIFT DETECTED: 7d accuracy %.1f%% (%d/%d) — below 52%% threshold",
                    accuracy * 100, correct, total,
                )
                try:
                    from mlb.alerts import AlertService
                    alerts = AlertService()
                    await alerts.send_alert(
                        f":warning: *Model Drift Alert*\n"
                        f"7-day accuracy: *{accuracy:.1%}* ({correct}/{total} correct)\n"
                        f"Threshold: 52% — consider early retrain"
                    )
                except Exception as e:
                    logger.warning("Drift alert failed: %s", e)

                # Trigger early retrain if it hasn't run in the last 3 days
                if self._last_retrain is None or (today - self._last_retrain).days >= 3:
                    logger.info("Triggering early retrain due to drift...")
                    await self._run_retrain()
                    self._last_retrain = today
        except Exception:
            logger.exception("Drift check failed")

    async def _run_predictions(self, today: date):
        """Generate predictions for today's games."""
        try:
            runner = DailyRunner(
                data_dir=self.data_dir,
                model_dir=self.model_dir,
            )
            result = await runner.run(today)

            n_preds = len(result.get("predictions", []))
            slip = result.get("betting_slip")
            n_bets = slip["num_bets"] if slip else 0

            logger.info(
                "Predictions completed: %d games, %d value bets",
                n_preds, n_bets,
            )
        except Exception:
            logger.exception("Predictions failed for %s", today)


# ── CLI ───────────────────────────────────────────────────────


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="MLB daily pipeline scheduler")
    parser.add_argument("--run-now", action="store_true", help="Run immediately, then schedule")
    parser.add_argument("--predict-hour", type=int, default=13, help="Hour to run predictions (0-23)")
    parser.add_argument("--backfill-hour", type=int, default=6, help="Hour to run backfill (0-23)")
    parser.add_argument("--settle-hour", type=int, default=1, help="Hour to run settlement (0-23)")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-dir", type=str, default="models")
    args = parser.parse_args()

    scheduler = DailyScheduler(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
        predict_hour=args.predict_hour,
        backfill_hour=args.backfill_hour,
        settle_hour=args.settle_hour,
    )

    loop = asyncio.new_event_loop()

    def _handle_signal(sig, frame):
        scheduler.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        loop.run_until_complete(scheduler.start(run_now=args.run_now))
    finally:
        loop.close()
        logger.info("Scheduler shut down")


if __name__ == "__main__":
    main()
