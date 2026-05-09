"""Scheduled daily pipeline — runs backfill, predictions, and rankings automatically.

Uses asyncio scheduling (no external dependencies like APScheduler).
Designed to run as a long-lived process alongside the API server.

Usage:
    python -m mlb.etl.scheduler                   # Run scheduler
    python -m mlb.etl.scheduler --run-now          # Run once immediately, then schedule
    python -m mlb.etl.scheduler --predict-hour 10  # Change prediction time (default 10 AM)
"""

from __future__ import annotations

import asyncio
import logging
import signal
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from mlb.etl.backfill import HistoricalBackfill
from mlb.etl.daily_runner import DailyRunner

logger = logging.getLogger(__name__)


class DailyScheduler:
    """Runs the daily MLB pipeline on a schedule."""

    def __init__(
        self,
        data_dir: Path = Path("data"),
        model_dir: Path = Path("models"),
        predict_hour: int = 10,   # 10 AM local time
        backfill_hour: int = 6,   # 6 AM local time
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.predict_hour = predict_hour
        self.backfill_hour = backfill_hour
        self._running = True
        self._last_backfill: date | None = None
        self._last_predict: date | None = None

    async def start(self, run_now: bool = False):
        """Start the scheduler loop."""
        logger.info(
            "Scheduler started — backfill at %02d:00, predictions at %02d:00",
            self.backfill_hour, self.predict_hour,
        )

        if run_now:
            await self._run_daily_cycle()

        while self._running:
            now = datetime.now()
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

            # Sleep until the next check (every 15 minutes)
            await asyncio.sleep(900)

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        logger.info("Scheduler stopping...")

    async def _run_daily_cycle(self):
        """Run the full daily cycle: backfill yesterday + predict today."""
        today = date.today()
        await self._run_backfill(today)
        await self._run_predictions(today)
        self._last_backfill = today
        self._last_predict = today

    async def _run_backfill(self, today: date):
        """Backfill yesterday's completed games."""
        yesterday = today - timedelta(days=1)
        try:
            backfill = HistoricalBackfill(delay=0.3, data_dir=self.data_dir)
            await backfill.backfill_range(yesterday, yesterday, season=today.year)
            logger.info("Backfill completed for %s", yesterday)
        except Exception:
            logger.exception("Backfill failed for %s", yesterday)

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
    parser.add_argument("--predict-hour", type=int, default=10, help="Hour to run predictions (0-23)")
    parser.add_argument("--backfill-hour", type=int, default=6, help="Hour to run backfill (0-23)")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-dir", type=str, default="models")
    args = parser.parse_args()

    scheduler = DailyScheduler(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
        predict_hour=args.predict_hour,
        backfill_hour=args.backfill_hour,
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
