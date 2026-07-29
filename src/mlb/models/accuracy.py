"""Prediction accuracy tracker — compares predictions against actual outcomes.

Loads a day's predictions, fetches actual scores from completed games,
and records whether each prediction was correct. Persists to
data/accuracy/{date}.json.

Usage:
    python -m mlb.models.accuracy                    # Track today
    python -m mlb.models.accuracy --date 2026-05-14  # Specific date
    python -m mlb.models.accuracy --backfill 7       # Last 7 days
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _today_et() -> date:
    """Today in US Eastern — naive date.today() is tomorrow on UTC hosts after 8 PM ET."""
    return datetime.now(ZoneInfo("America/New_York")).date()


async def track_accuracy(
    target_date: date,
    data_dir: Path = Path("data"),
) -> dict | None:
    """Compare predictions for *target_date* against actual outcomes.

    Grades the slate that was actually sent — the copy locked when the main card
    went out (mlb.etl.slate_record) — not the refreshed predictions later runs leave
    in data/predictions for the dashboard. Days from before the lock existed fall
    back to that cache.

    Returns the accuracy record dict, or None if no predictions/scores found.
    """
    from mlb.etl.slate_record import locked_predictions

    graded_from = "slate"
    predictions = locked_predictions(target_date, data_dir)
    if predictions is None:
        graded_from = "cache"
        pred_path = data_dir / "predictions" / f"{target_date.isoformat()}.json"
        if not pred_path.exists():
            logger.info("No predictions file for %s", target_date)
            return None

        with open(pred_path) as f:
            pred_data = json.load(f)
        predictions = pred_data.get("predictions", [])

    if not predictions:
        logger.info("No predictions for %s", target_date)
        return None

    # Fetch actual scores from MLB API
    from mlb.data.mlb_api import MLBApiClient

    client = MLBApiClient()
    try:
        games = await client.get_schedule(target_date)
    finally:
        await client.close()

    # Build lookup: (home_team, away_team) -> scores
    scores: dict[tuple[str, str], dict] = {}
    for g in games:
        if g["status"] == "Final" and g.get("home_score") is not None:
            key = (g["home_team_id"], g["away_team_id"])
            scores[key] = {
                "home_score": g["home_score"],
                "away_score": g["away_score"],
            }

    if not scores:
        logger.info("No final scores yet for %s", target_date)
        return None

    # Match predictions to scores
    results: list[dict] = []
    correct = 0
    total = 0
    brier_sum = 0.0

    for pred in predictions:
        home = pred.get("home_team")
        away = pred.get("away_team")
        score = scores.get((home, away))
        if not score:
            continue

        # Skip no-prediction placeholders (TBD pitchers → prob 0.5, confidence
        # 0): grading a coin-flip the model never actually called pollutes the
        # scoreboard with an arbitrary correct/incorrect.
        if not pred.get("confidence") and pred.get("home_win_prob", 0.5) == 0.5:
            continue

        home_score = score["home_score"]
        away_score = score["away_score"]
        home_won = home_score > away_score
        pred_home_win = pred.get("home_win_prob", 0.5) > 0.5

        is_correct = pred_home_win == home_won
        if is_correct:
            correct += 1
        total += 1

        # Brier score component: (predicted_prob - actual_outcome)^2
        actual = 1.0 if home_won else 0.0
        brier_sum += (pred.get("home_win_prob", 0.5) - actual) ** 2

        results.append({
            "game_id": pred.get("game_id"),
            "home_team": home,
            "away_team": away,
            "home_win_prob": pred.get("home_win_prob"),
            "confidence": pred.get("confidence"),
            "predicted_winner": pred.get("predicted_winner"),
            "actual_home_score": home_score,
            "actual_away_score": away_score,
            "actual_winner": home if home_won else away,
            "correct": is_correct,
        })

    if total == 0:
        return None

    accuracy = correct / total
    brier = brier_sum / total

    record = {
        "date": target_date.isoformat(),
        "tracked_at": datetime.now().isoformat(),
        "graded_from": graded_from,
        "results": results,
        "summary": {
            "total_games": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": round(accuracy, 4),
            "brier_score": round(brier, 4),
        },
    }

    # Save to disk
    acc_dir = data_dir / "accuracy"
    acc_dir.mkdir(parents=True, exist_ok=True)
    out_path = acc_dir / f"{target_date.isoformat()}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    logger.info(
        "Accuracy for %s (%s): %d/%d (%.1f%%), Brier: %.4f",
        target_date, graded_from, correct, total, accuracy * 100, brier,
    )
    return record


def load_accuracy(target_date: date, data_dir: Path = Path("data")) -> dict | None:
    """Load a single day's accuracy record."""
    path = data_dir / "accuracy" / f"{target_date.isoformat()}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_all_accuracy(data_dir: Path = Path("data")) -> list[dict]:
    """Load all accuracy records, sorted by date."""
    acc_dir = data_dir / "accuracy"
    if not acc_dir.exists():
        return []
    records = []
    for f in sorted(acc_dir.glob("*.json")):
        with open(f) as fp:
            records.append(json.load(fp))
    return records


# Confidence (0-100) at or above which a pick gets its own hit-rate line in the
# daily review, separate from the full-slate record.
HIGH_CONF_THRESHOLD = 65


def _record_from_results(results: list[dict], high_conf_only: bool = False) -> dict:
    """Tally a W-L record from accuracy `results`, optionally only >=65 conf games."""
    if high_conf_only:
        results = [r for r in results if (r.get("confidence") or 0) >= HIGH_CONF_THRESHOLD]
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    return {"correct": correct, "total": total, "incorrect": total - correct}


def build_daily_review(
    target_date: date,
    data_dir: Path = Path("data"),
    high_conf_window: int = 5,
) -> dict:
    """Assemble the retrospective block shown atop the daily Slack card.

    Covers the day *before* `target_date`: full-slate record, high-confidence
    (>=65) record, the trailing high-conf record over `high_conf_window` days,
    and yesterday's settled betting card. Sections are None when their file is
    missing (e.g. games not yet graded), so the caller can skip them cleanly.
    """
    from mlb.betting.settlement import load_settlement

    yesterday = target_date - timedelta(days=1)
    review: dict[str, Any] = {"yesterday": yesterday.isoformat()}

    y_acc = load_accuracy(yesterday, data_dir)
    if y_acc:
        results = y_acc.get("results", [])
        review["full"] = _record_from_results(results)
        review["high_conf"] = _record_from_results(results, high_conf_only=True)
    else:
        review["full"] = None
        review["high_conf"] = None

    # Trailing high-conf record (inclusive of yesterday, back high_conf_window days).
    window = {"correct": 0, "total": 0, "incorrect": 0, "days": high_conf_window}
    for i in range(1, high_conf_window + 1):
        acc = load_accuracy(target_date - timedelta(days=i), data_dir)
        if not acc:
            continue
        rec = _record_from_results(acc.get("results", []), high_conf_only=True)
        window["correct"] += rec["correct"]
        window["total"] += rec["total"]
        window["incorrect"] += rec["incorrect"]
    review["high_conf_window"] = window

    settlement = load_settlement(yesterday, data_dir)
    if settlement and settlement.get("summary", {}).get("bets_placed"):
        s = settlement["summary"]
        review["card"] = {
            "won": s["bets_won"],
            "lost": s["bets_lost"],
            "pushed": s["bets_pushed"],
            "staked": s["total_staked"],
            "pnl": s["daily_pnl"],
            "roi": s["roi"],
            "cumulative": s.get("cumulative_pnl"),
        }
    else:
        review["card"] = None

    return review


def get_rolling_accuracy(
    window: int = 30, data_dir: Path = Path("data")
) -> list[dict]:
    """Compute rolling accuracy over the last *window* days."""
    records = load_all_accuracy(data_dir)
    if not records:
        return []

    rolling: list[dict] = []
    for i, rec in enumerate(records):
        start = max(0, i - window + 1)
        window_recs = records[start : i + 1]
        total = sum(r["summary"]["total_games"] for r in window_recs)
        correct = sum(r["summary"]["correct"] for r in window_recs)
        rolling.append({
            "date": rec["date"],
            "accuracy": round(correct / total, 4) if total > 0 else 0.0,
            "games": total,
        })
    return rolling


# ── CLI ──────────────────────────────────────────────────────


async def _main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Track prediction accuracy")
    parser.add_argument("--date", type=str, default=None, help="Date (YYYY-MM-DD)")
    parser.add_argument("--backfill", type=int, default=0, help="Backfill N days")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.backfill > 0:
        today = _today_et()
        for i in range(args.backfill, 0, -1):
            d = today - timedelta(days=i)
            await track_accuracy(d, data_dir)
    else:
        target = date.fromisoformat(args.date) if args.date else _today_et()
        await track_accuracy(target, data_dir)


def main():
    import asyncio
    asyncio.run(_main())


if __name__ == "__main__":
    main()
