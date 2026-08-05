"""The day's slate of record — the snapshot that gets graded.

The pipeline runs several times a day (see the GitHub Actions workflow), and every
run recomputes the whole slate from whatever data exists at that moment. Only the
main run sends the betting card; the later wave runs are near-first-pitch
reminders. Without a lock, the last run of the night would silently replace the
picks and the bets that were actually alerted, and the next morning's review would
grade a card nobody ever saw.

So the main run locks ``data/slates/{date}.json`` and later runs only read it:

- ``mlb.models.accuracy`` grades its predictions (full slate + high-conf).
- ``mlb.betting.settlement`` settles its bets (the betting card).

``data/predictions/{date}.json`` keeps being refreshed by every run — that is the
live view the dashboard reads, not the record.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


def slate_path(target_date: date, data_dir: Path = Path("data")) -> Path:
    return data_dir / "slates" / f"{target_date.isoformat()}.json"


def load_slate(target_date: date, data_dir: Path = Path("data")) -> dict | None:
    """The locked slate for `target_date`, or None if the main run never wrote one."""
    path = slate_path(target_date, data_dir)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read slate record %s: %s", path, e)
        return None


def lock_slate(
    target_date: date,
    predictions: list[dict],
    betting_slip: dict | None,
    data_dir: Path = Path("data"),
    *,
    reconstructed_from: dict | None = None,
) -> dict:
    """Write the slate of record for `target_date` and return it.

    Called once a day, by the run that sends the main card. `betting_slip` is
    stored as-is — including None, which is the honest record of a main card that
    went out with no bets on it.

    `reconstructed_from` marks a slate recovered after the fact from git history
    (mlb.etl.slate_repair) rather than locked live. Such a slate is still far
    better than the degraded prediction cache, but it ranks below a genuine lock:
    a real run may replace it, and accuracy records it as `slate:reconstructed`.
    """
    record = {
        "date": target_date.isoformat(),
        "locked_at": datetime.now(ZoneInfo("America/New_York")).isoformat(
            timespec="seconds"
        ),
        "predictions": predictions,
        "betting_slip": betting_slip,
    }
    if reconstructed_from:
        record["reconstructed"] = reconstructed_from
    path = slate_path(target_date, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    slip_bets = (betting_slip or {}).get("num_bets", 0)
    logger.info(
        "Locked slate for %s: %d game(s), %d bet(s)",
        target_date, len(predictions), slip_bets,
    )
    return record


def locked_predictions(
    target_date: date, data_dir: Path = Path("data")
) -> list[dict] | None:
    """Predictions as sent on the main card, or None when the day isn't locked."""
    slate = load_slate(target_date, data_dir)
    if slate is None:
        return None
    return slate.get("predictions") or []


def locked_betting_slip(
    target_date: date, data_dir: Path = Path("data")
) -> dict | None:
    """The betting card as sent, or None when the day isn't locked / had no bets."""
    slate = load_slate(target_date, data_dir)
    if slate is None:
        return None
    slip: Any = slate.get("betting_slip")
    return slip if isinstance(slip, dict) else None
