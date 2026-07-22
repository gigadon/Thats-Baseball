"""Wave-send helpers: which games are due to send, and per-day de-dup.

The daily pipeline runs several times a day (see the GitHub Actions workflow). Each
run sends only the games whose first pitch is within the next few hours and that
haven't been sent yet today, so every game gets one card shortly before it starts —
with fresh starters, odds, lineups, and weather.

- `due_game_ids()` selects games in the `[now, now + lead_hours]` window.
- `data/sent/{date}.json` records game_ids already sent today so waves don't repeat.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def parse_game_time(game_time: Any) -> datetime | None:
    """Parse an MLB `game_time` (ISO first pitch, e.g. '2026-07-21T23:41:00Z').

    Returns a timezone-aware datetime, or None if missing/unparseable.
    """
    if not game_time or not isinstance(game_time, str):
        return None
    try:
        return datetime.fromisoformat(game_time.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def due_game_ids(
    predictions: list[dict], now: datetime, lead_hours: float
) -> set[str]:
    """Game_ids whose first pitch is within `[now, now + lead_hours]`.

    Games already underway (first pitch < now) are excluded — they're no longer
    bettable. Games with no parseable `game_time` are included as a safety net so a
    scheduling quirk never drops a game entirely.
    """
    horizon = now + timedelta(hours=lead_hours)
    due: set[str] = set()
    for p in predictions:
        gid = str(p.get("game_id"))
        dt = parse_game_time(p.get("game_time"))
        if dt is None:
            due.add(gid)
            continue
        if now <= dt <= horizon:
            due.add(gid)
    return due


def _sent_path(target_date: date, data_dir: Path) -> Path:
    return data_dir / "sent" / f"{target_date.isoformat()}.json"


def load_sent(target_date: date, data_dir: Path = Path("data")) -> list[str]:
    """Game_ids whose card has already been sent on `target_date`."""
    path = _sent_path(target_date, data_dir)
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return [str(g) for g in data.get("sent", [])]
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read sent log %s: %s", path, e)
        return []


def append_sent(
    target_date: date, game_ids: Iterable[Any], data_dir: Path = Path("data")
) -> list[str]:
    """Record `game_ids` as sent for `target_date`; returns the full merged list.

    De-duplicates while preserving order. Creates data/sent/ on first write.
    """
    existing = load_sent(target_date, data_dir)
    merged = list(dict.fromkeys([*existing, *(str(g) for g in game_ids)]))
    path = _sent_path(target_date, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"date": target_date.isoformat(), "sent": merged}, f, indent=2)
    return merged
