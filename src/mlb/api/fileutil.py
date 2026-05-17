"""Atomic JSON file read/write to prevent corruption from concurrent access."""

from __future__ import annotations

import fcntl
import json
import tempfile
from pathlib import Path


def safe_json_read(path: Path) -> dict:
    """Read a JSON file with shared lock. Returns {} if missing or corrupt."""
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                return json.loads(f.read())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except (json.JSONDecodeError, OSError):
        return {}


def safe_json_write(path: Path, data: dict):
    """Write JSON atomically: write to temp file, then rename (prevents corruption)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with open(tmp_fd, "w") as f:
            json.dump(data, f, default=str)
        Path(tmp_path).replace(path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def safe_json_merge(path: Path, key: str, value) -> dict:
    """Read existing JSON, merge a key, write atomically. Returns the merged dict."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Use exclusive lock for the read-modify-write cycle
    lock_path = path.with_suffix(".lock")
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            existing = safe_json_read(path)
            existing[key] = value
            safe_json_write(path, existing)
            return existing
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
