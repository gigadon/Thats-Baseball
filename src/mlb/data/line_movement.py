"""Line movement tracking — periodic odds snapshots.

Stores odds snapshots at regular intervals so the dashboard can show
how lines have moved over time for each game.

Storage: data/line_movement/{date}.json
Format: { "snapshots": [ { "timestamp": ..., "games": [ { "home_team", "away_team", "home_prob", ... } ] } ] }
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from mlb.betting.engine import american_to_decimal
from mlb.data.odds_api import OddsApiClient

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/line_movement")


async def capture_snapshot(data_dir: Path = Path("data")) -> int:
    """Fetch current odds and append a snapshot for today.

    Returns number of games captured.
    """
    client = OddsApiClient()
    odds = await client.get_mlb_odds()
    if not odds:
        logger.info("No odds available for snapshot")
        return 0

    now = datetime.now()
    today_str = date.today().isoformat()

    games = []
    for o in odds:
        home_ml = o.get("home_moneyline")
        away_ml = o.get("away_moneyline")
        if home_ml is None or away_ml is None:
            continue

        h_dec = american_to_decimal(home_ml)
        a_dec = american_to_decimal(away_ml)
        h_imp = 1.0 / h_dec
        a_imp = 1.0 / a_dec
        total_imp = h_imp + a_imp
        home_prob = h_imp / total_imp

        games.append({
            "home_team": o["home_team"],
            "away_team": o["away_team"],
            "home_prob": round(home_prob, 4),
            "home_moneyline": home_ml,
            "away_moneyline": away_ml,
            "total_line": o.get("total_line"),
        })

    snapshot = {
        "timestamp": now.isoformat(timespec="seconds"),
        "games": games,
    }

    # Load existing or create new
    lm_dir = data_dir / "line_movement"
    lm_dir.mkdir(parents=True, exist_ok=True)
    path = lm_dir / f"{today_str}.json"

    if path.exists():
        with open(path) as f:
            data = json.load(f)
    else:
        data = {"date": today_str, "snapshots": []}

    data["snapshots"].append(snapshot)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Captured line snapshot: %d games at %s", len(games), now.strftime("%H:%M"))
    return len(games)


def get_line_movement(
    game_date: str,
    home_team: str,
    away_team: str,
    data_dir: Path = Path("data"),
) -> list[float]:
    """Get line movement (home implied prob over time) for a specific game.

    Returns list of home_prob values at each snapshot time.
    """
    path = data_dir / "line_movement" / f"{game_date}.json"
    if not path.exists():
        return []

    with open(path) as f:
        data = json.load(f)

    probs = []
    for snap in data.get("snapshots", []):
        for g in snap.get("games", []):
            if g["home_team"] == home_team and g["away_team"] == away_team:
                probs.append(g["home_prob"])
                break

    return probs


def get_all_line_movements(
    game_date: str, data_dir: Path = Path("data")
) -> dict[tuple[str, str], list[float]]:
    """Get line movements for all games on a date.

    Returns {(home_team, away_team): [prob1, prob2, ...]}
    """
    path = data_dir / "line_movement" / f"{game_date}.json"
    if not path.exists():
        return {}

    with open(path) as f:
        data = json.load(f)

    movements: dict[tuple[str, str], list[float]] = {}
    for snap in data.get("snapshots", []):
        for g in snap.get("games", []):
            key = (g["home_team"], g["away_team"])
            movements.setdefault(key, []).append(g["home_prob"])

    return movements
