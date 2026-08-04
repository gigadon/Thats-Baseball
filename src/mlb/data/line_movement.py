"""Line movement tracking — periodic odds snapshots.

Stores odds snapshots at regular intervals so the dashboard can show
how lines have moved over time for each game.

Storage: data/line_movement/{date}.json
Format: { "snapshots": [ { "timestamp": ..., "games": [ { "game_id", "home_team",
          "away_team", "home_prob", ... } ] } ] }

Each snapshot game carries the MLB `game_id` (resolved via mlb.data.odds_match),
because the Odds API only knows the team pair and a pair is not unique on a
doubleheader date — without it, both legs collapse into one series and CLV grades
a bet against the other game's closing line. Snapshots written before this
carried no game_id and are still read, matched on the team pair.
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
    today = date.today()
    today_str = today.isoformat()

    # Attach MLB game_ids so doubleheader legs stay separable downstream.
    resolved = await _resolve_game_ids(odds, today)

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
            "game_id": resolved.get(o.get("odds_event_id"), ""),
            "home_team": o["home_team"],
            "away_team": o["away_team"],
            "commence_time": o.get("commence_time", ""),
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


async def _resolve_game_ids(odds: list[dict], game_date: date) -> dict[str, str]:
    """Odds event id -> MLB game_id, empty if the schedule can't be fetched."""
    from mlb.data.mlb_api import MLBApiClient
    from mlb.data.odds_match import resolve_odds_game_ids

    client = MLBApiClient()
    try:
        schedule = await client.get_schedule(game_date)
    except Exception as e:
        logger.warning("Schedule fetch failed, snapshot has no game_ids: %s", e)
        return {}
    finally:
        await client.close()

    return resolve_odds_game_ids(odds, schedule)


def _movement_key(game: dict) -> str:
    """Stable identity for a snapshot entry — the game_id when we have one."""
    return game.get("game_id") or f"{game.get('away_team', '')}@{game.get('home_team', '')}"


def get_line_movement(
    game_date: str,
    home_team: str,
    away_team: str,
    data_dir: Path = Path("data"),
    game_id: str | None = None,
) -> list[float]:
    """Home implied prob over time for one game.

    Pass `game_id` to pin a doubleheader leg; without it, a matchup that appears
    twice on the date returns nothing rather than an arbitrary leg's series.
    """
    movements = get_all_line_movements(game_date, data_dir)

    if game_id and str(game_id) in movements:
        return movements[str(game_id)]["probs"]

    matches = [
        m for m in movements.values()
        if m["home_team"] == home_team and m["away_team"] == away_team
    ]
    if len(matches) == 1:
        return matches[0]["probs"]
    if len(matches) > 1:
        logger.warning(
            "Ambiguous matchup %s@%s on %s (%d games) — pass game_id",
            away_team, home_team, game_date, len(matches),
        )
    return []


def get_all_line_movements(
    game_date: str, data_dir: Path = Path("data")
) -> dict[str, dict]:
    """Line movements for every game on a date, keyed by MLB game_id.

    Entries are {game_id, home_team, away_team, probs}. Snapshots written before
    game_ids were recorded fall back to an "AWY@HOME" key, which collapses a
    doubleheader — those files predate the fix and cannot be separated after
    the fact.
    """
    path = data_dir / "line_movement" / f"{game_date}.json"
    if not path.exists():
        return {}

    with open(path) as f:
        data = json.load(f)

    movements: dict[str, dict] = {}
    for snap in data.get("snapshots", []):
        for g in snap.get("games", []):
            key = _movement_key(g)
            entry = movements.setdefault(key, {
                "game_id": g.get("game_id", ""),
                "home_team": g.get("home_team", ""),
                "away_team": g.get("away_team", ""),
                "probs": [],
            })
            entry["probs"].append(g["home_prob"])

    return movements
