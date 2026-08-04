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
from datetime import date, datetime, timezone
from pathlib import Path

from mlb.betting.engine import american_to_decimal
from mlb.data.odds_api import OddsApiClient
from mlb.data.odds_match import parse_dt

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/line_movement")


def _no_vig_home_prob(home_ml: int, away_ml: int) -> float:
    """Home win probability implied by the two moneylines, vig removed."""
    h_imp = 1.0 / american_to_decimal(home_ml)
    a_imp = 1.0 / american_to_decimal(away_ml)
    return h_imp / (h_imp + a_imp)


def record_snapshot(
    odds_by_game: dict[str, dict],
    target_date: date,
    data_dir: Path = Path("data"),
    now: datetime | None = None,
) -> int:
    """Append one snapshot of `odds_by_game` to data/line_movement/{date}.json.

    Takes odds already fetched and already resolved to MLB game_ids so the daily
    pipeline can record its line for free — the Odds API free tier is 500 calls a
    month and the pipeline is the only thing that should be spending them.

    Only games that haven't started are recorded. The feed keeps some games after
    first pitch and switches them to in-play prices — a 3% home side two hours in
    is not a closing line, and grading a pre-game bet against it would make CLV
    meaningless.

    Returns the number of games recorded.
    """
    now = now or datetime.now()
    # A naive `now` is local wall clock, which is what datetime.now() gives.
    now_utc = now.astimezone(timezone.utc)

    games = []
    for game_id, o in odds_by_game.items():
        home_ml = o.get("home_moneyline")
        away_ml = o.get("away_moneyline")
        if home_ml is None or away_ml is None:
            continue

        start = parse_dt(o.get("commence_time"))
        if start is not None and start <= now_utc:
            logger.debug(
                "Skipping in-play line for %s (%s@%s, started %s)",
                game_id, o.get("away_team"), o.get("home_team"),
                o.get("commence_time"),
            )
            continue

        games.append({
            "game_id": game_id,
            "home_team": o["home_team"],
            "away_team": o["away_team"],
            "commence_time": o.get("commence_time", ""),
            "home_prob": round(_no_vig_home_prob(home_ml, away_ml), 4),
            "home_moneyline": home_ml,
            "away_moneyline": away_ml,
            "total_line": o.get("total_line"),
        })

    if not games:
        return 0

    date_str = target_date.isoformat()
    snapshot = {
        "timestamp": now.isoformat(timespec="seconds"),
        "games": games,
    }

    lm_dir = data_dir / "line_movement"
    lm_dir.mkdir(parents=True, exist_ok=True)
    path = lm_dir / f"{date_str}.json"

    if path.exists():
        with open(path) as f:
            data = json.load(f)
    else:
        data = {"date": date_str, "snapshots": []}

    data["snapshots"].append(snapshot)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return len(games)


async def capture_snapshot(data_dir: Path = Path("data")) -> int:
    """Fetch current odds and append a snapshot for today.

    Standalone entry point — costs one Odds API call. The daily pipeline does not
    use this; it records from the odds it has already fetched.

    Returns number of games captured.
    """
    client = OddsApiClient()
    odds = await client.get_mlb_odds()
    if not odds:
        logger.info("No odds available for snapshot")
        return 0

    now = datetime.now()
    today = date.today()

    # Resolve to MLB game_ids so doubleheader legs stay separable downstream.
    odds_by_game = await _index_by_game(odds, today)
    n = record_snapshot(odds_by_game, today, data_dir, now=now)

    logger.info("Captured line snapshot: %d games at %s", n, now.strftime("%H:%M"))
    return n


async def _index_by_game(odds: list[dict], game_date: date) -> dict[str, dict]:
    """Odds keyed by MLB game_id; empty if the schedule can't be fetched."""
    from mlb.data.mlb_api import MLBApiClient
    from mlb.data.odds_match import index_odds_by_game

    client = MLBApiClient()
    try:
        schedule = await client.get_schedule(game_date)
    except Exception as e:
        logger.warning("Schedule fetch failed, cannot snapshot lines: %s", e)
        return {}
    finally:
        await client.close()

    return index_odds_by_game(odds, schedule)


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
