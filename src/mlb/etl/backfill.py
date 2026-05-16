"""Historical data backfill — fetches games and boxscores from MLB API to local files.

Stores data as CSV files that can be loaded for training without needing Postgres.
Respects API rate limits with configurable delays between requests.

Usage:
    python -m mlb.etl.backfill --season 2024
    python -m mlb.etl.backfill --start 2024-04-01 --end 2024-09-30
    python -m mlb.etl.backfill --season 2023 --season 2024 --season 2025
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from mlb.data.mlb_api import MLBApiClient

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
GAMES_DIR = DATA_DIR / "games"
BOXSCORES_DIR = DATA_DIR / "boxscores"
PLAYERS_DIR = DATA_DIR / "players"

# MLB season date ranges (approximate)
SEASON_DATES = {
    2021: (date(2021, 4, 1), date(2021, 10, 3)),
    2022: (date(2022, 4, 7), date(2022, 10, 5)),
    2023: (date(2023, 3, 30), date(2023, 10, 1)),
    2024: (date(2024, 3, 28), date(2024, 9, 29)),
    2025: (date(2025, 3, 27), date(2025, 9, 28)),
    2026: (date(2026, 3, 26), date(2026, 9, 27)),
}


class HistoricalBackfill:
    """Fetches historical MLB data and writes to local CSV files."""

    def __init__(self, delay: float = 0.5, data_dir: Path | None = None):
        self.client = MLBApiClient()
        self.delay = delay  # Seconds between API calls
        self.data_dir = data_dir or DATA_DIR
        self._games_written = 0
        self._boxscores_written = 0
        self._errors = 0

    async def backfill_season(self, season: int):
        """Backfill an entire MLB season."""
        dates = SEASON_DATES.get(season)
        if not dates:
            logger.error("Unknown season: %d. Add dates to SEASON_DATES.", season)
            return
        start, end = dates
        # Clamp end to today if season is current
        if end > date.today():
            end = date.today() - timedelta(days=1)
        await self.backfill_range(start, end, season)

    async def backfill_range(self, start: date, end: date, season: int | None = None):
        """Backfill a date range."""
        season = season or start.year
        self._setup_dirs(season)

        logger.info("Backfilling %s to %s (season %d)", start, end, season)
        t0 = time.time()

        try:
            # Phase 1: Fetch all games in range
            logger.info("Phase 1: Fetching game schedule...")
            all_games = await self._fetch_all_games(start, end)
            logger.info("Found %d total games, %d final",
                        len(all_games), sum(1 for g in all_games if g["status"] == "Final"))

            # Write games CSV
            self._write_games_csv(all_games, season)

            # Phase 2: Fetch boxscores for completed games
            final_games = [g for g in all_games if g["status"] == "Final"]
            logger.info("Phase 2: Fetching %d boxscores...", len(final_games))
            await self._fetch_all_boxscores(final_games, season)

            elapsed = time.time() - t0
            logger.info(
                "Backfill complete: %d games, %d boxscores, %d errors in %.0fs",
                self._games_written, self._boxscores_written, self._errors, elapsed,
            )
        finally:
            await self.client.close()

    async def _fetch_all_games(self, start: date, end: date) -> list[dict]:
        """Fetch games using weekly chunks to reduce API calls."""
        all_games: list[dict] = []
        current = start

        while current <= end:
            chunk_end = min(current + timedelta(days=6), end)
            try:
                games = await self.client.get_schedule_range(current, chunk_end)
                all_games.extend(games)
                logger.info("  %s to %s: %d games", current, chunk_end, len(games))
            except Exception:
                logger.exception("Failed to fetch schedule %s to %s", current, chunk_end)
                self._errors += 1

            current = chunk_end + timedelta(days=1)
            await asyncio.sleep(self.delay)

        return all_games

    async def _fetch_all_boxscores(self, games: list[dict], season: int):
        """Fetch boxscores with progress logging."""
        total = len(games)
        batting_rows: list[dict] = []
        pitching_rows: list[dict] = []

        for i, game in enumerate(games):
            game_id = game["game_id"]
            game_date = game["game_date"]

            try:
                boxscore = await self.client.get_boxscore(game_id)

                for side in ("home", "away"):
                    side_data = boxscore.get(side, {})
                    team_id = side_data.get("team_id", "")

                    for batter in side_data.get("batters", []):
                        batting_rows.append({
                            "game_id": game_id,
                            "game_date": game_date,
                            "team_id": team_id,
                            "side": side,
                            "player_id": batter.get("player_id"),
                            "player_name": batter.get("name", ""),
                            "at_bats": batter.get("at_bats", 0),
                            "runs": batter.get("runs", 0),
                            "hits": batter.get("hits", 0),
                            "doubles": batter.get("doubles", 0),
                            "triples": batter.get("triples", 0),
                            "home_runs": batter.get("home_runs", 0),
                            "rbi": batter.get("rbi", 0),
                            "walks": batter.get("walks", 0),
                            "strikeouts": batter.get("strikeouts", 0),
                            "stolen_bases": batter.get("stolen_bases", 0),
                        })

                    for pitcher in side_data.get("pitchers", []):
                        pitching_rows.append({
                            "game_id": game_id,
                            "game_date": game_date,
                            "team_id": team_id,
                            "side": side,
                            "player_id": pitcher.get("player_id"),
                            "player_name": pitcher.get("name", ""),
                            "innings_pitched": pitcher.get("innings_pitched", 0),
                            "hits_allowed": pitcher.get("hits_allowed", 0),
                            "runs_allowed": pitcher.get("runs_allowed", 0),
                            "earned_runs": pitcher.get("earned_runs", 0),
                            "walks_allowed": pitcher.get("walks_allowed", 0),
                            "strikeouts_recorded": pitcher.get("strikeouts_recorded", 0),
                            "pitches_thrown": pitcher.get("pitches_thrown", 0),
                        })

                self._boxscores_written += 1

            except Exception:
                logger.exception("Failed boxscore for game %s", game_id)
                self._errors += 1

            if (i + 1) % 50 == 0 or i == total - 1:
                logger.info("  Boxscores: %d / %d (%.0f%%)", i + 1, total, (i + 1) / total * 100)

            await asyncio.sleep(self.delay)

        # Write CSVs
        self._write_csv(
            self.data_dir / f"batting_{season}.csv",
            batting_rows,
            batting_rows[0].keys() if batting_rows else [],
        )
        self._write_csv(
            self.data_dir / f"pitching_{season}.csv",
            pitching_rows,
            pitching_rows[0].keys() if pitching_rows else [],
        )
        logger.info("Wrote %d batting rows, %d pitching rows", len(batting_rows), len(pitching_rows))

    def _write_games_csv(self, games: list[dict], season: int):
        """Write games to CSV, merging with existing data."""
        if not games:
            return

        path = self.data_dir / f"games_{season}.csv"
        fields = [
            "game_id", "game_date", "home_team_id", "away_team_id",
            "home_score", "away_score", "status",
        ]

        # Load existing and merge by game_id (new data wins for updated scores/status)
        existing: dict[str, dict] = {}
        if path.exists():
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing[row["game_id"]] = row

        for g in games:
            existing[str(g["game_id"])] = {k: g.get(k, "") for k in fields}

        all_games = sorted(existing.values(), key=lambda r: (r.get("game_date", ""), r.get("game_id", "")))

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_games)

        self._games_written = len(games)
        logger.info("Wrote %d games (%d total) to %s", len(games), len(all_games), path)

    def _write_csv(self, path: Path, rows: list[dict], fields):
        """Write rows to CSV, merging with existing data by (game_id, player_id)."""
        if not rows:
            return
        fields = list(fields)

        # Load existing and merge
        existing: dict[tuple, dict] = {}
        if path.exists():
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row.get("game_id", ""), row.get("player_id", ""))
                    existing[key] = row

        for r in rows:
            key = (str(r.get("game_id", "")), str(r.get("player_id", "")))
            existing[key] = {k: r.get(k, "") for k in fields}

        all_rows = sorted(existing.values(), key=lambda r: (r.get("game_date", ""), r.get("game_id", "")))

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)

    def _setup_dirs(self, season: int):
        self.data_dir.mkdir(parents=True, exist_ok=True)


# ── CLI ───────────────────────────────────────────────────────


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Backfill historical MLB data")
    parser.add_argument("--season", type=int, action="append", help="Season year(s) to backfill")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls (seconds)")
    parser.add_argument("--data-dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    backfill = HistoricalBackfill(delay=args.delay, data_dir=Path(args.data_dir))

    if args.season:
        for s in args.season:
            asyncio.run(backfill.backfill_season(s))
    elif args.start and args.end:
        asyncio.run(backfill.backfill_range(
            date.fromisoformat(args.start),
            date.fromisoformat(args.end),
        ))
    else:
        parser.error("Provide --season or --start/--end")


if __name__ == "__main__":
    main()
