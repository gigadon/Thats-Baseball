"""ETL pipeline for daily MLB data ingestion."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from mlb.data.mlb_api import MLBApiClient
from mlb.data.validator import DataValidator
from mlb.db.engine import SessionLocal
from mlb.db.models import (
    BullpenUsage,
    Game,
    Player,
    PlayerGameStat,
    Team,
    TeamDailyStat,
)

logger = logging.getLogger(__name__)


class DailyETLPipeline:
    """Orchestrates daily data ingestion from MLB API into the database."""

    def __init__(self):
        self.client = MLBApiClient()
        self.validator = DataValidator()

    async def run(self, target_date: date | None = None):
        """Run the full daily ETL for a given date (defaults to yesterday)."""
        target_date = target_date or (date.today() - timedelta(days=1))
        logger.info("Starting daily ETL for %s", target_date)

        try:
            await self._sync_teams()
            games = await self._ingest_games(target_date)
            await self._ingest_boxscores(games)
            logger.info("Daily ETL completed for %s — %d games processed", target_date, len(games))
        finally:
            await self.client.close()

    async def run_range(self, start: date, end: date):
        """Backfill a date range."""
        logger.info("Backfilling %s to %s", start, end)
        try:
            await self._sync_teams()
            current = start
            while current <= end:
                games = await self._ingest_games(current)
                await self._ingest_boxscores(games)
                logger.info("Processed %s — %d games", current, len(games))
                current += timedelta(days=1)
        finally:
            await self.client.close()

    # ── Teams ──────────────────────────────────────────────────

    async def _sync_teams(self):
        """Sync the 30 MLB teams into the database."""
        teams = await self.client.get_teams()
        session = SessionLocal()
        try:
            for t in teams:
                stmt = pg_insert(Team).values(
                    team_id=t["team_id"],
                    team_name=t["team_name"],
                    division=t["division"],
                    league=t["league"],
                    stadium_name=t["stadium_name"],
                ).on_conflict_do_update(
                    index_elements=["team_id"],
                    set_={
                        "team_name": t["team_name"],
                        "division": t["division"],
                        "league": t["league"],
                        "stadium_name": t["stadium_name"],
                    },
                )
                session.execute(stmt)
            session.commit()
            logger.info("Synced %d teams", len(teams))
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ── Games ──────────────────────────────────────────────────

    async def _ingest_games(self, game_date: date) -> list[dict]:
        """Fetch and store games for a date. Returns list of final game dicts."""
        games = await self.client.get_schedule(game_date)
        valid_games, invalid = self.validator.validate_batch(games)

        if invalid:
            logger.warning("%d games failed validation on %s", len(invalid), game_date)

        session = SessionLocal()
        try:
            for g in valid_games:
                stmt = pg_insert(Game).values(
                    game_id=g["game_id"],
                    game_date=g["game_date"],
                    home_team_id=g["home_team_id"],
                    away_team_id=g["away_team_id"],
                    home_score=g.get("home_score"),
                    away_score=g.get("away_score"),
                    status=g["status"],
                ).on_conflict_do_update(
                    index_elements=["game_id"],
                    set_={
                        "home_score": g.get("home_score"),
                        "away_score": g.get("away_score"),
                        "status": g["status"],
                    },
                )
                session.execute(stmt)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        # Return only final games for boxscore ingestion
        return [g for g in valid_games if g["status"] == "Final"]

    # ── Boxscores ──────────────────────────────────────────────

    async def _ingest_boxscores(self, games: list[dict]):
        """Fetch and store boxscores for completed games."""
        for game in games:
            game_id = game["game_id"]
            try:
                boxscore = await self.client.get_boxscore(game_id)
                validation = self.validator.validate_boxscore(boxscore)
                if not validation.passed:
                    logger.warning(
                        "Boxscore %s failed validation: %s", game_id, validation.errors
                    )
                    continue

                self._store_boxscore(boxscore)
            except Exception:
                logger.exception("Failed to process boxscore for game %s", game_id)

    def _store_boxscore(self, boxscore: dict):
        """Store player stats from a boxscore."""
        session = SessionLocal()
        game_id = boxscore["game_id"]

        try:
            for side in ("home", "away"):
                side_data = boxscore[side]
                team_id = side_data["team_id"]

                # Ensure players exist
                for batter in side_data.get("batters", []):
                    self._upsert_player(session, batter, team_id)

                for pitcher in side_data.get("pitchers", []):
                    self._upsert_player(session, pitcher, team_id, position="P")

                # Store batting stats
                for batter in side_data.get("batters", []):
                    stmt = pg_insert(PlayerGameStat).values(
                        player_id=batter["player_id"],
                        game_id=game_id,
                        team_id=team_id,
                        at_bats=batter.get("at_bats"),
                        runs=batter.get("runs"),
                        hits=batter.get("hits"),
                        doubles=batter.get("doubles"),
                        triples=batter.get("triples"),
                        home_runs=batter.get("home_runs"),
                        rbi=batter.get("rbi"),
                        walks=batter.get("walks"),
                        strikeouts=batter.get("strikeouts"),
                        stolen_bases=batter.get("stolen_bases"),
                    ).on_conflict_do_nothing()
                    session.execute(stmt)

                # Store pitching stats + bullpen usage
                for pitcher in side_data.get("pitchers", []):
                    stmt = pg_insert(PlayerGameStat).values(
                        player_id=pitcher["player_id"],
                        game_id=game_id,
                        team_id=team_id,
                        innings_pitched=pitcher.get("innings_pitched"),
                        hits_allowed=pitcher.get("hits_allowed"),
                        runs_allowed=pitcher.get("runs_allowed"),
                        earned_runs=pitcher.get("earned_runs"),
                        walks_allowed=pitcher.get("walks_allowed"),
                        strikeouts_recorded=pitcher.get("strikeouts_recorded"),
                        pitches_thrown=pitcher.get("pitches_thrown"),
                    ).on_conflict_do_nothing()
                    session.execute(stmt)

                    # Track bullpen usage (non-starters)
                    ip = pitcher.get("innings_pitched") or 0
                    if ip < 5:  # Likely a reliever
                        stmt = pg_insert(BullpenUsage).values(
                            pitcher_id=pitcher["player_id"],
                            game_id=game_id,
                            team_id=team_id,
                            pitches_thrown=pitcher.get("pitches_thrown"),
                            innings_pitched=pitcher.get("innings_pitched"),
                        ).on_conflict_do_nothing()
                        session.execute(stmt)

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _upsert_player(
        self, session: Session, player_data: dict, team_id: str, position: str = ""
    ):
        """Ensure a player exists in the players table."""
        stmt = pg_insert(Player).values(
            player_id=player_data["player_id"],
            player_name=player_data.get("name", ""),
            team_id=team_id,
            position=position,
            bats="",
            throws="",
        ).on_conflict_do_update(
            index_elements=["player_id"],
            set_={"team_id": team_id},
        )
        session.execute(stmt)


# ── CLI Entry Point ────────────────────────────────────────────


def main():
    """Run daily ETL from the command line."""
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    target = None
    if len(sys.argv) > 1:
        target = date.fromisoformat(sys.argv[1])

    pipeline = DailyETLPipeline()
    asyncio.run(pipeline.run(target))


if __name__ == "__main__":
    main()
