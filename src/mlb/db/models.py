import uuid
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ─── Core Tables ───────────────────────────────────────────────


class Team(Base):
    __tablename__ = "teams"

    team_id: Mapped[str] = mapped_column(String(3), primary_key=True)
    team_name: Mapped[str] = mapped_column(String(50))
    division: Mapped[str] = mapped_column(String(20))
    league: Mapped[str] = mapped_column(String(2))
    stadium_name: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("NOW()")
    )

    players: Mapped[list["Player"]] = relationship(back_populates="team")
    daily_stats: Mapped[list["TeamDailyStat"]] = relationship(back_populates="team")
    stadium_factors: Mapped[list["StadiumFactor"]] = relationship(back_populates="team")


class Player(Base):
    __tablename__ = "players"

    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_name: Mapped[str] = mapped_column(String(100))
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.team_id"))
    position: Mapped[str] = mapped_column(String(3))
    bats: Mapped[str] = mapped_column(String(1))
    throws: Mapped[str] = mapped_column(String(1))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("NOW()")
    )

    team: Mapped["Team"] = relationship(back_populates="players")
    game_stats: Mapped[list["PlayerGameStat"]] = relationship(back_populates="player")
    bullpen_usage: Mapped[list["BullpenUsage"]] = relationship(back_populates="pitcher")


class Game(Base):
    __tablename__ = "games"

    game_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    game_date: Mapped[date] = mapped_column(Date)
    home_team_id: Mapped[str] = mapped_column(ForeignKey("teams.team_id"))
    away_team_id: Mapped[str] = mapped_column(ForeignKey("teams.team_id"))
    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(20))
    attendance: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("NOW()")
    )

    home_team: Mapped["Team"] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped["Team"] = relationship(foreign_keys=[away_team_id])
    player_stats: Mapped[list["PlayerGameStat"]] = relationship(back_populates="game")
    bullpen_usage: Mapped[list["BullpenUsage"]] = relationship(back_populates="game")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="game")


# ─── Stats Tables ──────────────────────────────────────────────


class TeamDailyStat(Base):
    __tablename__ = "team_daily_stats"
    __table_args__ = (
        UniqueConstraint("team_id", "stat_date"),
        Index("idx_team_daily_stats_date", "stat_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.team_id"))
    stat_date: Mapped[date] = mapped_column(Date)

    # Offense
    runs_scored: Mapped[int | None] = mapped_column(Integer, nullable=True)
    batting_avg: Mapped[float | None] = mapped_column(Numeric(4, 3), nullable=True)
    obp: Mapped[float | None] = mapped_column(Numeric(4, 3), nullable=True)
    slg: Mapped[float | None] = mapped_column(Numeric(4, 3), nullable=True)
    ops: Mapped[float | None] = mapped_column(Numeric(4, 3), nullable=True)
    woba: Mapped[float | None] = mapped_column(Numeric(4, 3), nullable=True)
    wrc_plus: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_runs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stolen_bases: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Pitching
    runs_allowed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    era: Mapped[float | None] = mapped_column(Numeric(5, 3), nullable=True)
    fip: Mapped[float | None] = mapped_column(Numeric(5, 3), nullable=True)
    xfip: Mapped[float | None] = mapped_column(Numeric(5, 3), nullable=True)
    whip: Mapped[float | None] = mapped_column(Numeric(4, 3), nullable=True)
    k9: Mapped[float | None] = mapped_column(Numeric(4, 2), nullable=True)
    bb9: Mapped[float | None] = mapped_column(Numeric(4, 2), nullable=True)
    hr9: Mapped[float | None] = mapped_column(Numeric(4, 2), nullable=True)

    # Defense
    fielding_pct: Mapped[float | None] = mapped_column(Numeric(4, 3), nullable=True)
    drs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    uzr: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    oaa: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Advanced
    hard_hit_pct: Mapped[float | None] = mapped_column(Numeric(4, 3), nullable=True)
    barrel_pct: Mapped[float | None] = mapped_column(Numeric(4, 3), nullable=True)
    ground_ball_pct: Mapped[float | None] = mapped_column(Numeric(4, 3), nullable=True)

    team: Mapped["Team"] = relationship(back_populates="daily_stats")


class PlayerGameStat(Base):
    __tablename__ = "player_game_stats"
    __table_args__ = (Index("idx_player_game_stats_game", "game_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.player_id"))
    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"))
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.team_id"))

    # Batting
    at_bats: Mapped[int | None] = mapped_column(Integer, nullable=True)
    runs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hits: Mapped[int | None] = mapped_column(Integer, nullable=True)
    doubles: Mapped[int | None] = mapped_column(Integer, nullable=True)
    triples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_runs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rbi: Mapped[int | None] = mapped_column(Integer, nullable=True)
    walks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    strikeouts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stolen_bases: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Pitching (if applicable)
    innings_pitched: Mapped[float | None] = mapped_column(Numeric(4, 1), nullable=True)
    hits_allowed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    runs_allowed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    earned_runs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    walks_allowed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    strikeouts_recorded: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pitches_thrown: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Advanced
    exit_velocity_avg: Mapped[float | None] = mapped_column(Numeric(4, 1), nullable=True)
    launch_angle_avg: Mapped[float | None] = mapped_column(Numeric(4, 1), nullable=True)
    hard_hit_pct: Mapped[float | None] = mapped_column(Numeric(4, 3), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("NOW()")
    )

    player: Mapped["Player"] = relationship(back_populates="game_stats")
    game: Mapped["Game"] = relationship(back_populates="player_stats")


class BullpenUsage(Base):
    __tablename__ = "bullpen_usage"
    __table_args__ = (Index("idx_bullpen_usage_game", "game_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pitcher_id: Mapped[int] = mapped_column(ForeignKey("players.player_id"))
    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"))
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.team_id"))

    pitches_thrown: Mapped[int | None] = mapped_column(Integer, nullable=True)
    innings_pitched: Mapped[float | None] = mapped_column(Numeric(3, 1), nullable=True)
    leverage_index_avg: Mapped[float | None] = mapped_column(Numeric(4, 2), nullable=True)
    rest_days_before: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Performance
    era_after: Mapped[float | None] = mapped_column(Numeric(5, 3), nullable=True)
    velocity_drop: Mapped[float | None] = mapped_column(Numeric(3, 1), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("NOW()")
    )

    pitcher: Mapped["Player"] = relationship(back_populates="bullpen_usage")
    game: Mapped["Game"] = relationship(back_populates="bullpen_usage")


class StadiumFactor(Base):
    __tablename__ = "stadium_factors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.team_id"))
    season: Mapped[int] = mapped_column(Integer)

    overall_pf: Mapped[int | None] = mapped_column(Integer, nullable=True)
    runs_pf: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hr_pf: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hits_pf: Mapped[int | None] = mapped_column(Integer, nullable=True)
    doubles_pf: Mapped[int | None] = mapped_column(Integer, nullable=True)
    triples_pf: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bb_pf: Mapped[int | None] = mapped_column(Integer, nullable=True)
    k_pf: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lh_hr_pf: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rh_hr_pf: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("NOW()")
    )

    team: Mapped["Team"] = relationship(back_populates="stadium_factors")


# ─── Prediction & Betting Tables ──────────────────────────────


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (Index("idx_predictions_game", "game_id"),)

    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"))
    model_version: Mapped[str] = mapped_column(String(10))
    home_team_id: Mapped[str] = mapped_column(String(3))
    away_team_id: Mapped[str] = mapped_column(String(3))
    home_win_prob: Mapped[float] = mapped_column(Numeric(5, 4))
    predicted_home_runs: Mapped[float] = mapped_column(Numeric(4, 2))
    predicted_away_runs: Mapped[float] = mapped_column(Numeric(4, 2))
    predicted_total: Mapped[float] = mapped_column(Numeric(4, 2))
    confidence: Mapped[float] = mapped_column(Numeric(4, 2))
    features_used: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    stadium_impact: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    bullpen_impact: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("NOW()")
    )

    game: Mapped["Game"] = relationship(back_populates="predictions")
    bets: Mapped[list["BettingResult"]] = relationship(back_populates="prediction")


class BettingResult(Base):
    __tablename__ = "betting_results"
    __table_args__ = (Index("idx_betting_results_date", "placed_at"),)

    bet_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("predictions.prediction_id")
    )
    bet_type: Mapped[str] = mapped_column(String(20))  # moneyline, total, runline
    selection: Mapped[str] = mapped_column(String(10))  # home, away, over, under
    odds: Mapped[float] = mapped_column(Numeric(7, 2))
    stake: Mapped[float] = mapped_column(Numeric(10, 2))
    kelly_fraction: Mapped[float] = mapped_column(Numeric(4, 2))
    result: Mapped[str | None] = mapped_column(String(10), nullable=True)  # win, loss, push
    profit_loss: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)
    placed_at: Mapped[datetime] = mapped_column(DateTime)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    prediction: Mapped["Prediction"] = relationship(back_populates="bets")
