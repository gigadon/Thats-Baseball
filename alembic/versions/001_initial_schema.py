"""Initial schema — all core tables.

Revision ID: 001
Revises: None
Create Date: 2026-05-09
"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Teams ──────────────────────────────────────────────
    op.create_table(
        "teams",
        sa.Column("team_id", sa.String(3), primary_key=True),
        sa.Column("team_name", sa.String(50), nullable=False),
        sa.Column("division", sa.String(20), nullable=False),
        sa.Column("league", sa.String(2), nullable=False),
        sa.Column("stadium_name", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
    )

    # ── Players ────────────────────────────────────────────
    op.create_table(
        "players",
        sa.Column("player_id", sa.Integer, primary_key=True),
        sa.Column("player_name", sa.String(100), nullable=False),
        sa.Column("team_id", sa.String(3), sa.ForeignKey("teams.team_id"), nullable=False),
        sa.Column("position", sa.String(3), nullable=False),
        sa.Column("bats", sa.String(1), nullable=False),
        sa.Column("throws", sa.String(1), nullable=False),
        sa.Column("active", sa.Boolean, default=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
    )

    # ── Games ──────────────────────────────────────────────
    op.create_table(
        "games",
        sa.Column("game_id", sa.String(20), primary_key=True),
        sa.Column("game_date", sa.Date, nullable=False),
        sa.Column("home_team_id", sa.String(3), sa.ForeignKey("teams.team_id"), nullable=False),
        sa.Column("away_team_id", sa.String(3), sa.ForeignKey("teams.team_id"), nullable=False),
        sa.Column("home_score", sa.Integer, nullable=True),
        sa.Column("away_score", sa.Integer, nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("attendance", sa.Integer, nullable=True),
        sa.Column("duration_minutes", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
    )

    # ── Team Daily Stats ───────────────────────────────────
    op.create_table(
        "team_daily_stats",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("team_id", sa.String(3), sa.ForeignKey("teams.team_id"), nullable=False),
        sa.Column("stat_date", sa.Date, nullable=False),
        # Offense
        sa.Column("runs_scored", sa.Integer),
        sa.Column("batting_avg", sa.Numeric(4, 3)),
        sa.Column("obp", sa.Numeric(4, 3)),
        sa.Column("slg", sa.Numeric(4, 3)),
        sa.Column("ops", sa.Numeric(4, 3)),
        sa.Column("woba", sa.Numeric(4, 3)),
        sa.Column("wrc_plus", sa.Integer),
        sa.Column("home_runs", sa.Integer),
        sa.Column("stolen_bases", sa.Integer),
        # Pitching
        sa.Column("runs_allowed", sa.Integer),
        sa.Column("era", sa.Numeric(5, 3)),
        sa.Column("fip", sa.Numeric(5, 3)),
        sa.Column("xfip", sa.Numeric(5, 3)),
        sa.Column("whip", sa.Numeric(4, 3)),
        sa.Column("k9", sa.Numeric(4, 2)),
        sa.Column("bb9", sa.Numeric(4, 2)),
        sa.Column("hr9", sa.Numeric(4, 2)),
        # Defense
        sa.Column("fielding_pct", sa.Numeric(4, 3)),
        sa.Column("drs", sa.Integer),
        sa.Column("uzr", sa.Numeric(5, 2)),
        sa.Column("oaa", sa.Integer),
        # Advanced
        sa.Column("hard_hit_pct", sa.Numeric(4, 3)),
        sa.Column("barrel_pct", sa.Numeric(4, 3)),
        sa.Column("ground_ball_pct", sa.Numeric(4, 3)),
        sa.UniqueConstraint("team_id", "stat_date"),
    )
    op.create_index("idx_team_daily_stats_date", "team_daily_stats", ["stat_date"])

    # ── Player Game Stats ──────────────────────────────────
    op.create_table(
        "player_game_stats",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("player_id", sa.Integer, sa.ForeignKey("players.player_id"), nullable=False),
        sa.Column("game_id", sa.String(20), sa.ForeignKey("games.game_id"), nullable=False),
        sa.Column("team_id", sa.String(3), sa.ForeignKey("teams.team_id"), nullable=False),
        # Batting
        sa.Column("at_bats", sa.Integer),
        sa.Column("runs", sa.Integer),
        sa.Column("hits", sa.Integer),
        sa.Column("doubles", sa.Integer),
        sa.Column("triples", sa.Integer),
        sa.Column("home_runs", sa.Integer),
        sa.Column("rbi", sa.Integer),
        sa.Column("walks", sa.Integer),
        sa.Column("strikeouts", sa.Integer),
        sa.Column("stolen_bases", sa.Integer),
        # Pitching
        sa.Column("innings_pitched", sa.Numeric(4, 1)),
        sa.Column("hits_allowed", sa.Integer),
        sa.Column("runs_allowed", sa.Integer),
        sa.Column("earned_runs", sa.Integer),
        sa.Column("walks_allowed", sa.Integer),
        sa.Column("strikeouts_recorded", sa.Integer),
        sa.Column("pitches_thrown", sa.Integer),
        # Advanced
        sa.Column("exit_velocity_avg", sa.Numeric(4, 1)),
        sa.Column("launch_angle_avg", sa.Numeric(4, 1)),
        sa.Column("hard_hit_pct", sa.Numeric(4, 3)),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
    )
    op.create_index("idx_player_game_stats_game", "player_game_stats", ["game_id"])

    # ── Bullpen Usage ──────────────────────────────────────
    op.create_table(
        "bullpen_usage",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("pitcher_id", sa.Integer, sa.ForeignKey("players.player_id"), nullable=False),
        sa.Column("game_id", sa.String(20), sa.ForeignKey("games.game_id"), nullable=False),
        sa.Column("team_id", sa.String(3), sa.ForeignKey("teams.team_id"), nullable=False),
        sa.Column("pitches_thrown", sa.Integer),
        sa.Column("innings_pitched", sa.Numeric(3, 1)),
        sa.Column("leverage_index_avg", sa.Numeric(4, 2)),
        sa.Column("rest_days_before", sa.Integer),
        sa.Column("era_after", sa.Numeric(5, 3)),
        sa.Column("velocity_drop", sa.Numeric(3, 1)),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
    )
    op.create_index("idx_bullpen_usage_game", "bullpen_usage", ["game_id"])

    # ── Stadium Factors ────────────────────────────────────
    op.create_table(
        "stadium_factors",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("team_id", sa.String(3), sa.ForeignKey("teams.team_id"), nullable=False),
        sa.Column("season", sa.Integer, nullable=False),
        sa.Column("overall_pf", sa.Integer),
        sa.Column("runs_pf", sa.Integer),
        sa.Column("hr_pf", sa.Integer),
        sa.Column("hits_pf", sa.Integer),
        sa.Column("doubles_pf", sa.Integer),
        sa.Column("triples_pf", sa.Integer),
        sa.Column("bb_pf", sa.Integer),
        sa.Column("k_pf", sa.Integer),
        sa.Column("lh_hr_pf", sa.Integer),
        sa.Column("rh_hr_pf", sa.Integer),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
    )

    # ── Predictions ────────────────────────────────────────
    op.create_table(
        "predictions",
        sa.Column("prediction_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("game_id", sa.String(20), sa.ForeignKey("games.game_id"), nullable=False),
        sa.Column("model_version", sa.String(10), nullable=False),
        sa.Column("home_team_id", sa.String(3), nullable=False),
        sa.Column("away_team_id", sa.String(3), nullable=False),
        sa.Column("home_win_prob", sa.Numeric(5, 4), nullable=False),
        sa.Column("predicted_home_runs", sa.Numeric(4, 2), nullable=False),
        sa.Column("predicted_away_runs", sa.Numeric(4, 2), nullable=False),
        sa.Column("predicted_total", sa.Numeric(4, 2), nullable=False),
        sa.Column("confidence", sa.Numeric(4, 2), nullable=False),
        sa.Column("features_used", postgresql.JSONB),
        sa.Column("stadium_impact", postgresql.JSONB),
        sa.Column("bullpen_impact", postgresql.JSONB),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
    )
    op.create_index("idx_predictions_game", "predictions", ["game_id"])

    # ── Betting Results ────────────────────────────────────
    op.create_table(
        "betting_results",
        sa.Column("bet_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "prediction_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("predictions.prediction_id"),
            nullable=False,
        ),
        sa.Column("bet_type", sa.String(20), nullable=False),
        sa.Column("selection", sa.String(10), nullable=False),
        sa.Column("odds", sa.Numeric(7, 2), nullable=False),
        sa.Column("stake", sa.Numeric(10, 2), nullable=False),
        sa.Column("kelly_fraction", sa.Numeric(4, 2), nullable=False),
        sa.Column("result", sa.String(10)),
        sa.Column("profit_loss", sa.Numeric(10, 2)),
        sa.Column("placed_at", sa.DateTime, nullable=False),
        sa.Column("settled_at", sa.DateTime),
    )
    op.create_index("idx_betting_results_date", "betting_results", ["placed_at"])


def downgrade() -> None:
    op.drop_table("betting_results")
    op.drop_table("predictions")
    op.drop_table("stadium_factors")
    op.drop_table("bullpen_usage")
    op.drop_table("player_game_stats")
    op.drop_table("team_daily_stats")
    op.drop_table("games")
    op.drop_table("players")
    op.drop_table("teams")
