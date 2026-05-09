"""Data quality validation for incoming MLB data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)


class DataValidator:
    """Validates incoming MLB data for quality and consistency."""

    # Reasonable stat ranges for outlier detection
    BATTING_RANGES = {
        "batting_avg": (0.0, 0.500),
        "obp": (0.0, 0.600),
        "slg": (0.0, 1.000),
        "ops": (0.0, 1.600),
    }

    PITCHING_RANGES = {
        "era": (0.0, 15.0),
        "whip": (0.0, 3.5),
        "k9": (0.0, 20.0),
        "bb9": (0.0, 12.0),
    }

    def validate_game(self, game: dict[str, Any]) -> ValidationResult:
        """Validate a single game record."""
        result = ValidationResult(passed=True)

        # Required fields
        for field_name in ("game_id", "game_date", "home_team_id", "away_team_id", "status"):
            if not game.get(field_name):
                result.add_error(f"Missing required field: {field_name}")

        # Same team check
        if game.get("home_team_id") == game.get("away_team_id"):
            result.add_error("Home and away team cannot be the same")

        # Score validation for completed games
        if game.get("status") == "Final":
            if game.get("home_score") is None or game.get("away_score") is None:
                result.add_error("Final game missing scores")
            elif game["home_score"] == game["away_score"]:
                result.add_warning("Tie game — check if suspended or extra innings")

        # Date validation
        game_date = game.get("game_date")
        if game_date:
            if isinstance(game_date, str):
                try:
                    game_date = date.fromisoformat(game_date)
                except ValueError:
                    result.add_error(f"Invalid date format: {game_date}")
                    return result
            # MLB season is roughly March–November
            if game_date.month < 2 or game_date.month > 11:
                result.add_warning(f"Game date {game_date} outside normal season window")

        return result

    def validate_boxscore(self, boxscore: dict[str, Any]) -> ValidationResult:
        """Validate a boxscore for consistency."""
        result = ValidationResult(passed=True)

        if not boxscore.get("game_id"):
            result.add_error("Boxscore missing game_id")
            return result

        for side in ("home", "away"):
            side_data = boxscore.get(side, {})
            if not side_data.get("team_id"):
                result.add_error(f"Missing {side} team_id in boxscore")

            # Verify at least some batters
            batters = side_data.get("batters", [])
            if len(batters) < 9:
                result.add_warning(f"{side} has fewer than 9 batters ({len(batters)})")

            # Verify at least one pitcher
            pitchers = side_data.get("pitchers", [])
            if len(pitchers) < 1:
                result.add_error(f"{side} has no pitchers in boxscore")

            # Validate individual stat lines
            for batter in batters:
                self._validate_batter_line(batter, result)
            for pitcher in pitchers:
                self._validate_pitcher_line(pitcher, result)

        return result

    def validate_team_stats(
        self, stats: dict[str, Any], stat_type: str = "hitting"
    ) -> ValidationResult:
        """Validate team aggregate statistics."""
        result = ValidationResult(passed=True)

        ranges = self.BATTING_RANGES if stat_type == "hitting" else self.PITCHING_RANGES
        for field_name, (lo, hi) in ranges.items():
            val = stats.get(field_name)
            if val is not None:
                try:
                    val = float(val)
                    if val < lo or val > hi:
                        result.add_warning(
                            f"{field_name}={val} outside expected range [{lo}, {hi}]"
                        )
                except (ValueError, TypeError):
                    result.add_error(f"{field_name} is not numeric: {val}")

        return result

    def validate_batch(
        self, games: list[dict[str, Any]]
    ) -> tuple[list[dict], list[dict]]:
        """Validate a batch of games. Returns (valid, invalid) lists."""
        valid = []
        invalid = []
        for game in games:
            result = self.validate_game(game)
            if result.passed:
                valid.append(game)
            else:
                logger.warning(
                    "Game %s failed validation: %s",
                    game.get("game_id", "?"),
                    result.errors,
                )
                invalid.append({"game": game, "errors": result.errors})
        return valid, invalid

    def _validate_batter_line(self, batter: dict, result: ValidationResult):
        pid = batter.get("player_id", "?")
        ab = batter.get("at_bats") or 0
        hits = batter.get("hits") or 0
        if hits > ab:
            result.add_error(f"Player {pid}: hits ({hits}) > at_bats ({ab})")

        hr = batter.get("home_runs") or 0
        if hr > hits:
            result.add_warning(f"Player {pid}: home_runs ({hr}) > hits ({hits})")

    def _validate_pitcher_line(self, pitcher: dict, result: ValidationResult):
        pid = pitcher.get("player_id", "?")
        ip = pitcher.get("innings_pitched")
        if ip is not None and ip < 0:
            result.add_error(f"Pitcher {pid}: negative innings pitched")

        pitches = pitcher.get("pitches_thrown")
        if pitches is not None and pitches > 150:
            result.add_warning(f"Pitcher {pid}: unusually high pitch count ({pitches})")
