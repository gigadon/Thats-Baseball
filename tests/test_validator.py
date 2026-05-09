"""Tests for the data validator."""

import pytest
from datetime import date

from mlb.data.validator import DataValidator


class TestGameValidation:
    def setup_method(self):
        self.v = DataValidator()

    def test_valid_game(self):
        game = {
            "game_id": "123456",
            "game_date": date(2026, 7, 15),
            "home_team_id": "NYY",
            "away_team_id": "BOS",
            "status": "Final",
            "home_score": 5,
            "away_score": 3,
        }
        result = self.v.validate_game(game)
        assert result.passed is True
        assert len(result.errors) == 0

    def test_missing_game_id(self):
        game = {
            "game_date": date(2026, 7, 15),
            "home_team_id": "NYY",
            "away_team_id": "BOS",
            "status": "Final",
        }
        result = self.v.validate_game(game)
        assert result.passed is False
        assert any("game_id" in e for e in result.errors)

    def test_same_team(self):
        game = {
            "game_id": "123456",
            "game_date": date(2026, 7, 15),
            "home_team_id": "NYY",
            "away_team_id": "NYY",
            "status": "Final",
            "home_score": 5,
            "away_score": 3,
        }
        result = self.v.validate_game(game)
        assert result.passed is False
        assert any("same" in e.lower() for e in result.errors)

    def test_final_game_missing_scores(self):
        game = {
            "game_id": "123456",
            "game_date": date(2026, 7, 15),
            "home_team_id": "NYY",
            "away_team_id": "BOS",
            "status": "Final",
            "home_score": None,
            "away_score": None,
        }
        result = self.v.validate_game(game)
        assert result.passed is False

    def test_date_outside_season(self):
        game = {
            "game_id": "123456",
            "game_date": date(2026, 1, 15),
            "home_team_id": "NYY",
            "away_team_id": "BOS",
            "status": "Scheduled",
        }
        result = self.v.validate_game(game)
        assert len(result.warnings) > 0


class TestBoxscoreValidation:
    def setup_method(self):
        self.v = DataValidator()

    def test_valid_boxscore(self):
        box = {
            "game_id": "123456",
            "home": {
                "team_id": "NYY",
                "batters": [
                    {"player_id": 1, "at_bats": 4, "hits": 2, "home_runs": 0},
                ],
                "pitchers": [
                    {"player_id": 2, "innings_pitched": 7.0, "pitches_thrown": 95},
                ],
            },
            "away": {
                "team_id": "BOS",
                "batters": [
                    {"player_id": 3, "at_bats": 4, "hits": 1, "home_runs": 0},
                ],
                "pitchers": [
                    {"player_id": 4, "innings_pitched": 6.0, "pitches_thrown": 88},
                ],
            },
        }
        result = self.v.validate_boxscore(box)
        assert len(result.errors) == 0

    def test_hits_exceed_at_bats(self):
        box = {
            "game_id": "123456",
            "home": {
                "team_id": "NYY",
                "batters": [
                    {"player_id": 1, "at_bats": 3, "hits": 5, "home_runs": 0},
                ],
                "pitchers": [
                    {"player_id": 2, "innings_pitched": 7.0, "pitches_thrown": 95},
                ],
            },
            "away": {
                "team_id": "BOS",
                "batters": [
                    {"player_id": 3, "at_bats": 4, "hits": 1, "home_runs": 0},
                ],
                "pitchers": [
                    {"player_id": 4, "innings_pitched": 6.0, "pitches_thrown": 88},
                ],
            },
        }
        result = self.v.validate_boxscore(box)
        assert any("hits" in e.lower() for e in result.errors)

    def test_no_pitchers_error(self):
        box = {
            "game_id": "123456",
            "home": {
                "team_id": "NYY",
                "batters": [],
                "pitchers": [],
            },
            "away": {
                "team_id": "BOS",
                "batters": [],
                "pitchers": [],
            },
        }
        result = self.v.validate_boxscore(box)
        assert result.passed is False


class TestBatchValidation:
    def setup_method(self):
        self.v = DataValidator()

    def test_batch_splits_valid_invalid(self):
        games = [
            {"game_id": "1", "game_date": date(2026, 7, 15), "home_team_id": "NYY", "away_team_id": "BOS", "status": "Final", "home_score": 5, "away_score": 3},
            {"game_date": date(2026, 7, 15), "home_team_id": "NYY", "away_team_id": "BOS", "status": "Final"},  # missing game_id
        ]
        valid, invalid = self.v.validate_batch(games)
        assert len(valid) == 1
        assert len(invalid) == 1
