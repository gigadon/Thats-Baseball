"""Tests for the Odds API client."""

import pytest

from mlb.data.odds_api import OddsApiClient, _team_abbrev


class TestTeamMapping:
    def test_known_teams(self):
        assert _team_abbrev("New York Yankees") == "NYY"
        assert _team_abbrev("Los Angeles Dodgers") == "LAD"
        assert _team_abbrev("Chicago Cubs") == "CHC"
        assert _team_abbrev("Tampa Bay Rays") == "TB"

    def test_unknown_team_returns_input(self):
        assert _team_abbrev("Unknown Team") == "Unknown Team"


class TestOddsParser:
    def setup_method(self):
        self.client = OddsApiClient(api_key="test_key")

    def test_parse_empty(self):
        assert self.client._parse_odds([]) == []

    def test_parse_single_game(self):
        raw = [
            {
                "id": "abc123",
                "home_team": "New York Yankees",
                "away_team": "Boston Red Sox",
                "commence_time": "2026-05-09T23:05:00Z",
                "bookmakers": [
                    {
                        "key": "fanduel",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "New York Yankees", "price": -150},
                                    {"name": "Boston Red Sox", "price": 130},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "price": -110, "point": 8.5},
                                    {"name": "Under", "price": -110},
                                ],
                            },
                        ],
                    }
                ],
            }
        ]

        result = self.client._parse_odds(raw)
        assert len(result) == 1

        game = result[0]
        assert game["home_team"] == "NYY"
        assert game["away_team"] == "BOS"
        assert game["home_moneyline"] == -150
        assert game["away_moneyline"] == 130
        assert game["total_line"] == 8.5
        assert game["over_odds"] == -110
        assert game["under_odds"] == -110

    def test_parse_missing_totals(self):
        raw = [
            {
                "id": "abc123",
                "home_team": "Los Angeles Dodgers",
                "away_team": "San Francisco Giants",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Los Angeles Dodgers", "price": -200},
                                    {"name": "San Francisco Giants", "price": 170},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]

        result = self.client._parse_odds(raw)
        game = result[0]
        assert game["home_moneyline"] == -200
        assert game["total_line"] is None

    def test_explicit_api_key(self):
        client = OddsApiClient(api_key="my_test_key")
        assert client.api_key == "my_test_key"
