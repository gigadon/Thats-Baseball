"""Tests for the weather client."""

import pytest

from mlb.data.weather import (
    STADIUM_COORDS,
    WeatherClient,
    _classify_wind_direction,
)


class TestStadiumCoords:
    def test_all_30_teams(self):
        expected = {
            "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE",
            "COL", "DET", "HOU", "KC", "LAA", "LAD", "MIA", "MIL",
            "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SD", "SF",
            "SEA", "STL", "TB", "TEX", "TOR", "WSH",
        }
        assert set(STADIUM_COORDS.keys()) == expected

    def test_coords_are_valid(self):
        for team, (lat, lon) in STADIUM_COORDS.items():
            assert -90 <= lat <= 90, f"{team} latitude out of range: {lat}"
            assert -180 <= lon <= 180, f"{team} longitude out of range: {lon}"


class TestWindDirection:
    def test_out(self):
        assert _classify_wind_direction(0) == "out"
        assert _classify_wind_direction(350) == "out"

    def test_in(self):
        assert _classify_wind_direction(180) == "in"

    def test_cross(self):
        assert _classify_wind_direction(90) == "cross"
        assert _classify_wind_direction(270) == "cross"


class TestWeatherParser:
    def setup_method(self):
        self.client = WeatherClient(api_key="test_key")

    def test_parse_normal_weather(self):
        data = {
            "main": {"temp": 75.0, "humidity": 60},
            "wind": {"speed": 10, "deg": 180},
        }
        wc = self.client._parse_weather(data, "NYY", is_dome=False)
        assert wc.temperature_f == 75.0
        assert wc.wind_speed_mph == 10.0
        assert wc.wind_direction == "in"
        assert wc.humidity_pct == 0.60
        assert wc.is_dome is False

    def test_parse_dome_team(self):
        data = {
            "main": {"temp": 95.0, "humidity": 80},
            "wind": {"speed": 15, "deg": 0},
        }
        wc = self.client._parse_weather(data, "TB", is_dome=True)
        assert wc.is_dome is True

    def test_retractable_roof_closes_in_bad_weather(self):
        data = {
            "main": {"temp": 40.0, "humidity": 90},
            "wind": {"speed": 25, "deg": 0},
            "rain": {"1h": 5},
        }
        wc = self.client._parse_weather(data, "HOU", is_dome=False)
        # Houston is retractable — should close due to cold + rain
        assert wc.is_dome is True

    def test_calm_wind(self):
        data = {
            "main": {"temp": 72.0, "humidity": 50},
            "wind": {"speed": 1, "deg": 90},
        }
        wc = self.client._parse_weather(data, "LAD", is_dome=False)
        assert wc.wind_direction == "calm"
