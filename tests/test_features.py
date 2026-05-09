"""Tests for feature engineering modules."""

import pytest

from mlb.features.stadium import (
    STADIUM_INFO,
    StadiumFeatures,
    WeatherConditions,
    calculate_stadium_factor,
    compute_stadium_features,
)


class TestStadiumInfo:
    def test_all_30_teams(self):
        expected = {
            "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE",
            "COL", "DET", "HOU", "KC", "LAA", "LAD", "MIA", "MIL",
            "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SD", "SF",
            "SEA", "STL", "TB", "TEX", "TOR", "WSH",
        }
        assert set(STADIUM_INFO.keys()) == expected

    def test_coors_field_altitude(self):
        assert STADIUM_INFO["COL"]["altitude"] == 5280

    def test_tropicana_is_dome(self):
        assert STADIUM_INFO["TB"]["roof"] == "dome"

    def test_turf_surfaces(self):
        turf_teams = [t for t, info in STADIUM_INFO.items() if info["surface"] == "turf"]
        assert "TB" in turf_teams
        assert "TOR" in turf_teams


class TestStadiumFactor:
    def test_neutral_stadium(self):
        f = StadiumFeatures(
            overall_pf=100, runs_pf=100, hr_pf=100, hits_pf=100,
            doubles_pf=100, triples_pf=100, lh_hr_pf=100, rh_hr_pf=100,
            avg_dimension=395, dimension_score=0.79,
            altitude_ft=0, altitude_factor=1.0,
            temp_factor=1.0, wind_factor=1.0, humidity_factor=1.0,
            is_turf=False, surface_speed_factor=1.0,
            roof_status="open", weather_neutralized=False,
            home_field_advantage=0.537,
        )
        factor = calculate_stadium_factor(f)
        assert factor == pytest.approx(1.0, abs=0.02)

    def test_hitter_friendly(self):
        f = StadiumFeatures(
            overall_pf=110, runs_pf=115, hr_pf=120, hits_pf=105,
            doubles_pf=105, triples_pf=100, lh_hr_pf=125, rh_hr_pf=115,
            avg_dimension=370, dimension_score=0.58,
            altitude_ft=5280, altitude_factor=1.053,
            temp_factor=1.01, wind_factor=1.05, humidity_factor=1.0,
            is_turf=False, surface_speed_factor=1.0,
            roof_status="open", weather_neutralized=False,
            home_field_advantage=0.537,
        )
        factor = calculate_stadium_factor(f)
        assert factor > 1.0  # Hitter friendly

    def test_pitcher_friendly(self):
        f = StadiumFeatures(
            overall_pf=90, runs_pf=88, hr_pf=85, hits_pf=92,
            doubles_pf=95, triples_pf=100, lh_hr_pf=82, rh_hr_pf=88,
            avg_dimension=410, dimension_score=0.92,
            altitude_ft=0, altitude_factor=1.0,
            temp_factor=0.99, wind_factor=0.96, humidity_factor=1.0,
            is_turf=False, surface_speed_factor=1.0,
            roof_status="open", weather_neutralized=False,
            home_field_advantage=0.537,
        )
        factor = calculate_stadium_factor(f)
        assert factor < 1.0  # Pitcher friendly


class TestComputeStadiumFeatures:
    def test_default_park_factors(self):
        f = compute_stadium_features("NYY", None, None)
        assert f.overall_pf == 100  # Default
        assert f.altitude_ft == 55  # Yankee Stadium
        assert f.roof_status == "open"

    def test_dome_neutralizes_weather(self):
        weather = WeatherConditions(
            temperature_f=40, wind_speed_mph=30,
            wind_direction="out", humidity_pct=0.9,
            precipitation_prob=0.5, is_dome=True,
        )
        f = compute_stadium_features("TB", None, weather)
        assert f.weather_neutralized is True
        assert f.temp_factor == 1.0
        assert f.wind_factor == 1.0

    def test_retractable_closes_in_rain(self):
        weather = WeatherConditions(
            temperature_f=72, wind_speed_mph=5,
            wind_direction="out", humidity_pct=0.5,
            precipitation_prob=0.0, is_dome=True,
        )
        f = compute_stadium_features("HOU", None, weather)
        assert f.weather_neutralized is True

    def test_coors_high_altitude(self):
        f = compute_stadium_features("COL", None, None)
        assert f.altitude_ft == 5280
        assert f.altitude_factor > 1.0
