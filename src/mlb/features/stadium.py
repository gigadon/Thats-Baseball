"""Stadium adjustment factor calculations.

Stadium_Factor =
    Base_Park_Factor  × 0.35 +
    Run_Environment   × 0.25 +
    Dimension_Impact  × 0.15 +
    Altitude_Impact   × 0.10 +
    Weather_Impact    × 0.10 +
    Surface_Impact    × 0.05
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ─── Stadium Reference Data ───────────────────────────────────

# Altitude in feet, dimensions in feet, surface type
STADIUM_INFO: dict[str, dict] = {
    "COL": {"altitude": 5280, "lf": 347, "cf": 415, "rf": 350, "surface": "grass", "roof": "open"},
    "ARI": {"altitude": 1082, "lf": 330, "cf": 407, "rf": 334, "surface": "grass", "roof": "retractable"},
    "ATL": {"altitude": 1050, "lf": 335, "cf": 400, "rf": 325, "surface": "grass", "roof": "open"},
    "BAL": {"altitude": 30, "lf": 333, "cf": 400, "rf": 318, "surface": "grass", "roof": "open"},
    "BOS": {"altitude": 20, "lf": 310, "cf": 390, "rf": 302, "surface": "grass", "roof": "open"},
    "CHC": {"altitude": 595, "lf": 355, "cf": 400, "rf": 353, "surface": "grass", "roof": "open"},
    "CWS": {"altitude": 595, "lf": 330, "cf": 400, "rf": 335, "surface": "grass", "roof": "open"},
    "CIN": {"altitude": 490, "lf": 328, "cf": 404, "rf": 325, "surface": "grass", "roof": "open"},
    "CLE": {"altitude": 653, "lf": 325, "cf": 405, "rf": 325, "surface": "grass", "roof": "open"},
    "DET": {"altitude": 600, "lf": 345, "cf": 420, "rf": 330, "surface": "grass", "roof": "open"},
    "HOU": {"altitude": 50, "lf": 315, "cf": 409, "rf": 326, "surface": "grass", "roof": "retractable"},
    "KC": {"altitude": 750, "lf": 330, "cf": 410, "rf": 330, "surface": "grass", "roof": "open"},
    "LAA": {"altitude": 160, "lf": 347, "cf": 396, "rf": 350, "surface": "grass", "roof": "open"},
    "LAD": {"altitude": 515, "lf": 330, "cf": 395, "rf": 330, "surface": "grass", "roof": "open"},
    "MIA": {"altitude": 6, "lf": 344, "cf": 407, "rf": 335, "surface": "grass", "roof": "retractable"},
    "MIL": {"altitude": 600, "lf": 344, "cf": 400, "rf": 345, "surface": "grass", "roof": "retractable"},
    "MIN": {"altitude": 841, "lf": 339, "cf": 411, "rf": 328, "surface": "grass", "roof": "open"},
    "NYM": {"altitude": 12, "lf": 335, "cf": 408, "rf": 330, "surface": "grass", "roof": "open"},
    "NYY": {"altitude": 55, "lf": 318, "cf": 408, "rf": 314, "surface": "grass", "roof": "open"},
    "OAK": {"altitude": 25, "lf": 330, "cf": 400, "rf": 330, "surface": "grass", "roof": "open"},
    "PHI": {"altitude": 20, "lf": 329, "cf": 401, "rf": 330, "surface": "grass", "roof": "open"},
    "PIT": {"altitude": 730, "lf": 325, "cf": 399, "rf": 320, "surface": "grass", "roof": "open"},
    "SD": {"altitude": 17, "lf": 334, "cf": 396, "rf": 322, "surface": "grass", "roof": "open"},
    "SF": {"altitude": 0, "lf": 339, "cf": 399, "rf": 309, "surface": "grass", "roof": "open"},
    "SEA": {"altitude": 17, "lf": 331, "cf": 401, "rf": 326, "surface": "grass", "roof": "retractable"},
    "STL": {"altitude": 455, "lf": 336, "cf": 400, "rf": 335, "surface": "grass", "roof": "open"},
    "TB": {"altitude": 40, "lf": 315, "cf": 404, "rf": 322, "surface": "turf", "roof": "dome"},
    "TEX": {"altitude": 551, "lf": 329, "cf": 407, "rf": 326, "surface": "grass", "roof": "retractable"},
    "TOR": {"altitude": 266, "lf": 328, "cf": 400, "rf": 328, "surface": "turf", "roof": "retractable"},
    "WSH": {"altitude": 25, "lf": 336, "cf": 402, "rf": 335, "surface": "grass", "roof": "open"},
}


@dataclass
class WeatherConditions:
    """Weather at game time."""

    temperature_f: float  # Fahrenheit
    wind_speed_mph: float
    wind_direction: str  # "in", "out", "cross", "calm"
    humidity_pct: float  # 0-1
    precipitation_prob: float  # 0-1
    is_dome: bool


@dataclass
class StadiumFeatures:
    """Stadium-related features for a game."""

    # Park factors (100 = neutral)
    overall_pf: float
    runs_pf: float
    hr_pf: float
    hits_pf: float
    doubles_pf: float
    triples_pf: float
    lh_hr_pf: float  # HR park factor for LHB
    rh_hr_pf: float  # HR park factor for RHB

    # Dimensions
    avg_dimension: float  # Average of LF/CF/RF
    dimension_score: float  # Smaller = more hitter friendly

    # Altitude
    altitude_ft: float
    altitude_factor: float  # >1 = ball carries more

    # Weather impact
    temp_factor: float  # Warm = ball carries
    wind_factor: float  # Positive = out, negative = in
    humidity_factor: float

    # Surface
    is_turf: bool  # Turf vs grass
    surface_speed_factor: float  # Turf = faster

    # Roof
    roof_status: str  # open, retractable, dome
    weather_neutralized: bool  # True if dome/closed roof

    # Home field advantage base
    home_field_advantage: float  # Typically 0.530-0.545 win rate


def calculate_stadium_factor(f: StadiumFeatures) -> float:
    """Calculate composite stadium adjustment factor.

    Returns a multiplier centered on 1.0 where:
      >1.0 = hitter-friendly environment
      <1.0 = pitcher-friendly environment
    """
    # Base park factor (normalize around 1.0)
    base = f.overall_pf / 100.0

    # Run environment
    run_env = f.runs_pf / 100.0

    # Dimensions: smaller avg = more hitter friendly
    # League avg ~395 ft average dimension
    dim_factor = 1.0 + (395 - f.avg_dimension) / 200.0

    # Altitude: ~1% increase in HR per 1000 ft
    alt_factor = f.altitude_factor

    # Weather (only applies if not dome)
    if f.weather_neutralized:
        weather = 1.0
    else:
        weather = f.temp_factor * f.wind_factor * f.humidity_factor

    # Surface: turf slightly boosts hits
    surface = f.surface_speed_factor

    composite = (
        base * 0.35
        + run_env * 0.25
        + dim_factor * 0.15
        + alt_factor * 0.10
        + weather * 0.10
        + surface * 0.05
    )
    return float(composite)


def compute_stadium_features(
    team_id: str,
    park_factors: dict | None,
    weather: WeatherConditions | None,
) -> StadiumFeatures:
    """Build StadiumFeatures from park factors and weather data.

    Args:
        team_id: Home team abbreviation (e.g. "NYY").
        park_factors: Dict from stadium_factors table (or None for defaults).
        weather: WeatherConditions for game time (or None for neutral).
    """
    info = STADIUM_INFO.get(team_id, {})
    altitude = info.get("altitude", 0)
    lf = info.get("lf", 330)
    cf = info.get("cf", 400)
    rf = info.get("rf", 330)
    surface = info.get("surface", "grass")
    roof = info.get("roof", "open")

    # Park factors from DB or defaults
    pf = park_factors or {}
    overall_pf = float(pf.get("overall_pf", 100) or 100)
    runs_pf = float(pf.get("runs_pf", 100) or 100)
    hr_pf = float(pf.get("hr_pf", 100) or 100)
    hits_pf = float(pf.get("hits_pf", 100) or 100)
    doubles_pf = float(pf.get("doubles_pf", 100) or 100)
    triples_pf = float(pf.get("triples_pf", 100) or 100)
    lh_hr_pf = float(pf.get("lh_hr_pf", 100) or 100)
    rh_hr_pf = float(pf.get("rh_hr_pf", 100) or 100)

    # Dimensions
    avg_dim = (lf + cf + rf) / 3.0
    dim_score = (avg_dim - 300) / 120.0  # 0-1 scale, bigger = more pitcher friendly

    # Altitude factor: ball carries ~1% more per 1000 ft
    alt_factor = 1.0 + altitude / 100000.0

    # Weather
    is_dome = roof == "dome"
    weather_neutralized = is_dome
    if weather and roof == "retractable" and weather.is_dome:
        weather_neutralized = True

    if weather and not weather_neutralized:
        # Temperature: warmer = ball carries. ~70°F neutral, each 10°F = ~1-2% effect
        temp_factor = 1.0 + (weather.temperature_f - 70) / 1000.0

        # Wind: blowing out boosts offense, in suppresses
        wind_dir_mult = {"out": 1.0, "cross": 0.3, "calm": 0.0, "in": -0.8}
        wind_mult = wind_dir_mult.get(weather.wind_direction, 0.0)
        wind_factor = 1.0 + (weather.wind_speed_mph * wind_mult) / 200.0

        # Humidity: higher humidity = slightly less ball carry (denser air myth is reversed)
        humidity_factor = 1.0 + (weather.humidity_pct - 0.5) * 0.02
    else:
        temp_factor = 1.0
        wind_factor = 1.0
        humidity_factor = 1.0

    # Surface
    is_turf = surface == "turf"
    surface_factor = 1.02 if is_turf else 1.0  # Turf slightly boosts hits

    return StadiumFeatures(
        overall_pf=overall_pf,
        runs_pf=runs_pf,
        hr_pf=hr_pf,
        hits_pf=hits_pf,
        doubles_pf=doubles_pf,
        triples_pf=triples_pf,
        lh_hr_pf=lh_hr_pf,
        rh_hr_pf=rh_hr_pf,
        avg_dimension=avg_dim,
        dimension_score=dim_score,
        altitude_ft=altitude,
        altitude_factor=alt_factor,
        temp_factor=temp_factor,
        wind_factor=wind_factor,
        humidity_factor=humidity_factor,
        is_turf=is_turf,
        surface_speed_factor=surface_factor,
        roof_status=roof,
        weather_neutralized=weather_neutralized,
        home_field_advantage=0.537,  # MLB historical HFA
    )
