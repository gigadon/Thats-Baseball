"""Weather client — fetches game-time weather from OpenWeatherMap.

Docs: https://openweathermap.org/current
Free tier: 1,000 calls/day.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from mlb.config import settings
from mlb.features.stadium import WeatherConditions

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# Stadium lat/lon for weather lookups
STADIUM_COORDS: dict[str, tuple[float, float]] = {
    "ARI": (33.4455, -112.0667),  # Chase Field, Phoenix
    "ATL": (33.8907, -84.4677),   # Truist Park, Atlanta
    "BAL": (39.2838, -76.6218),   # Camden Yards, Baltimore
    "BOS": (42.3467, -71.0972),   # Fenway Park, Boston
    "CHC": (41.9484, -87.6553),   # Wrigley Field, Chicago
    "CWS": (41.8299, -87.6338),   # Guaranteed Rate, Chicago
    "CIN": (39.0974, -84.5082),   # Great American, Cincinnati
    "CLE": (41.4962, -81.6852),   # Progressive Field, Cleveland
    "COL": (39.7559, -104.9942),  # Coors Field, Denver
    "DET": (42.3390, -83.0485),   # Comerica Park, Detroit
    "HOU": (29.7573, -95.3555),   # Minute Maid, Houston
    "KC":  (39.0517, -94.4803),   # Kauffman, Kansas City
    "LAA": (33.8003, -117.8827),  # Angel Stadium, Anaheim
    "LAD": (34.0739, -118.2400),  # Dodger Stadium, LA
    "MIA": (25.7781, -80.2197),   # LoanDepot Park, Miami
    "MIL": (43.0280, -87.9712),   # American Family, Milwaukee
    "MIN": (44.9817, -93.2776),   # Target Field, Minneapolis
    "NYM": (40.7571, -73.8458),   # Citi Field, New York
    "NYY": (40.8296, -73.9262),   # Yankee Stadium, New York
    "OAK": (37.7516, -122.2005),  # Oakland Coliseum
    "PHI": (39.9061, -75.1665),   # Citizens Bank, Philadelphia
    "PIT": (40.4469, -80.0058),   # PNC Park, Pittsburgh
    "SD":  (32.7076, -117.1570),  # Petco Park, San Diego
    "SF":  (37.7786, -122.3893),  # Oracle Park, San Francisco
    "SEA": (47.5914, -122.3325),  # T-Mobile Park, Seattle
    "STL": (38.6226, -90.1928),   # Busch Stadium, St. Louis
    "TB":  (27.7682, -82.6534),   # Tropicana Field, Tampa Bay
    "TEX": (32.7512, -97.0832),   # Globe Life, Arlington
    "TOR": (43.6414, -79.3894),   # Rogers Centre, Toronto
    "WSH": (38.8730, -77.0074),   # Nationals Park, Washington
}

# Dome/retractable teams — weather is neutralized
_DOME_TEAMS = {"TB"}
_RETRACTABLE_TEAMS = {"ARI", "HOU", "MIA", "MIL", "SEA", "TEX", "TOR"}


def _classify_wind_direction(degrees: float, stadium_orientation: float = 0) -> str:
    """Classify wind as 'in', 'out', 'cross', or 'calm' relative to home plate."""
    # Simplified: assume home plate faces roughly north-northeast (typical)
    # Wind blowing from behind home plate → out to CF
    # This is a rough heuristic; real parks have varied orientations
    relative = (degrees - stadium_orientation) % 360
    if 315 <= relative or relative < 45:
        return "out"
    elif 135 <= relative < 225:
        return "in"
    elif 45 <= relative < 135 or 225 <= relative < 315:
        return "cross"
    return "calm"


class WeatherClient:
    """Fetches current weather for MLB stadiums."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.weather_api_key
        if not self.api_key:
            logger.warning("No WEATHER_API_KEY set — weather data unavailable")

    async def get_game_weather(self, team_id: str) -> WeatherConditions | None:
        """Fetch current weather for a team's stadium.

        Returns None if the API key is missing or the team has a dome.
        """
        if not self.api_key:
            return None

        coords = STADIUM_COORDS.get(team_id)
        if not coords:
            logger.warning("No coordinates for team %s", team_id)
            return None

        is_dome = team_id in _DOME_TEAMS

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    BASE_URL,
                    params={
                        "lat": coords[0],
                        "lon": coords[1],
                        "appid": self.api_key,
                        "units": "imperial",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            logger.exception("Weather fetch failed for %s", team_id)
            return None

        return self._parse_weather(data, team_id, is_dome)

    async def get_bulk_weather(
        self, team_ids: list[str]
    ) -> dict[str, WeatherConditions]:
        """Fetch weather for multiple stadiums."""
        results: dict[str, WeatherConditions] = {}
        for team_id in team_ids:
            weather = await self.get_game_weather(team_id)
            if weather:
                results[team_id] = weather
        return results

    def _parse_weather(
        self, data: dict, team_id: str, is_dome: bool
    ) -> WeatherConditions:
        """Parse OpenWeatherMap response into WeatherConditions."""
        main = data.get("main", {})
        wind = data.get("wind", {})

        temp_f = float(main.get("temp", 70))
        wind_speed = float(wind.get("speed", 0))
        wind_deg = float(wind.get("deg", 0))
        humidity = float(main.get("humidity", 50)) / 100.0

        # Precipitation probability from rain/snow
        rain = data.get("rain", {})
        snow = data.get("snow", {})
        precip = 0.0
        if rain or snow:
            precip = 0.5  # Approximate — OWM current doesn't give probability directly

        # Check for retractable roof (assume closed if bad weather)
        if team_id in _RETRACTABLE_TEAMS:
            if temp_f < 55 or temp_f > 95 or wind_speed > 20 or precip > 0.3:
                is_dome = True

        wind_dir = "calm" if wind_speed < 3 else _classify_wind_direction(wind_deg)

        return WeatherConditions(
            temperature_f=round(temp_f, 1),
            wind_speed_mph=round(wind_speed, 1),
            wind_direction=wind_dir,
            humidity_pct=round(humidity, 2),
            precipitation_prob=round(precip, 2),
            is_dome=is_dome,
        )
