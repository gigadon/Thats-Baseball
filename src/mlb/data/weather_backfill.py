"""Backfill historical weather data for MLB training set.

Uses the Open-Meteo Archive API (free, no key required) to fetch
hourly temperature, wind speed, and humidity at each stadium for
every game date.  Results are cached to data/weather_history.csv
so subsequent runs only fetch missing data.

Usage:
    PYTHONPATH=src python -m mlb.data.weather_backfill
"""

from __future__ import annotations

import csv
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import httpx

from mlb.data.weather import STADIUM_COORDS, _DOME_TEAMS, _RETRACTABLE_TEAMS

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Team timezone UTC offsets (standard time — matches build_training_data.py)
TEAM_TIMEZONES: dict[str, int] = {
    "NYY": -5, "NYM": -5, "BOS": -5, "BAL": -5, "TB": -5,
    "PHI": -5, "PIT": -5, "MIA": -5, "WSH": -5, "ATL": -5,
    "CIN": -5, "CLE": -5, "DET": -5, "TOR": -5,
    "CHC": -6, "CWS": -6, "MIL": -6, "STL": -6,
    "MIN": -6, "KC": -6, "HOU": -6, "TEX": -6,
    "ARI": -7, "COL": -7,
    "LAD": -8, "LAA": -8, "SF": -8, "SD": -8, "SEA": -8, "OAK": -8,
}

# Approximate MLB season windows per year
SEASON_WINDOWS: dict[int, tuple[str, str]] = {
    2021: ("2021-04-01", "2021-11-03"),
    2022: ("2022-03-31", "2022-11-06"),
    2023: ("2023-03-30", "2023-11-02"),
    2024: ("2024-03-28", "2024-11-03"),
    2025: ("2025-03-27", "2025-11-02"),
    2026: ("2026-03-26", "2026-05-27"),
}

CACHE_FILE = Path("data/weather_history.csv")


def _game_hour_utc(team: str) -> int:
    """Approximate UTC hour for a ~7pm local first pitch."""
    tz_offset = TEAM_TIMEZONES.get(team, -5)
    return (19 - tz_offset) % 24


def _load_existing_cache() -> dict[tuple[str, str], dict]:
    """Load cached records from CSV.  Returns {(date, team): row_dict}."""
    if not CACHE_FILE.exists():
        return {}
    result: dict[tuple[str, str], dict] = {}
    with open(CACHE_FILE) as f:
        for row in csv.DictReader(f):
            result[(row["date"], row["team"])] = row
    logger.info("Loaded %d cached weather records", len(result))
    return result


def fetch_weather_batch(
    team: str, start_date: str, end_date: str,
) -> list[dict[str, str | float]]:
    """Fetch hourly weather for a stadium over a date range.

    Returns one record per date at the approximate game-time hour.
    """
    lat, lon = STADIUM_COORDS[team]
    target_hour = _game_hour_utc(team)

    resp = httpx.get(
        OPEN_METEO_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,relative_humidity_2m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    records: list[dict[str, str | float]] = []
    for i, time_str in enumerate(hourly["time"]):
        hour = int(time_str[11:13])
        if hour == target_hour:
            temp = hourly["temperature_2m"][i]
            wind = hourly["wind_speed_10m"][i]
            wdir = hourly["wind_direction_10m"][i]
            hum = hourly["relative_humidity_2m"][i]
            # Skip if any value is None (missing data)
            if temp is None or wind is None or hum is None or wdir is None:
                continue
            records.append({
                "date": time_str[:10],
                "team": team,
                "temperature": round(float(temp), 1),
                "wind_speed": round(float(wind), 1),
                "wind_direction": round(float(wdir)),  # degrees, 0=N 90=E (FROM)
                "humidity": round(float(hum) / 100.0, 3),  # 0-100 → 0-1
            })
    return records


def backfill_all_weather(seasons: list[int] | None = None):
    """Fetch weather for all outdoor stadiums for all seasons."""
    if seasons is None:
        seasons = sorted(SEASON_WINDOWS.keys())

    existing = _load_existing_cache()

    # Track which (team, season) combos are fully cached
    # by checking if any record exists for that team in that season window
    def _season_cached(team: str, season: int) -> bool:
        start, end = SEASON_WINDOWS[season]
        # Cached only if a record in range already has wind_direction (so old
        # caches without direction are re-fetched to add it).
        for (d, t), row in existing.items():
            if t == team and start <= d <= end and row.get("wind_direction"):
                return True
        return False

    all_records = list(existing.values())
    fetched = 0

    outdoor_teams = [t for t in sorted(STADIUM_COORDS) if t not in _DOME_TEAMS]

    for team in outdoor_teams:
        for season in seasons:
            if season not in SEASON_WINDOWS:
                continue
            if _season_cached(team, season):
                continue

            start, end = SEASON_WINDOWS[season]
            logger.info("Fetching weather: %s %d (%s to %s)", team, season, start, end)
            try:
                records = fetch_weather_batch(team, start, end)
                all_records.extend(records)
                fetched += len(records)
                logger.info("  → %d records", len(records))
            except Exception as e:
                logger.warning("Failed to fetch %s %d: %s", team, season, e)

            time.sleep(0.3)  # Rate limit courtesy

    # Deduplicate by (date, team)
    seen: dict[tuple[str, str], dict] = {}
    for rec in all_records:
        key = (rec["date"], rec["team"])
        seen[key] = rec

    # Write CSV
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["date", "team", "temperature", "wind_speed", "wind_direction", "humidity"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for key in sorted(seen):
            writer.writerow(seen[key])

    logger.info(
        "Weather backfill complete: %d total records (%d newly fetched)",
        len(seen), fetched,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    backfill_all_weather()
