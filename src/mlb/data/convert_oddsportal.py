"""Convert OddsPortal scraped CSV to our odds_history.csv format.

Reads data/oddsportal_{year}_raw.csv files, parses dates and team names,
computes market_home_prob, and merges into data/odds_history.csv.

Usage:
    PYTHONPATH=src python -m mlb.data.convert_oddsportal
"""

from __future__ import annotations

import csv
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
ODDS_FILE = DATA_DIR / "odds_history.csv"

# Map OddsPortal full names -> our abbreviations
TEAM_MAP = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Athletics": "OAK",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St.Louis Cardinals": "STL",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


def _parse_date(date_str: str, reference_date: date | None = None) -> str | None:
    """Parse OddsPortal date formats to YYYY-MM-DD."""
    if reference_date is None:
        reference_date = date.today()

    date_str = date_str.strip()

    if date_str.startswith("Today"):
        return reference_date.isoformat()
    if date_str.startswith("Yesterday"):
        return (reference_date - timedelta(days=1)).isoformat()

    # "26 May 2026" format
    try:
        dt = datetime.strptime(date_str, "%d %b %Y")
        return dt.date().isoformat()
    except ValueError:
        pass

    logger.warning("Could not parse date: %s", date_str)
    return None


def _american_to_decimal(american: float) -> float:
    if american >= 100:
        return 1.0 + american / 100.0
    else:
        return 1.0 + 100.0 / abs(american)


def convert_oddsportal_file(csv_path: Path) -> list[dict]:
    """Convert a single OddsPortal raw CSV to our format."""
    df = pd.read_csv(csv_path)
    rows = []

    for _, row in df.iterrows():
        game_date = _parse_date(str(row["date"]))
        if game_date is None:
            continue

        home_name = str(row["home_team"]).strip()
        away_name = str(row["away_team"]).strip()

        home_abbr = TEAM_MAP.get(home_name)
        away_abbr = TEAM_MAP.get(away_name)

        if home_abbr is None or away_abbr is None:
            logger.warning("Unknown team: home=%s away=%s", home_name, away_name)
            continue

        home_ml = row.get("home_odds")
        away_ml = row.get("away_odds")

        if pd.isna(home_ml) or pd.isna(away_ml) or home_ml == 0 or away_ml == 0:
            continue

        home_ml = float(home_ml)
        away_ml = float(away_ml)

        h_dec = _american_to_decimal(home_ml)
        a_dec = _american_to_decimal(away_ml)
        h_imp = 1.0 / h_dec
        a_imp = 1.0 / a_dec
        total_imp = h_imp + a_imp
        market_home_prob = round(h_imp / total_imp, 4)

        rows.append({
            "game_date": game_date,
            "home_team": home_abbr,
            "away_team": away_abbr,
            "home_moneyline": int(home_ml),
            "away_moneyline": int(away_ml),
            "total_line": "",  # OddsPortal scraper doesn't include totals
            "market_home_prob": market_home_prob,
        })

    return rows


def merge_into_odds_history(new_rows: list[dict]):
    """Merge new rows into odds_history.csv, deduplicating."""
    existing = {}
    if ODDS_FILE.exists():
        with open(ODDS_FILE) as f:
            for row in csv.DictReader(f):
                key = (row["game_date"], row["home_team"], row["away_team"])
                existing[key] = row

    logger.info("Existing odds records: %d", len(existing))

    added = 0
    for row in new_rows:
        key = (row["game_date"], row["home_team"], row["away_team"])
        if key not in existing:
            existing[key] = row
            added += 1

    # Write sorted
    fieldnames = [
        "game_date", "home_team", "away_team",
        "home_moneyline", "away_moneyline", "total_line", "market_home_prob",
    ]

    with open(ODDS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(existing):
            writer.writerow(existing[key])

    logger.info("Added %d new records, total: %d", added, len(existing))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    all_rows = []
    for raw_file in sorted(DATA_DIR.glob("oddsportal_*_raw.csv")):
        logger.info("Processing %s", raw_file.name)
        rows = convert_oddsportal_file(raw_file)
        logger.info("  -> %d valid records", len(rows))
        all_rows.extend(rows)

    if all_rows:
        merge_into_odds_history(all_rows)
    else:
        logger.warning("No records to merge")


if __name__ == "__main__":
    main()
