"""Convert ArnavSaraogi/mlb-odds-scraper JSON to our odds_history.csv format.

Reads data/mlb_odds_dataset.json (80MB, covers 2021-04-01 to 2025-08-16),
extracts closing moneyline and totals from the first available sportsbook,
computes market_home_prob, and writes to data/odds_history.csv.

Usage:
    PYTHONPATH=src python -m mlb.data.convert_odds_json
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "mlb_odds_dataset.json"
OUTPUT_FILE = DATA_DIR / "odds_history.csv"

FIELDNAMES = [
    "game_date", "home_team", "away_team",
    "home_moneyline", "away_moneyline", "total_line", "market_home_prob",
]

# Map dataset abbreviations → our pipeline abbreviations
TEAM_MAP = {
    "CHW": "CWS",
    "WAS": "WSH",
    "ATH": "OAK",
    "AZ": "ARI",
}

# Skip All-Star game teams
SKIP_TEAMS = {"AL", "NL"}

# Preferred sportsbook order (use first available)
PREFERRED_BOOKS = ["fanduel", "draftkings", "caesars", "bet365", "betmgm", "betrivers"]


def _map_team(abbrev: str) -> str:
    return TEAM_MAP.get(abbrev, abbrev)


def _american_to_decimal(american: float) -> float:
    if american >= 100:
        return 1.0 + american / 100.0
    else:
        return 1.0 + 100.0 / abs(american)


def _pick_line(odds_list: list[dict], market_type: str) -> dict | None:
    """Pick closing line from preferred sportsbook."""
    book_map = {entry["sportsbook"]: entry for entry in odds_list}
    for book in PREFERRED_BOOKS:
        if book in book_map:
            return book_map[book].get("currentLine")
    # Fallback: use first available
    if odds_list:
        return odds_list[0].get("currentLine")
    return None


def convert():
    logger.info("Loading %s ...", INPUT_FILE)
    with open(INPUT_FILE) as f:
        data = json.load(f)

    logger.info("Processing %d dates ...", len(data))

    rows = []
    skipped = 0

    for game_date, games in sorted(data.items()):
        for game in games:
            gv = game["gameView"]
            home_abbr = _map_team(gv["homeTeam"]["shortName"])
            away_abbr = _map_team(gv["awayTeam"]["shortName"])

            if home_abbr in SKIP_TEAMS or away_abbr in SKIP_TEAMS:
                skipped += 1
                continue

            # Skip non-regular-season games
            if gv.get("gameType") != "R":
                skipped += 1
                continue

            odds = game.get("odds", {})

            # Moneyline
            ml_line = _pick_line(odds.get("moneyline", []), "moneyline")
            if ml_line is None:
                skipped += 1
                continue

            home_ml = ml_line.get("homeOdds")
            away_ml = ml_line.get("awayOdds")
            if home_ml is None or away_ml is None or home_ml == 0 or away_ml == 0:
                skipped += 1
                continue

            # Totals
            totals_line = _pick_line(odds.get("totals", []), "totals")
            total_val = totals_line.get("total") if totals_line else None

            # Compute implied probability
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
                "home_moneyline": home_ml,
                "away_moneyline": away_ml,
                "total_line": total_val,
                "market_home_prob": market_home_prob,
            })

    # Deduplicate by (date, home, away) — keep first (preferred book)
    seen = {}
    for row in rows:
        key = (row["game_date"], row["home_team"], row["away_team"])
        if key not in seen:
            seen[key] = row

    deduped = sorted(seen.values(), key=lambda r: (r["game_date"], r["home_team"]))

    # Write CSV
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(deduped)

    logger.info(
        "Wrote %d odds records to %s (skipped %d non-regular/missing)",
        len(deduped), OUTPUT_FILE, skipped,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    convert()
