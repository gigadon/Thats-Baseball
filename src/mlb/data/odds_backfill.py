"""Backfill historical odds from The Odds API.

The historical endpoint (v4/historical/sports/{sport}/odds) requires a paid plan.
Each call returns odds for a single point-in-time snapshot.

Usage:
    python -m mlb.data.odds_backfill --start 2024-03-28 --end 2024-09-29
    python -m mlb.data.odds_backfill --season 2025
    python -m mlb.data.odds_backfill --season 2026 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx

from mlb.config import settings
from mlb.data.odds_api import _ODDS_TEAM_MAP, BASE_URL, SPORT

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
ODDS_FILE = DATA_DIR / "odds_history.csv"
FIELDNAMES = [
    "game_date", "home_team", "away_team",
    "home_moneyline", "away_moneyline", "total_line", "market_home_prob",
]

# Season start dates (Opening Day)
SEASON_STARTS = {
    2021: "2021-04-01",
    2022: "2022-04-07",
    2023: "2023-03-30",
    2024: "2024-03-28",
    2025: "2025-03-27",
    2026: "2026-03-26",
}


def _team_abbrev(name: str) -> str:
    return _ODDS_TEAM_MAP.get(name, name)


def _american_to_decimal(american: float) -> float:
    if american >= 100:
        return 1.0 + american / 100.0
    else:
        return 1.0 + 100.0 / abs(american)


def _load_existing_dates() -> set[str]:
    """Load dates already in odds_history.csv to skip."""
    if not ODDS_FILE.exists():
        return set()
    dates = set()
    with open(ODDS_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates.add(row["game_date"])
    return dates


def _get_game_dates(start: str, end: str) -> list[str]:
    """Get all dates between start and end that have games in our data."""
    game_dates = set()
    for year in range(2021, 2027):
        path = DATA_DIR / f"games_{year}.csv"
        if not path.exists():
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = row.get("game_date", "")
                if start <= d <= end and row.get("status") == "Final":
                    game_dates.add(d)
    return sorted(game_dates)


def fetch_historical_odds(target_date: str, api_key: str) -> list[dict]:
    """Fetch odds for a specific historical date.

    Uses the historical endpoint: GET /v4/historical/sports/{sport}/odds/
    """
    # Request odds at 5pm ET on game day (close to game time for most games)
    dt = datetime.fromisoformat(target_date + "T17:00:00-04:00")
    date_iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,totals",
        "oddsFormat": "american",
        "date": date_iso,
    }

    resp = httpx.get(
        f"{BASE_URL}/historical/sports/{SPORT}/odds/",
        params=params,
        timeout=20.0,
    )

    if resp.status_code == 422:
        logger.warning("Date %s not available in historical API", target_date)
        return []
    if resp.status_code == 401:
        logger.error("API key unauthorized for historical endpoint (requires paid plan)")
        return []

    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining", "?")
    logger.info("Requests remaining: %s", remaining)

    data = resp.json()
    events = data.get("data", [])

    results = []
    for event in events:
        home_team = _team_abbrev(event.get("home_team", ""))
        away_team = _team_abbrev(event.get("away_team", ""))

        home_ml = None
        away_ml = None
        total_line = None

        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                key = market.get("key")
                if key == "h2h" and home_ml is None:
                    for outcome in market.get("outcomes", []):
                        team = _team_abbrev(outcome.get("name", ""))
                        price = outcome.get("price")
                        if team == home_team:
                            home_ml = price
                        elif team == away_team:
                            away_ml = price
                elif key == "totals" and total_line is None:
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name", "").lower() == "over":
                            total_line = outcome.get("point")
            if home_ml is not None:
                break

        if home_ml is not None and away_ml is not None:
            h_dec = _american_to_decimal(home_ml)
            a_dec = _american_to_decimal(away_ml)
            h_imp = 1.0 / h_dec
            a_imp = 1.0 / a_dec
            total_imp = h_imp + a_imp
            market_home_prob = h_imp / total_imp

            results.append({
                "game_date": target_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_moneyline": home_ml,
                "away_moneyline": away_ml,
                "total_line": total_line,
                "market_home_prob": round(market_home_prob, 4),
            })

    return results


def backfill(start: str, end: str, dry_run: bool = False, delay: float = 1.0):
    """Backfill odds for all game dates in range."""
    api_key = settings.odds_api_key
    if not api_key:
        logger.error("No ODDS_API_KEY set in .env — cannot backfill")
        return

    existing = _load_existing_dates()
    game_dates = _get_game_dates(start, end)

    # Filter out dates we already have
    to_fetch = [d for d in game_dates if d not in existing]
    logger.info(
        "Backfill: %d total game dates, %d already have odds, %d to fetch",
        len(game_dates), len(game_dates) - len(to_fetch), len(to_fetch),
    )

    if dry_run:
        print(f"DRY RUN: would fetch {len(to_fetch)} dates")
        for d in to_fetch[:10]:
            print(f"  {d}")
        if len(to_fetch) > 10:
            print(f"  ... and {len(to_fetch) - 10} more")
        return

    file_exists = ODDS_FILE.exists()
    total_rows = 0

    with open(ODDS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
            file_exists = True

        for i, game_date in enumerate(to_fetch):
            logger.info("[%d/%d] Fetching odds for %s", i + 1, len(to_fetch), game_date)
            try:
                rows = fetch_historical_odds(game_date, api_key)
                if rows:
                    writer.writerows(rows)
                    f.flush()
                    total_rows += len(rows)
                    logger.info("  -> %d games with odds", len(rows))
                else:
                    logger.info("  -> no odds available")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("Rate limited — stopping. Got %d rows so far.", total_rows)
                    break
                logger.error("HTTP error for %s: %s", game_date, e)
            except Exception as e:
                logger.error("Error fetching %s: %s", game_date, e)

            # Respect rate limits
            if i < len(to_fetch) - 1:
                time.sleep(delay)

    logger.info("Backfill complete: %d new odds records saved", total_rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Backfill historical MLB odds")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--season", type=int, help="Season year (sets start/end automatically)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fetched")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between API calls")
    args = parser.parse_args()

    if args.season:
        start = SEASON_STARTS.get(args.season, f"{args.season}-03-28")
        # End at today or Sep 29, whichever is earlier
        season_end = f"{args.season}-09-29"
        today = date.today().isoformat()
        end = min(season_end, today)
    elif args.start and args.end:
        start, end = args.start, args.end
    else:
        parser.error("Provide --season or both --start and --end")
        return

    print(f"Backfilling odds: {start} to {end}")
    backfill(start, end, dry_run=args.dry_run, delay=args.delay)


if __name__ == "__main__":
    main()
