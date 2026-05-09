"""Client for The Odds API — fetches live MLB betting odds.

Docs: https://the-odds-api.com/liveAPI/guides/v4/
Free tier: 500 requests/month.

Usage:
    client = OddsApiClient("your_api_key")
    odds = await client.get_mlb_odds()
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from mlb.config import settings
from mlb.data.mlb_api import TEAM_IDS

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"

# Map common Odds API team names → our abbreviations
_ODDS_TEAM_MAP: dict[str, str] = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
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
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


def _team_abbrev(name: str) -> str:
    return _ODDS_TEAM_MAP.get(name, name)


class OddsApiClient:
    """Fetches live MLB odds from The Odds API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.odds_api_key
        if not self.api_key:
            logger.warning("No ODDS_API_KEY set — odds fetching will be unavailable")

    async def get_mlb_odds(
        self,
        markets: str = "h2h,totals",
        regions: str = "us",
        odds_format: str = "american",
    ) -> list[dict[str, Any]]:
        """Fetch current MLB odds for all upcoming games.

        Returns a list of dicts, each with:
            game_id, home_team, away_team,
            home_moneyline, away_moneyline,
            total_line, over_odds, under_odds
        """
        if not self.api_key:
            logger.warning("No API key — returning empty odds")
            return []

        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{BASE_URL}/sports/{SPORT}/odds/",
                params=params,
            )
            resp.raise_for_status()

            remaining = resp.headers.get("x-requests-remaining", "?")
            logger.info("Odds API requests remaining: %s", remaining)

            raw = resp.json()

        return self._parse_odds(raw)

    def _parse_odds(self, raw: list[dict]) -> list[dict[str, Any]]:
        """Parse Odds API response into our format."""
        results = []

        for event in raw:
            home_team = _team_abbrev(event.get("home_team", ""))
            away_team = _team_abbrev(event.get("away_team", ""))
            event_id = event.get("id", "")

            game_odds: dict[str, Any] = {
                "odds_event_id": event_id,
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": event.get("commence_time", ""),
                "home_moneyline": None,
                "away_moneyline": None,
                "total_line": None,
                "over_odds": None,
                "under_odds": None,
            }

            # Use the first bookmaker (typically consensus/best available)
            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    key = market.get("key")

                    if key == "h2h" and game_odds["home_moneyline"] is None:
                        for outcome in market.get("outcomes", []):
                            team = _team_abbrev(outcome.get("name", ""))
                            price = outcome.get("price")
                            if team == home_team:
                                game_odds["home_moneyline"] = price
                            elif team == away_team:
                                game_odds["away_moneyline"] = price

                    elif key == "totals" and game_odds["total_line"] is None:
                        for outcome in market.get("outcomes", []):
                            name = outcome.get("name", "").lower()
                            if name == "over":
                                game_odds["total_line"] = outcome.get("point")
                                game_odds["over_odds"] = outcome.get("price")
                            elif name == "under":
                                game_odds["under_odds"] = outcome.get("price")

                # Break after first bookmaker with both markets
                if game_odds["home_moneyline"] is not None:
                    break

            results.append(game_odds)

        logger.info("Parsed odds for %d games", len(results))
        return results

    async def get_scores(self, days_from: int = 1) -> list[dict[str, Any]]:
        """Fetch recent scores for settling bets."""
        if not self.api_key:
            return []

        params = {
            "apiKey": self.api_key,
            "daysFrom": days_from,
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{BASE_URL}/sports/{SPORT}/scores/",
                params=params,
            )
            resp.raise_for_status()
            raw = resp.json()

        results = []
        for event in raw:
            if not event.get("completed"):
                continue

            scores = event.get("scores", [])
            home_name = event.get("home_team", "")
            home_abbrev = _team_abbrev(home_name)
            away_abbrev = _team_abbrev(event.get("away_team", ""))

            home_score = away_score = 0
            for s in scores:
                team = s.get("name", "")
                score = int(s.get("score", 0))
                if team == home_name:
                    home_score = score
                else:
                    away_score = score

            results.append({
                "event_id": event.get("id"),
                "home_team": home_abbrev,
                "away_team": away_abbrev,
                "home_score": home_score,
                "away_score": away_score,
                "completed": True,
            })

        return results
