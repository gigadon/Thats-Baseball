"""Client for MLB StatsAPI (statsapi.mlb.com)."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import httpx

from mlb.config import settings

logger = logging.getLogger(__name__)

# MLB StatsAPI team abbreviation lookup (API uses numeric IDs internally)
TEAM_ABBREVS: dict[int, str] = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC",  119: "LAD", 120: "WSH", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SD",  136: "SEA", 137: "SF",  138: "STL",
    139: "TB",  140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}

TEAM_IDS: dict[str, int] = {v: k for k, v in TEAM_ABBREVS.items()}


class MLBApiClient:
    """Async client for the MLB Stats API."""

    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or settings.mlb_api_base_url
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=30.0,
                headers={"User-Agent": "ThatsBaseball/0.1"},
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _get(self, endpoint: str, params: dict | None = None) -> dict[str, Any]:
        client = await self._get_client()
        resp = await client.get(endpoint, params=params)
        resp.raise_for_status()
        return resp.json()

    # ── Schedule & Games ───────────────────────────────────────

    async def get_schedule(
        self, game_date: date, sport_id: int = 1
    ) -> list[dict[str, Any]]:
        """Fetch all MLB games for a given date."""
        data = await self._get(
            "/schedule",
            params={
                "sportId": sport_id,
                "date": game_date.isoformat(),
                "hydrate": "team,linescore,probablePitcher",
            },
        )
        games = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                games.append(self._parse_schedule_game(game))
        return games

    async def get_schedule_range(
        self, start: date, end: date, sport_id: int = 1
    ) -> list[dict[str, Any]]:
        """Fetch all games in a date range."""
        data = await self._get(
            "/schedule",
            params={
                "sportId": sport_id,
                "startDate": start.isoformat(),
                "endDate": end.isoformat(),
                "hydrate": "team,linescore",
            },
        )
        games = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                games.append(self._parse_schedule_game(game))
        return games

    def _parse_schedule_game(self, game: dict) -> dict[str, Any]:
        teams = game.get("teams", {})
        home = teams.get("home", {})
        away = teams.get("away", {})
        linescore = game.get("linescore", {})

        home_team_id = home.get("team", {}).get("id")
        away_team_id = away.get("team", {}).get("id")

        # Extract home plate umpire ID from officials array if available
        home_plate_umpire_id = self._extract_home_plate_umpire(linescore)

        return {
            "game_id": str(game["gamePk"]),
            "game_date": game.get("officialDate", game.get("gameDate", "")[:10]),
            "game_time": game.get("gameDate", ""),
            "home_team_id": TEAM_ABBREVS.get(home_team_id, str(home_team_id)),
            "away_team_id": TEAM_ABBREVS.get(away_team_id, str(away_team_id)),
            "home_score": home.get("score"),
            "away_score": away.get("score"),
            "status": game.get("status", {}).get("detailedState", "Unknown"),
            "home_probable_pitcher": self._extract_pitcher(home),
            "away_probable_pitcher": self._extract_pitcher(away),
            "innings": linescore.get("currentInning"),
            "home_plate_umpire_id": home_plate_umpire_id,
        }

    def _extract_home_plate_umpire(self, linescore: dict) -> int | None:
        """Extract the home plate umpire ID from the linescore officials array.

        The officials array contains entries like:
            {"official": {"id": 427072, "fullName": "Angel Hernandez"},
             "officialType": "Home Plate"}
        Returns the umpire ID or None if not available.
        """
        officials = linescore.get("officials", [])
        for official_entry in officials:
            if official_entry.get("officialType") == "Home Plate":
                official = official_entry.get("official", {})
                return official.get("id")
        return None

    def _extract_pitcher(self, team_data: dict) -> dict[str, Any] | None:
        pitcher = team_data.get("probablePitcher")
        if not pitcher:
            return None
        return {
            "player_id": pitcher["id"],
            "name": pitcher.get("fullName", ""),
        }

    # ── Boxscores ──────────────────────────────────────────────

    async def get_boxscore(self, game_id: str | int) -> dict[str, Any]:
        """Fetch full boxscore for a completed game."""
        data = await self._get(f"/game/{game_id}/boxscore")
        return self._parse_boxscore(data, str(game_id))

    def _parse_boxscore(self, data: dict, game_id: str) -> dict[str, Any]:
        result: dict[str, Any] = {"game_id": game_id, "home": {}, "away": {}}

        for side in ("home", "away"):
            team_data = data.get("teams", {}).get(side, {})
            team_info = team_data.get("team", {})
            team_id = TEAM_ABBREVS.get(team_info.get("id"), str(team_info.get("id", "")))

            batters = []
            pitchers = []
            for player_id_str, player in team_data.get("players", {}).items():
                pid = player.get("person", {}).get("id")
                name = player.get("person", {}).get("fullName", "")
                stats = player.get("stats", {})

                batting = stats.get("batting", {})
                if batting:
                    batters.append({
                        "player_id": pid,
                        "name": name,
                        "at_bats": _int(batting.get("atBats")),
                        "runs": _int(batting.get("runs")),
                        "hits": _int(batting.get("hits")),
                        "doubles": _int(batting.get("doubles")),
                        "triples": _int(batting.get("triples")),
                        "home_runs": _int(batting.get("homeRuns")),
                        "rbi": _int(batting.get("rbi")),
                        "walks": _int(batting.get("baseOnBalls")),
                        "strikeouts": _int(batting.get("strikeOuts")),
                        "stolen_bases": _int(batting.get("stolenBases")),
                    })

                pitching = stats.get("pitching", {})
                if pitching and pitching.get("inningsPitched"):
                    pitchers.append({
                        "player_id": pid,
                        "name": name,
                        "innings_pitched": _ip(pitching.get("inningsPitched")),
                        "hits_allowed": _int(pitching.get("hits")),
                        "runs_allowed": _int(pitching.get("runs")),
                        "earned_runs": _int(pitching.get("earnedRuns")),
                        "walks_allowed": _int(pitching.get("baseOnBalls")),
                        "strikeouts_recorded": _int(pitching.get("strikeOuts")),
                        "pitches_thrown": _int(pitching.get("numberOfPitches")),
                    })

            result[side] = {
                "team_id": team_id,
                "batters": batters,
                "pitchers": pitchers,
            }

        return result

    # ── Team Stats ─────────────────────────────────────────────

    async def get_team_stats(
        self, team_id: int, season: int, group: str = "hitting"
    ) -> dict[str, Any]:
        """Fetch season stats for a team. group: hitting | pitching | fielding."""
        data = await self._get(
            f"/teams/{team_id}/stats",
            params={
                "stats": "season",
                "group": group,
                "season": season,
            },
        )
        splits = data.get("stats", [{}])[0].get("splits", [])
        if splits:
            return splits[0].get("stat", {})
        return {}

    async def get_team_season_stats(
        self, team_abbrev: str, season: int
    ) -> dict[str, dict[str, Any]]:
        """Fetch hitting, pitching, and fielding stats for a team."""
        team_id = TEAM_IDS.get(team_abbrev)
        if team_id is None:
            raise ValueError(f"Unknown team abbreviation: {team_abbrev}")

        hitting = await self.get_team_stats(team_id, season, "hitting")
        pitching = await self.get_team_stats(team_id, season, "pitching")
        fielding = await self.get_team_stats(team_id, season, "fielding")

        return {"hitting": hitting, "pitching": pitching, "fielding": fielding}

    # ── Player Stats ───────────────────────────────────────────

    async def get_player_stats(
        self, player_id: int, season: int, group: str = "hitting"
    ) -> dict[str, Any]:
        """Fetch season stats for a player."""
        data = await self._get(
            f"/people/{player_id}/stats",
            params={
                "stats": "season",
                "group": group,
                "season": season,
            },
        )
        splits = data.get("stats", [{}])[0].get("splits", [])
        if splits:
            return splits[0].get("stat", {})
        return {}

    async def get_player_game_log(
        self, player_id: int, season: int, group: str = "hitting"
    ) -> list[dict[str, Any]]:
        """Fetch game-by-game stats for a player."""
        data = await self._get(
            f"/people/{player_id}/stats",
            params={
                "stats": "gameLog",
                "group": group,
                "season": season,
            },
        )
        splits = data.get("stats", [{}])[0].get("splits", [])
        return splits

    # ── Pitcher Season Stats (convenience) ──────────────────

    async def get_pitcher_season_stats(
        self, player_id: int, season: int
    ) -> dict[str, Any]:
        """Fetch season pitching stats for a pitcher, returning parsed key metrics."""
        raw = await self.get_player_stats(player_id, season, group="pitching")
        if not raw:
            return {}
        return {
            "player_id": player_id,
            "era": float(raw.get("era", 0)),
            "whip": float(raw.get("whip", 0)),
            "innings_pitched": _ip(raw.get("inningsPitched", "0")),
            "wins": _int(raw.get("wins")) or 0,
            "losses": _int(raw.get("losses")) or 0,
            "strikeouts": _int(raw.get("strikeOuts")) or 0,
            "walks": _int(raw.get("baseOnBalls")) or 0,
            "hits_allowed": _int(raw.get("hits")) or 0,
            "home_runs_allowed": _int(raw.get("homeRuns")) or 0,
            "games_started": _int(raw.get("gamesStarted")) or 0,
            "k_per_9": float(raw.get("strikeoutsPer9Inn", 0)),
            "bb_per_9": float(raw.get("walksPer9Inn", 0)),
            "h_per_9": float(raw.get("hitsPer9Inn", 0)),
            "hr_per_9": float(raw.get("homeRunsPer9Inn", 0)),
            "ops_against": float(raw.get("ops", 0)),
            "avg_against": float(raw.get("avg", 0)),
        }

    # ── Pitcher Advanced Stats (Statcast-adjacent) ─────────────

    async def get_pitcher_advanced_stats(
        self, pitcher_id: int, season: int
    ) -> dict[str, Any] | None:
        """Get pitcher's advanced/Statcast-adjacent stats for the season.

        Fetches from the MLB Stats API and derives Statcast proxy metrics:
        - k_rate: strikeout percentage (strong Statcast proxy)
        - bb_rate: walk percentage
        - k_minus_bb: K% - BB% (one of the best predictive stats)
        - gb_ao_ratio: groundouts-to-airouts (proxy for groundball tendency)
        - hr_per_9: home runs per 9 innings
        - whip: walks + hits per inning pitched
        - era: earned run average
        """
        data = await self._get(
            f"/people/{pitcher_id}/stats",
            params={
                "stats": "statsSingleSeason",
                "group": "pitching",
                "season": season,
                "gameType": "R",
            },
        )
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return None

        stat = splits[0].get("stat", {})
        ip = _ip(stat.get("inningsPitched", "0")) or 0.0
        if ip == 0.0:
            return None

        # Parse percentage strings (e.g. ".285" -> 28.5%)
        k_pct_raw = stat.get("strikeoutPercentage", stat.get("strikeoutsPer9Inn"))
        bb_pct_raw = stat.get("walkPercentage", stat.get("walksPer9Inn"))

        # strikeoutPercentage/walkPercentage come as decimals like ".285"
        try:
            k_rate = float(k_pct_raw) * 100 if float(k_pct_raw) < 1.0 else float(k_pct_raw)
        except (TypeError, ValueError):
            k_rate = float(stat.get("strikeoutsPer9Inn", 8.0))

        try:
            bb_rate = float(bb_pct_raw) * 100 if float(bb_pct_raw) < 1.0 else float(bb_pct_raw)
        except (TypeError, ValueError):
            bb_rate = float(stat.get("walksPer9Inn", 3.0))

        return {
            "player_id": pitcher_id,
            "k_rate": k_rate,
            "bb_rate": bb_rate,
            "k_minus_bb": k_rate - bb_rate,
            "gb_ao_ratio": float(stat.get("groundOutsToAirouts", 1.0)),
            "hr_per_9": float(stat.get("homeRunsPer9Inn", 1.0)),
            "whip": float(stat.get("whip", 1.30)),
            "era": float(stat.get("era", 4.50)),
            "innings_pitched": ip,
        }

    # ── Pitcher Game Log ──────────────────────────────────────

    async def get_pitcher_game_log(
        self, pitcher_id: int, season: int
    ) -> list[dict[str, Any]]:
        """Get pitcher's game log for the season.

        Calls the gameLog endpoint filtered to regular-season pitching and
        returns a simplified list of dicts with date and innings pitched.
        Returns an empty list on failure.
        """
        try:
            data = await self._get(
                f"/people/{pitcher_id}/stats",
                params={
                    "stats": "gameLog",
                    "group": "pitching",
                    "season": season,
                    "gameType": "R",
                },
            )
            splits = data.get("stats", [{}])[0].get("splits", [])
            entries: list[dict[str, Any]] = []
            for split in splits:
                game_date = split.get("date", "")
                stat = split.get("stat", {})
                ip_raw = stat.get("inningsPitched", "0")
                entries.append({
                    "date": game_date,
                    "innings_pitched": _ip(ip_raw) or 0.0,
                    "earned_runs": int(stat.get("earnedRuns", 0)),
                    "hits": int(stat.get("hits", 0)),
                    "walks": int(stat.get("baseOnBalls", 0)),
                    "strikeouts": int(stat.get("strikeOuts", 0)),
                })
            return entries
        except Exception as e:
            logger.debug(
                "Failed to fetch pitcher game log for %d/%d: %s",
                pitcher_id, season, e,
            )
            return []

    # ── Standings ──────────────────────────────────────────────

    async def get_standings(self, season: int | None = None) -> dict[str, dict[str, Any]]:
        """Fetch current MLB standings. Returns {team_abbr: {wins, losses, pct, streak, l10}}."""
        params: dict[str, Any] = {"leagueId": "103,104"}
        if season:
            params["season"] = season
        data = await self._get("/standings", params=params)

        standings: dict[str, dict[str, Any]] = {}
        for record in data.get("records", []):
            for team_rec in record.get("teamRecords", []):
                team_id = team_rec.get("team", {}).get("id")
                abbr = TEAM_ABBREVS.get(team_id)
                if not abbr:
                    continue

                streak_obj = team_rec.get("streak", {})
                streak_str = streak_obj.get("streakCode", "-")  # e.g. "W4", "L2"

                # L10 record from records -> splitRecords or from the API
                l10 = team_rec.get("records", {}).get("splitRecords", [])
                l10_str = "-"
                for sr in l10:
                    if sr.get("type") == "lastTen":
                        l10_str = f"{sr.get('wins', 0)}-{sr.get('losses', 0)}"
                        break

                standings[abbr] = {
                    "wins": team_rec.get("wins", 0),
                    "losses": team_rec.get("losses", 0),
                    "pct": float(team_rec.get("winningPercentage", ".500")),
                    "streak": streak_str,
                    "l10": l10_str,
                    "run_diff": team_rec.get("runDifferential", 0),
                }
        return standings

    # ── Rosters ────────────────────────────────────────────────

    async def get_roster(
        self, team_id: int, roster_type: str = "active"
    ) -> list[dict[str, Any]]:
        """Fetch team roster. roster_type: active | fullSeason | 40Man."""
        data = await self._get(
            f"/teams/{team_id}/roster",
            params={"rosterType": roster_type},
        )
        players = []
        for entry in data.get("roster", []):
            person = entry.get("person", {})
            pos = entry.get("position", {})
            players.append({
                "player_id": person.get("id"),
                "name": person.get("fullName", ""),
                "position": pos.get("abbreviation", ""),
                "bats": person.get("batSide", {}).get("code", ""),
                "throws": person.get("pitchHand", {}).get("code", ""),
            })
        return players

    async def get_all_player_positions(self) -> dict[int, str]:
        """Fetch positions for all MLB players by iterating team rosters.

        Returns {player_id: position_abbreviation} e.g. {12345: "SS", 67890: "CF"}.
        Normalizes OF positions (LF/CF/RF -> OF).
        """
        positions: dict[int, str] = {}
        for team_abbr, team_id in TEAM_IDS.items():
            try:
                roster = await self.get_roster(team_id, roster_type="fullSeason")
                for player in roster:
                    pid = player.get("player_id")
                    pos = player.get("position", "")
                    if pid and pos:
                        # Normalize outfield positions
                        if pos in ("LF", "CF", "RF"):
                            pos = "OF"
                        # Normalize TWP/UT to DH
                        if pos in ("TWP", "UT", "PH", "PR"):
                            pos = "DH"
                        positions[pid] = pos
            except Exception:
                continue
        return positions

    async def get_injuries(self, team_id: int) -> list[dict[str, Any]]:
        """Fetch current injuries/IL for a team.

        Uses /teams/{id}/roster?rosterType=depthChart and checks status,
        or the injuries endpoint at /injuries.
        Returns list of {"player_id", "name", "position", "injury", "status"}.
        """
        try:
            data = await self._get(
                f"/teams/{team_id}/roster",
                params={"rosterType": "fullSeason"},
            )
            injured = []
            for entry in data.get("roster", []):
                status = entry.get("status", {}).get("code", "A")
                if status in ("D10", "D15", "D60"):  # IL stints
                    person = entry.get("person", {})
                    pos = entry.get("position", {})
                    injured.append({
                        "player_id": person.get("id"),
                        "name": person.get("fullName", ""),
                        "position": pos.get("abbreviation", ""),
                        "status": status,
                    })
            return injured
        except Exception as e:
            logger.debug("Injuries fetch failed for team %d: %s", team_id, e)
            return []

    # ── Lineups ────────────────────────────────────────────

    async def get_game_lineups(
        self, game_date: date
    ) -> dict[str, dict[str, list[dict]]]:
        """Fetch starting lineups for all games on a date.

        Returns {game_id: {"home": [player_dicts], "away": [player_dicts]}}
        where each player has id, name, batSide, pitchHand.
        """
        data = await self._get(
            "/schedule",
            params={
                "sportId": 1,
                "date": game_date.isoformat(),
                "hydrate": "lineups",
            },
        )
        result: dict[str, dict[str, list[dict]]] = {}
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                gid = str(game["gamePk"])
                lineups = game.get("lineups", {})
                home_players = lineups.get("homePlayers", [])
                away_players = lineups.get("awayPlayers", [])
                if home_players or away_players:
                    result[gid] = {
                        "home": [{"id": p["id"], "name": p.get("fullName", "")} for p in home_players],
                        "away": [{"id": p["id"], "name": p.get("fullName", "")} for p in away_players],
                    }
        return result

    async def get_players_info(
        self, player_ids: list[int]
    ) -> dict[int, dict[str, str]]:
        """Fetch batSide and pitchHand for multiple players.

        Returns {player_id: {"bats": "R"/"L"/"S", "throws": "R"/"L"}}
        """
        if not player_ids:
            return {}
        ids_str = ",".join(str(pid) for pid in player_ids)
        data = await self._get("/people", params={"personIds": ids_str})
        result: dict[int, dict[str, str]] = {}
        for p in data.get("people", []):
            result[p["id"]] = {
                "bats": p.get("batSide", {}).get("code", "R"),
                "throws": p.get("pitchHand", {}).get("code", "R"),
            }
        return result

    async def get_player_batting_stats(
        self, player_id: int, season: int
    ) -> dict[str, Any]:
        """Fetch season batting stats for a player."""
        raw = await self.get_player_stats(player_id, season, group="hitting")
        if not raw:
            return {}
        ab = _int(raw.get("atBats")) or 0
        h = _int(raw.get("hits")) or 0
        bb = _int(raw.get("baseOnBalls")) or 0
        hr = _int(raw.get("homeRuns")) or 0
        doubles = _int(raw.get("doubles")) or 0
        triples = _int(raw.get("triples")) or 0
        pa = ab + bb + _int(raw.get("hitByPitch")) or 0 + _int(raw.get("sacFlies")) or 0
        obp = float(raw.get("obp", 0))
        slg = float(raw.get("slg", 0))
        return {
            "player_id": player_id,
            "obp": obp,
            "slg": slg,
            "ops": obp + slg,
            "at_bats": ab,
        }

    async def get_all_pitcher_records(self, season: int) -> dict[int, dict[str, int]]:
        """Fetch official W-L-SV records for all MLB pitchers in a season.

        Uses the /stats/leaders endpoint which returns all qualified pitchers.
        Returns {player_id: {"wins": int, "losses": int, "saves": int}}.
        """
        records: dict[int, dict[str, int]] = {}
        try:
            # Fetch wins leaders (returns up to 1000 pitchers)
            data = await self._get(
                "/stats",
                params={
                    "stats": "season",
                    "group": "pitching",
                    "season": season,
                    "sportId": 1,
                    "limit": 800,
                    "offset": 0,
                    "sortStat": "gamesPlayed",
                    "order": "desc",
                },
            )
            for split in data.get("stats", [{}])[0].get("splits", []):
                player = split.get("player", {})
                pid = player.get("id")
                stat = split.get("stat", {})
                if pid:
                    records[pid] = {
                        "wins": _int(stat.get("wins")) or 0,
                        "losses": _int(stat.get("losses")) or 0,
                        "saves": _int(stat.get("saves")) or 0,
                    }
        except Exception as e:
            logger.warning("Failed to fetch pitcher records: %s", e)
        return records

    # ─�� Batter vs Pitcher ──────────────────────────────────────

    async def get_batter_vs_pitcher(
        self, batter_id: int, pitcher_id: int
    ) -> dict[str, Any] | None:
        """Fetch career stats of a batter against a specific pitcher.

        Returns {"at_bats": int, "hits": int, "home_runs": int, "avg": float, "ops": float}
        or None if no data exists.
        """
        try:
            data = await self._get(
                f"/people/{batter_id}/stats",
                params={
                    "stats": "vsPlayer",
                    "opposingPlayerId": pitcher_id,
                    "group": "hitting",
                },
            )
            splits = data.get("stats", [{}])[0].get("splits", [])
            if not splits:
                return None
            stat = splits[0].get("stat", {})
            ab = _int(stat.get("atBats")) or 0
            if ab == 0:
                return None
            return {
                "at_bats": ab,
                "hits": _int(stat.get("hits")) or 0,
                "home_runs": _int(stat.get("homeRuns")) or 0,
                "avg": float(stat.get("avg", 0)),
                "ops": float(stat.get("ops", 0)),
            }
        except Exception as e:
            logger.debug("BvP lookup failed for %d vs %d: %s", batter_id, pitcher_id, e)
            return None

    # ── Teams List ─────────────────────────────────────────────

    async def get_teams(self, season: int | None = None) -> list[dict[str, Any]]:
        """Fetch all MLB teams."""
        params: dict[str, Any] = {"sportId": 1}
        if season:
            params["season"] = season
        data = await self._get("/teams", params=params)
        teams = []
        for t in data.get("teams", []):
            teams.append({
                "team_id": TEAM_ABBREVS.get(t["id"], str(t["id"])),
                "team_api_id": t["id"],
                "team_name": t.get("name", ""),
                "division": t.get("division", {}).get("name", ""),
                "league": t.get("league", {}).get("abbreviation", ""),
                "stadium_name": t.get("venue", {}).get("name", ""),
            })
        return teams


# ── Helpers ────────────────────────────────────────────────────


def _int(val: Any) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _ip(val: Any) -> float | None:
    """Convert innings pitched string (e.g. '6.2' meaning 6 and 2/3) to float."""
    if val is None:
        return None
    try:
        s = str(val)
        if "." in s:
            whole, frac = s.split(".")
            return int(whole) + int(frac) / 3.0
        return float(s)
    except (ValueError, TypeError):
        return None
