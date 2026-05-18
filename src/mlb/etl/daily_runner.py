"""Daily prediction pipeline — the main entry point for daily operations.

Ties together: schedule fetch, feature building from CSV rolling stats,
model predictions, live odds, weather, and betting recommendations.

Usage:
    python -m mlb.etl.daily_runner                   # Today's games
    python -m mlb.etl.daily_runner --date 2026-05-09  # Specific date
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlb.data.mlb_api import MLBApiClient
from mlb.data.odds_api import OddsApiClient
from mlb.data.weather import WeatherClient
from mlb.etl.build_training_data import (
    DOME_TEAMS, PARK_DEFAULT, PARK_FACTORS, RETRACTABLE_TEAMS,
    TEAM_TIMEZONES, TrainingDataBuilder, _classify_day_game,
    _compute_travel_fatigue, _update_elo,
)
from mlb.features.stadium import compute_stadium_features, calculate_stadium_factor
from mlb.models.predict import GamePrediction, PredictionService
from mlb.features.assembler import GameFeatureVector
from mlb.betting.engine import BettingEngine, BettingSlip
from mlb.alerts import AlertService
from mlb.api.routes.rankings import cache_team_rankings, cache_player_rankings

logger = logging.getLogger(__name__)


class DailyRunner:
    """Orchestrates the full daily prediction pipeline."""

    def __init__(
        self,
        data_dir: Path = Path("data"),
        model_dir: Path = Path("models"),
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.mlb_client = MLBApiClient()
        self.odds_client = OddsApiClient()
        self.weather_client = WeatherClient()
        self.prediction_service = PredictionService(model_dir)
        self.betting_engine = BettingEngine()
        self.alert_service = AlertService()
        self._sp_rest_cache: dict[int, int] = {}
        self._sp_form_cache: dict[int, dict[str, float]] = {}

    async def run(self, target_date: date | None = None) -> dict[str, Any]:
        """Run the full daily pipeline.

        Returns a summary dict with predictions, betting slip, and metadata.
        """
        target_date = target_date or date.today()
        logger.info("=== Daily Runner: %s ===", target_date)

        result: dict[str, Any] = {
            "date": target_date.isoformat(),
            "predictions": [],
            "betting_slip": None,
            "weather": {},
            "errors": [],
        }

        try:
            # 1. Load the trained model
            self.prediction_service.load()
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            result["errors"].append(f"Model load failed: {e}")
            return result

        try:
            # 2. Fetch today's schedule (all games, regardless of status)
            games = await self.mlb_client.get_schedule(target_date)
            all_games = [g for g in games if g.get("game_id")]
            # Only generate new predictions for games not yet final
            scheduled = [
                g for g in all_games
                if g["status"] in ("Scheduled", "Pre-Game", "Warmup", "In Progress", "Preview")
            ]
            logger.info("Found %d games for %s (%d predictable)", len(all_games), target_date, len(scheduled))

            if not all_games:
                logger.info("No games found for %s", target_date)
                return result

            # 3. Build rolling team stats from CSV data
            team_features = self._build_team_features(target_date)

            # 4. Fetch starting pitcher stats
            sp_stats = await self._fetch_sp_stats(scheduled, target_date.year)

            # 4a. Fetch SP rest days (days since last start)
            await self._fetch_sp_rest_days(scheduled, target_date)

            # 4b. Fetch lineups and compute lineup/platoon features
            lineup_features = await self._fetch_lineup_features(
                scheduled, target_date
            )

            # 4c. Fetch IL counts for all teams in today's games
            injury_counts = await self._fetch_injury_counts(scheduled)

            # 5. Fetch weather for home stadiums
            home_teams = list({g["home_team_id"] for g in scheduled})
            weather_data = await self.weather_client.get_bulk_weather(home_teams)
            result["weather"] = {
                k: {
                    "temp": v.temperature_f,
                    "wind": v.wind_speed_mph,
                    "wind_dir": v.wind_direction,
                    "humidity": v.humidity_pct,
                }
                for k, v in weather_data.items()
            }

            # 5b. Fetch live odds early so they can be used as model features
            odds_data: list[dict] = []
            try:
                odds_data = await self.odds_client.get_mlb_odds() or []
                if odds_data:
                    logger.info("Fetched %d odds records for features", len(odds_data))
            except Exception as e:
                logger.warning("Early odds fetch failed, continuing without: %s", e)

            # Build odds lookup keyed by (home_team, away_team) for feature injection
            odds_by_matchup: dict[tuple[str, str], dict] = {}
            for odds in odds_data:
                key = (odds["home_team"], odds["away_team"])
                odds_by_matchup[key] = odds

            # Persist odds for future retraining
            if odds_data:
                self._save_odds_history(target_date, odds_data)

            # 6. Generate predictions for each game
            predictions: list[GamePrediction] = []
            for game in scheduled:
                try:
                    matched_odds = odds_by_matchup.get(
                        (game["home_team_id"], game["away_team_id"])
                    )
                    pred = self._predict_game(
                        game, team_features, weather_data, target_date,
                        sp_stats=sp_stats,
                        lineup_features=lineup_features,
                        game_odds=matched_odds,
                        injury_counts=injury_counts,
                    )
                    if pred:
                        predictions.append(pred)
                except Exception as e:
                    logger.warning("Prediction failed for %s: %s", game["game_id"], e)
                    result["errors"].append(f"Game {game['game_id']}: {e}")

            logger.info("Generated %d predictions", len(predictions))
            game_lookup = {g["game_id"]: g for g in all_games}

            # Fetch live standings for current records/streaks
            try:
                live_standings = await self.mlb_client.get_standings(target_date.year)
                logger.info("Fetched live standings for %d teams", len(live_standings))
            except Exception as e:
                logger.warning("Live standings fetch failed, falling back to CSV: %s", e)
                live_standings = {}

            result["predictions"] = [
                self._prediction_to_dict(
                    p, game_lookup.get(p.game_id, {}),
                    sp_stats=sp_stats,
                    live_standings=live_standings,
                    weather_data=weather_data,
                    game_odds=odds_by_matchup.get(
                        (p.home_team_id, p.away_team_id)
                    ),
                )
                for p in predictions
            ]

            # Add stub entries for games already started/finished (not predicted)
            # so they still show up on the dashboard
            predicted_ids = {p.game_id for p in predictions}
            for game in all_games:
                gid = game["game_id"]
                if gid not in predicted_ids:
                    home_id = game.get("home_team_id", "")
                    away_id = game.get("away_team_id", "")
                    h_std = live_standings.get(home_id, {})
                    a_std = live_standings.get(away_id, {})
                    result["predictions"].append({
                        "game_id": gid,
                        "game_date": target_date.isoformat(),
                        "home_team": home_id,
                        "away_team": away_id,
                        "home_win_prob": 0.5,
                        "away_win_prob": 0.5,
                        "predicted_home_runs": 0.0,
                        "predicted_away_runs": 0.0,
                        "predicted_total": 0.0,
                        "confidence": 0.0,
                        "model_agreement": 0.0,
                        "predicted_winner": home_id,
                        "model_predictions": {},
                        "top_factors": [],
                        "home_wins": h_std.get("wins", 0),
                        "home_losses": h_std.get("losses", 0),
                        "away_wins": a_std.get("wins", 0),
                        "away_losses": a_std.get("losses", 0),
                        "home_streak": h_std.get("streak", ""),
                        "away_streak": a_std.get("streak", ""),
                        "game_time": game.get("game_time", ""),
                        "home_moneyline": odds_by_matchup.get((home_id, away_id), {}).get("home_moneyline"),
                        "away_moneyline": odds_by_matchup.get((home_id, away_id), {}).get("away_moneyline"),
                        "total_line": odds_by_matchup.get((home_id, away_id), {}).get("total_line"),
                        "status": game.get("status", ""),
                    })

            # 7. Generate betting slip from already-fetched odds
            try:
                if odds_data:
                    # Only bet on games where both starters are announced
                    eligible_ids = {
                        p["game_id"]
                        for p in result["predictions"]
                        if p.get("home_sp_name", "TBD") != "TBD"
                        and p.get("away_sp_name", "TBD") != "TBD"
                    }
                    bet_preds = [p for p in predictions if p.game_id in eligible_ids]
                    odds_matched = self._match_odds_to_games(bet_preds, odds_data)
                    slip = self.betting_engine.find_value_bets(bet_preds, odds_matched)
                    result["betting_slip"] = self._slip_to_dict(slip)
                    logger.info(
                        "Betting: %d value bets, $%.2f total stake, $%.2f EV",
                        slip.num_bets, slip.total_stake, slip.total_ev,
                    )
                else:
                    logger.info("No odds data available")
            except Exception as e:
                logger.warning("Odds fetch failed: %s", e)
                result["errors"].append(f"Odds: {e}")

            # 8. Cache predictions for the API
            self._cache_results(target_date, result)

            # 9. Generate team rankings from rolling stats (after cache so it merges into file)
            self._generate_rankings(team_features, target_date, live_standings)

            # 9b. Generate player rankings from season CSV data
            await self._generate_player_rankings(target_date)

            # 10. Send alerts (Slack/email) if configured
            try:
                await self.alert_service.send_betting_alert(result)
            except Exception as e:
                logger.warning("Alert send failed: %s", e)

        except Exception as e:
            logger.exception("Daily runner failed")
            result["errors"].append(str(e))
        finally:
            await self.mlb_client.close()

        return result

    def _build_team_features(self, target_date: date) -> dict[str, dict[str, float]]:
        """Build current team features from CSV rolling stats."""
        builder = TrainingDataBuilder(data_dir=self.data_dir)

        # Determine which seasons to load
        current_year = target_date.year
        seasons = [current_year - 1, current_year]

        try:
            games_df = builder._load_games(seasons)
            batting_df = builder._load_batting(seasons)
            pitching_df = builder._load_pitching(seasons)
        except FileNotFoundError:
            logger.warning("CSV data not found, trying current year only")
            seasons = [current_year]
            games_df = builder._load_games(seasons)
            batting_df = builder._load_batting(seasons)
            pitching_df = builder._load_pitching(seasons)

        team_rolling = builder._compute_team_rolling_stats(games_df, batting_df, pitching_df)

        # Get latest features for each team
        features = {}
        for team_id, stats_list in team_rolling.items():
            feat = builder._get_team_features(stats_list, target_date)
            if feat:
                features[team_id] = feat

        # Compute Elo ratings, rest days, and last venue from game history
        self._elo_ratings: dict[str, float] = {}
        self._last_game_date: dict[str, date] = {}
        self._last_game_venue: dict[str, str] = {}
        completed = games_df[
            (games_df["status"] == "Final")
            & games_df["home_score"].notna()
        ].sort_values("game_date")
        for _, g in completed.iterrows():
            h, a = g["home_team_id"], g["away_team_id"]
            gd = g["game_date"]
            if gd >= target_date:
                break
            home_won = g["home_score"] > g["away_score"]
            _update_elo(self._elo_ratings, h, a, home_won)
            self._last_game_date[h] = gd
            self._last_game_date[a] = gd
            self._last_game_venue[h] = h  # home team was at their own venue
            self._last_game_venue[a] = h  # away team was at home team's venue

        logger.info("Built features for %d teams (Elo for %d)", len(features), len(self._elo_ratings))
        return features

    async def _fetch_sp_stats(
        self, games: list[dict], season: int
    ) -> dict[int, dict]:
        """Fetch season stats for all probable starters in today's games."""
        sp_stats: dict[int, dict] = {}
        pitcher_ids: list[int] = []

        for game in games:
            for side in ("home_probable_pitcher", "away_probable_pitcher"):
                pitcher = game.get(side)
                if pitcher and pitcher.get("player_id"):
                    pitcher_ids.append(pitcher["player_id"])

        # Deduplicate
        pitcher_ids = list(set(pitcher_ids))
        logger.info("Fetching stats for %d probable starters...", len(pitcher_ids))

        import asyncio
        for pid in pitcher_ids:
            try:
                stats = await self.mlb_client.get_pitcher_season_stats(pid, season)
                if stats:
                    sp_stats[pid] = stats
                await asyncio.sleep(0.2)  # Rate limit
            except Exception as e:
                logger.debug("Failed to fetch SP stats for %d: %s", pid, e)

        logger.info("Got stats for %d / %d starters", len(sp_stats), len(pitcher_ids))
        return sp_stats

    async def _fetch_lineup_features(
        self, games: list[dict], target_date: date,
    ) -> dict[str, dict[str, float]]:
        """Fetch lineups and compute lineup OPS + platoon advantage per game side.

        Returns {game_id: {"h_lineup_ops": ..., "a_lineup_ops": ..., "h_platoon_adv": ..., ...}}
        """
        result: dict[str, dict[str, float]] = {}
        try:
            lineups = await self.mlb_client.get_game_lineups(target_date)
        except Exception as e:
            logger.warning("Lineup fetch failed: %s", e)
            return result

        if not lineups:
            return result

        # Collect all player IDs to fetch info (handedness) and batting stats
        all_pids: set[int] = set()
        for gid, sides in lineups.items():
            for side in ("home", "away"):
                for p in sides.get(side, []):
                    all_pids.add(p["id"])

        # Fetch handedness for all players
        try:
            player_info = await self.mlb_client.get_players_info(list(all_pids))
        except Exception as e:
            logger.warning("Player info fetch failed: %s", e)
            player_info = {}

        # Fetch batting stats for lineup players
        season = target_date.year
        batting_stats: dict[int, dict] = {}
        for pid in all_pids:
            try:
                stats = await self.mlb_client.get_player_batting_stats(pid, season)
                if stats and stats.get("at_bats", 0) >= 20:
                    batting_stats[pid] = stats
                await asyncio.sleep(0.1)
            except Exception:
                pass

        logger.info("Got batting stats for %d / %d lineup players", len(batting_stats), len(all_pids))

        # Compute features per game
        for game in games:
            gid = game["game_id"]
            lineup_data = lineups.get(gid)
            if not lineup_data:
                continue

            game_feats: dict[str, float] = {}
            for prefix, side in [("h", "home"), ("a", "away")]:
                players = lineup_data.get(side, [])
                ops_vals = []
                for p in players:
                    bs = batting_stats.get(p["id"])
                    if bs:
                        ops_vals.append(bs["ops"])

                game_feats[f"{prefix}_lineup_ops"] = np.mean(ops_vals) if ops_vals else 0.720
                game_feats[f"{prefix}_lineup_obp"] = np.mean(
                    [batting_stats[p["id"]]["obp"] for p in players if p["id"] in batting_stats]
                ) if any(p["id"] in batting_stats for p in players) else 0.320
                game_feats[f"{prefix}_lineup_slg"] = np.mean(
                    [batting_stats[p["id"]]["slg"] for p in players if p["id"] in batting_stats]
                ) if any(p["id"] in batting_stats for p in players) else 0.400

                # Platoon advantage vs opposing SP
                opp_side = "away" if side == "home" else "home"
                sp_key = f"{opp_side}_probable_pitcher"
                opp_sp = game.get(sp_key)
                sp_throws = "R"
                if opp_sp and opp_sp.get("player_id"):
                    sp_info = player_info.get(opp_sp["player_id"])
                    if sp_info:
                        sp_throws = sp_info.get("throws", "R")

                adv_count = 0
                total = 0
                for p in players:
                    pi = player_info.get(p["id"])
                    if not pi:
                        continue
                    total += 1
                    bats = pi.get("bats", "R")
                    if bats == "S" or (bats == "L" and sp_throws == "R") or (bats == "R" and sp_throws == "L"):
                        adv_count += 1

                game_feats[f"{prefix}_platoon_adv"] = adv_count / total if total >= 3 else 0.5

            # Batter-vs-pitcher matchup OPS
            for prefix, side in [("h", "home"), ("a", "away")]:
                players = lineup_data.get(side, [])
                opp_side = "away" if side == "home" else "home"
                sp_key = f"{opp_side}_probable_pitcher"
                opp_sp = game.get(sp_key)
                opp_sp_id = opp_sp.get("player_id") if opp_sp else None

                bvp_ops_vals = []
                if opp_sp_id:
                    for p in players:
                        try:
                            bvp = await self.mlb_client.get_batter_vs_pitcher(
                                p["id"], opp_sp_id
                            )
                            if bvp and bvp["at_bats"] >= 5:
                                bvp_ops_vals.append(bvp["ops"])
                            await asyncio.sleep(0.1)
                        except Exception:
                            pass

                if len(bvp_ops_vals) >= 3:
                    game_feats[f"{prefix}_bvp_ops"] = np.mean(bvp_ops_vals)
                else:
                    game_feats[f"{prefix}_bvp_ops"] = 0.750  # league average default

            game_feats["diff_bvp_ops"] = game_feats["h_bvp_ops"] - game_feats["a_bvp_ops"]

            # Approximate recent form from season stats (precise 7-day form is
            # only available in training data from CSVs — the live API would
            # require ~18 game-log calls per game which is too slow).
            # Season OPS is a reasonable proxy; the model will weight this
            # feature based on training data where true 7-day form is available.
            for prefix, side in [("h", "home"), ("a", "away")]:
                players = lineup_data.get(side, [])
                ops_vals = [batting_stats[p["id"]]["ops"] for p in players if p["id"] in batting_stats]
                game_feats[f"{prefix}_lineup_ops_7d"] = np.mean(ops_vals) if ops_vals else 0.720
                hot = sum(1 for o in ops_vals if o > 0.800)
                game_feats[f"{prefix}_lineup_hot_pct"] = hot / len(ops_vals) if ops_vals else 0.4

            # Differentials
            for k in ("lineup_ops", "lineup_obp", "lineup_slg", "platoon_adv",
                       "lineup_ops_7d", "lineup_hot_pct"):
                game_feats[f"diff_{k}"] = game_feats[f"h_{k}"] - game_feats[f"a_{k}"]

            result[gid] = game_feats

        logger.info("Computed lineup features for %d games", len(result))
        return result

    async def _fetch_sp_rest_days(
        self, games: list[dict], target_date: date,
    ) -> dict[int, int]:
        """Fetch rest days for all probable starters by checking their game logs.

        Populates and returns self._sp_rest_cache: {pitcher_id: rest_days}.
        """
        pitcher_ids: list[int] = []
        for game in games:
            for side in ("home_probable_pitcher", "away_probable_pitcher"):
                pitcher = game.get(side)
                if pitcher and pitcher.get("player_id"):
                    pitcher_ids.append(pitcher["player_id"])

        # Deduplicate and skip already-cached pitchers
        pitcher_ids = [
            pid for pid in set(pitcher_ids) if pid not in self._sp_rest_cache
        ]
        if not pitcher_ids:
            return self._sp_rest_cache

        logger.info("Fetching rest days for %d probable starters...", len(pitcher_ids))
        season = target_date.year

        for pid in pitcher_ids:
            try:
                game_log = await self.mlb_client.get_pitcher_game_log(pid, season)
                rest = self._compute_rest_days(game_log, target_date)
                self._sp_rest_cache[pid] = rest
                # Also compute recent form from the same game log
                self._sp_form_cache[pid] = self._compute_recent_form(game_log, target_date)
                await asyncio.sleep(0.1)  # Rate limit
            except Exception as e:
                logger.debug("Failed to fetch game log for SP %d: %s", pid, e)
                self._sp_rest_cache[pid] = 5  # default on failure

        logger.info(
            "Got rest days for %d starters", len(self._sp_rest_cache),
        )
        return self._sp_rest_cache

    @staticmethod
    def _compute_recent_form(
        game_log: list[dict], target_date: date, n_starts: int = 3
    ) -> dict[str, float]:
        """Compute ERA, WHIP, K/9 from the pitcher's last N starts.

        Only considers entries before target_date. Returns empty dict
        if fewer than 1 qualifying start found.
        """
        # Filter to starts before target_date, sorted most recent first
        past_starts = []
        for entry in game_log:
            date_str = entry.get("date", "")
            if not date_str:
                continue
            try:
                game_date = date.fromisoformat(date_str)
            except ValueError:
                continue
            if game_date >= target_date:
                continue
            # Only count starts where pitcher threw at least 3 IP
            if entry.get("innings_pitched", 0) >= 3.0:
                past_starts.append(entry)

        if not past_starts:
            return {}

        # Sort by date descending, take last N
        past_starts.sort(key=lambda e: e["date"], reverse=True)
        recent = past_starts[:n_starts]

        total_ip = sum(e.get("innings_pitched", 0) for e in recent)
        if total_ip == 0:
            return {}

        total_er = sum(e.get("earned_runs", 0) for e in recent)
        total_h = sum(e.get("hits", 0) for e in recent)
        total_bb = sum(e.get("walks", 0) for e in recent)
        total_k = sum(e.get("strikeouts", 0) for e in recent)

        era = (total_er / total_ip) * 9.0
        whip = (total_h + total_bb) / total_ip
        k9 = (total_k / total_ip) * 9.0

        return {"era": round(era, 2), "whip": round(whip, 2), "k9": round(k9, 1)}

    @staticmethod
    def _compute_rest_days(game_log: list[dict], target_date: date) -> int:
        """Compute days since the pitcher's most recent appearance.

        Walks the game log entries (which contain a ``date`` string in
        ``YYYY-MM-DD`` format) and returns the number of days between
        *target_date* and the most recent entry.  Returns 5 (normal rest)
        when no entries are found.
        """
        last_start: date | None = None
        for entry in game_log:
            date_str = entry.get("date", "")
            if not date_str:
                continue
            try:
                game_date = date.fromisoformat(date_str)
            except ValueError:
                continue
            if game_date >= target_date:
                continue
            if last_start is None or game_date > last_start:
                last_start = game_date

        if last_start is None:
            return 5
        return (target_date - last_start).days

    def _get_sp_rest_days(self, pitcher_id: int, target_date: date) -> int:
        """Return cached rest days for a pitcher, defaulting to 5."""
        return self._sp_rest_cache.get(pitcher_id, 5)

    def _get_sp_recent_form(self, pitcher_id: int) -> dict[str, float]:
        """Return recent form stats (last 3 starts) from cached game log.

        Computes ERA, WHIP, K/9 from the pitcher's most recent 3 game log
        entries. The game log is already fetched by _fetch_sp_rest_days.
        Returns empty dict if no data available.
        """
        if pitcher_id in self._sp_form_cache:
            return self._sp_form_cache[pitcher_id]

        # Game log was populated during _fetch_sp_rest_days
        # Re-use the cached game log by computing from the rest cache data
        # The game log isn't stored directly, so we return empty and let
        # the prefetch populate this cache
        return self._sp_form_cache.get(pitcher_id, {})

    async def _fetch_injury_counts(
        self, games: list[dict]
    ) -> dict[str, int]:
        """Fetch IL player count for each team in today's games.

        Returns {team_abbrev: number_of_players_on_IL}.
        """
        from mlb.data.mlb_api import TEAM_IDS

        teams = set()
        for g in games:
            teams.add(g["home_team_id"])
            teams.add(g["away_team_id"])

        counts: dict[str, int] = {}
        for team_abbrev in teams:
            team_numeric_id = TEAM_IDS.get(team_abbrev)
            if not team_numeric_id:
                counts[team_abbrev] = 0
                continue
            try:
                injured = await self.mlb_client.get_injuries(team_numeric_id)
                counts[team_abbrev] = len(injured)
                await asyncio.sleep(0.1)
            except Exception:
                counts[team_abbrev] = 0

        logger.info("Fetched IL counts for %d teams (avg %.1f on IL)",
                    len(counts), sum(counts.values()) / max(len(counts), 1))
        return counts

    def _predict_game(
        self,
        game: dict,
        team_features: dict[str, dict[str, float]],
        weather_data: dict,
        target_date: date,
        sp_stats: dict[int, dict] | None = None,
        lineup_features: dict[str, dict[str, float]] | None = None,
        game_odds: dict | None = None,
        injury_counts: dict[str, int] | None = None,
    ) -> GamePrediction | None:
        """Generate a prediction for a single game."""
        home = game["home_team_id"]
        away = game["away_team_id"]

        home_feat = team_features.get(home)
        away_feat = team_features.get(away)

        if not home_feat or not away_feat:
            logger.debug("Missing features for %s vs %s", home, away)
            return None

        # Build stadium features with weather
        weather = weather_data.get(home)
        stadium_feat = compute_stadium_features(home, None, weather)
        stadium_factor = calculate_stadium_factor(stadium_feat)

        # Build a GameFeatureVector compatible with our trained model
        features: dict[str, float] = {}

        # Home features
        for k, v in home_feat.items():
            features[f"h_{k}"] = v
        # Away features
        for k, v in away_feat.items():
            features[f"a_{k}"] = v
        # Differentials
        for k in home_feat:
            if k in away_feat:
                features[f"diff_{k}"] = home_feat[k] - away_feat[k]

        # Ballpark factors
        park = PARK_FACTORS.get(home, PARK_DEFAULT)
        features["park_runs_factor"] = park["runs"]
        features["park_hr_factor"] = park["hr"]

        # Weather proxy features
        features["is_dome"] = 1 if home in DOME_TEAMS or home in RETRACTABLE_TEAMS else 0
        features["game_month"] = target_date.month

        # Real weather features
        is_dome = home in DOME_TEAMS or home in RETRACTABLE_TEAMS
        wx = weather_data.get(home)
        features["temperature"] = 72.0 if is_dome or not wx else wx.temperature_f
        features["wind_speed"] = 5.0 if is_dome or not wx else wx.wind_speed_mph
        features["humidity"] = 50.0 if is_dome or not wx else wx.humidity_pct
        features["is_outdoor"] = 0 if is_dome else 1

        # Market odds features
        if game_odds:
            h_ml = game_odds.get("home_moneyline")
            a_ml = game_odds.get("away_moneyline")
            if h_ml is not None and h_ml != 0:
                features["market_home_prob"] = (
                    abs(h_ml) / (abs(h_ml) + 100) if h_ml < 0
                    else 100 / (h_ml + 100)
                )
                features["has_real_odds"] = True
            else:
                features["market_home_prob"] = 0.5
                features["has_real_odds"] = False
            features["market_total"] = game_odds.get("total_line", 8.5) or 8.5
        else:
            features["market_home_prob"] = 0.5
            features["market_total"] = 8.5
            features["has_real_odds"] = False

        # Elo ratings
        h_elo = self._elo_ratings.get(home, 1500.0)
        a_elo = self._elo_ratings.get(away, 1500.0)
        features["h_elo"] = h_elo
        features["a_elo"] = a_elo
        features["elo_diff"] = h_elo - a_elo

        # Rest days
        h_last = self._last_game_date.get(home)
        a_last = self._last_game_date.get(away)
        features["h_rest_days"] = (target_date - h_last).days if h_last else 5
        features["a_rest_days"] = (target_date - a_last).days if a_last else 5
        features["rest_diff"] = features["h_rest_days"] - features["a_rest_days"]

        # Venue-specific rolling features (home team's home record, away team's road record)
        features["h_home_win_pct"] = home_feat.get("venue_home_win_pct", 0.536)
        features["a_away_win_pct"] = away_feat.get("venue_away_win_pct", 0.464)
        features["diff_venue_win_pct"] = features["h_home_win_pct"] - features["a_away_win_pct"]
        features["h_home_rs_per_game"] = home_feat.get("venue_home_rs_per_game", 4.5)
        features["a_away_rs_per_game"] = away_feat.get("venue_away_rs_per_game", 4.5)

        # Starting pitcher season stats — passed to model as features
        sp_data = sp_stats or {}
        home_sp = game.get("home_probable_pitcher")
        away_sp = game.get("away_probable_pitcher")
        h_sp_stats = sp_data.get(home_sp["player_id"]) if home_sp else None
        a_sp_stats = sp_data.get(away_sp["player_id"]) if away_sp else None

        sp_defaults = {"era": 4.50, "whip": 1.30, "k_per_nine": 8.0, "bb_per_nine": 3.0, "innings_pitched": 0.0}
        sp_key_map = {"era": "sp_season_era", "whip": "sp_season_whip",
                      "k_per_nine": "sp_season_k9", "bb_per_nine": "sp_season_bb9",
                      "innings_pitched": "sp_season_ip"}
        for api_key, feat_key in sp_key_map.items():
            h_val = h_sp_stats.get(api_key, sp_defaults[api_key]) if h_sp_stats else sp_defaults[api_key]
            a_val = a_sp_stats.get(api_key, sp_defaults[api_key]) if a_sp_stats else sp_defaults[api_key]
            features[f"h_{feat_key}"] = h_val
            features[f"a_{feat_key}"] = a_val
            features[f"diff_{feat_key}"] = h_val - a_val

        # Derived Statcast-proxy feature: K-BB% (one of the strongest
        # predictors of future pitcher performance)
        h_k_minus_bb = features.get("h_sp_season_k9", 8.0) - features.get("h_sp_season_bb9", 3.0)
        a_k_minus_bb = features.get("a_sp_season_k9", 8.0) - features.get("a_sp_season_bb9", 3.0)
        features["h_sp_k_minus_bb"] = h_k_minus_bb
        features["a_sp_k_minus_bb"] = a_k_minus_bb
        features["diff_sp_k_minus_bb"] = h_k_minus_bb - a_k_minus_bb

        # Pitcher rest days (days since SP last started) — live lookup
        h_rest = self._get_sp_rest_days(home_sp["player_id"], target_date) if home_sp else 5
        a_rest = self._get_sp_rest_days(away_sp["player_id"], target_date) if away_sp else 5
        features["h_sp_rest_days"] = h_rest
        features["a_sp_rest_days"] = a_rest
        features["diff_sp_rest_days"] = h_rest - a_rest

        # Pitcher recent form (last 3 starts) — derived from game log
        h_recent = self._get_sp_recent_form(home_sp["player_id"]) if home_sp else {}
        a_recent = self._get_sp_recent_form(away_sp["player_id"]) if away_sp else {}
        features["h_sp_recent_era"] = h_recent.get("era", features.get("h_sp_season_era", 4.50))
        features["a_sp_recent_era"] = a_recent.get("era", features.get("a_sp_season_era", 4.50))
        features["diff_sp_recent_era"] = features["h_sp_recent_era"] - features["a_sp_recent_era"]
        features["h_sp_recent_whip"] = h_recent.get("whip", features.get("h_sp_season_whip", 1.30))
        features["a_sp_recent_whip"] = a_recent.get("whip", features.get("a_sp_season_whip", 1.30))
        features["diff_sp_recent_whip"] = features["h_sp_recent_whip"] - features["a_sp_recent_whip"]
        features["h_sp_recent_k9"] = h_recent.get("k9", features.get("h_sp_season_k9", 8.0))
        features["a_sp_recent_k9"] = a_recent.get("k9", features.get("a_sp_season_k9", 8.0))
        features["diff_sp_recent_k9"] = features["h_sp_recent_k9"] - features["a_sp_recent_k9"]

        # Lineup and platoon features
        lf = (lineup_features or {}).get(game["game_id"], {})
        for k in ("lineup_ops", "lineup_obp", "lineup_slg", "platoon_adv"):
            features[f"h_{k}"] = lf.get(f"h_{k}", 0.720 if "ops" in k else 0.320 if "obp" in k else 0.400 if "slg" in k else 0.5)
            features[f"a_{k}"] = lf.get(f"a_{k}", 0.720 if "ops" in k else 0.320 if "obp" in k else 0.400 if "slg" in k else 0.5)
            features[f"diff_{k}"] = features[f"h_{k}"] - features[f"a_{k}"]

        # Lineup recent form (7-day OPS)
        for k in ("lineup_ops_7d", "lineup_hot_pct"):
            default = 0.720 if "ops" in k else 0.4
            features[f"h_{k}"] = lf.get(f"h_{k}", default)
            features[f"a_{k}"] = lf.get(f"a_{k}", default)
            features[f"diff_{k}"] = features[f"h_{k}"] - features[f"a_{k}"]

        # Batter-vs-pitcher matchup OPS
        features["h_bvp_ops"] = lf.get("h_bvp_ops", 0.750)
        features["a_bvp_ops"] = lf.get("a_bvp_ops", 0.750)
        features["diff_bvp_ops"] = features["h_bvp_ops"] - features["a_bvp_ops"]

        # Bullpen availability (approximated from rolling stats)
        # bp_ip_3d is already in the rolling features from _compute_team_rolling_stats
        h_bp_ip_3d = home_feat.get("bp_ip_3d", 6.0)
        a_bp_ip_3d = away_feat.get("bp_ip_3d", 6.0)
        # Freshness: 15 IP in 3 days = fully depleted (freshness 0)
        h_bp_freshness = max(0.0, 1.0 - h_bp_ip_3d / 15.0)
        a_bp_freshness = max(0.0, 1.0 - a_bp_ip_3d / 15.0)
        # Approximate relievers used from IP (avg ~1.5 IP per reliever appearance)
        h_bp_relievers_used_3d = round(h_bp_ip_3d / 1.5, 1)
        a_bp_relievers_used_3d = round(a_bp_ip_3d / 1.5, 1)

        features["h_bp_freshness"] = h_bp_freshness
        features["a_bp_freshness"] = a_bp_freshness
        features["diff_bp_freshness"] = h_bp_freshness - a_bp_freshness
        features["h_bp_relievers_used_3d"] = h_bp_relievers_used_3d
        features["a_bp_relievers_used_3d"] = a_bp_relievers_used_3d
        features["diff_bp_relievers_used_3d"] = h_bp_relievers_used_3d - a_bp_relievers_used_3d

        # Travel fatigue
        h_prev_venue = self._last_game_venue.get(home)
        a_prev_venue = self._last_game_venue.get(away)
        features["h_travel_fatigue"] = _compute_travel_fatigue(h_prev_venue, home) if h_prev_venue else 0.0
        features["a_travel_fatigue"] = _compute_travel_fatigue(a_prev_venue, home) if a_prev_venue else 0.0
        features["diff_travel_fatigue"] = features["h_travel_fatigue"] - features["a_travel_fatigue"]

        # Day/night classification
        game_time_str = game.get("game_time", "")
        features["is_day_game"] = _classify_day_game(game_time_str, home)

        # ── Feature 1: Umpire Assignment ──
        # Default to neutral; can be enhanced with real umpire K-rate data later
        features["ump_k_rate_effect"] = 0.0

        # ── Feature 2: Run Differential Trends (7-game) ──
        features["h_rd_7d"] = home_feat.get("rd_7d", 0.0)
        features["a_rd_7d"] = away_feat.get("rd_7d", 0.0)
        features["diff_rd_7d"] = features["h_rd_7d"] - features["a_rd_7d"]

        # ── Feature 3: Defensive Metrics Proxy ──
        features["h_def_proxy"] = home_feat.get("def_proxy", 0.0)
        features["a_def_proxy"] = away_feat.get("def_proxy", 0.0)
        features["diff_def_proxy"] = features["h_def_proxy"] - features["a_def_proxy"]

        # ── Feature 4: Platoon Splits (SP handedness) ──
        # Encode SP throws hand: 0=RHP, 1=LHP
        h_sp_throws = 0.0
        a_sp_throws = 0.0
        if home_sp and home_sp.get("player_id"):
            pi = getattr(self, '_player_info_cache', {}).get(home_sp["player_id"])
            if pi:
                h_sp_throws = 1.0 if pi.get("throws") == "L" else 0.0
        if away_sp and away_sp.get("player_id"):
            pi = getattr(self, '_player_info_cache', {}).get(away_sp["player_id"])
            if pi:
                a_sp_throws = 1.0 if pi.get("throws") == "L" else 0.0
        features["h_sp_throws"] = h_sp_throws
        features["a_sp_throws"] = a_sp_throws
        features["platoon_advantage_home"] = 1.0 if lf.get("h_platoon_adv", 0.5) > 0.5 else 0.0
        features["platoon_advantage_away"] = 1.0 if lf.get("a_platoon_adv", 0.5) > 0.5 else 0.0

        # ── Feature 5: Recent Form Weighting (Exponential Decay) ──
        features["h_ewm_win_pct"] = home_feat.get("ewm_win_pct", home_feat.get("win_pct", 0.500))
        features["a_ewm_win_pct"] = away_feat.get("ewm_win_pct", away_feat.get("win_pct", 0.500))
        features["diff_ewm_win_pct"] = features["h_ewm_win_pct"] - features["a_ewm_win_pct"]
        features["h_ewm_rs_per_game"] = home_feat.get("ewm_rs_per_game", home_feat.get("rs_per_game", 4.5))
        features["a_ewm_rs_per_game"] = away_feat.get("ewm_rs_per_game", away_feat.get("rs_per_game", 4.5))
        features["diff_ewm_rs_per_game"] = features["h_ewm_rs_per_game"] - features["a_ewm_rs_per_game"]
        features["h_ewm_ra_per_game"] = home_feat.get("ewm_ra_per_game", home_feat.get("ra_per_game", 4.5))
        features["a_ewm_ra_per_game"] = away_feat.get("ewm_ra_per_game", away_feat.get("ra_per_game", 4.5))
        features["diff_ewm_ra_per_game"] = features["h_ewm_ra_per_game"] - features["a_ewm_ra_per_game"]
        features["h_momentum"] = home_feat.get("momentum", 0.0)
        features["a_momentum"] = away_feat.get("momentum", 0.0)
        features["diff_momentum"] = features["h_momentum"] - features["a_momentum"]

        # ── Feature 6: Bullpen Fatigue Tracking ──
        features["h_bp_ip_3d"] = home_feat.get("bp_ip_3d", home_feat.get("bullpen_ip_3d", 6.0))
        features["a_bp_ip_3d"] = away_feat.get("bp_ip_3d", away_feat.get("bullpen_ip_3d", 6.0))
        features["diff_bp_ip_3d"] = features["h_bp_ip_3d"] - features["a_bp_ip_3d"]

        # ── Feature 7: Interaction Features (mismatch detection) ──
        elo_gap = features.get("elo_diff", 0.0) / 100.0
        sp_gap = features.get("diff_sp_season_era", 0.0)
        bp_gap = features.get("diff_bp_freshness", 0.0)
        mom_gap = features.get("diff_momentum", 0.0)
        form_gap = features.get("diff_ewm_win_pct", 0.0) * 10

        features["interact_elo_x_sp"] = elo_gap * (-sp_gap)
        features["interact_elo_x_bp"] = elo_gap * bp_gap
        features["interact_sp_x_bp"] = (-sp_gap) * bp_gap
        features["interact_elo_x_momentum"] = elo_gap * mom_gap

        # SP quality × opposing offense
        h_off_ops = home_feat.get("ops_14", 0.720)
        a_off_ops = away_feat.get("ops_14", 0.720)
        features["interact_hsp_vs_aoff"] = (-sp_gap) * (a_off_ops - 0.720) * 10
        features["interact_asp_vs_hoff"] = sp_gap * (h_off_ops - 0.720) * 10

        # Rest × form
        rest_gap = features.get("rest_diff", 0.0)
        features["interact_rest_x_form"] = rest_gap * form_gap

        # Park factor × SP quality
        park = PARK_FACTORS.get(home, PARK_DEFAULT)
        pf = park["runs"]
        features["interact_park_x_sp"] = (pf - 1.0) * 10 * (-sp_gap)

        # Bullpen × pitching duel likelihood
        h_sp_era_val = features.get("h_sp_season_era", 4.50)
        a_sp_era_val = features.get("a_sp_season_era", 4.50)
        pitching_duel = max(0, (9.0 - h_sp_era_val - a_sp_era_val) / 4.0)
        features["interact_bp_x_duel"] = bp_gap * pitching_duel

        # ── Feature 8: Injury/IL Signals ──
        ic = injury_counts or {}
        h_il = ic.get(home, 0)
        a_il = ic.get(away, 0)
        features["h_il_count"] = float(h_il)
        features["a_il_count"] = float(a_il)
        features["diff_il_count"] = float(h_il - a_il)

        # Compute composite scores from features
        home_off_score = _compute_offense_score(home_feat)
        away_off_score = _compute_offense_score(away_feat)
        home_pit_score = _compute_pitching_score(home_feat)
        away_pit_score = _compute_pitching_score(away_feat)

        # Adjust pitching scores with SP quality (affects run predictions + confidence)
        if h_sp_stats:
            home_pit_score = _adjust_pitching_with_sp(home_pit_score, h_sp_stats)
        if a_sp_stats:
            away_pit_score = _adjust_pitching_with_sp(away_pit_score, a_sp_stats)

        h_power = _power_score(home_feat)
        a_power = _power_score(away_feat)

        game_fv = GameFeatureVector(
            game_id=game["game_id"],
            game_date=target_date.isoformat(),
            home_team_id=home,
            away_team_id=away,
            features=features,
            home_power_score=h_power,
            away_power_score=a_power,
            home_offense_score=home_off_score,
            away_offense_score=away_off_score,
            home_pitching_score=home_pit_score,
            away_pitching_score=away_pit_score,
            home_bullpen_score=50.0,
            home_defense_score=50.0,
            home_momentum_score=50.0,
            home_sp_index=50.0,
            away_bullpen_score=50.0,
            away_defense_score=50.0,
            away_momentum_score=50.0,
            away_sp_index=50.0,
            offense_diff=home_off_score - away_off_score,
            pitching_diff=home_pit_score - away_pit_score,
            bullpen_diff=0.0,
            defense_diff=0.0,
            momentum_diff=0.0,
            sp_diff=0.0,
            power_diff=h_power - a_power,
            stadium_factor=stadium_factor,
        )

        return self.prediction_service.predict_game(game_fv)

    def _match_odds_to_games(
        self,
        predictions: list[GamePrediction],
        odds_data: list[dict],
    ) -> list[dict]:
        """Match odds data to predictions by team matchups."""
        matched = []
        for odds in odds_data:
            for pred in predictions:
                if (
                    odds["home_team"] == pred.home_team_id
                    and odds["away_team"] == pred.away_team_id
                ):
                    matched.append({
                        "game_id": pred.game_id,
                        "home_moneyline": odds["home_moneyline"],
                        "away_moneyline": odds["away_moneyline"],
                        "total_line": odds["total_line"],
                        "over_odds": odds.get("over_odds", -110),
                        "under_odds": odds.get("under_odds", -110),
                    })
                    break
        return matched

    def _generate_rankings(
        self,
        team_features: dict[str, dict[str, float]],
        target_date: date,
        live_standings: dict[str, dict] | None = None,
    ):
        """Generate team power rankings from rolling stats."""
        date_str = target_date.isoformat()
        live_standings = live_standings or {}

        # Fall back to CSV records only if live standings unavailable
        if not live_standings:
            csv_records = self._compute_season_records(target_date.year)
        else:
            csv_records = {}

        rankings_data = []
        for team_id, feat in team_features.items():
            off = _compute_offense_score(feat)
            pit = _compute_pitching_score(feat)
            bp = _compute_bullpen_score(feat)
            defense = _compute_defense_score(feat)
            power = _power_score(feat)

            std = live_standings.get(team_id, {})
            rec = csv_records.get(team_id, {})
            wins = std.get("wins", rec.get("wins", 0))
            losses = std.get("losses", rec.get("losses", 0))
            rd = std.get("run_diff", rec.get("run_diff", 0))
            streak_str = std.get("streak", "-")
            l10_str = std.get("l10", "")

            # Parse streak for momentum calc
            streak = 0
            if streak_str.startswith("W"):
                streak = int(streak_str[1:]) if len(streak_str) > 1 else 0
            elif streak_str.startswith("L"):
                streak = -int(streak_str[1:]) if len(streak_str) > 1 else 0

            # Parse L10 for momentum calc
            if l10_str and "-" in l10_str:
                l10_w = int(l10_str.split("-")[0])
            else:
                l10_w = int(feat.get("l10_wpct", 0.5) * 10)

            # Tier from power score
            if power >= 60:
                tier = "elite"
            elif power >= 50:
                tier = "contender"
            elif power >= 40:
                tier = "average"
            elif power >= 30:
                tier = "below_average"
            else:
                tier = "rebuilding"

            rankings_data.append({
                "team_id": team_id,
                "team_name": _TEAM_NAMES.get(team_id, team_id),
                "division": _TEAM_DIVISIONS.get(team_id, ""),
                "league": _TEAM_LEAGUES.get(team_id, ""),
                "power_score": round(power, 1),
                "offense_score": round(off, 1),
                "pitching_score": round(pit, 1),
                "defense_score": round(defense, 1),
                "bullpen_score": round(bp, 1),
                "momentum_score": round(min(100, max(0, 50 + streak * 3 + (l10_w - 5) * 4)), 1),
                "wins": wins,
                "losses": losses,
                "win_pct": round(wins / max(wins + losses, 1), 3),
                "run_diff": rd,
                "last_10_record": l10_str or f"{l10_w}-{10 - l10_w}",
                "streak": streak_str,
                "rank_change": 0,
                "tier": tier,
            })

        # Sort by each category and cache (deep copy to avoid shared references)
        import copy
        for category, sort_key in [
            ("power", "power_score"),
            ("offense", "offense_score"),
            ("pitching", "pitching_score"),
            ("defense", "defense_score"),
            ("bullpen", "bullpen_score"),
            ("momentum", "momentum_score"),
        ]:
            sorted_teams = sorted(rankings_data, key=lambda t: t[sort_key], reverse=True)
            ranked = []
            for i, team in enumerate(sorted_teams):
                t = copy.copy(team)
                t["rank"] = i + 1
                ranked.append(t)

            cache_team_rankings(date_str, category, {
                "ranking_date": date_str,
                "category": category,
                "rankings": ranked,
            })

        logger.info("Generated rankings for %d teams", len(rankings_data))

    async def _generate_player_rankings(self, target_date: date):
        """Generate player rankings from season batting/pitching CSV data."""
        season = target_date.year
        date_str = target_date.isoformat()

        batting_path = self.data_dir / f"batting_{season}.csv"
        pitching_path = self.data_dir / f"pitching_{season}.csv"

        # Fetch real positions from MLB API rosters
        position_map: dict[int, str] = {}
        try:
            position_map = await self.mlb_client.get_all_player_positions()
            logger.info("Fetched positions for %d players from rosters", len(position_map))
        except Exception as e:
            logger.warning("Could not fetch player positions: %s", e)

        # ── Batting rankings (position players) ──
        if batting_path.exists():
            bat_df = pd.read_csv(batting_path)
            bat_df["game_date"] = pd.to_datetime(bat_df["game_date"])
            bat_df = bat_df[bat_df["game_date"] <= str(target_date)]

            # Aggregate season stats per player
            player_bat = bat_df.groupby(["player_id", "player_name", "team_id"]).agg(
                games=("game_id", "nunique"),
                ab=("at_bats", "sum"),
                hits=("hits", "sum"),
                hr=("home_runs", "sum"),
                rbi=("rbi", "sum"),
                runs=("runs", "sum"),
                walks=("walks", "sum"),
                so=("strikeouts", "sum"),
                sb=("stolen_bases", "sum"),
                doubles=("doubles", "sum"),
                triples=("triples", "sum"),
            ).reset_index()

            # Filter to players with meaningful playing time
            player_bat = player_bat[player_bat["ab"] >= 30].copy()

            if len(player_bat) > 0:
                # Compute derived stats
                player_bat["avg"] = player_bat["hits"] / player_bat["ab"]
                player_bat["obp"] = (player_bat["hits"] + player_bat["walks"]) / (player_bat["ab"] + player_bat["walks"])
                player_bat["slg"] = (
                    (player_bat["hits"] - player_bat["doubles"] - player_bat["triples"] - player_bat["hr"])
                    + player_bat["doubles"] * 2
                    + player_bat["triples"] * 3
                    + player_bat["hr"] * 4
                ) / player_bat["ab"]
                player_bat["ops"] = player_bat["obp"] + player_bat["slg"]

                # Composite score: weighted OPS + power + speed
                player_bat["score"] = (
                    player_bat["ops"] * 40
                    + (player_bat["hr"] / player_bat["games"]) * 15
                    + (player_bat["rbi"] / player_bat["games"]) * 5
                    + (player_bat["sb"] / player_bat["games"]) * 3
                )

                # Tier assignment
                def _bat_tier(score):
                    if score >= 50: return "elite"
                    if score >= 40: return "contender"
                    if score >= 32: return "average"
                    if score >= 25: return "below_average"
                    return "rebuilding"

                # Assign real positions from API roster data
                player_bat["position"] = player_bat["player_id"].map(
                    lambda pid: position_map.get(pid, "DH")
                )
                # Exclude pitchers from batting rankings
                player_bat = player_bat[~player_bat["position"].isin(["P", "SP", "RP"])].copy()

                # Group by position and rank within each
                positions = ["C", "1B", "2B", "3B", "SS", "OF", "DH"]
                for pos in positions:
                    pos_players = player_bat[player_bat["position"] == pos].copy()
                    pos_players = pos_players.sort_values("score", ascending=False)

                    ranked = []
                    for i, (_, row) in enumerate(pos_players.iterrows()):
                        ranked.append({
                            "rank": i + 1,
                            "player_id": int(row["player_id"]),
                            "player_name": row["player_name"],
                            "team_id": row["team_id"],
                            "position": pos,
                            "score": round(float(row["score"]), 1),
                            "key_stats": {
                                "avg": round(float(row["avg"]), 3),
                                "obp": round(float(row["obp"]), 3),
                                "slg": round(float(row["slg"]), 3),
                                "ops": round(float(row["ops"]), 3),
                                "hr": int(row["hr"]),
                                "rbi": int(row["rbi"]),
                                "r": int(row["runs"]),
                                "h": int(row["hits"]),
                                "sb": int(row["sb"]),
                                "so": int(row["so"]),
                            },
                            "games_played": int(row["games"]),
                            "rank_change": 0,
                            "tier": _bat_tier(row["score"]),
                        })
                        if i >= 149:
                            break

                    cache_player_rankings(date_str, pos, {
                        "position": pos,
                        "ranking_date": date_str,
                        "rankings": ranked,
                    })

                logger.info("Generated batting player rankings: %d players", len(sorted_bat))

        # ── Pitching rankings (SP, RP) ──
        if pitching_path.exists():
            pit_df = pd.read_csv(pitching_path)
            pit_df["game_date"] = pd.to_datetime(pit_df["game_date"])
            pit_df = pit_df[pit_df["game_date"] <= str(target_date)]

            # Fetch official W-L records from MLB Stats API
            pitcher_records: dict[int, dict] = {}
            try:
                pitcher_records = await self.mlb_client.get_all_pitcher_records(season)
                logger.info("Fetched official pitcher records for %d pitchers", len(pitcher_records))
            except Exception as e:
                logger.warning("Could not fetch pitcher records from API: %s", e)

            player_pit = pit_df.groupby(["player_id", "player_name", "team_id"]).agg(
                games=("game_id", "nunique"),
                ip=("innings_pitched", "sum"),
                er=("earned_runs", "sum"),
                hits_a=("hits_allowed", "sum"),
                bb=("walks_allowed", "sum"),
                k=("strikeouts_recorded", "sum"),
                pitches=("pitches_thrown", "sum"),
                runs_a=("runs_allowed", "sum"),
            ).reset_index()

            if len(player_pit) > 0:
                # Compute derived stats
                player_pit["era"] = (player_pit["er"] / player_pit["ip"].clip(lower=0.1)) * 9
                player_pit["whip"] = (player_pit["hits_a"] + player_pit["bb"]) / player_pit["ip"].clip(lower=0.1)
                player_pit["k9"] = (player_pit["k"] / player_pit["ip"].clip(lower=0.1)) * 9
                player_pit["ip_per_game"] = player_pit["ip"] / player_pit["games"]
                player_pit["h9"] = (player_pit["hits_a"] / player_pit["ip"].clip(lower=0.1)) * 9
                player_pit["bb9"] = (player_pit["bb"] / player_pit["ip"].clip(lower=0.1)) * 9

                # Add official W/L from MLB Stats API
                player_pit["wins"] = player_pit["player_id"].map(
                    lambda pid: pitcher_records.get(pid, {}).get("wins", 0)
                )
                player_pit["losses"] = player_pit["player_id"].map(
                    lambda pid: pitcher_records.get(pid, {}).get("losses", 0)
                )
                player_pit["saves"] = player_pit["player_id"].map(
                    lambda pid: pitcher_records.get(pid, {}).get("saves", 0)
                )

                # Classify SP vs RP based on avg IP per game
                player_pit["is_sp"] = player_pit["ip_per_game"] >= 4.0

                def _pit_score(row):
                    era_score = max(0, (5.0 - row["era"]) * 12)
                    whip_score = max(0, (1.6 - row["whip"]) * 20)
                    k_score = row["k9"] * 3
                    ip_score = row["ip"] * 0.3
                    return era_score + whip_score + k_score + ip_score

                player_pit["score"] = player_pit.apply(_pit_score, axis=1)

                def _pit_tier(score):
                    if score >= 70: return "elite"
                    if score >= 50: return "contender"
                    if score >= 35: return "average"
                    if score >= 20: return "below_average"
                    return "rebuilding"

                for pos, is_sp in [("SP", True), ("RP", False)]:
                    subset = player_pit[player_pit["is_sp"] == is_sp].copy()
                    # Minimum thresholds
                    if is_sp:
                        subset = subset[subset["ip"] >= 15]
                    else:
                        subset = subset[subset["games"] >= 5]

                    sorted_pit = subset.sort_values("score", ascending=False)

                    ranked = []
                    for i, (_, row) in enumerate(sorted_pit.iterrows()):
                        ranked.append({
                            "rank": i + 1,
                            "player_id": int(row["player_id"]),
                            "player_name": row["player_name"],
                            "team_id": row["team_id"],
                            "position": pos,
                            "score": round(float(row["score"]), 1),
                            "key_stats": {
                                "w": int(row["wins"]),
                                "l": int(row["losses"]),
                                "sv": int(row["saves"]),
                                "era": round(float(row["era"]), 2),
                                "ip": round(float(row["ip"]), 1),
                                "k9": round(float(row["k9"]), 1),
                                "whip": round(float(row["whip"]), 2),
                                "h": int(row["hits_a"]),
                                "bb": int(row["bb"]),
                                "k": int(row["k"]),
                            },
                            "games_played": int(row["games"]),
                            "rank_change": 0,
                            "tier": _pit_tier(row["score"]),
                        })
                        if i >= 149:
                            break

                    cache_player_rankings(date_str, pos, {
                        "position": pos,
                        "ranking_date": date_str,
                        "rankings": ranked,
                    })

                logger.info("Generated pitching player rankings")

    def _compute_season_records(self, season: int) -> dict[str, dict]:
        """Compute W-L records from current season games CSV only."""
        import pandas as pd

        path = self.data_dir / f"games_{season}.csv"
        if not path.exists():
            return {}

        df = pd.read_csv(path)
        records: dict[str, dict] = {}

        for team_id in set(df["home_team_id"]) | set(df["away_team_id"]):
            home = df[df["home_team_id"] == team_id]
            away = df[df["away_team_id"] == team_id]
            wins = len(home[home["home_score"] > home["away_score"]]) + \
                   len(away[away["away_score"] > away["home_score"]])
            losses = len(home[home["home_score"] < home["away_score"]]) + \
                     len(away[away["away_score"] < away["home_score"]])
            rs = home["home_score"].sum() + away["away_score"].sum()
            ra = home["away_score"].sum() + away["home_score"].sum()
            records[team_id] = {
                "wins": int(wins),
                "losses": int(losses),
                "run_diff": int(rs - ra),
            }

        return records

    def _save_odds_history(self, target_date: date, odds_data: list[dict]):
        """Persist fetched odds to CSV for future model retraining."""
        import csv

        odds_file = self.data_dir / "odds_history.csv"
        file_exists = odds_file.exists()

        rows = []
        for odds in odds_data:
            home_ml = odds.get("home_moneyline")
            away_ml = odds.get("away_moneyline")
            if home_ml is None or away_ml is None:
                continue
            # Compute implied probabilities (removing vig with power method)
            from mlb.betting.engine import american_to_decimal
            h_dec = american_to_decimal(home_ml)
            a_dec = american_to_decimal(away_ml)
            h_imp = 1.0 / h_dec
            a_imp = 1.0 / a_dec
            total_imp = h_imp + a_imp
            h_prob = h_imp / total_imp  # Vig-removed
            rows.append({
                "game_date": target_date.isoformat(),
                "home_team": odds["home_team"],
                "away_team": odds["away_team"],
                "home_moneyline": home_ml,
                "away_moneyline": away_ml,
                "total_line": odds.get("total_line"),
                "market_home_prob": round(h_prob, 4),
            })

        if rows:
            with open(odds_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerows(rows)
            logger.info("Saved %d odds records to history", len(rows))

    def _cache_results(self, target_date: date, result: dict):
        """Push results into the API cache."""
        try:
            from mlb.api.routes.predictions import cache_predictions
            from mlb.api.routes.betting import cache_betting_slip

            date_str = target_date.isoformat()
            cache_predictions(date_str, result["predictions"])

            if result["betting_slip"]:
                cache_betting_slip(date_str, result["betting_slip"])
        except ImportError:
            pass  # API not installed / not running

    @staticmethod
    def _prediction_to_dict(
        pred: GamePrediction,
        game: dict = None,
        sp_stats: dict[int, dict] | None = None,
        live_standings: dict[str, dict] | None = None,
        weather_data: dict | None = None,
        game_odds: dict | None = None,
    ) -> dict:
        game = game or {}
        home_sp = game.get("home_probable_pitcher") or {}
        away_sp = game.get("away_probable_pitcher") or {}
        sp_stats = sp_stats or {}
        game_odds = game_odds or {}
        live_standings = live_standings or {}

        # Live team records and streaks from MLB API standings
        h_std = live_standings.get(pred.home_team_id, {})
        a_std = live_standings.get(pred.away_team_id, {})

        # SP stats
        h_sp_data = sp_stats.get(home_sp.get("player_id")) if home_sp else None
        a_sp_data = sp_stats.get(away_sp.get("player_id")) if away_sp else None

        result = {
            "game_id": pred.game_id,
            "game_date": pred.game_date,
            "home_team": pred.home_team_id,
            "away_team": pred.away_team_id,
            "home_win_prob": pred.home_win_prob,
            "away_win_prob": pred.away_win_prob,
            "predicted_home_runs": pred.predicted_home_runs,
            "predicted_away_runs": pred.predicted_away_runs,
            "predicted_total": pred.predicted_total,
            "confidence": pred.confidence,
            "model_agreement": pred.model_agreement,
            "predicted_winner": pred.predicted_winner,
            "model_predictions": pred.model_predictions,
            "top_factors": pred.top_factors,
            "home_power_score": pred.home_power_score,
            "away_power_score": pred.away_power_score,
            "home_sp_name": home_sp.get("name", "TBD"),
            "away_sp_name": away_sp.get("name", "TBD"),
            "home_wins": h_std.get("wins", 0),
            "home_losses": h_std.get("losses", 0),
            "away_wins": a_std.get("wins", 0),
            "away_losses": a_std.get("losses", 0),
            "home_streak": h_std.get("streak", "-"),
            "away_streak": a_std.get("streak", "-"),
            "home_sp_era": h_sp_data.get("era") if h_sp_data else None,
            "away_sp_era": a_sp_data.get("era") if a_sp_data else None,
            "home_sp_wins": h_sp_data.get("wins") if h_sp_data else None,
            "home_sp_losses": h_sp_data.get("losses") if h_sp_data else None,
            "away_sp_wins": a_sp_data.get("wins") if a_sp_data else None,
            "away_sp_losses": a_sp_data.get("losses") if a_sp_data else None,
            "game_time": game.get("game_time", ""),
            "home_moneyline": game_odds.get("home_moneyline"),
            "away_moneyline": game_odds.get("away_moneyline"),
            "total_line": game_odds.get("total_line"),
            "over_odds": game_odds.get("over_odds"),
            "under_odds": game_odds.get("under_odds"),
            "weather": None,
        }
        # Add weather if available
        weather_data = weather_data or {}
        wx = weather_data.get(pred.home_team_id)
        if wx:
            result["weather"] = {
                "temp": wx.temperature_f,
                "wind": wx.wind_speed_mph,
                "wind_dir": wx.wind_direction,
                "humidity": wx.humidity_pct,
                "is_dome": wx.is_dome,
            }
        return result

    @staticmethod
    def _slip_to_dict(slip: BettingSlip) -> dict:
        return {
            "slip_date": slip.slip_date,
            "bankroll": slip.bankroll,
            "total_stake": slip.total_stake,
            "num_bets": slip.num_bets,
            "bets": [
                {
                    "game_id": b.game_id,
                    "game_date": b.game_date,
                    "home_team": b.home_team,
                    "away_team": b.away_team,
                    "bet_type": b.bet_type,
                    "selection": b.selection,
                    "odds": b.odds,
                    "model_prob": b.model_prob,
                    "implied_prob": b.implied_prob,
                    "edge": b.edge,
                    "edge_pct": b.edge_pct,
                    "kelly_fraction": b.kelly_fraction,
                    "recommended_stake": b.recommended_stake,
                    "confidence": b.confidence,
                    "ev_per_dollar": b.ev_per_dollar,
                    "decimal_odds": b.decimal_odds,
                    "total_line": b.total_line,
                }
                for b in slip.bets
            ],
            "total_ev": slip.total_ev,
            "max_exposure": slip.max_exposure,
            "risk_level": slip.risk_level,
        }


# ── Team Reference Data ───────────────────────────────────────

_TEAM_NAMES: dict[str, str] = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
    "DET": "Detroit Tigers", "HOU": "Houston Astros", "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Oakland Athletics", "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates", "SD": "San Diego Padres", "SF": "San Francisco Giants",
    "SEA": "Seattle Mariners", "STL": "St. Louis Cardinals", "TB": "Tampa Bay Rays",
    "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays", "WSH": "Washington Nationals",
}

_TEAM_DIVISIONS: dict[str, str] = {
    "BAL": "AL East", "BOS": "AL East", "NYY": "AL East", "TB": "AL East", "TOR": "AL East",
    "CLE": "AL Central", "CWS": "AL Central", "DET": "AL Central", "KC": "AL Central", "MIN": "AL Central",
    "HOU": "AL West", "LAA": "AL West", "OAK": "AL West", "SEA": "AL West", "TEX": "AL West",
    "ATL": "NL East", "MIA": "NL East", "NYM": "NL East", "PHI": "NL East", "WSH": "NL East",
    "CHC": "NL Central", "CIN": "NL Central", "MIL": "NL Central", "PIT": "NL Central", "STL": "NL Central",
    "ARI": "NL West", "COL": "NL West", "LAD": "NL West", "SD": "NL West", "SF": "NL West",
}

_TEAM_LEAGUES: dict[str, str] = {t: d[:2] for t, d in _TEAM_DIVISIONS.items()}


# ── SP Feature Helpers ─────────────────────────────────────────


def _add_sp_features(features: dict, sp: dict | None, prefix: str):
    """Add starting pitcher features to the feature dict."""
    if sp:
        features[f"{prefix}_era"] = sp.get("era", 4.50)
        features[f"{prefix}_whip"] = sp.get("whip", 1.30)
        features[f"{prefix}_k_per_9"] = sp.get("k_per_9", 8.0)
        features[f"{prefix}_bb_per_9"] = sp.get("bb_per_9", 3.0)
        features[f"{prefix}_h_per_9"] = sp.get("h_per_9", 8.5)
        features[f"{prefix}_hr_per_9"] = sp.get("hr_per_9", 1.2)
        features[f"{prefix}_avg_against"] = sp.get("avg_against", 0.250)
        features[f"{prefix}_ip"] = sp.get("innings_pitched", 0)
        features[f"{prefix}_wins"] = sp.get("wins", 0)
        features[f"{prefix}_losses"] = sp.get("losses", 0)
        # Derived
        ip = sp.get("innings_pitched", 1) or 1
        features[f"{prefix}_k_bb_ratio"] = sp.get("k_per_9", 8) / max(sp.get("bb_per_9", 3), 0.1)
        features[f"{prefix}_fip_approx"] = (
            (13 * sp.get("hr_per_9", 1.2) + 3 * sp.get("bb_per_9", 3) - 2 * sp.get("k_per_9", 8)) / 3 + 3.2
        )
    else:
        # League-average defaults when no SP announced
        features[f"{prefix}_era"] = 4.50
        features[f"{prefix}_whip"] = 1.30
        features[f"{prefix}_k_per_9"] = 8.0
        features[f"{prefix}_bb_per_9"] = 3.0
        features[f"{prefix}_h_per_9"] = 8.5
        features[f"{prefix}_hr_per_9"] = 1.2
        features[f"{prefix}_avg_against"] = 0.250
        features[f"{prefix}_ip"] = 0.0
        features[f"{prefix}_wins"] = 0.0
        features[f"{prefix}_losses"] = 0.0
        features[f"{prefix}_k_bb_ratio"] = 2.67
        features[f"{prefix}_fip_approx"] = 4.20


def _adjust_pitching_with_sp(team_pit_score: float, sp: dict) -> float:
    """Blend team pitching score with individual SP quality (60/40 split)."""
    era = sp.get("era", 4.50)
    whip = sp.get("whip", 1.30)
    k9 = sp.get("k_per_9", 8.0)
    sp_score = _compute_sp_score(era, whip, k9)
    return round(team_pit_score * 0.4 + sp_score * 0.6, 1)


def _compute_sp_score(era: float, whip: float, k9: float) -> float:
    """Individual SP quality score (0-100, higher = better)."""
    era_score = min(100, max(0, (7.0 - era) / 7.0 * 100))
    whip_score = min(100, max(0, (2.0 - whip) / 2.0 * 100))
    k9_score = min(100, max(0, k9 / 14.0 * 100))
    return era_score * 0.4 + whip_score * 0.35 + k9_score * 0.25


# ── Helper score functions ────────────────────────────────────


def _compute_offense_score(feat: dict[str, float]) -> float:
    """Offense composite from season-long stats (0-100)."""
    ops = feat.get("ops_season", feat.get("ops_14", 0.700))
    rs = feat.get("rs_per_game", 4.5)
    hr = feat.get("hr_season", feat.get("hr_7", 1.0))
    # Normalize: OPS .700 = 50, RS 4.5 = 50, HR 1.0 = 50
    ops_score = min(100, max(0, (ops - 0.500) / 0.400 * 100))
    rs_score = min(100, max(0, rs / 9.0 * 100))
    hr_score = min(100, max(0, hr / 3.0 * 100))
    return round(ops_score * 0.5 + rs_score * 0.3 + hr_score * 0.2, 1)


def _compute_pitching_score(feat: dict[str, float]) -> float:
    """Pitching composite from season-long stats (0-100, higher = better)."""
    sp_era = feat.get("sp_era_season", feat.get("sp_era_14", 4.50))
    sp_whip = feat.get("sp_whip_season", feat.get("sp_whip_14", 1.30))
    bp_era = feat.get("bp_era_season", feat.get("bp_era_14", 4.50))
    bp_whip = feat.get("bp_whip_season", feat.get("bp_whip_14", 1.30))
    k9 = feat.get("sp_k9_season", feat.get("sp_k9_14", 8.0))
    # Weighted: SP 60%, bullpen 40%
    era = sp_era * 0.6 + bp_era * 0.4
    whip = sp_whip * 0.6 + bp_whip * 0.4
    # Lower ERA/WHIP = better, higher K9 = better
    era_score = min(100, max(0, (7.0 - era) / 7.0 * 100))
    whip_score = min(100, max(0, (2.0 - whip) / 2.0 * 100))
    k9_score = min(100, max(0, k9 / 14.0 * 100))
    return round(era_score * 0.4 + whip_score * 0.35 + k9_score * 0.25, 1)


def _compute_bullpen_score(feat: dict[str, float]) -> float:
    """Bullpen quality from season-long stats (0-100, higher = better)."""
    bp_era = feat.get("bp_era_season", feat.get("bp_era_14", 4.50))
    bp_whip = feat.get("bp_whip_season", feat.get("bp_whip_14", 1.30))
    bp_k9 = feat.get("bp_k9_season", feat.get("bp_k9_14", 8.0))
    era_score = min(100, max(0, (7.0 - bp_era) / 7.0 * 100))
    whip_score = min(100, max(0, (2.0 - bp_whip) / 2.0 * 100))
    k9_score = min(100, max(0, bp_k9 / 14.0 * 100))
    return round(era_score * 0.4 + whip_score * 0.35 + k9_score * 0.25, 1)


def _compute_defense_score(feat: dict[str, float]) -> float:
    """Defense proxy from season runs allowed vs pitching quality (0-100).

    Teams that allow fewer runs than their pitching stats suggest have good defense.
    """
    ra = feat.get("ra_per_game", 4.5)
    sp_era = feat.get("sp_era_season", feat.get("sp_era_14", 4.50))
    bp_era = feat.get("bp_era_season", feat.get("bp_era_14", 4.50))
    pit_era = sp_era * 0.6 + bp_era * 0.4
    # Runs allowed score (lower = better)
    ra_score = min(100, max(0, (7.0 - ra) / 7.0 * 100))
    # Defensive efficiency: if RA < expected from ERA, defense is helping
    era_ra_diff = pit_era - ra  # positive = defense saving runs
    def_bonus = max(-15, min(15, era_ra_diff * 10))
    return round(min(100, max(0, ra_score + def_bonus)), 1)


def _power_score(feat: dict[str, float]) -> float:
    """Overall team power score (0-100)."""
    off = _compute_offense_score(feat)
    pit = _compute_pitching_score(feat)
    wpct = feat.get("win_pct", 0.500)
    wpct_score = wpct * 100
    return round(off * 0.35 + pit * 0.35 + wpct_score * 0.30, 1)


# ── CLI ───────────────────────────────────────────────────────


def main():
    import argparse
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run daily MLB predictions")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD), default today")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-dir", type=str, default="models")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()

    runner = DailyRunner(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
    )

    result = asyncio.run(runner.run(target))

    # Print summary
    print(f"\n{'='*60}")
    print(f"  DAILY PREDICTIONS — {result['date']}")
    print(f"{'='*60}")

    preds = result["predictions"]
    if not preds:
        print("  No predictions generated.")
    else:
        for p in preds:
            winner = p["predicted_winner"]
            prob = max(p["home_win_prob"], p["away_win_prob"])
            h_sp = p.get("home_sp_name", "TBD")
            a_sp = p.get("away_sp_name", "TBD")
            print(
                f"  {p['away_team']:>3} @ {p['home_team']:<3}  |  "
                f"{winner} {prob:.1%}  |  "
                f"Total: {p['predicted_total']:.1f}  |  "
                f"Conf: {p['confidence']:.0f}"
            )
            print(f"    SP: {a_sp} vs {h_sp}")

    slip = result.get("betting_slip")
    if slip and slip["num_bets"] > 0:
        print(f"\n  BETTING SLIP ({slip['num_bets']} bets)")
        print(f"  Total Stake: ${slip['total_stake']:.2f}  |  EV: ${slip['total_ev']:.2f}")
        for b in slip["bets"]:
            print(
                f"    {b['bet_type'].upper():>10} {b['selection']:>5}  "
                f"{b['home_team']}v{b['away_team']}  "
                f"Edge: {b['edge_pct']:.1f}%  "
                f"Stake: ${b['recommended_stake']:.2f}"
            )

    if result["weather"]:
        print(f"\n  WEATHER")
        for team, w in result["weather"].items():
            print(f"    {team}: {w['temp']:.0f}F, wind {w['wind']:.0f}mph {w['wind_dir']}")

    if result["errors"]:
        print(f"\n  ERRORS: {len(result['errors'])}")
        for e in result["errors"]:
            print(f"    - {e}")

    print()


if __name__ == "__main__":
    main()
