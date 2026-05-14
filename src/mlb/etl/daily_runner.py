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
    TrainingDataBuilder, _update_elo,
)
from mlb.features.stadium import compute_stadium_features, calculate_stadium_factor
from mlb.models.predict import GamePrediction, PredictionService
from mlb.features.assembler import GameFeatureVector
from mlb.betting.engine import BettingEngine, BettingSlip
from mlb.alerts import AlertService
from mlb.api.routes.rankings import cache_team_rankings

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
            # 2. Fetch today's schedule
            games = await self.mlb_client.get_schedule(target_date)
            scheduled = [
                g for g in games
                if g["status"] in ("Scheduled", "Pre-Game", "Warmup", "In Progress")
                or g["status"] == "Preview"
            ]
            logger.info("Found %d scheduled games for %s", len(scheduled), target_date)

            if not scheduled:
                logger.info("No games scheduled for %s", target_date)
                return result

            # 3. Build rolling team stats from CSV data
            team_features = self._build_team_features(target_date)

            # 4. Fetch starting pitcher stats
            sp_stats = await self._fetch_sp_stats(scheduled, target_date.year)

            # 4b. Fetch lineups and compute lineup/platoon features
            lineup_features = await self._fetch_lineup_features(
                scheduled, target_date
            )

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

            # 6. Generate predictions for each game
            predictions: list[GamePrediction] = []
            for game in scheduled:
                try:
                    pred = self._predict_game(
                        game, team_features, weather_data, target_date,
                        sp_stats=sp_stats,
                        lineup_features=lineup_features,
                    )
                    if pred:
                        predictions.append(pred)
                except Exception as e:
                    logger.warning("Prediction failed for %s: %s", game["game_id"], e)
                    result["errors"].append(f"Game {game['game_id']}: {e}")

            logger.info("Generated %d predictions", len(predictions))
            game_lookup = {g["game_id"]: g for g in scheduled}
            season_records = self._compute_season_records(target_date.year)
            result["predictions"] = [
                self._prediction_to_dict(
                    p, game_lookup.get(p.game_id, {}),
                    sp_stats=sp_stats,
                    team_features=team_features,
                    season_records=season_records,
                )
                for p in predictions
            ]

            # 7. Fetch live odds and generate betting slip
            try:
                odds_data = await self.odds_client.get_mlb_odds()
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
            self._generate_rankings(team_features, target_date)

            # 10. Send alerts (Slack/email) if configured
            try:
                await self.alert_service.send_betting_alert(result)
            except Exception as e:
                logger.debug("Alert send skipped: %s", e)

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

        # Compute Elo ratings and rest days from game history
        self._elo_ratings: dict[str, float] = {}
        self._last_game_date: dict[str, date] = {}
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

            # Differentials
            for k in ("lineup_ops", "lineup_obp", "lineup_slg", "platoon_adv"):
                game_feats[f"diff_{k}"] = game_feats[f"h_{k}"] - game_feats[f"a_{k}"]

            result[gid] = game_feats

        logger.info("Computed lineup features for %d games", len(result))
        return result

    def _predict_game(
        self,
        game: dict,
        team_features: dict[str, dict[str, float]],
        weather_data: dict,
        target_date: date,
        sp_stats: dict[int, dict] | None = None,
        lineup_features: dict[str, dict[str, float]] | None = None,
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

        # Lineup and platoon features
        lf = (lineup_features or {}).get(game["game_id"], {})
        for k in ("lineup_ops", "lineup_obp", "lineup_slg", "platoon_adv"):
            features[f"h_{k}"] = lf.get(f"h_{k}", 0.720 if "ops" in k else 0.320 if "obp" in k else 0.400 if "slg" in k else 0.5)
            features[f"a_{k}"] = lf.get(f"a_{k}", 0.720 if "ops" in k else 0.320 if "obp" in k else 0.400 if "slg" in k else 0.5)
            features[f"diff_{k}"] = features[f"h_{k}"] - features[f"a_{k}"]

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
    ):
        """Generate team power rankings from rolling stats."""
        date_str = target_date.isoformat()

        # Load current season games only for W-L records
        season_records = self._compute_season_records(target_date.year)

        rankings_data = []
        for team_id, feat in team_features.items():
            off = _compute_offense_score(feat)
            pit = _compute_pitching_score(feat)
            bp = _compute_bullpen_score(feat)
            defense = _compute_defense_score(feat)
            power = _power_score(feat)
            rec = season_records.get(team_id, {})
            wins = rec.get("wins", 0)
            losses = rec.get("losses", 0)
            l10_w = int(feat.get("l10_wpct", 0.5) * 10)
            streak = int(feat.get("streak", 0))
            rd = rec.get("run_diff", 0)

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
                "last_10_record": f"{l10_w}-{10 - l10_w}",
                "streak": f"W{streak}" if streak > 0 else f"L{abs(streak)}" if streak < 0 else "-",
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
        team_features: dict[str, dict] | None = None,
        season_records: dict[str, dict] | None = None,
    ) -> dict:
        game = game or {}
        home_sp = game.get("home_probable_pitcher") or {}
        away_sp = game.get("away_probable_pitcher") or {}
        sp_stats = sp_stats or {}
        season_records = season_records or {}
        team_features = team_features or {}

        # Team records
        h_rec = season_records.get(pred.home_team_id, {})
        a_rec = season_records.get(pred.away_team_id, {})

        # Team streaks from rolling features
        h_feat = team_features.get(pred.home_team_id, {})
        a_feat = team_features.get(pred.away_team_id, {})
        h_streak_val = int(h_feat.get("streak", 0))
        a_streak_val = int(a_feat.get("streak", 0))
        h_streak = f"W{h_streak_val}" if h_streak_val > 0 else f"L{abs(h_streak_val)}" if h_streak_val < 0 else "-"
        a_streak = f"W{a_streak_val}" if a_streak_val > 0 else f"L{abs(a_streak_val)}" if a_streak_val < 0 else "-"

        # SP stats
        h_sp_data = sp_stats.get(home_sp.get("player_id")) if home_sp else None
        a_sp_data = sp_stats.get(away_sp.get("player_id")) if away_sp else None

        return {
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
            "home_wins": h_rec.get("wins", 0),
            "home_losses": h_rec.get("losses", 0),
            "away_wins": a_rec.get("wins", 0),
            "away_losses": a_rec.get("losses", 0),
            "home_streak": h_streak,
            "away_streak": a_streak,
            "home_sp_era": h_sp_data.get("era") if h_sp_data else None,
            "away_sp_era": a_sp_data.get("era") if a_sp_data else None,
            "home_sp_wins": h_sp_data.get("wins") if h_sp_data else None,
            "home_sp_losses": h_sp_data.get("losses") if h_sp_data else None,
            "away_sp_wins": a_sp_data.get("wins") if a_sp_data else None,
            "away_sp_losses": a_sp_data.get("losses") if a_sp_data else None,
        }

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
    """Quick offense composite from rolling stats (0-100)."""
    ops = feat.get("ops_7", 0.700)
    rs = feat.get("rs_per_game", 4.5)
    hr = feat.get("hr_7", 1.0)
    # Normalize: OPS .700 = 50, RS 4.5 = 50, HR 1.0 = 50
    ops_score = min(100, max(0, (ops - 0.500) / 0.400 * 100))
    rs_score = min(100, max(0, rs / 9.0 * 100))
    hr_score = min(100, max(0, hr / 3.0 * 100))
    return round(ops_score * 0.5 + rs_score * 0.3 + hr_score * 0.2, 1)


def _compute_pitching_score(feat: dict[str, float]) -> float:
    """Quick pitching composite from SP + bullpen rolling stats (0-100, higher = better)."""
    sp_era = feat.get("sp_era_7", 4.50)
    sp_whip = feat.get("sp_whip_7", 1.30)
    bp_era = feat.get("bp_era_7", 4.50)
    bp_whip = feat.get("bp_whip_7", 1.30)
    k9 = feat.get("sp_k9_7", 8.0)
    # Weighted: SP 60%, bullpen 40%
    era = sp_era * 0.6 + bp_era * 0.4
    whip = sp_whip * 0.6 + bp_whip * 0.4
    # Lower ERA/WHIP = better, higher K9 = better
    era_score = min(100, max(0, (7.0 - era) / 7.0 * 100))
    whip_score = min(100, max(0, (2.0 - whip) / 2.0 * 100))
    k9_score = min(100, max(0, k9 / 14.0 * 100))
    return round(era_score * 0.4 + whip_score * 0.35 + k9_score * 0.25, 1)


def _compute_bullpen_score(feat: dict[str, float]) -> float:
    """Bullpen quality from rolling stats (0-100, higher = better)."""
    bp_era = feat.get("bp_era_7", 4.50)
    bp_whip = feat.get("bp_whip_7", 1.30)
    bp_k9 = feat.get("bp_k9_7", 8.0)
    fatigue = feat.get("bp_ip_3d", 6.0)
    era_score = min(100, max(0, (7.0 - bp_era) / 7.0 * 100))
    whip_score = min(100, max(0, (2.0 - bp_whip) / 2.0 * 100))
    k9_score = min(100, max(0, bp_k9 / 14.0 * 100))
    # Penalize heavy recent usage (>10 IP in 3 days = overworked)
    fatigue_penalty = max(0, (fatigue - 10) * 3)
    return round(max(0, era_score * 0.4 + whip_score * 0.35 + k9_score * 0.25 - fatigue_penalty), 1)


def _compute_defense_score(feat: dict[str, float]) -> float:
    """Defense proxy from runs allowed vs pitching quality (0-100).

    Teams that allow fewer runs than their pitching stats suggest have good defense.
    """
    ra = feat.get("ra_per_game", 4.5)
    pit_era = feat.get("sp_era_7", 4.50) * 0.6 + feat.get("bp_era_7", 4.50) * 0.4
    # Runs allowed score (lower = better)
    ra_score = min(100, max(0, (7.0 - ra) / 7.0 * 100))
    # Defensive efficiency: if RA < expected from ERA, defense is helping
    era_ra_diff = (pit_era / 9 * 9) - ra  # positive = defense saving runs
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
