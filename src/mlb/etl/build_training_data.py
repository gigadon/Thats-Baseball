"""Build training dataset from backfilled CSV files.

Reads games, batting, and pitching CSVs and computes rolling team-level
features for each game to produce a model-ready training dataset.

Usage:
    python -m mlb.etl.build_training_data --data-dir data --seasons 2023 2024 2025
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Park factors: runs factor relative to league average (1.00).
# Source: multi-year averages. >1.0 = hitter-friendly, <1.0 = pitcher-friendly.
PARK_FACTORS: dict[str, dict[str, float]] = {
    "COL": {"runs": 1.38, "hr": 1.29},  # Coors Field
    "CIN": {"runs": 1.12, "hr": 1.18},  # Great American
    "TEX": {"runs": 1.10, "hr": 1.11},  # Globe Life
    "BOS": {"runs": 1.08, "hr": 1.04},  # Fenway Park
    "CHC": {"runs": 1.06, "hr": 1.10},  # Wrigley Field
    "ARI": {"runs": 1.05, "hr": 1.04},  # Chase Field
    "PHI": {"runs": 1.04, "hr": 1.09},  # Citizens Bank
    "ATL": {"runs": 1.03, "hr": 1.10},  # Truist Park
    "MIN": {"runs": 1.03, "hr": 1.06},  # Target Field
    "TOR": {"runs": 1.02, "hr": 1.07},  # Rogers Centre
    "LAA": {"runs": 1.01, "hr": 0.99},  # Angel Stadium
    "BAL": {"runs": 1.01, "hr": 1.10},  # Camden Yards
    "DET": {"runs": 1.00, "hr": 0.97},  # Comerica Park
    "CLE": {"runs": 1.00, "hr": 1.00},  # Progressive Field
    "WSH": {"runs": 1.00, "hr": 1.02},  # Nationals Park
    "CWS": {"runs": 0.99, "hr": 1.05},  # Guaranteed Rate
    "NYY": {"runs": 0.99, "hr": 1.09},  # Yankee Stadium
    "KC":  {"runs": 0.98, "hr": 0.87},  # Kauffman Stadium
    "HOU": {"runs": 0.98, "hr": 1.00},  # Minute Maid
    "STL": {"runs": 0.97, "hr": 0.95},  # Busch Stadium
    "LAD": {"runs": 0.97, "hr": 0.96},  # Dodger Stadium
    "PIT": {"runs": 0.96, "hr": 0.91},  # PNC Park
    "MIL": {"runs": 0.96, "hr": 1.02},  # American Family
    "SEA": {"runs": 0.95, "hr": 0.93},  # T-Mobile Park
    "SD":  {"runs": 0.94, "hr": 0.87},  # Petco Park
    "TB":  {"runs": 0.93, "hr": 0.88},  # Tropicana Field
    "NYM": {"runs": 0.93, "hr": 0.91},  # Citi Field
    "SF":  {"runs": 0.92, "hr": 0.85},  # Oracle Park
    "OAK": {"runs": 0.92, "hr": 0.88},  # Oakland Coliseum
    "MIA": {"runs": 0.90, "hr": 0.86},  # loanDepot park
}
PARK_DEFAULT = {"runs": 1.00, "hr": 1.00}

# Dome/retractable stadiums — weather neutralized
DOME_TEAMS = {"TB"}
RETRACTABLE_TEAMS = {"ARI", "HOU", "MIA", "MIL", "SEA", "TEX", "TOR"}

# Team timezone UTC offsets (standard time, no DST adjustment needed for relative diffs)
TEAM_TIMEZONES: dict[str, int] = {
    # Eastern (-5)
    "NYY": -5, "NYM": -5, "BOS": -5, "BAL": -5, "TB": -5,
    "PHI": -5, "PIT": -5, "MIA": -5, "WSH": -5, "ATL": -5,
    "CIN": -5, "CLE": -5, "DET": -5, "TOR": -5,
    # Central (-6)
    "CHC": -6, "CWS": -6, "MIL": -6, "STL": -6,
    "MIN": -6, "KC": -6, "HOU": -6, "TEX": -6,
    # Mountain (-7)
    "ARI": -7, "COL": -7,
    # Pacific (-8)
    "LAD": -8, "LAA": -8, "SF": -8, "SD": -8, "SEA": -8, "OAK": -8,
}


def _compute_travel_fatigue(prev_venue: str, current_venue: str) -> float:
    """Compute travel fatigue score based on timezone change.

    Eastward travel (positive tz_change) is penalized more heavily.
    Returns 0 if same timezone or unknown teams.
    """
    prev_tz = TEAM_TIMEZONES.get(prev_venue)
    cur_tz = TEAM_TIMEZONES.get(current_venue)
    if prev_tz is None or cur_tz is None:
        return 0.0
    tz_change = cur_tz - prev_tz
    if tz_change == 0:
        return 0.0
    return max(0, tz_change) * 1.5 + abs(tz_change) * 0.5


def _classify_day_game(game_time_str: str, home_team: str) -> int:
    """Classify a game as day (1) or night (0) from ISO timestamp.

    Day game = first pitch before 5pm local time (based on home team timezone).
    """
    if not game_time_str or len(game_time_str) < 16:
        return 0  # default to night
    try:
        from datetime import datetime
        # Parse ISO: "2026-05-11T18:05:00Z"
        dt = datetime.fromisoformat(game_time_str.replace("Z", "+00:00"))
        utc_hour = dt.hour
        tz_offset = TEAM_TIMEZONES.get(home_team, -5)
        local_hour = (utc_hour + tz_offset) % 24
        return 1 if local_hour < 17 else 0
    except (ValueError, AttributeError):
        return 0


# Elo constants
ELO_K = 6  # Update speed — moderate for baseball (many games)
ELO_HOME_ADVANTAGE = 24  # ~54% expected win rate for home team


def _update_elo(
    ratings: dict[str, float], home: str, away: str, home_won: bool
) -> None:
    """Update Elo ratings in-place after a game."""
    h_elo = ratings.get(home, 1500.0)
    a_elo = ratings.get(away, 1500.0)
    # Expected score (home gets advantage)
    exp_h = 1.0 / (1.0 + 10 ** ((a_elo - h_elo - ELO_HOME_ADVANTAGE) / 400))
    result = 1.0 if home_won else 0.0
    delta = ELO_K * (result - exp_h)
    ratings[home] = h_elo + delta
    ratings[away] = a_elo - delta


class TrainingDataBuilder:
    """Builds feature vectors from historical CSV files."""

    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = data_dir

    def build(self, seasons: list[int], output: str = "training_data.parquet") -> pd.DataFrame:
        """Build training data for the given seasons.

        Returns a DataFrame where each row is one game with:
          - Feature columns for home and away teams
          - Target column: home_win (1/0)
        """
        # Load raw data
        games_df = self._load_games(seasons)
        batting_df = self._load_batting(seasons)
        pitching_df = self._load_pitching(seasons)

        logger.info(
            "Loaded %d games, %d batting rows, %d pitching rows",
            len(games_df), len(batting_df), len(pitching_df),
        )

        # Build team rolling stats
        team_stats = self._compute_team_rolling_stats(games_df, batting_df, pitching_df)

        # Build Elo ratings and rest days chronologically
        elo_ratings, last_game_date = self._compute_elo_and_rest(games_df)

        # Build SP season stats lookup
        sp_season = self._build_sp_season_stats(pitching_df)

        # Build SP lookup for rest days tracking (pitcher_id per game/team)
        sp_lookup = self._build_sp_lookup(pitching_df)

        # Track each pitcher's last start date for rest days calculation
        last_sp_start: dict[int, date] = {}  # pitcher_id -> last game_date they started

        # Build lineup aggregate season stats
        lineup_season = self._build_lineup_season_stats(batting_df)

        # Build platoon features (requires handedness cache)
        platoon = self._build_platoon_features(batting_df, pitching_df)

        # Build lineup recent form (7-day rolling OPS)
        lineup_recent = self._build_lineup_recent_form(batting_df)

        # Build SP recent form (rolling 3-start ERA/WHIP/K9)
        sp_recent = self._build_sp_recent_form(pitching_df)

        # Build bullpen availability (per-pitcher usage tracking)
        bp_avail = self._build_bullpen_availability(pitching_df)

        # Track last venue for travel fatigue
        last_game_venue: dict[str, str] = {}  # team_id -> venue (home_team_id) of last game

        # Load handedness cache for platoon splits
        hand_cache = self._load_handedness_cache()

        # Load real odds history if available
        odds_history = self._load_odds_history()

        # Check if game_time column exists for day/night classification
        has_game_time = "game_time" in games_df.columns

        # Generate feature rows for each game
        feature_rows = []
        games_sorted = games_df.sort_values("game_date").reset_index(drop=True)

        for idx, game in games_sorted.iterrows():
            if game["status"] != "Final":
                continue
            if game["home_score"] is None or game["away_score"] is None:
                continue

            game_date = game["game_date"]
            home = game["home_team_id"]
            away = game["away_team_id"]

            home_stats = team_stats.get(home)
            away_stats = team_stats.get(away)
            if not home_stats or not away_stats:
                continue

            # Get most recent stats BEFORE this game
            home_feat = self._get_team_features(home_stats, game_date)
            away_feat = self._get_team_features(away_stats, game_date)

            if home_feat is None or away_feat is None:
                continue  # Not enough history yet

            # Build feature row
            row = {"game_id": game["game_id"], "game_date": game_date}
            row["home_team"] = home
            row["away_team"] = away
            row["home_win"] = 1 if game["home_score"] > game["away_score"] else 0

            # Regression targets (not features — dropped before feature selection)
            row["home_score"] = int(game["home_score"])
            row["away_score"] = int(game["away_score"])
            row["total_runs"] = int(game["home_score"]) + int(game["away_score"])

            # Home features
            for k, v in home_feat.items():
                row[f"h_{k}"] = v

            # Away features
            for k, v in away_feat.items():
                row[f"a_{k}"] = v

            # Differentials
            for k in home_feat:
                row[f"diff_{k}"] = home_feat[k] - away_feat[k]

            # Ballpark factors (keyed by home team = venue)
            park = PARK_FACTORS.get(home, PARK_DEFAULT)
            row["park_runs_factor"] = park["runs"]
            row["park_hr_factor"] = park["hr"]

            # Weather proxy features (available historically)
            is_dome = home in DOME_TEAMS or home in RETRACTABLE_TEAMS
            row["is_dome"] = 1 if is_dome else 0
            row["game_month"] = game_date.month

            # Weather features — historical defaults are not useful (constant),
            # so we skip them for training. Live predictions inject real weather.

            # Elo ratings (pre-game)
            h_elo = elo_ratings.get(home, 1500.0)
            a_elo = elo_ratings.get(away, 1500.0)
            row["h_elo"] = h_elo
            row["a_elo"] = a_elo
            row["elo_diff"] = h_elo - a_elo

            # Market odds features — use real odds if available, else
            # a multi-factor synthetic proxy that's intentionally different
            # from raw Elo (which is already a feature) to avoid collinearity.
            odds_key = (game_date.isoformat() if hasattr(game_date, 'isoformat') else str(game_date), home, away)
            real_odds = odds_history.get(odds_key)
            if real_odds:
                row["market_home_prob"] = real_odds["market_home_prob"]
            else:
                # Multi-factor proxy blending several independent signals:
                #   40% Elo, 20% recent form, 20% SP matchup, 10% bullpen, 10% momentum
                # This creates a market-like composite that differs from raw Elo.

                # 1) Elo component (with HFA)
                elo_home_adj = h_elo + 24
                elo_comp = 1.0 / (1.0 + 10 ** ((a_elo - elo_home_adj) / 400.0))

                # 2) Recent form component (EWM win pct)
                h_ewm = home_feat.get("ewm_win_pct", 0.500)
                a_ewm = away_feat.get("ewm_win_pct", 0.500)
                form_comp = h_ewm / (h_ewm + a_ewm) if (h_ewm + a_ewm) > 0 else 0.5

                # 3) SP matchup component (ERA → win prob adjustment)
                h_sp_era = row.get("h_sp_season_era", 4.50)
                a_sp_era = row.get("a_sp_season_era", 4.50)
                # Transform ERA difference to a probability-like score
                era_diff = a_sp_era - h_sp_era  # positive = home SP better
                sp_comp = 1.0 / (1.0 + np.exp(-era_diff * 0.5))

                # 4) Bullpen component
                h_bp_era = home_feat.get("bp_era", 4.00)
                a_bp_era = away_feat.get("bp_era", 4.00)
                bp_diff = a_bp_era - h_bp_era
                bp_comp = 1.0 / (1.0 + np.exp(-bp_diff * 0.3))

                # 5) Momentum component
                h_mom = home_feat.get("momentum", 0.0)
                a_mom = away_feat.get("momentum", 0.0)
                mom_diff = h_mom - a_mom
                mom_comp = 1.0 / (1.0 + np.exp(-mom_diff * 0.5))

                # Weighted blend — Elo kept low to avoid collinearity with elo_diff feature.
                # SP matchup is the strongest differentiator vs raw Elo.
                market_prob = (
                    elo_comp * 0.25 +
                    form_comp * 0.20 +
                    sp_comp * 0.30 +
                    bp_comp * 0.15 +
                    mom_comp * 0.10
                )

                # Deterministic jitter
                noise_seed = hash(str(game["game_id"])) % 10000
                noise = ((noise_seed / 10000.0) - 0.5) * 0.03  # ±0.015
                market_prob = max(0.15, min(0.85, market_prob + noise))

                row["market_home_prob"] = round(market_prob, 4)

            # Rest days
            h_last = last_game_date.get(home)
            a_last = last_game_date.get(away)
            row["h_rest_days"] = (game_date - h_last).days if h_last else 5
            row["a_rest_days"] = (game_date - a_last).days if a_last else 5
            row["rest_diff"] = row["h_rest_days"] - row["a_rest_days"]

            # Venue-specific win pct differential (already in h_/a_ from rolling stats)
            h_home_wpct = home_feat.get("venue_home_win_pct", 0.536)
            a_away_wpct = away_feat.get("venue_away_win_pct", 0.464)
            row["diff_venue_win_pct"] = h_home_wpct - a_away_wpct

            # Starting pitcher season stats (entering this game)
            h_sp = sp_season.get((game["game_id"], home))
            a_sp = sp_season.get((game["game_id"], away))
            sp_defaults = {"sp_season_era": 4.50, "sp_season_whip": 1.30,
                           "sp_season_k9": 8.0, "sp_season_bb9": 3.0, "sp_season_ip": 0.0}
            for k, default in sp_defaults.items():
                row[f"h_{k}"] = h_sp[k] if h_sp else default
                row[f"a_{k}"] = a_sp[k] if a_sp else default
                row[f"diff_{k}"] = row[f"h_{k}"] - row[f"a_{k}"]

            # Derived Statcast-proxy feature: K-BB% (one of the strongest
            # predictors of future pitcher performance)
            h_k_minus_bb = row["h_sp_season_k9"] - row["h_sp_season_bb9"]
            a_k_minus_bb = row["a_sp_season_k9"] - row["a_sp_season_bb9"]
            row["h_sp_k_minus_bb"] = h_k_minus_bb
            row["a_sp_k_minus_bb"] = a_k_minus_bb
            row["diff_sp_k_minus_bb"] = h_k_minus_bb - a_k_minus_bb

            # SP recent form (rolling 3-start ERA/WHIP/K9)
            h_sp_recent = sp_recent.get((game["game_id"], home))
            a_sp_recent = sp_recent.get((game["game_id"], away))
            row["h_sp_recent_era"] = h_sp_recent["sp_recent_era"] if h_sp_recent else row.get("h_sp_season_era", 4.50)
            row["a_sp_recent_era"] = a_sp_recent["sp_recent_era"] if a_sp_recent else row.get("a_sp_season_era", 4.50)
            row["diff_sp_recent_era"] = row["h_sp_recent_era"] - row["a_sp_recent_era"]
            row["h_sp_recent_whip"] = h_sp_recent["sp_recent_whip"] if h_sp_recent else row.get("h_sp_season_whip", 1.30)
            row["a_sp_recent_whip"] = a_sp_recent["sp_recent_whip"] if a_sp_recent else row.get("a_sp_season_whip", 1.30)
            row["diff_sp_recent_whip"] = row["h_sp_recent_whip"] - row["a_sp_recent_whip"]
            row["h_sp_recent_k9"] = h_sp_recent["sp_recent_k9"] if h_sp_recent else row.get("h_sp_season_k9", 8.0)
            row["a_sp_recent_k9"] = a_sp_recent["sp_recent_k9"] if a_sp_recent else row.get("a_sp_season_k9", 8.0)
            row["diff_sp_recent_k9"] = row["h_sp_recent_k9"] - row["a_sp_recent_k9"]

            # Pitcher rest days (days since SP last started a game)
            h_sp_pid = sp_lookup.get((game["game_id"], home))
            a_sp_pid = sp_lookup.get((game["game_id"], away))
            h_sp_last = last_sp_start.get(h_sp_pid) if h_sp_pid else None
            a_sp_last = last_sp_start.get(a_sp_pid) if a_sp_pid else None
            row["h_sp_rest_days"] = (game_date - h_sp_last).days if h_sp_last else 5
            row["a_sp_rest_days"] = (game_date - a_sp_last).days if a_sp_last else 5
            row["diff_sp_rest_days"] = row["h_sp_rest_days"] - row["a_sp_rest_days"]

            # Lineup aggregate season stats
            h_lineup = lineup_season.get((game["game_id"], home))
            a_lineup = lineup_season.get((game["game_id"], away))
            lineup_defaults = {"lineup_ops": 0.720, "lineup_obp": 0.320, "lineup_slg": 0.400}
            for k, default in lineup_defaults.items():
                row[f"h_{k}"] = h_lineup[k] if h_lineup else default
                row[f"a_{k}"] = a_lineup[k] if a_lineup else default
                row[f"diff_{k}"] = row[f"h_{k}"] - row[f"a_{k}"]

            # Platoon advantage
            h_platoon = platoon.get((game["game_id"], home))
            a_platoon = platoon.get((game["game_id"], away))
            row["h_platoon_adv"] = h_platoon["platoon_adv"] if h_platoon else 0.5
            row["a_platoon_adv"] = a_platoon["platoon_adv"] if a_platoon else 0.5
            row["diff_platoon_adv"] = row["h_platoon_adv"] - row["a_platoon_adv"]

            # Lineup recent form (7-day rolling OPS)
            h_recent = lineup_recent.get((game["game_id"], home))
            a_recent = lineup_recent.get((game["game_id"], away))
            recent_defaults = {"lineup_ops_7d": 0.720, "lineup_hot_pct": 0.4}
            for k, default in recent_defaults.items():
                row[f"h_{k}"] = h_recent[k] if h_recent else default
                row[f"a_{k}"] = a_recent[k] if a_recent else default
                row[f"diff_{k}"] = row[f"h_{k}"] - row[f"a_{k}"]

            # BvP matchup OPS — historical data not in CSVs, so omitted from
            # training. Live predictions inject real BvP data when available.

            # Bullpen availability
            h_bp_avail = bp_avail.get((game["game_id"], home))
            a_bp_avail = bp_avail.get((game["game_id"], away))
            bp_avail_defaults = {
                "bp_relievers_used_3d": 4, "bp_freshness": 0.5,
            }
            for k, default in bp_avail_defaults.items():
                row[f"h_{k}"] = h_bp_avail[k] if h_bp_avail else default
                row[f"a_{k}"] = a_bp_avail[k] if a_bp_avail else default
                row[f"diff_{k}"] = row[f"h_{k}"] - row[f"a_{k}"]

            # Travel fatigue
            h_prev_venue = last_game_venue.get(home)
            a_prev_venue = last_game_venue.get(away)
            # Home team fatigue: previous venue -> their home stadium
            row["h_travel_fatigue"] = (
                _compute_travel_fatigue(h_prev_venue, home) if h_prev_venue else 0.0
            )
            # Away team fatigue: previous venue -> current game's home stadium
            row["a_travel_fatigue"] = (
                _compute_travel_fatigue(a_prev_venue, home) if a_prev_venue else 0.0
            )
            row["diff_travel_fatigue"] = row["h_travel_fatigue"] - row["a_travel_fatigue"]

            # Day/night — only useful if game_time is in the CSV.
            # Omit entirely when constant to avoid noise.
            if has_game_time:
                game_time_str = game.get("game_time", "")
                row["is_day_game"] = _classify_day_game(game_time_str, home)

            # Umpire data — not in historical CSVs, omitted from training.
            # Live predictions inject real umpire effects when available.

            # ── Feature 2: Run Differential Trends (7-game) ──
            row["h_rd_7d"] = home_feat.get("rd_7d", 0.0)
            row["a_rd_7d"] = away_feat.get("rd_7d", 0.0)
            row["diff_rd_7d"] = row["h_rd_7d"] - row["a_rd_7d"]

            # ── Feature 3: Defensive Metrics Proxy ──
            # def_proxy = ra_per_game - rolling ERA (positive = defense costing
            # runs, negative = defense saving runs).  Invert so higher = better.
            row["h_def_proxy"] = home_feat.get("def_proxy", 0.0)
            row["a_def_proxy"] = away_feat.get("def_proxy", 0.0)
            row["diff_def_proxy"] = row["h_def_proxy"] - row["a_def_proxy"]

            # ── Feature 4: Platoon Splits (SP handedness) ──
            # Encode starting pitcher handedness as a feature.
            # 0 = RHP, 1 = LHP. Uses the handedness cache.
            h_sp_hand = None
            a_sp_hand = None
            if h_sp_pid:
                h_info = hand_cache.get(h_sp_pid)
                if h_info:
                    h_sp_hand = h_info.get("throws")
            if a_sp_pid:
                a_info = hand_cache.get(a_sp_pid)
                if a_info:
                    a_sp_hand = a_info.get("throws")
            row["h_sp_throws"] = 1.0 if h_sp_hand == "L" else 0.0
            row["a_sp_throws"] = 1.0 if a_sp_hand == "L" else 0.0
            # Platoon advantage indicator: 1 if SP throws opposite hand from
            # majority of opposing lineup (approximated by platoon_adv > 0.5),
            # 0 otherwise. Falls back to a simple lefty-pitcher flag since
            # lefty starters have historically different profiles.
            row["platoon_advantage_home"] = 1.0 if row.get("h_platoon_adv", 0.5) > 0.5 else 0.0
            row["platoon_advantage_away"] = 1.0 if row.get("a_platoon_adv", 0.5) > 0.5 else 0.0

            # ── Feature 5: Recent Form Weighting (Exponential Decay) ──
            row["h_ewm_win_pct"] = home_feat.get("ewm_win_pct", 0.500)
            row["a_ewm_win_pct"] = away_feat.get("ewm_win_pct", 0.500)
            row["diff_ewm_win_pct"] = row["h_ewm_win_pct"] - row["a_ewm_win_pct"]
            row["h_ewm_rs_per_game"] = home_feat.get("ewm_rs_per_game", 4.5)
            row["a_ewm_rs_per_game"] = away_feat.get("ewm_rs_per_game", 4.5)
            row["diff_ewm_rs_per_game"] = row["h_ewm_rs_per_game"] - row["a_ewm_rs_per_game"]
            row["h_ewm_ra_per_game"] = home_feat.get("ewm_ra_per_game", 4.5)
            row["a_ewm_ra_per_game"] = away_feat.get("ewm_ra_per_game", 4.5)
            row["diff_ewm_ra_per_game"] = row["h_ewm_ra_per_game"] - row["a_ewm_ra_per_game"]
            row["h_momentum"] = home_feat.get("momentum", 0.0)
            row["a_momentum"] = away_feat.get("momentum", 0.0)
            row["diff_momentum"] = row["h_momentum"] - row["a_momentum"]

            # ── Feature 6: Bullpen Fatigue Tracking ──
            # Bullpen IP is already in h_bp_ip_3d / a_bp_ip_3d from rolling stats

            # IL signals — not in historical CSVs, omitted from training.
            # Live predictions inject real IL data when available.

            # ── Interaction Features (mismatch detection) ──
            # Normalized component diffs for interaction terms
            elo_gap = row.get("elo_diff", 0.0) / 100.0  # ~±2 range
            sp_era_gap = row.get("diff_sp_season_era", 0.0)  # Negative = home SP better ERA
            bp_fresh = row.get("diff_bp_freshness", 0.0)
            mom_gap = row.get("diff_momentum", 0.0)
            form_gap = row.get("diff_ewm_win_pct", 0.0) * 10  # Scale up

            # Core interactions
            row["interact_elo_x_sp"] = elo_gap * (-sp_era_gap)  # Both favor home → positive
            row["interact_elo_x_bp"] = elo_gap * bp_fresh
            row["interact_sp_x_bp"] = (-sp_era_gap) * bp_fresh
            row["interact_elo_x_momentum"] = elo_gap * mom_gap

            # NEW: SP quality × opposing offense (mismatch detector)
            h_off_ops = home_feat.get("ops_14", 0.720)
            a_off_ops = away_feat.get("ops_14", 0.720)
            row["interact_hsp_vs_aoff"] = (-sp_era_gap) * (a_off_ops - 0.720) * 10
            row["interact_asp_vs_hoff"] = sp_era_gap * (h_off_ops - 0.720) * 10

            # NEW: Rest × form (rested team on a hot streak)
            rest_gap = row.get("rest_diff", 0.0)
            row["interact_rest_x_form"] = rest_gap * form_gap

            # NEW: Park factor × SP quality (elite SP in hitter park)
            pf = park["runs"]
            row["interact_park_x_sp"] = (pf - 1.0) * 10 * (-sp_era_gap)

            # NEW: Bullpen fatigue × close-game likelihood
            # If both SPs are good (low ERA), game likely close → bullpen matters more
            h_sp_era_val = row.get("h_sp_season_era", 4.50)
            a_sp_era_val = row.get("a_sp_season_era", 4.50)
            pitching_duel = max(0, (9.0 - h_sp_era_val - a_sp_era_val) / 4.0)
            row["interact_bp_x_duel"] = bp_fresh * pitching_duel

            feature_rows.append(row)

            # Update Elo AFTER recording pre-game values
            home_won = game["home_score"] > game["away_score"]
            _update_elo(elo_ratings, home, away, home_won)

            # Update last game dates and venues
            last_game_date[home] = game_date
            last_game_date[away] = game_date
            last_game_venue[home] = home  # venue is the home team's stadium
            last_game_venue[away] = home

            # Update pitcher last start dates
            if h_sp_pid:
                last_sp_start[h_sp_pid] = game_date
            if a_sp_pid:
                last_sp_start[a_sp_pid] = game_date

            if (idx + 1) % 500 == 0:
                logger.info("Processed %d / %d games", idx + 1, len(games_sorted))

        df = pd.DataFrame(feature_rows)
        logger.info("Built %d training samples with %d features", len(df), len(df.columns) - 9)

        # Save
        out_path = self.data_dir / output
        df.to_parquet(out_path, index=False)
        logger.info("Saved training data to %s", out_path)

        return df

    # ── Data Loading ──────────────────────────────────────

    def _load_games(self, seasons: list[int]) -> pd.DataFrame:
        frames = []
        for s in seasons:
            path = self.data_dir / f"games_{s}.csv"
            if path.exists():
                df = pd.read_csv(path)
                frames.append(df)
                logger.info("Loaded %d games from %s", len(df), path)
            else:
                logger.warning("Games file not found: %s", path)
        if not frames:
            raise FileNotFoundError("No games CSVs found")
        df = pd.concat(frames, ignore_index=True)
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
        df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
        return df

    def _load_batting(self, seasons: list[int]) -> pd.DataFrame:
        frames = []
        for s in seasons:
            path = self.data_dir / f"batting_{s}.csv"
            if path.exists():
                frames.append(pd.read_csv(path))
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        return df

    def _load_odds_history(self) -> dict[tuple, dict]:
        """Load historical odds from odds_history.csv.

        Returns {(date_str, home_team, away_team): {"market_home_prob": ..., "total_line": ...}}
        """
        odds_file = self.data_dir / "odds_history.csv"
        if not odds_file.exists():
            logger.info("No odds history file found — using Elo-derived proxies")
            return {}

        import csv
        result: dict[tuple, dict] = {}
        with open(odds_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["game_date"], row["home_team"], row["away_team"])
                result[key] = {
                    "market_home_prob": float(row["market_home_prob"]),
                    "total_line": float(row["total_line"]) if row.get("total_line") else None,
                }

        logger.info("Loaded %d historical odds records", len(result))
        return result

    def _load_pitching(self, seasons: list[int]) -> pd.DataFrame:
        frames = []
        for s in seasons:
            path = self.data_dir / f"pitching_{s}.csv"
            if path.exists():
                frames.append(pd.read_csv(path))
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        return df

    # ── Elo & Rest ────────────────────────────────────────

    def _compute_elo_and_rest(
        self, games_df: pd.DataFrame
    ) -> tuple[dict[str, float], dict[str, date]]:
        """Initialize Elo ratings and last-game-date trackers.

        Returns (elo_ratings, last_game_date) dicts to be updated
        during the main build loop.
        """
        elo_ratings: dict[str, float] = {}  # team_id -> current Elo
        last_game_date: dict[str, date] = {}  # team_id -> date of last game

        # Pre-seed by replaying all completed games chronologically
        # so Elo is warmed up before the feature-generation loop.
        # (The build loop will also update, but this seeds for teams
        # that appear before the 15-game feature threshold.)
        # Actually, we return empty dicts — the build loop handles
        # both reading and updating in one pass.
        return elo_ratings, last_game_date

    def _build_sp_season_stats(
        self, pitching_df: pd.DataFrame
    ) -> dict[tuple, dict[str, float]]:
        """Build per-game SP season stats entering each game.

        Returns {(game_id, team_id): {"sp_season_era": ..., ...}}
        where stats reflect the SP's cumulative season performance
        BEFORE the current game.
        """
        if pitching_df.empty:
            return {}

        # Identify SP (highest IP) for each (game_id, team_id)
        sp_lookup: dict[tuple, int] = {}  # (game_id, team_id) -> player_id
        for (gid, tid), grp in pitching_df.groupby(["game_id", "team_id"]):
            sp_row = grp.nlargest(1, "innings_pitched").iloc[0]
            sp_lookup[(gid, tid)] = sp_row["player_id"]

        # Accumulate each pitcher's season stats game-by-game
        # Sort by date to process chronologically
        sorted_pit = pitching_df.sort_values("game_date")
        pitcher_totals: dict[tuple, dict] = {}  # (player_id, year) -> cumulative stats
        result: dict[tuple, dict[str, float]] = {}

        for _, row in sorted_pit.iterrows():
            pid = row["player_id"]
            gid = row["game_id"]
            tid = row["team_id"]
            year = row["game_date"].year

            # Only process if this pitcher was the SP for this game
            if sp_lookup.get((gid, tid)) != pid:
                continue

            key = (pid, year)
            prev = pitcher_totals.get(key)

            if prev and prev["ip"] > 0:
                # Stats ENTERING this game
                era = prev["er"] * 9 / prev["ip"]
                whip = (prev["ha"] + prev["bb"]) / prev["ip"]
                k9 = prev["k"] * 9 / prev["ip"]
                bb9 = prev["bb"] * 9 / prev["ip"]
                result[(gid, tid)] = {
                    "sp_season_era": era,
                    "sp_season_whip": whip,
                    "sp_season_k9": k9,
                    "sp_season_bb9": bb9,
                    "sp_season_ip": prev["ip"],
                }
            # else: first start of season, no prior stats

            # Update cumulative totals AFTER recording pre-game stats
            if prev is None:
                pitcher_totals[key] = {
                    "ip": float(row["innings_pitched"]),
                    "er": int(row["earned_runs"]),
                    "ha": int(row["hits_allowed"]),
                    "bb": int(row["walks_allowed"]),
                    "k": int(row["strikeouts_recorded"]),
                }
            else:
                prev["ip"] += float(row["innings_pitched"])
                prev["er"] += int(row["earned_runs"])
                prev["ha"] += int(row["hits_allowed"])
                prev["bb"] += int(row["walks_allowed"])
                prev["k"] += int(row["strikeouts_recorded"])

        logger.info("Built SP season stats for %d game-team pairs", len(result))
        return result

    def _build_sp_recent_form(
        self, pitching_df: pd.DataFrame, n_starts: int = 3
    ) -> dict[tuple, dict[str, float]]:
        """Build rolling N-start ERA/WHIP/K9 for each SP entering each game.

        Returns {(game_id, team_id): {"sp_recent_era": ..., "sp_recent_whip": ..., "sp_recent_k9": ...}}
        Only includes starts where the pitcher threw 3+ IP (quality starts filter).
        """
        if pitching_df.empty:
            return {}

        # Identify SP (highest IP) for each (game_id, team_id)
        sp_lookup: dict[tuple, int] = {}
        for (gid, tid), grp in pitching_df.groupby(["game_id", "team_id"]):
            sp_row = grp.nlargest(1, "innings_pitched").iloc[0]
            sp_lookup[(gid, tid)] = sp_row["player_id"]

        # Process chronologically, tracking each pitcher's recent starts
        sorted_pit = pitching_df.sort_values("game_date")
        # pitcher_id -> deque of last N starts: [(ip, er, h, bb, k), ...]
        from collections import deque
        pitcher_history: dict[int, deque] = {}
        result: dict[tuple, dict[str, float]] = {}

        for _, row in sorted_pit.iterrows():
            pid = row["player_id"]
            gid = row["game_id"]
            tid = row["team_id"]

            # Only process if this pitcher was the SP for this game
            if sp_lookup.get((gid, tid)) != pid:
                continue

            ip = float(row["innings_pitched"])

            # Record pre-game recent form (from previous starts)
            history = pitcher_history.get(pid)
            if history and len(history) > 0:
                total_ip = sum(s[0] for s in history)
                if total_ip > 0:
                    total_er = sum(s[1] for s in history)
                    total_h = sum(s[2] for s in history)
                    total_bb = sum(s[3] for s in history)
                    total_k = sum(s[4] for s in history)
                    result[(gid, tid)] = {
                        "sp_recent_era": round((total_er / total_ip) * 9.0, 2),
                        "sp_recent_whip": round((total_h + total_bb) / total_ip, 2),
                        "sp_recent_k9": round((total_k / total_ip) * 9.0, 1),
                    }

            # Update history AFTER recording pre-game stats
            # Only count starts with 3+ IP
            if ip >= 3.0:
                if pid not in pitcher_history:
                    pitcher_history[pid] = deque(maxlen=n_starts)
                pitcher_history[pid].append((
                    ip,
                    int(row["earned_runs"]),
                    int(row["hits_allowed"]),
                    int(row["walks_allowed"]),
                    int(row["strikeouts_recorded"]),
                ))

        logger.info("Built SP recent form for %d game-team pairs", len(result))
        return result

    def _build_sp_lookup(
        self, pitching_df: pd.DataFrame
    ) -> dict[tuple, int]:
        """Identify the starting pitcher (highest IP) for each (game_id, team_id).

        Returns {(game_id, team_id): player_id}
        """
        if pitching_df.empty:
            return {}

        sp_lookup: dict[tuple, int] = {}
        for (gid, tid), grp in pitching_df.groupby(["game_id", "team_id"]):
            sp_row = grp.nlargest(1, "innings_pitched").iloc[0]
            sp_lookup[(gid, tid)] = sp_row["player_id"]

        logger.info("Built SP lookup for %d game-team pairs", len(sp_lookup))
        return sp_lookup

    # ── Player Handedness Cache ─────────────────────────────

    def _load_handedness_cache(self) -> dict[int, dict[str, str]]:
        """Load player handedness from cache file.

        Returns {player_id: {"bats": "R"/"L"/"S", "throws": "R"/"L"}}
        """
        import json
        cache_path = self.data_dir / "player_handedness.json"
        if cache_path.exists():
            with open(cache_path) as f:
                raw = json.load(f)
            return {int(k): v for k, v in raw.items()}
        return {}

    def _save_handedness_cache(self, cache: dict[int, dict[str, str]]):
        import json
        cache_path = self.data_dir / "player_handedness.json"
        with open(cache_path, "w") as f:
            json.dump({str(k): v for k, v in cache.items()}, f)

    def _build_platoon_features(
        self, batting_df: pd.DataFrame, pitching_df: pd.DataFrame
    ) -> dict[tuple, dict[str, float]]:
        """Compute platoon advantage for each (game_id, team_id).

        Platoon advantage = fraction of lineup batters with the handedness advantage
        vs the opposing SP. LHB vs RHP and RHB vs LHP have a historical edge.

        Returns {(game_id, team_id): {"platoon_adv": 0.0-1.0}}
        """
        if batting_df.empty or pitching_df.empty:
            return {}

        hand_cache = self._load_handedness_cache()

        # Identify SP for each (game_id, team_id)
        sp_lookup: dict[tuple, int] = {}
        for (gid, tid), grp in pitching_df.groupby(["game_id", "team_id"]):
            sp_row = grp.nlargest(1, "innings_pitched").iloc[0]
            sp_lookup[(gid, tid)] = sp_row["player_id"]

        # Build game_id -> {home_team, away_team} mapping
        game_teams: dict[str, dict[str, str]] = {}
        for (gid, tid), _ in batting_df.groupby(["game_id", "team_id"]):
            if gid not in game_teams:
                game_teams[gid] = {}
            side = batting_df[
                (batting_df["game_id"] == gid) & (batting_df["team_id"] == tid)
            ]["side"].iloc[0]
            game_teams[gid][side] = tid

        result: dict[tuple, dict[str, float]] = {}

        for (gid, tid), grp in batting_df.groupby(["game_id", "team_id"]):
            # Find opposing SP
            sides = game_teams.get(gid, {})
            batting_side = grp["side"].iloc[0]
            opp_side = "away" if batting_side == "home" else "home"
            opp_tid = sides.get(opp_side)
            if not opp_tid:
                continue

            opp_sp_id = sp_lookup.get((gid, opp_tid))
            if not opp_sp_id:
                continue

            sp_hand_info = hand_cache.get(opp_sp_id)
            if not sp_hand_info:
                continue
            sp_throws = sp_hand_info.get("throws", "R")

            # Count lineup batters with platoon advantage
            adv_count = 0
            total = 0
            for _, row in grp.iterrows():
                if int(row["at_bats"]) == 0:
                    continue
                pid = row["player_id"]
                batter_info = hand_cache.get(pid)
                if not batter_info:
                    continue
                bats = batter_info.get("bats", "R")
                total += 1
                # Platoon advantage: LHB vs RHP, RHB vs LHP, switch hitters always have it
                if bats == "S" or (bats == "L" and sp_throws == "R") or (bats == "R" and sp_throws == "L"):
                    adv_count += 1

            if total >= 5:
                result[(gid, tid)] = {
                    "platoon_adv": adv_count / total,
                }

        logger.info("Built platoon features for %d game-team pairs", len(result))
        return result

    # ── Lineup Season Stats ────────────────────────────────

    def _build_lineup_season_stats(
        self, batting_df: pd.DataFrame
    ) -> dict[tuple, dict[str, float]]:
        """Build per-game lineup aggregate season stats entering each game.

        For each (game_id, team_id), computes the mean OPS/OBP/SLG of the
        lineup batters based on their cumulative season stats BEFORE that game.

        Returns {(game_id, team_id): {"lineup_ops": ..., "lineup_obp": ..., ...}}
        """
        if batting_df.empty:
            return {}

        sorted_bat = batting_df.sort_values("game_date")

        # Accumulate each batter's season stats game-by-game
        batter_totals: dict[tuple, dict] = {}  # (player_id, year) -> cumulative
        result: dict[tuple, dict[str, float]] = {}

        # Group by game to process all batters in a game together
        for (gid, tid), grp in sorted_bat.groupby(["game_id", "team_id"]):
            year = grp.iloc[0]["game_date"].year

            # Collect pre-game stats for each batter in this lineup
            lineup_ops_vals = []
            for _, row in grp.iterrows():
                pid = row["player_id"]
                ab = int(row["at_bats"])
                if ab == 0:
                    continue  # pinch runner, etc.

                key = (pid, year)
                prev = batter_totals.get(key)

                if prev and prev["ab"] >= 20:
                    # Stats entering this game
                    p_ab = prev["ab"]
                    p_h = prev["h"]
                    p_2b = prev["2b"]
                    p_3b = prev["3b"]
                    p_hr = prev["hr"]
                    p_bb = prev["bb"]
                    avg = p_h / p_ab
                    obp = (p_h + p_bb) / (p_ab + p_bb) if (p_ab + p_bb) > 0 else 0
                    slg = (p_h + p_2b + 2 * p_3b + 3 * p_hr) / p_ab
                    lineup_ops_vals.append({"obp": obp, "slg": slg, "ops": obp + slg})

                # Update cumulative AFTER recording
                if prev is None:
                    batter_totals[key] = {
                        "ab": int(row["at_bats"]),
                        "h": int(row["hits"]),
                        "2b": int(row["doubles"]),
                        "3b": int(row["triples"]),
                        "hr": int(row["home_runs"]),
                        "bb": int(row["walks"]),
                    }
                else:
                    prev["ab"] += int(row["at_bats"])
                    prev["h"] += int(row["hits"])
                    prev["2b"] += int(row["doubles"])
                    prev["3b"] += int(row["triples"])
                    prev["hr"] += int(row["home_runs"])
                    prev["bb"] += int(row["walks"])

            if len(lineup_ops_vals) >= 5:
                result[(gid, tid)] = {
                    "lineup_ops": np.mean([v["ops"] for v in lineup_ops_vals]),
                    "lineup_obp": np.mean([v["obp"] for v in lineup_ops_vals]),
                    "lineup_slg": np.mean([v["slg"] for v in lineup_ops_vals]),
                }

        logger.info("Built lineup season stats for %d game-team pairs", len(result))
        return result

    # ── Bullpen Availability ───────────────────────────────

    def _build_bullpen_availability(
        self, pitching_df: pd.DataFrame
    ) -> dict[tuple, dict[str, float]]:
        """Build per-game bullpen availability for each team.

        For each (game_id, team_id), looks at the prior 3 calendar days to
        measure how many unique relievers pitched and total reliever IP.

        The pitcher with the most IP in a game for a team is the starter;
        everyone else is a reliever.

        Returns {(game_id, team_id): {
            "bp_relievers_used_1d": int,
            "bp_relievers_used_3d": int,
            "bp_ip_recent": float,
            "bp_freshness": float,
        }}
        """
        if pitching_df.empty:
            return {}

        sorted_pit = pitching_df.sort_values("game_date")

        # Step 1: Identify starters and collect reliever appearances per game
        # {(game_id, team_id): set of reliever player_ids}
        game_relievers: dict[tuple, set[int]] = {}
        # {(game_id, team_id): total reliever IP}
        game_reliever_ip: dict[tuple, float] = {}
        # {(game_id, team_id): game_date}
        game_dates: dict[tuple, date] = {}
        # Track all unique relievers per team (across the season so far)
        team_all_relievers: dict[str, set[int]] = {}

        for (gid, tid), grp in sorted_pit.groupby(["game_id", "team_id"]):
            game_date = grp.iloc[0]["game_date"]
            game_dates[(gid, tid)] = game_date

            # Starter = pitcher with max IP in this game for this team
            sp_row = grp.nlargest(1, "innings_pitched").iloc[0]
            sp_pid = sp_row["player_id"]

            relievers = set()
            reliever_ip = 0.0
            for _, row in grp.iterrows():
                pid = row["player_id"]
                if pid != sp_pid:
                    relievers.add(pid)
                    reliever_ip += float(row["innings_pitched"])
                    if tid not in team_all_relievers:
                        team_all_relievers[tid] = set()
                    team_all_relievers[tid].add(pid)

            game_relievers[(gid, tid)] = relievers
            game_reliever_ip[(gid, tid)] = reliever_ip

        # Step 2: For each game-team, look back at the prior 1 and 3 days
        # Build a per-team chronological index of (game_date, game_id) for lookback
        team_game_log: dict[str, list[tuple]] = {}  # tid -> [(date, gid), ...]
        for (gid, tid), gd in sorted(game_dates.items(), key=lambda x: x[1]):
            if tid not in team_game_log:
                team_game_log[tid] = []
            team_game_log[tid].append((gd, gid))

        result: dict[tuple, dict[str, float]] = {}

        for tid, log in team_game_log.items():
            total_relievers = len(team_all_relievers.get(tid, set()))
            if total_relievers == 0:
                total_relievers = 1  # avoid division by zero

            for i, (gd, gid) in enumerate(log):
                # Look at previous games (not including current) within 1 and 3 days
                relievers_1d: set[int] = set()
                relievers_3d: set[int] = set()
                ip_3d = 0.0

                for j in range(i - 1, -1, -1):
                    prev_date, prev_gid = log[j]
                    days_ago = (gd - prev_date).days
                    if days_ago > 3:
                        break
                    key = (prev_gid, tid)
                    prev_relievers = game_relievers.get(key, set())
                    prev_ip = game_reliever_ip.get(key, 0.0)

                    relievers_3d |= prev_relievers
                    ip_3d += prev_ip
                    if days_ago <= 1:
                        relievers_1d |= prev_relievers

                freshness = max(0.0, (total_relievers - len(relievers_3d)) / total_relievers)

                result[(gid, tid)] = {
                    "bp_relievers_used_1d": len(relievers_1d),
                    "bp_relievers_used_3d": len(relievers_3d),
                    "bp_ip_recent": ip_3d,
                    "bp_freshness": freshness,
                }

        logger.info("Built bullpen availability for %d game-team pairs", len(result))
        return result

    # ── Lineup Recent Form ──────────────────────────────

    def _build_lineup_recent_form(
        self, batting_df: pd.DataFrame
    ) -> dict[tuple, dict[str, float]]:
        """Build per-game lineup 7-day rolling OPS for hot/cold streak detection.

        For each (game_id, team_id), computes mean OPS of team's batters over
        their last 7 calendar days (minimum 5 AB in window).

        Returns {(game_id, team_id): {"lineup_ops_7d": ..., "lineup_hot_pct": ...}}
        """
        if batting_df.empty:
            return {}

        sorted_bat = batting_df.sort_values("game_date")
        from datetime import timedelta

        # Per-player game-by-game batting log: {player_id: [(date, ab, h, 2b, 3b, hr, bb), ...]}
        player_log: dict[int, list[tuple]] = {}
        result: dict[tuple, dict[str, float]] = {}

        for (gid, tid), grp in sorted_bat.groupby(["game_id", "team_id"]):
            game_date = grp.iloc[0]["game_date"]
            cutoff = game_date - timedelta(days=7)

            ops_vals = []
            for _, row in grp.iterrows():
                pid = row["player_id"]
                ab = int(row["at_bats"])

                # Compute 7-day OPS from previous games
                if pid in player_log:
                    recent = [
                        e for e in player_log[pid] if cutoff <= e[0] < game_date
                    ]
                    tot_ab = sum(e[1] for e in recent)
                    if tot_ab >= 5:
                        tot_h = sum(e[2] for e in recent)
                        tot_2b = sum(e[3] for e in recent)
                        tot_3b = sum(e[4] for e in recent)
                        tot_hr = sum(e[5] for e in recent)
                        tot_bb = sum(e[6] for e in recent)
                        obp = (tot_h + tot_bb) / (tot_ab + tot_bb) if (tot_ab + tot_bb) else 0
                        slg = (tot_h + tot_2b + 2 * tot_3b + 3 * tot_hr) / tot_ab
                        ops_vals.append(obp + slg)

                # Record this game AFTER computing (so we don't include current game)
                if ab > 0:
                    if pid not in player_log:
                        player_log[pid] = []
                    player_log[pid].append((
                        game_date, ab, int(row["hits"]),
                        int(row["doubles"]), int(row["triples"]),
                        int(row["home_runs"]), int(row["walks"]),
                    ))

            if len(ops_vals) >= 3:
                mean_ops = np.mean(ops_vals)
                hot_pct = sum(1 for o in ops_vals if o > 0.800) / len(ops_vals)
                result[(gid, tid)] = {
                    "lineup_ops_7d": mean_ops,
                    "lineup_hot_pct": hot_pct,
                }

        logger.info("Built lineup recent form for %d game-team pairs", len(result))
        return result

    # ── Rolling Stats ─────────────────────────────────────

    def _compute_team_rolling_stats(
        self,
        games_df: pd.DataFrame,
        batting_df: pd.DataFrame,
        pitching_df: pd.DataFrame,
    ) -> dict[str, list[dict]]:
        """Compute game-by-game rolling stats for each team.

        Returns {team_id: [{"date": ..., "stat": ...}, ...]} sorted by date.
        """
        teams = set(games_df["home_team_id"].unique()) | set(games_df["away_team_id"].unique())
        team_stats: dict[str, list[dict]] = {}

        for team_id in sorted(teams):
            # Get all games for this team (home or away)
            team_games = games_df[
                ((games_df["home_team_id"] == team_id) | (games_df["away_team_id"] == team_id))
                & (games_df["status"] == "Final")
            ].sort_values("game_date").reset_index(drop=True)

            if len(team_games) < 10:
                continue

            # Batting stats for this team
            team_batting = batting_df[batting_df["team_id"] == team_id] if len(batting_df) else pd.DataFrame()
            team_pitching = pitching_df[pitching_df["team_id"] == team_id] if len(pitching_df) else pd.DataFrame()

            stats_list = []
            wins = losses = total_rs = total_ra = 0
            results_buffer: list[dict] = []  # For rolling calculations
            home_buffer: list[dict] = []  # Last N home games (venue splits)
            away_buffer: list[dict] = []  # Last N away games (venue splits)
            rd_7_buffer: list[int] = []  # Last 7 game run differentials

            for _, game in team_games.iterrows():
                is_home = game["home_team_id"] == team_id
                rs = game["home_score"] if is_home else game["away_score"]
                ra = game["away_score"] if is_home else game["home_score"]

                if pd.isna(rs) or pd.isna(ra):
                    continue

                rs, ra = int(rs), int(ra)
                won = rs > ra
                if won:
                    wins += 1
                else:
                    losses += 1
                total_rs += rs
                total_ra += ra

                results_buffer.append({
                    "date": game["game_date"],
                    "won": won,
                    "rs": rs,
                    "ra": ra,
                    "run_diff": rs - ra,
                    "is_home": is_home,
                    "venue": game["home_team_id"],  # venue is always the home team
                    "bp_ip": 0.0,  # filled after pitching calc below
                    "win_int": 1 if won else 0,  # for EWM calculations
                })

                # Track 7-game run differential buffer
                rd_7_buffer.append(rs - ra)
                if len(rd_7_buffer) > 7:
                    rd_7_buffer = rd_7_buffer[-7:]

                # Track venue-specific results (last 20 home/away games)
                venue_entry = {"won": won, "rs": rs, "ra": ra}
                if is_home:
                    home_buffer.append(venue_entry)
                    if len(home_buffer) > 20:
                        home_buffer = home_buffer[-20:]
                else:
                    away_buffer.append(venue_entry)
                    if len(away_buffer) > 20:
                        away_buffer = away_buffer[-20:]

                gp = wins + losses

                # Game-level batting aggregates for this game
                game_bat = team_batting[team_batting["game_id"] == game["game_id"]] if len(team_batting) else pd.DataFrame()
                g_ab = int(game_bat["at_bats"].sum()) if len(game_bat) else 0
                g_h = int(game_bat["hits"].sum()) if len(game_bat) else 0
                g_hr = int(game_bat["home_runs"].sum()) if len(game_bat) else 0
                g_bb = int(game_bat["walks"].sum()) if len(game_bat) else 0
                g_so = int(game_bat["strikeouts"].sum()) if len(game_bat) else 0
                g_2b = int(game_bat["doubles"].sum()) if len(game_bat) else 0
                g_3b = int(game_bat["triples"].sum()) if len(game_bat) else 0
                g_sb = int(game_bat["stolen_bases"].sum()) if len(game_bat) else 0

                # Game-level pitching aggregates
                game_pit = team_pitching[team_pitching["game_id"] == game["game_id"]] if len(team_pitching) else pd.DataFrame()
                g_ip = float(game_pit["innings_pitched"].sum()) if len(game_pit) else 0
                g_er = int(game_pit["earned_runs"].sum()) if len(game_pit) else 0
                g_ha = int(game_pit["hits_allowed"].sum()) if len(game_pit) else 0
                g_bba = int(game_pit["walks_allowed"].sum()) if len(game_pit) else 0
                g_ka = int(game_pit["strikeouts_recorded"].sum()) if len(game_pit) else 0

                # Separate SP (highest IP) from bullpen
                if len(game_pit) > 0:
                    sp_row = game_pit.nlargest(1, "innings_pitched").iloc[0]
                    bp_rows = game_pit.drop(sp_row.name)
                    sp_ip = float(sp_row["innings_pitched"])
                    sp_er = int(sp_row["earned_runs"])
                    sp_ha = int(sp_row["hits_allowed"])
                    sp_bb = int(sp_row["walks_allowed"])
                    sp_k = int(sp_row["strikeouts_recorded"])
                    bp_ip = float(bp_rows["innings_pitched"].sum())
                    bp_er = int(bp_rows["earned_runs"].sum())
                    bp_ha = int(bp_rows["hits_allowed"].sum())
                    bp_bb = int(bp_rows["walks_allowed"].sum())
                    bp_k = int(bp_rows["strikeouts_recorded"].sum())
                else:
                    sp_ip = sp_er = sp_ha = sp_bb = sp_k = 0
                    bp_ip = bp_er = bp_ha = bp_bb = bp_k = 0

                # Store bp_ip in results_buffer for fatigue calc
                results_buffer[-1]["bp_ip"] = bp_ip

                # Bullpen fatigue: IP in last 3 calendar days
                cur_date = game["game_date"]
                bp_ip_3d = sum(
                    r["bp_ip"] for r in results_buffer[:-1]
                    if (cur_date - r["date"]).days <= 3
                )
                bp_games_5d = sum(
                    1 for r in results_buffer[:-1]
                    if (cur_date - r["date"]).days <= 5
                )

                # Rolling windows
                last_10 = results_buffer[-10:]
                last_20 = results_buffer[-20:]
                l10_w = sum(1 for r in last_10 if r["won"])
                l20_w = sum(1 for r in last_20 if r["won"])
                l10_rd = sum(r["run_diff"] for r in last_10)

                # Streak
                streak = 0
                for r in reversed(results_buffer):
                    if r["won"]:
                        if streak >= 0:
                            streak += 1
                        else:
                            break
                    else:
                        if streak <= 0:
                            streak -= 1
                        else:
                            break

                # Exponentially-weighted recent form (halflife ~10 games)
                win_vals = [r["win_int"] for r in results_buffer]
                rs_vals = [float(r["rs"]) for r in results_buffer]
                ra_vals = [float(r["ra"]) for r in results_buffer]
                if len(win_vals) >= 5:
                    ewm_win_pct = pd.Series(win_vals).ewm(halflife=10).mean().iloc[-1]
                    ewm_rs_per_game = pd.Series(rs_vals).ewm(halflife=10).mean().iloc[-1]
                    ewm_ra_per_game = pd.Series(ra_vals).ewm(halflife=10).mean().iloc[-1]
                else:
                    ewm_win_pct = wins / max(gp, 1)
                    ewm_rs_per_game = total_rs / max(gp, 1)
                    ewm_ra_per_game = total_ra / max(gp, 1)

                # Momentum: win rate last 5 minus last 20 (hot/cold streak)
                last_5 = results_buffer[-5:]
                l5_w = sum(1 for r in last_5 if r["won"])
                l5_wpct = l5_w / max(len(last_5), 1)
                l20_wpct = l20_w / max(len(last_20), 1)
                momentum = l5_wpct - l20_wpct

                # Pythagorean
                rs2 = max(total_rs, 1) ** 2
                ra2 = max(total_ra, 1) ** 2
                pythag = rs2 / (rs2 + ra2)

                # Season batting averages
                # Compute cumulative from results_buffer length
                avg = g_h / max(g_ab, 1)
                obp = (g_h + g_bb) / max(g_ab + g_bb, 1)
                slg_num = g_h + g_2b + 2 * g_3b + 3 * g_hr
                slg = slg_num / max(g_ab, 1)

                stats_list.append({
                    "date": game["game_date"],
                    "gp": gp,
                    "wins": wins,
                    "losses": losses,
                    "win_pct": wins / max(gp, 1),
                    "rs_per_game": total_rs / max(gp, 1),
                    "ra_per_game": total_ra / max(gp, 1),
                    "run_diff": total_rs - total_ra,
                    "run_diff_per_game": (total_rs - total_ra) / max(gp, 1),
                    "pythag_wpct": pythag,
                    "luck": (wins / max(gp, 1)) - pythag,
                    # Recent
                    "l10_wins": l10_w,
                    "l10_losses": len(last_10) - l10_w,
                    "l10_run_diff": l10_rd,
                    "l20_wins": l20_w,
                    "l20_losses": len(last_20) - l20_w,
                    "streak": streak,
                    "rs_last_10": sum(r["rs"] for r in last_10) / max(len(last_10), 1),
                    "ra_last_10": sum(r["ra"] for r in last_10) / max(len(last_10), 1),
                    # Game-level offense
                    "g_avg": avg,
                    "g_obp": obp,
                    "g_slg": slg,
                    "g_ops": obp + slg,
                    "g_hr": g_hr,
                    "g_bb": g_bb,
                    "g_so": g_so,
                    "g_sb": g_sb,
                    # Game-level pitching
                    "g_ip": g_ip,
                    "g_er": g_er,
                    "g_ha": g_ha,
                    "g_bba": g_bba,
                    "g_ka": g_ka,
                    "g_era": g_er * 9 / max(g_ip, 0.1),
                    "g_whip": (g_ha + g_bba) / max(g_ip, 0.1),
                    "g_k9": g_ka * 9 / max(g_ip, 0.1),
                    # Home/away
                    "home_wpct": (
                        sum(1 for r in results_buffer if r["is_home"] and r["won"])
                        / max(sum(1 for r in results_buffer if r["is_home"]), 1)
                    ),
                    "away_wpct": (
                        sum(1 for r in results_buffer if not r["is_home"] and r["won"])
                        / max(sum(1 for r in results_buffer if not r["is_home"]), 1)
                    ),
                    # Venue-specific rolling stats (last 20 home/away games)
                    "venue_home_win_pct": (
                        sum(1 for r in home_buffer if r["won"]) / len(home_buffer)
                        if home_buffer else 0.536
                    ),
                    "venue_away_win_pct": (
                        sum(1 for r in away_buffer if r["won"]) / len(away_buffer)
                        if away_buffer else 0.464
                    ),
                    "venue_home_rs_per_game": (
                        sum(r["rs"] for r in home_buffer) / len(home_buffer)
                        if home_buffer else 4.5
                    ),
                    "venue_away_rs_per_game": (
                        sum(r["rs"] for r in away_buffer) / len(away_buffer)
                        if away_buffer else 4.5
                    ),
                    # SP game stats
                    "sp_era": sp_er * 9 / max(sp_ip, 0.1),
                    "sp_whip": (sp_ha + sp_bb) / max(sp_ip, 0.1),
                    "sp_k9": sp_k * 9 / max(sp_ip, 0.1),
                    "sp_bb9": sp_bb * 9 / max(sp_ip, 0.1),
                    "sp_ip": sp_ip,
                    # Bullpen game stats
                    "bp_era": bp_er * 9 / max(bp_ip, 0.1),
                    "bp_whip": (bp_ha + bp_bb) / max(bp_ip, 0.1),
                    "bp_k9": bp_k * 9 / max(bp_ip, 0.1),
                    "bp_bb9": bp_bb * 9 / max(bp_ip, 0.1),
                    # Bullpen fatigue
                    "bp_ip_3d": bp_ip_3d,
                    "bp_games_5d": bp_games_5d,
                    # 7-game run differential
                    "rd_7d": sum(rd_7_buffer) / len(rd_7_buffer) if rd_7_buffer else 0.0,
                    # Exponentially-weighted recent form
                    "ewm_win_pct": ewm_win_pct,
                    "ewm_rs_per_game": ewm_rs_per_game,
                    "ewm_ra_per_game": ewm_ra_per_game,
                    "momentum": momentum,
                })

            team_stats[team_id] = stats_list

        logger.info("Computed rolling stats for %d teams", len(team_stats))
        return team_stats

    def _get_team_features(
        self, stats_list: list[dict], game_date: date
    ) -> dict[str, float] | None:
        """Get the most recent team stats BEFORE game_date.

        Requires at least 15 games of history.
        """
        # Find the latest entry before game_date
        candidates = [s for s in stats_list if s["date"] < game_date]
        if len(candidates) < 15:
            return None

        latest = candidates[-1]

        # Also compute rolling averages from last N entries
        last_7 = candidates[-7:]
        last_14 = candidates[-14:]

        def _roll_avg(entries: list[dict], field: str) -> float:
            vals = [e.get(field, 0) for e in entries]
            return sum(vals) / len(vals) if vals else 0

        features = {
            # Season
            "win_pct": latest["win_pct"],
            "rs_per_game": latest["rs_per_game"],
            "ra_per_game": latest["ra_per_game"],
            "run_diff_pg": latest["run_diff_per_game"],
            "pythag": latest["pythag_wpct"],
            "luck": latest["luck"],
            # Recent form
            "l10_wpct": latest["l10_wins"] / max(latest["l10_wins"] + latest["l10_losses"], 1),
            "l10_rd": latest["l10_run_diff"],
            "l20_wpct": latest["l20_wins"] / max(latest["l20_wins"] + latest["l20_losses"], 1),
            "streak": latest["streak"],
            "rs_last_10": latest["rs_last_10"],
            "ra_last_10": latest["ra_last_10"],
            # Rolling batting (7-game)
            "avg_7": _roll_avg(last_7, "g_avg"),
            "obp_7": _roll_avg(last_7, "g_obp"),
            "slg_7": _roll_avg(last_7, "g_slg"),
            "ops_7": _roll_avg(last_7, "g_ops"),
            "hr_7": _roll_avg(last_7, "g_hr"),
            "bb_7": _roll_avg(last_7, "g_bb"),
            "so_7": _roll_avg(last_7, "g_so"),
            # Rolling batting (14-game)
            "avg_14": _roll_avg(last_14, "g_avg"),
            "obp_14": _roll_avg(last_14, "g_obp"),
            "slg_14": _roll_avg(last_14, "g_slg"),
            "ops_14": _roll_avg(last_14, "g_ops"),
            # Rolling SP pitching (7-game)
            "sp_era_7": _roll_avg(last_7, "sp_era"),
            "sp_whip_7": _roll_avg(last_7, "sp_whip"),
            "sp_k9_7": _roll_avg(last_7, "sp_k9"),
            "sp_bb9_7": _roll_avg(last_7, "sp_bb9"),
            "sp_ip_7": _roll_avg(last_7, "sp_ip"),
            # Rolling SP pitching (14-game)
            "sp_era_14": _roll_avg(last_14, "sp_era"),
            "sp_whip_14": _roll_avg(last_14, "sp_whip"),
            "sp_k9_14": _roll_avg(last_14, "sp_k9"),
            # Rolling bullpen (7-game)
            "bp_era_7": _roll_avg(last_7, "bp_era"),
            "bp_whip_7": _roll_avg(last_7, "bp_whip"),
            "bp_k9_7": _roll_avg(last_7, "bp_k9"),
            "bp_bb9_7": _roll_avg(last_7, "bp_bb9"),
            # Rolling bullpen (14-game)
            "bp_era_14": _roll_avg(last_14, "bp_era"),
            "bp_whip_14": _roll_avg(last_14, "bp_whip"),
            "bp_k9_14": _roll_avg(last_14, "bp_k9"),
            # Bullpen fatigue
            "bp_ip_3d": latest.get("bp_ip_3d", 0.0),
            "bp_games_5d": latest.get("bp_games_5d", 0),
            # 7-game run differential
            "rd_7d": latest.get("rd_7d", 0.0),
            # Exponentially-weighted recent form
            "ewm_win_pct": latest.get("ewm_win_pct", latest.get("win_pct", 0.500)),
            "ewm_rs_per_game": latest.get("ewm_rs_per_game", latest.get("rs_per_game", 4.5)),
            "ewm_ra_per_game": latest.get("ewm_ra_per_game", latest.get("ra_per_game", 4.5)),
            "momentum": latest.get("momentum", 0.0),
            # Defensive proxy: ERA-gap (runs allowed vs pitching-predicted)
            # Positive = defense is costing runs, Negative = defense is saving runs
            "def_proxy": (
                latest.get("ra_per_game", 4.5)
                - _roll_avg(last_14, "g_era")
            ),
            # Home/away
            "home_wpct": latest["home_wpct"],
            "away_wpct": latest["away_wpct"],
            # Venue-specific rolling stats (last 20 home/away games)
            "venue_home_win_pct": latest.get("venue_home_win_pct", 0.536),
            "venue_away_win_pct": latest.get("venue_away_win_pct", 0.464),
            "venue_home_rs_per_game": latest.get("venue_home_rs_per_game", 4.5),
            "venue_away_rs_per_game": latest.get("venue_away_rs_per_game", 4.5),
            # Games played (proxy for sample stability)
            "gp": latest["gp"],
        }

        return features


# ── CLI ───────────────────────────────────────────────────────


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build training data from backfilled CSVs")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=str, default="training_data.parquet")
    args = parser.parse_args()

    builder = TrainingDataBuilder(data_dir=Path(args.data_dir))
    df = builder.build(args.seasons, args.output)
    print(f"\nTraining data: {len(df)} samples, {len(df.columns)} columns")
    print(f"Home win rate: {df['home_win'].mean():.3f}")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")


if __name__ == "__main__":
    main()
