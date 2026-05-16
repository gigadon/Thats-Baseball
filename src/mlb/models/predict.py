"""Prediction service — generates game predictions from trained models.

Takes a GameFeatureVector, runs it through the pipeline, and produces
win probability, predicted runs, and confidence scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mlb.features.assembler import GameFeatureVector
from mlb.models.pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


@dataclass
class GamePrediction:
    """Complete prediction for a single game."""

    game_id: str
    game_date: str
    home_team_id: str
    away_team_id: str

    # Core predictions
    home_win_prob: float
    away_win_prob: float
    predicted_home_runs: float
    predicted_away_runs: float
    predicted_total: float

    # Confidence
    confidence: float  # 0-100
    model_agreement: float  # 0-1, how much models agree

    # Per-model breakdown
    model_predictions: dict[str, float]

    # Key factors
    top_factors: list[tuple[str, float]]  # (feature_name, contribution)

    # Power scores
    home_power_score: float
    away_power_score: float

    @property
    def predicted_winner(self) -> str:
        return self.home_team_id if self.home_win_prob > 0.5 else self.away_team_id

    @property
    def predicted_margin(self) -> float:
        return abs(self.predicted_home_runs - self.predicted_away_runs)

    @property
    def edge(self) -> float:
        """How far the prediction is from 50/50."""
        return abs(self.home_win_prob - 0.5)


class PredictionService:
    """Generates game predictions using trained models."""

    def __init__(self, model_dir: str | Path = "models"):
        self.model_dir = Path(model_dir)
        self.pipeline: TrainingPipeline | None = None
        # Runs regression model (loaded from joblib artifacts)
        self._runs_regressor: object | None = None
        self._runs_scaler: object | None = None
        self._runs_feature_names: list[str] | None = None

    def load(self):
        """Load trained models from disk."""
        import joblib

        self.pipeline = TrainingPipeline()
        self.pipeline.load(self.model_dir / "win_model")

        # Runs regression model (optional — may not exist yet)
        runs_path = self.model_dir / "runs_model"
        regressor_file = runs_path / "runs_regressor.joblib"
        if regressor_file.exists():
            try:
                self._runs_regressor = joblib.load(regressor_file)
                self._runs_scaler = joblib.load(runs_path / "runs_scaler.joblib")
                self._runs_feature_names = joblib.load(runs_path / "runs_feature_names.joblib")
                logger.info("Runs regression model loaded from %s", runs_path)
            except Exception as e:
                logger.warning("Failed to load runs model: %s", e)
                self._runs_regressor = None

        logger.info("Prediction service loaded")

    def predict_game(self, game_fv: GameFeatureVector) -> GamePrediction:
        """Generate a prediction for a single game."""
        if self.pipeline is None:
            raise RuntimeError("Call load() first")

        # Build single-row DataFrame
        features_df = pd.DataFrame([game_fv.features])

        # Get detailed predictions from each model
        detailed = self.pipeline.predict_detailed(features_df)

        raw_home_prob = float(detailed["ensemble"][0])

        # Post-hoc calibration: backtest showed the model undervalues home-field
        # advantage — overconfident on away picks, underconfident on home picks.
        # Shift predictions toward home slightly (historical home win rate ~53.5%).
        # The adjustment is stronger near 0.5 (where calibration error is worst)
        # and tapers off at extremes where the model is already more accurate.
        home_bias = 0.015  # ~1.5% nudge toward home
        distance_from_center = abs(raw_home_prob - 0.5)
        taper = max(0.0, 1.0 - distance_from_center * 4)  # Full effect at 50%, zero at 75%+
        home_win_prob = raw_home_prob + home_bias * taper
        home_win_prob = max(0.01, min(0.99, home_win_prob))

        model_preds = {
            name: float(vals[0])
            for name, vals in detailed.items()
            if name not in ("ensemble", "std")
        }

        # Model agreement (1 - normalized std)
        model_std = float(detailed["std"][0])
        agreement = max(0.0, 1.0 - model_std * 4)  # Scale: 0.25 std → 0 agreement

        # Confidence: combines prediction strength, model agreement, and data quality.
        # In MLB, even top teams win ~60%, so a 60% prediction IS very confident.
        edge = abs(home_win_prob - 0.5)

        # 1) Prediction strength (0-50 pts): rescaled so 55%→15, 60%→30, 65%→42, 70%→50
        strength = min(50.0, edge * 300)

        # 2) Model agreement (0-25 pts): all 6 models agree → 25
        agreement_pts = agreement * 25

        # 3) Data quality (0-25 pts): do we have real SP stats, weather, odds, lineup?
        fv = game_fv.features
        dq = 0.0
        # SP data available (not defaults): +8 if both SPs have real IP > 0
        if fv.get("h_sp_season_ip", 0) > 0:
            dq += 4
        if fv.get("a_sp_season_ip", 0) > 0:
            dq += 4
        # Market odds available (not default 0.5): +6
        if fv.get("market_home_prob", 0.5) != 0.5:
            dq += 6
        # Weather data available (not default 72): +3
        if fv.get("temperature", 72.0) != 72.0 or fv.get("is_outdoor", 0) == 0:
            dq += 3
        # Lineup/BvP data (not defaults): +4
        if fv.get("h_bvp_ops", 0.75) != 0.75:
            dq += 4

        confidence = min(100.0, strength + agreement_pts + dq)

        # Run predictions
        home_runs, away_runs = self._predict_runs(game_fv, home_win_prob)

        # Top contributing factors
        top_factors = self._get_top_factors(game_fv)

        return GamePrediction(
            game_id=game_fv.game_id,
            game_date=game_fv.game_date,
            home_team_id=game_fv.home_team_id,
            away_team_id=game_fv.away_team_id,
            home_win_prob=round(home_win_prob, 4),
            away_win_prob=round(1 - home_win_prob, 4),
            predicted_home_runs=round(home_runs, 1),
            predicted_away_runs=round(away_runs, 1),
            predicted_total=round(home_runs + away_runs, 1),
            confidence=round(confidence, 1),
            model_agreement=round(agreement, 3),
            model_predictions=model_preds,
            top_factors=top_factors,
            home_power_score=game_fv.home_power_score,
            away_power_score=game_fv.away_power_score,
        )

    def predict_slate(self, games: list[GameFeatureVector]) -> list[GamePrediction]:
        """Predict a full slate of games."""
        return [self.predict_game(g) for g in games]

    def _predict_runs(
        self, game_fv: GameFeatureVector, home_win_prob: float
    ) -> tuple[float, float]:
        """Predict runs for each team.

        If a runs regression model is loaded, use it for differentiated
        totals (Coors games 10+, pitcher duels 6-7, etc.) and split by
        win probability.  Falls back to a heuristic if no model exists.
        """
        if self._runs_regressor is not None and self._runs_feature_names is not None:
            # Build a feature row matching the training feature set
            fv = game_fv.features
            row = {col: fv.get(col, 0.0) for col in self._runs_feature_names}
            X = np.array([[row[c] for c in self._runs_feature_names]], dtype=float)
            X = np.nan_to_num(X, nan=0.0)

            if self._runs_scaler is not None:
                X = self._runs_scaler.transform(X)

            total = float(self._runs_regressor.predict(X)[0])
            # Clamp to reasonable MLB range
            total = max(3.0, min(total, 30.0))

            # Split home/away: predicted winner gets a larger share.
            # At 50/50 the split is 45/55-ish (slight away edge),
            # each 10% of win-prob shifts ~3% of runs.
            home_share = 0.45 + (home_win_prob - 0.5) * 0.3
            home_share = max(0.30, min(home_share, 0.70))

            home_runs = total * home_share
            away_runs = total * (1 - home_share)
            return round(max(0.5, home_runs), 1), round(max(0.5, away_runs), 1)

        # Heuristic estimation from feature scores
        # Average MLB game: ~4.5 runs per team, ~9 total
        base_rpg = 4.5

        # Adjust by offensive/pitching scores (0-100 scale, 50 = average)
        home_off_adj = (game_fv.home_offense_score - 50) / 100  # ±0.50
        home_pit_opp = (50 - game_fv.away_pitching_score) / 100  # Opponent pitching
        away_off_adj = (game_fv.away_offense_score - 50) / 100
        away_pit_opp = (50 - game_fv.home_pitching_score) / 100

        # Stadium factor
        stadium_adj = (game_fv.stadium_factor - 1.0) * 2  # ±0.10 typically

        home_runs = max(0.5, base_rpg + home_off_adj + home_pit_opp + stadium_adj)
        away_runs = max(0.5, base_rpg + away_off_adj + away_pit_opp + stadium_adj)

        # Blend with win probability so runs always agree with the predicted winner.
        # The model's win prob is authoritative; scale the run margin to match.
        edge = home_win_prob - 0.5  # positive = home favored
        prob_margin = edge * 3.0  # ±1.5 runs at extremes (e.g. 60% → +0.3)
        heuristic_margin = home_runs - away_runs
        # Weighted blend: 40% heuristic, 60% win-prob-implied margin
        target_margin = heuristic_margin * 0.4 + prob_margin * 0.6
        midpoint = (home_runs + away_runs) / 2
        home_runs = max(0.5, midpoint + target_margin / 2)
        away_runs = max(0.5, midpoint - target_margin / 2)

        # Final safety: if runs still disagree with win prob, force consistency
        if home_win_prob > 0.5 and home_runs <= away_runs:
            avg = (home_runs + away_runs) / 2
            home_runs = avg + 0.1
            away_runs = avg - 0.1
        elif home_win_prob < 0.5 and away_runs <= home_runs:
            avg = (home_runs + away_runs) / 2
            away_runs = avg + 0.1
            home_runs = avg - 0.1

        return round(max(0.5, home_runs), 1), round(max(0.5, away_runs), 1)

    def _get_top_factors(
        self, game_fv: GameFeatureVector, n: int = 10
    ) -> list[tuple[str, float]]:
        """Identify top factors driving the prediction.

        Uses feature importance from ensemble weighted by feature values.
        """
        if self.pipeline is None or self.pipeline.ensemble is None:
            return []

        importance = self.pipeline.ensemble.feature_importance()
        if not importance:
            return []

        # Score each feature by importance × deviation from neutral
        scored = []
        for feat, imp in importance.items():
            val = game_fv.features.get(feat, 0.0)
            # Features centered around 0 after scaling; larger magnitude = more impact
            contribution = imp * abs(val)
            scored.append((feat, round(contribution, 4)))

        scored.sort(key=lambda x: -x[1])
        return scored[:n]
