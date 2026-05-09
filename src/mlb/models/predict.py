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
        self.runs_model: TrainingPipeline | None = None

    def load(self):
        """Load trained models from disk."""
        self.pipeline = TrainingPipeline()
        self.pipeline.load(self.model_dir / "win_model")

        # Run prediction model (optional — may not exist yet)
        runs_path = self.model_dir / "runs_model"
        if runs_path.exists():
            self.runs_model = TrainingPipeline()
            self.runs_model.load(runs_path)

        logger.info("Prediction service loaded")

    def predict_game(self, game_fv: GameFeatureVector) -> GamePrediction:
        """Generate a prediction for a single game."""
        if self.pipeline is None:
            raise RuntimeError("Call load() first")

        # Build single-row DataFrame
        features_df = pd.DataFrame([game_fv.features])

        # Get detailed predictions from each model
        detailed = self.pipeline.predict_detailed(features_df)

        home_win_prob = float(detailed["ensemble"][0])
        model_preds = {
            name: float(vals[0])
            for name, vals in detailed.items()
            if name not in ("ensemble", "std")
        }

        # Model agreement (1 - normalized std)
        model_std = float(detailed["std"][0])
        agreement = max(0.0, 1.0 - model_std * 4)  # Scale: 0.25 std → 0 agreement

        # Confidence: based on distance from 0.5 and model agreement
        edge = abs(home_win_prob - 0.5)
        confidence = min(100.0, (edge * 150 + agreement * 30))

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

        If a runs model is trained, use it. Otherwise, estimate from
        offensive features and win probability.
        """
        if self.runs_model is not None:
            features_df = pd.DataFrame([game_fv.features])
            # Runs model would predict total runs; split by win prob
            total = float(self.runs_model.predict(features_df)[0])
            home_share = 0.45 + home_win_prob * 0.10  # Winner scores slightly more
            return total * home_share, total * (1 - home_share)

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

        # Nudge toward win probability implication
        if home_win_prob > 0.5:
            home_runs += (home_win_prob - 0.5) * 0.5
            away_runs -= (home_win_prob - 0.5) * 0.3
        else:
            away_runs += (0.5 - home_win_prob) * 0.5
            home_runs -= (0.5 - home_win_prob) * 0.3

        return max(0.5, home_runs), max(0.5, away_runs)

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
