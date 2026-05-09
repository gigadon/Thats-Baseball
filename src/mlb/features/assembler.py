"""Feature assembler — combines all sub-features into a single game feature vector.

Produces a flat dict of ~213 features for model input, keyed by descriptive names.
Each game produces one vector from the perspective of the home team.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mlb.features.bullpen import BullpenFeatures, calculate_bullpen_score
from mlb.features.momentum import (
    DefenseFeatures,
    MatchupFeatures,
    MomentumFeatures,
    calculate_defense_score,
    calculate_momentum_score,
)
from mlb.features.offense import OffenseFeatures, calculate_offense_score
from mlb.features.pitching import (
    PitchingFeatures,
    SPFeatures,
    calculate_pitching_score,
    calculate_sp_index,
)
from mlb.features.stadium import StadiumFeatures, calculate_stadium_factor


@dataclass
class TeamFeatureSet:
    """All features for one side of a matchup."""

    offense: OffenseFeatures
    pitching: PitchingFeatures
    bullpen: BullpenFeatures
    defense: DefenseFeatures
    momentum: MomentumFeatures
    starting_pitcher: SPFeatures


@dataclass
class GameFeatureVector:
    """The full feature vector for a game prediction."""

    game_id: str
    game_date: str
    home_team_id: str
    away_team_id: str

    # Composite scores (0-100)
    home_offense_score: float
    home_pitching_score: float
    home_bullpen_score: float
    home_defense_score: float
    home_momentum_score: float
    home_sp_index: float
    home_power_score: float  # Weighted composite

    away_offense_score: float
    away_pitching_score: float
    away_bullpen_score: float
    away_defense_score: float
    away_momentum_score: float
    away_sp_index: float
    away_power_score: float

    # Differentials (home - away)
    offense_diff: float
    pitching_diff: float
    bullpen_diff: float
    defense_diff: float
    momentum_diff: float
    sp_diff: float
    power_diff: float

    # Stadium
    stadium_factor: float

    # Raw features dict (full ~213 features for model input)
    features: dict[str, float]

    @property
    def feature_names(self) -> list[str]:
        return sorted(self.features.keys())

    def to_array(self) -> np.ndarray:
        """Convert to numpy array with consistent feature ordering."""
        return np.array([self.features[k] for k in self.feature_names])


def _team_power_score(
    offense: float,
    pitching: float,
    defense: float,
    bullpen: float,
    momentum: float,
) -> float:
    """Team Power Score from the architecture spec.

    Team_Power_Score =
        Offense  × 0.30 +
        Pitching × 0.30 +
        Defense  × 0.15 +
        Bullpen  × 0.15 +
        Momentum × 0.10
    """
    return (
        offense * 0.30
        + pitching * 0.30
        + defense * 0.15
        + bullpen * 0.15
        + momentum * 0.10
    )


def _prefix_dict(d: dict, prefix: str) -> dict[str, float]:
    """Add a prefix to all keys in a dict, keeping only numeric values."""
    result = {}
    for k, v in d.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            result[f"{prefix}{k}"] = float(v)
        elif isinstance(v, bool):
            result[f"{prefix}{k}"] = 1.0 if v else 0.0
    return result


def assemble_game_features(
    game_id: str,
    game_date: str,
    home_team_id: str,
    away_team_id: str,
    home: TeamFeatureSet,
    away: TeamFeatureSet,
    matchup: MatchupFeatures,
    stadium: StadiumFeatures,
) -> GameFeatureVector:
    """Assemble all sub-features into a single GameFeatureVector.

    This is the main entry point for feature generation before model prediction.
    """
    # ── Composite scores ──────────────────────────────────
    h_off = calculate_offense_score(home.offense)
    h_pit = calculate_pitching_score(home.pitching)
    h_bp = calculate_bullpen_score(home.bullpen)
    h_def = calculate_defense_score(home.defense)
    h_mom = calculate_momentum_score(home.momentum)
    h_sp = calculate_sp_index(home.starting_pitcher)
    h_power = _team_power_score(h_off, h_pit, h_def, h_bp, h_mom)

    a_off = calculate_offense_score(away.offense)
    a_pit = calculate_pitching_score(away.pitching)
    a_bp = calculate_bullpen_score(away.bullpen)
    a_def = calculate_defense_score(away.defense)
    a_mom = calculate_momentum_score(away.momentum)
    a_sp = calculate_sp_index(away.starting_pitcher)
    a_power = _team_power_score(a_off, a_pit, a_def, a_bp, a_mom)

    stadium_factor = calculate_stadium_factor(stadium)

    # ── Build flat feature dict ───────────────────────────
    features: dict[str, float] = {}

    # Home team raw features
    features.update(_prefix_dict(asdict(home.offense), "h_off_"))
    features.update(_prefix_dict(asdict(home.pitching), "h_pit_"))
    features.update(_prefix_dict(_bullpen_features_dict(home.bullpen), "h_bp_"))
    features.update(_prefix_dict(asdict(home.defense), "h_def_"))
    features.update(_prefix_dict(asdict(home.momentum), "h_mom_"))
    features.update(_prefix_dict(asdict(home.starting_pitcher), "h_sp_"))

    # Away team raw features
    features.update(_prefix_dict(asdict(away.offense), "a_off_"))
    features.update(_prefix_dict(asdict(away.pitching), "a_pit_"))
    features.update(_prefix_dict(_bullpen_features_dict(away.bullpen), "a_bp_"))
    features.update(_prefix_dict(asdict(away.defense), "a_def_"))
    features.update(_prefix_dict(asdict(away.momentum), "a_mom_"))
    features.update(_prefix_dict(asdict(away.starting_pitcher), "a_sp_"))

    # Matchup features
    features.update(_prefix_dict(asdict(matchup), "mu_"))

    # Stadium features
    features.update(_prefix_dict(_stadium_features_dict(stadium), "std_"))

    # Composite scores
    features["h_offense_score"] = h_off
    features["h_pitching_score"] = h_pit
    features["h_bullpen_score"] = h_bp
    features["h_defense_score"] = h_def
    features["h_momentum_score"] = h_mom
    features["h_sp_index"] = h_sp
    features["h_power_score"] = h_power

    features["a_offense_score"] = a_off
    features["a_pitching_score"] = a_pit
    features["a_bullpen_score"] = a_bp
    features["a_defense_score"] = a_def
    features["a_momentum_score"] = a_mom
    features["a_sp_index"] = a_sp
    features["a_power_score"] = a_power

    # Differentials (home - away)
    features["diff_offense"] = h_off - a_off
    features["diff_pitching"] = h_pit - a_pit
    features["diff_bullpen"] = h_bp - a_bp
    features["diff_defense"] = h_def - a_def
    features["diff_momentum"] = h_mom - a_mom
    features["diff_sp"] = h_sp - a_sp
    features["diff_power"] = h_power - a_power

    features["stadium_factor"] = stadium_factor

    # ── Interaction features ──────────────────────────────
    features["sp_vs_opp_offense"] = h_sp - a_off  # Home SP dominance over away offense
    features["opp_sp_vs_offense"] = a_sp - h_off  # Away SP dominance over home offense
    features["bp_advantage_late"] = h_bp - a_bp  # Who has the bullpen edge
    features["total_power"] = h_power + a_power  # Game quality indicator
    features["power_ratio"] = h_power / max(a_power, 1)  # Relative strength

    # Pitching matchup quality
    features["pitching_duel_score"] = (h_sp + a_sp) / 2  # Both SPs good = low scoring
    features["offensive_explosion_risk"] = (h_off + a_off) / 2 - features["pitching_duel_score"]

    # Rest / travel interaction
    features["rest_advantage"] = float(matchup.home_rest_days - matchup.away_rest_days)
    features["travel_fatigue"] = matchup.away_travel_distance_miles / 1000.0

    return GameFeatureVector(
        game_id=game_id,
        game_date=game_date,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_offense_score=h_off,
        home_pitching_score=h_pit,
        home_bullpen_score=h_bp,
        home_defense_score=h_def,
        home_momentum_score=h_mom,
        home_sp_index=h_sp,
        home_power_score=h_power,
        away_offense_score=a_off,
        away_pitching_score=a_pit,
        away_bullpen_score=a_bp,
        away_defense_score=a_def,
        away_momentum_score=a_mom,
        away_sp_index=a_sp,
        away_power_score=a_power,
        offense_diff=h_off - a_off,
        pitching_diff=h_pit - a_pit,
        bullpen_diff=h_bp - a_bp,
        defense_diff=h_def - a_def,
        momentum_diff=h_mom - a_mom,
        sp_diff=h_sp - a_sp,
        power_diff=h_power - a_power,
        stadium_factor=stadium_factor,
        features=features,
    )


def _bullpen_features_dict(bp: BullpenFeatures) -> dict[str, Any]:
    """Extract numeric bullpen features, excluding the relievers list."""
    d = asdict(bp)
    d.pop("relievers", None)
    return d


def _stadium_features_dict(sf: StadiumFeatures) -> dict[str, Any]:
    """Extract stadium features as a dict."""
    return asdict(sf)
