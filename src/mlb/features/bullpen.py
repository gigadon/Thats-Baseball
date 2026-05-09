"""Bullpen aggregate score with fatigue tracking.

Bullpen_Score =
    Closer_Score   × 0.30 +
    Setup_Score    × 0.25 +
    Middle_Score   × 0.20 +
    Depth_Score    × 0.15 +
    Fatigue_Score  × 0.10
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RelieverProfile:
    """Stats for a single reliever."""

    player_id: int
    role: str  # "closer", "setup", "middle", "long"
    era: float
    fip: float
    k_per_9: float
    bb_per_9: float
    whip: float
    gb_pct: float
    leverage_index: float  # Average leverage of situations faced
    holds: int
    saves: int
    blown_saves: int

    # Fatigue inputs
    pitches_last_3_days: int
    appearances_last_7_days: int
    innings_last_7_days: float
    rest_days: int  # Days since last appearance
    season_innings: float
    season_appearances: int


@dataclass
class BullpenFeatures:
    """Aggregate bullpen features for a team."""

    # Role group scores (0-100)
    closer_score: float
    setup_score: float
    middle_relief_score: float
    depth_score: float
    fatigue_score: float

    # Aggregate stats
    bullpen_era: float
    bullpen_fip: float
    bullpen_k_per_9: float
    bullpen_bb_per_9: float
    bullpen_whip: float
    bullpen_gb_pct: float

    # Availability
    relievers_available: int  # Number with adequate rest
    high_leverage_available: int  # Closer + setup available
    total_relievers: int

    # Usage patterns
    avg_pitches_per_appearance: float
    bullpen_innings_last_3: float
    bullpen_innings_last_7: float

    # Individual top-line
    closer_era: float
    closer_saves: int
    closer_blown_saves: int

    relievers: list[RelieverProfile] = field(default_factory=list)


def _score_reliever(r: RelieverProfile) -> float:
    """Score an individual reliever on a 0-100 scale."""
    # FIP (lower = better)
    fip_norm = np.clip((5.5 - r.fip) / 3.5, 0, 1) * 100

    # K/9 (higher = better)
    k_norm = np.clip(r.k_per_9 / 15.0, 0, 1) * 100

    # BB/9 (lower = better)
    bb_norm = np.clip((6.0 - r.bb_per_9) / 6.0, 0, 1) * 100

    # GB% (higher = better)
    gb_norm = np.clip((r.gb_pct - 0.25) / 0.35, 0, 1) * 100

    return float(
        fip_norm * 0.35
        + k_norm * 0.30
        + bb_norm * 0.20
        + gb_norm * 0.15
    )


def _fatigue_factor(r: RelieverProfile) -> float:
    """Calculate fatigue factor for a reliever (0-1, where 1 = fully rested)."""
    # Penalize recent heavy usage
    rest_bonus = min(r.rest_days / 3.0, 1.0)  # Full rest at 3+ days

    # Penalize high pitch count in last 3 days
    pitch_penalty = max(0, 1.0 - (r.pitches_last_3_days / 60.0))

    # Penalize many appearances in last 7 days
    app_penalty = max(0, 1.0 - (r.appearances_last_7_days / 5.0))

    # Season workload (penalize as relievers approach ~70+ IP)
    workload = max(0, 1.0 - max(0, r.season_innings - 50) / 30.0)

    return float(
        rest_bonus * 0.35
        + pitch_penalty * 0.30
        + app_penalty * 0.20
        + workload * 0.15
    )


def calculate_bullpen_score(f: BullpenFeatures) -> float:
    """Weighted composite bullpen score (0-100)."""
    return float(np.clip(
        f.closer_score * 0.30
        + f.setup_score * 0.25
        + f.middle_relief_score * 0.20
        + f.depth_score * 0.15
        + f.fatigue_score * 0.10,
        0,
        100,
    ))


def compute_bullpen_features(relievers: list[RelieverProfile]) -> BullpenFeatures:
    """Compute aggregate bullpen features from individual reliever profiles."""
    if not relievers:
        return _empty_bullpen_features()

    # Group by role
    closers = [r for r in relievers if r.role == "closer"]
    setups = [r for r in relievers if r.role == "setup"]
    middles = [r for r in relievers if r.role == "middle"]
    longs = [r for r in relievers if r.role == "long"]

    def _group_score(group: list[RelieverProfile]) -> float:
        if not group:
            return 50.0  # Neutral if no one in role
        scores = [_score_reliever(r) for r in group]
        return float(np.mean(scores))

    closer_score = _group_score(closers)
    setup_score = _group_score(setups)
    middle_score = _group_score(middles)

    # Depth: how many quality relievers (score > 50)?
    all_scores = [_score_reliever(r) for r in relievers]
    quality_count = sum(1 for s in all_scores if s > 50)
    depth_score = float(np.clip(quality_count / 5.0, 0, 1) * 100)

    # Fatigue: average freshness across the pen
    fatigue_factors = [_fatigue_factor(r) for r in relievers]
    fatigue_score = float(np.mean(fatigue_factors) * 100)

    # Availability
    available = [r for r in relievers if r.rest_days >= 1 or r.pitches_last_3_days < 30]
    hl_available = [r for r in available if r.role in ("closer", "setup")]

    # Aggregates
    def _wavg(vals: list[float], weights: list[float]) -> float:
        if not vals:
            return 0.0
        total_w = sum(weights)
        if total_w == 0:
            return float(np.mean(vals))
        return sum(v * w for v, w in zip(vals, weights)) / total_w

    ips = [r.season_innings for r in relievers]
    bp_era = _wavg([r.era for r in relievers], ips)
    bp_fip = _wavg([r.fip for r in relievers], ips)
    bp_k9 = _wavg([r.k_per_9 for r in relievers], ips)
    bp_bb9 = _wavg([r.bb_per_9 for r in relievers], ips)
    bp_whip = _wavg([r.whip for r in relievers], ips)
    bp_gb = _wavg([r.gb_pct for r in relievers], ips)

    primary_closer = closers[0] if closers else None

    return BullpenFeatures(
        closer_score=closer_score,
        setup_score=setup_score,
        middle_relief_score=middle_score,
        depth_score=depth_score,
        fatigue_score=fatigue_score,
        bullpen_era=bp_era,
        bullpen_fip=bp_fip,
        bullpen_k_per_9=bp_k9,
        bullpen_bb_per_9=bp_bb9,
        bullpen_whip=bp_whip,
        bullpen_gb_pct=bp_gb,
        relievers_available=len(available),
        high_leverage_available=len(hl_available),
        total_relievers=len(relievers),
        avg_pitches_per_appearance=(
            sum(r.pitches_last_3_days for r in relievers)
            / max(sum(r.appearances_last_7_days for r in relievers), 1)
        ),
        bullpen_innings_last_3=sum(r.innings_last_7_days * 3 / 7 for r in relievers),
        bullpen_innings_last_7=sum(r.innings_last_7_days for r in relievers),
        closer_era=primary_closer.era if primary_closer else 0.0,
        closer_saves=primary_closer.saves if primary_closer else 0,
        closer_blown_saves=primary_closer.blown_saves if primary_closer else 0,
        relievers=relievers,
    )


def _empty_bullpen_features() -> BullpenFeatures:
    return BullpenFeatures(
        closer_score=50.0,
        setup_score=50.0,
        middle_relief_score=50.0,
        depth_score=0.0,
        fatigue_score=50.0,
        bullpen_era=4.50,
        bullpen_fip=4.50,
        bullpen_k_per_9=8.0,
        bullpen_bb_per_9=3.5,
        bullpen_whip=1.35,
        bullpen_gb_pct=0.43,
        relievers_available=0,
        high_leverage_available=0,
        total_relievers=0,
        avg_pitches_per_appearance=15.0,
        bullpen_innings_last_3=0.0,
        bullpen_innings_last_7=0.0,
        closer_era=0.0,
        closer_saves=0,
        closer_blown_saves=0,
    )
