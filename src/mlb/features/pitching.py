"""Team pitching feature calculations.

Produces a Pitching Score from weighted components:
  FIP, xFIP, SIERA, K%, BB%

Also provides the Starting Pitcher Index for individual matchups.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ─── Team Pitching Features ───────────────────────────────────


@dataclass
class PitchingFeatures:
    """Aggregate pitching features for a team on a given date."""

    # Core metrics
    era: float
    fip: float
    xfip: float
    siera: float  # Skill-Interactive ERA (if available, else xFIP)
    whip: float

    # Rate stats
    k_per_9: float
    bb_per_9: float
    hr_per_9: float
    k_pct: float  # K / batters faced
    bb_pct: float  # BB / batters faced
    k_minus_bb_pct: float  # K% - BB% (elite predictor)

    # Batted ball
    ground_ball_pct: float
    fly_ball_pct: float
    hard_hit_pct: float
    barrel_pct: float

    # Quality starts
    qs_pct: float  # % of starts that are quality starts

    # Rolling windows
    era_last_14: float
    fip_last_14: float
    k_per_9_last_14: float
    bb_per_9_last_14: float


def calculate_pitching_score(f: PitchingFeatures) -> float:
    """Combine pitching features into a single 0-100 score.

    Weights:
      FIP            25%  — best ERA predictor
      xFIP           20%  — stabilized FIP
      K-BB%          20%  — strikeout minus walk rate
      Hard hit %     10%  — contact quality allowed (inverted)
      GB%            10%  — ground ball tendency
      Recent trend   15%  — last-14 FIP
    """
    # FIP: lower is better. League avg ~4.00. Scale: 2.0 = 100, 6.0 = 0
    fip_norm = np.clip((6.0 - f.fip) / 4.0, 0, 1) * 100

    # xFIP: same scale as FIP
    xfip_norm = np.clip((6.0 - f.xfip) / 4.0, 0, 1) * 100

    # K-BB%: higher is better. League avg ~10-12%. Elite ~20%+
    kbb_norm = np.clip(f.k_minus_bb_pct / 0.25, 0, 1) * 100

    # Hard hit % allowed: lower is better. Avg ~35-38%
    hh_norm = np.clip((0.50 - f.hard_hit_pct) / 0.25, 0, 1) * 100

    # GB%: higher is generally better. Avg ~43-45%
    gb_norm = np.clip((f.ground_ball_pct - 0.30) / 0.25, 0, 1) * 100

    # Recent FIP trend
    trend_norm = np.clip((6.0 - f.fip_last_14) / 4.0, 0, 1) * 100

    score = (
        fip_norm * 0.25
        + xfip_norm * 0.20
        + kbb_norm * 0.20
        + hh_norm * 0.10
        + gb_norm * 0.10
        + trend_norm * 0.15
    )
    return float(np.clip(score, 0, 100))


def compute_pitching_features_from_stats(
    season_stats: dict,
    recent_games: list[dict],
) -> PitchingFeatures:
    """Build PitchingFeatures from raw season stats and recent game logs."""
    era = float(season_stats.get("era", 4.50) or 4.50)
    fip = float(season_stats.get("fip", 4.50) or 4.50)
    xfip = float(season_stats.get("xfip", 4.50) or 4.50)
    whip = float(season_stats.get("whip", 1.30) or 1.30)
    k9 = float(season_stats.get("k9", 8.0) or 8.0)
    bb9 = float(season_stats.get("bb9", 3.0) or 3.0)
    hr9 = float(season_stats.get("hr9", 1.2) or 1.2)
    hard_hit = float(season_stats.get("hard_hit_pct", 0.36) or 0.36)
    barrel = float(season_stats.get("barrel_pct", 0.07) or 0.07)
    gb_pct = float(season_stats.get("ground_ball_pct", 0.43) or 0.43)

    # Estimate K% and BB% from per-9 rates (roughly: rate / 9 * ~4.1 PA/IP)
    k_pct = min(k9 / 9.0 * 0.95, 0.45)  # approximate
    bb_pct = min(bb9 / 9.0 * 0.95, 0.20)

    def _avg_field(games: list[dict], field: str, n: int) -> float:
        subset = games[-n:] if len(games) >= n else games
        if not subset:
            return 0.0
        vals = [float(g.get(field, 0) or 0) for g in subset]
        return sum(vals) / len(vals)

    era_14 = _avg_field(recent_games, "era", 14) or era
    fip_14 = _avg_field(recent_games, "fip", 14) or fip
    k9_14 = _avg_field(recent_games, "k9", 14) or k9
    bb9_14 = _avg_field(recent_games, "bb9", 14) or bb9

    return PitchingFeatures(
        era=era,
        fip=fip,
        xfip=xfip,
        siera=xfip,  # SIERA not in base stats; fallback to xFIP
        whip=whip,
        k_per_9=k9,
        bb_per_9=bb9,
        hr_per_9=hr9,
        k_pct=k_pct,
        bb_pct=bb_pct,
        k_minus_bb_pct=k_pct - bb_pct,
        ground_ball_pct=gb_pct,
        fly_ball_pct=1.0 - gb_pct - 0.20,  # rough: 1 - GB% - ~20% line drives
        hard_hit_pct=hard_hit,
        barrel_pct=barrel,
        qs_pct=0.0,  # populated when start-level data available
        era_last_14=era_14,
        fip_last_14=fip_14,
        k_per_9_last_14=k9_14,
        bb_per_9_last_14=bb9_14,
    )


# ─── Starting Pitcher Index ───────────────────────────────────


@dataclass
class SPFeatures:
    """Features for an individual starting pitcher matchup."""

    # Stuff metrics
    velocity_avg: float  # Average fastball velocity
    spin_rate_avg: float  # Average spin rate
    whiff_pct: float  # Swing-and-miss rate
    stuff_plus: float  # Stuff+ (100 = avg, if available)

    # Command metrics
    zone_pct: float  # % pitches in zone
    edge_pct: float  # % pitches on edge of zone
    bb_per_9: float
    command_plus: float  # Command+ (100 = avg, if available)

    # Recent performance
    recent_fip: float  # FIP over last 5 starts
    recent_era: float  # ERA over last 5 starts

    # Batted ball
    gb_rate: float
    hard_hit_pct: float

    # Context
    park_adjusted_era: float  # ERA adjusted for park
    opponent_wrc_plus: float  # Opposing team's wRC+
    innings_per_start: float  # Average IP per start
    pitches_per_start: float  # Average pitches per start

    # Platoon
    vs_lhb_woba: float  # wOBA against left-handed batters
    vs_rhb_woba: float  # wOBA against right-handed batters

    # Season workload
    season_innings: float
    season_starts: int
    days_rest: int


def calculate_sp_index(f: SPFeatures) -> float:
    """Calculate Starting Pitcher Index (0-100).

    SP_Index =
        Stuff+      × 0.25 +
        Command+    × 0.25 +
        Recent_FIP  × 0.20 +
        GB_Rate     × 0.10 +
        Park_ERA    × 0.10 +
        Opp_Quality × 0.10
    """
    # Stuff+ (100 = avg → 50 on our scale)
    stuff_norm = np.clip((f.stuff_plus - 50) / 100, 0, 1) * 100

    # Command+ same scale
    cmd_norm = np.clip((f.command_plus - 50) / 100, 0, 1) * 100

    # Recent FIP (lower = better, same scale as team pitching)
    fip_norm = np.clip((6.0 - f.recent_fip) / 4.0, 0, 1) * 100

    # GB rate (higher = better)
    gb_norm = np.clip((f.gb_rate - 0.30) / 0.30, 0, 1) * 100

    # Park-adjusted ERA (lower = better)
    park_era_norm = np.clip((6.0 - f.park_adjusted_era) / 4.0, 0, 1) * 100

    # Opponent quality: lower opposing wRC+ = easier matchup = higher score
    opp_norm = np.clip((150 - f.opponent_wrc_plus) / 100, 0, 1) * 100

    index = (
        stuff_norm * 0.25
        + cmd_norm * 0.25
        + fip_norm * 0.20
        + gb_norm * 0.10
        + park_era_norm * 0.10
        + opp_norm * 0.10
    )
    return float(np.clip(index, 0, 100))


def compute_sp_features_from_stats(
    pitcher_stats: dict,
    recent_starts: list[dict],
    opponent_wrc_plus: float,
    park_factor: float,
    days_rest: int,
) -> SPFeatures:
    """Build SPFeatures from pitcher stats, recent starts, and context."""
    era = float(pitcher_stats.get("era", 4.50) or 4.50)
    fip = float(pitcher_stats.get("fip", 4.50) or 4.50)
    bb9 = float(pitcher_stats.get("bb_per_9", 3.0) or 3.0)
    ip = float(pitcher_stats.get("innings_pitched", 0) or 0)
    starts = int(pitcher_stats.get("games_started", 0) or 0)
    gb = float(pitcher_stats.get("ground_ball_pct", 0.43) or 0.43)
    hh = float(pitcher_stats.get("hard_hit_pct", 0.36) or 0.36)

    # Stuff+ and Command+ default to 100 (league average) if not available
    stuff_plus = float(pitcher_stats.get("stuff_plus", 100) or 100)
    command_plus = float(pitcher_stats.get("command_plus", 100) or 100)

    # Recent performance from last 5 starts
    def _avg(games: list[dict], field: str) -> float:
        vals = [float(g.get(field, 0) or 0) for g in games]
        return sum(vals) / len(vals) if vals else 0.0

    recent_fip = _avg(recent_starts[-5:], "fip") if recent_starts else fip
    recent_era = _avg(recent_starts[-5:], "era") if recent_starts else era

    # Park-adjust ERA
    pf = max(park_factor, 80) / 100.0  # park factor as multiplier
    park_adj_era = era / pf if pf > 0 else era

    ip_per_start = (ip / starts) if starts > 0 else 5.0

    return SPFeatures(
        velocity_avg=float(pitcher_stats.get("velocity_avg", 92.0) or 92.0),
        spin_rate_avg=float(pitcher_stats.get("spin_rate_avg", 2200) or 2200),
        whiff_pct=float(pitcher_stats.get("whiff_pct", 0.24) or 0.24),
        stuff_plus=stuff_plus,
        zone_pct=float(pitcher_stats.get("zone_pct", 0.45) or 0.45),
        edge_pct=float(pitcher_stats.get("edge_pct", 0.18) or 0.18),
        bb_per_9=bb9,
        command_plus=command_plus,
        recent_fip=recent_fip,
        recent_era=recent_era,
        gb_rate=gb,
        hard_hit_pct=hh,
        park_adjusted_era=park_adj_era,
        opponent_wrc_plus=opponent_wrc_plus,
        innings_per_start=ip_per_start,
        pitches_per_start=float(pitcher_stats.get("pitches_per_start", 90) or 90),
        vs_lhb_woba=float(pitcher_stats.get("vs_lhb_woba", 0.320) or 0.320),
        vs_rhb_woba=float(pitcher_stats.get("vs_rhb_woba", 0.310) or 0.310),
        season_innings=ip,
        season_starts=starts,
        days_rest=days_rest,
    )
