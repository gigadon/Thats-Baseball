"""Team offensive feature calculations.

Produces an Offense Score from weighted components:
  wRC+, OBP, SLG, ISO, BsR (baserunning runs)

All inputs come from team_daily_stats or computed from player_game_stats.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OffenseFeatures:
    """Raw offensive features for a team on a given date."""

    # Core rate stats (season-to-date)
    batting_avg: float
    obp: float
    slg: float
    ops: float
    woba: float
    wrc_plus: float  # 100 = league average
    iso: float  # Isolated power = SLG - AVG

    # Counting / rate
    runs_per_game: float
    home_runs_per_game: float
    stolen_bases_per_game: float
    walk_rate: float  # BB / PA
    strikeout_rate: float  # K / PA
    babip: float  # Batting avg on balls in play

    # Quality of contact
    hard_hit_pct: float
    barrel_pct: float
    ground_ball_pct: float

    # Baserunning composite (BsR proxy)
    baserunning_score: float

    # Clutch / situational
    risp_avg: float  # AVG with runners in scoring position
    leverage_ops: float  # OPS in high-leverage situations

    # Rolling windows
    runs_per_game_last_7: float
    runs_per_game_last_14: float
    runs_per_game_last_30: float
    ops_last_14: float
    wrc_plus_last_14: float


def calculate_offense_score(f: OffenseFeatures) -> float:
    """Combine offensive features into a single 0-100 score.

    Weights:
      wRC+          25%  — best single offensive metric
      OBP           15%  — on-base ability
      SLG           15%  — power
      ISO           10%  — isolated power
      Baserunning   10%  — stolen bases + advancement
      Hard hit %    10%  — contact quality
      Recent trend  15%  — momentum (last-14 wRC+)
    """
    # Normalize wRC+ (100 = average → 50 on our scale)
    wrc_norm = np.clip((f.wrc_plus - 50) / 100, 0, 1) * 100

    # Normalize OBP (league avg ~.310-.320)
    obp_norm = np.clip((f.obp - 0.200) / 0.200, 0, 1) * 100

    # Normalize SLG (league avg ~.390-.410)
    slg_norm = np.clip((f.slg - 0.250) / 0.300, 0, 1) * 100

    # Normalize ISO (league avg ~.140-.160)
    iso_norm = np.clip(f.iso / 0.300, 0, 1) * 100

    # Baserunning (proxy: SB/game + some base advancement)
    bsr_norm = np.clip(f.baserunning_score / 5.0, 0, 1) * 100

    # Hard hit % (league avg ~35-38%)
    hh_norm = np.clip((f.hard_hit_pct - 0.20) / 0.30, 0, 1) * 100

    # Recent wRC+ trend
    trend_norm = np.clip((f.wrc_plus_last_14 - 50) / 100, 0, 1) * 100

    score = (
        wrc_norm * 0.25
        + obp_norm * 0.15
        + slg_norm * 0.15
        + iso_norm * 0.10
        + bsr_norm * 0.10
        + hh_norm * 0.10
        + trend_norm * 0.15
    )
    return float(np.clip(score, 0, 100))


def compute_offense_features_from_stats(
    season_stats: dict,
    recent_games: list[dict],
    games_played: int,
) -> OffenseFeatures:
    """Build OffenseFeatures from raw season stats and recent game logs.

    Args:
        season_stats: Aggregate season stats from team_daily_stats.
        recent_games: Last 30 game dicts with runs_scored, ops, etc.
        games_played: Number of games played this season.
    """
    gp = max(games_played, 1)

    avg = float(season_stats.get("batting_avg", 0) or 0)
    obp = float(season_stats.get("obp", 0) or 0)
    slg = float(season_stats.get("slg", 0) or 0)
    ops = float(season_stats.get("ops", 0) or 0)
    woba = float(season_stats.get("woba", 0) or 0)
    wrc_plus = float(season_stats.get("wrc_plus", 100) or 100)
    total_runs = float(season_stats.get("runs_scored", 0) or 0)
    total_hr = float(season_stats.get("home_runs", 0) or 0)
    total_sb = float(season_stats.get("stolen_bases", 0) or 0)
    hard_hit = float(season_stats.get("hard_hit_pct", 0.35) or 0.35)
    barrel = float(season_stats.get("barrel_pct", 0.06) or 0.06)
    gb_pct = float(season_stats.get("ground_ball_pct", 0.43) or 0.43)

    iso = slg - avg

    # Rolling windows from recent games
    def _avg_field(games: list[dict], field: str, n: int) -> float:
        subset = games[-n:] if len(games) >= n else games
        if not subset:
            return 0.0
        vals = [float(g.get(field, 0) or 0) for g in subset]
        return sum(vals) / len(vals)

    rpg_7 = _avg_field(recent_games, "runs_scored", 7)
    rpg_14 = _avg_field(recent_games, "runs_scored", 14)
    rpg_30 = _avg_field(recent_games, "runs_scored", 30)
    ops_14 = _avg_field(recent_games, "ops", 14)
    wrc_14 = _avg_field(recent_games, "wrc_plus", 14) or wrc_plus

    # Baserunning proxy: SB/game scaled
    bsr_proxy = (total_sb / gp) * 10.0  # rough scale

    return OffenseFeatures(
        batting_avg=avg,
        obp=obp,
        slg=slg,
        ops=ops,
        woba=woba,
        wrc_plus=wrc_plus,
        iso=iso,
        runs_per_game=total_runs / gp,
        home_runs_per_game=total_hr / gp,
        stolen_bases_per_game=total_sb / gp,
        walk_rate=obp - avg,  # rough proxy: OBP - AVG ≈ unintentional BB rate
        strikeout_rate=0.0,  # populated when K data available
        babip=0.0,  # populated when BIP data available
        hard_hit_pct=hard_hit,
        barrel_pct=barrel,
        ground_ball_pct=gb_pct,
        baserunning_score=bsr_proxy,
        risp_avg=avg,  # default to overall avg until situational data loaded
        leverage_ops=ops,
        runs_per_game_last_7=rpg_7,
        runs_per_game_last_14=rpg_14,
        runs_per_game_last_30=rpg_30,
        ops_last_14=ops_14,
        wrc_plus_last_14=wrc_14,
    )
