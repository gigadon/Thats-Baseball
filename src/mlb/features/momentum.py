"""Momentum, matchup, and situational features.

Covers:
  - Win/loss streaks and last-10 record
  - Run differential trends
  - Head-to-head history
  - Travel and rest effects
  - Umpire tendencies
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np


@dataclass
class MomentumFeatures:
    """Momentum and trend features for a team."""

    # Recent record
    last_5_wins: int
    last_5_losses: int
    last_10_wins: int
    last_10_losses: int
    last_20_wins: int
    last_20_losses: int

    # Streaks
    current_streak: int  # Positive = wins, negative = losses
    longest_win_streak_30d: int
    longest_loss_streak_30d: int

    # Run differential
    run_diff_season: int
    run_diff_last_10: int
    run_diff_per_game: float
    run_diff_last_10_per_game: float

    # Pythagorean expectation
    pythag_win_pct: float  # RS^2 / (RS^2 + RA^2)
    actual_win_pct: float
    luck_factor: float  # Actual - Pythagorean (positive = lucky)


@dataclass
class MatchupFeatures:
    """Head-to-head and situational features for a specific game."""

    # H2H this season
    h2h_wins: int
    h2h_losses: int
    h2h_run_diff: int
    h2h_games: int

    # Travel / rest
    home_rest_days: int  # Days since last game for home team
    away_rest_days: int
    away_travel_distance_miles: float  # Distance traveled for road trip
    away_games_in_row_on_road: int
    away_timezone_change: int  # +/- hours from home timezone

    # Day/night
    is_day_game: bool
    is_doubleheader: bool
    game_number: int  # 1 or 2 in doubleheader

    # Home/away splits
    home_team_home_win_pct: float
    away_team_away_win_pct: float

    # Division / interleague
    is_division_game: bool
    is_interleague: bool


@dataclass
class DefenseFeatures:
    """Team defensive features."""

    fielding_pct: float
    drs: int  # Defensive Runs Saved
    uzr: float  # Ultimate Zone Rating
    oaa: int  # Outs Above Average
    errors_per_game: float
    double_plays_per_game: float
    catcher_framing_runs: float  # Catcher framing value


def calculate_momentum_score(f: MomentumFeatures) -> float:
    """Calculate momentum score (0-100).

    Components:
      Recent wins    40%  — last-10 win %
      Streak         25%  — current streak value
      Run diff trend 20%  — recent run differential
      Luck factor    15%  — Pythagorean vs actual (regression expected)
    """
    # Last-10 win %
    l10_pct = f.last_10_wins / max(f.last_10_wins + f.last_10_losses, 1)
    l10_norm = l10_pct * 100

    # Streak: cap at ±10 games
    streak_norm = np.clip((f.current_streak + 10) / 20.0, 0, 1) * 100

    # Run diff per game last 10 (range roughly -5 to +5)
    rdiff_norm = np.clip((f.run_diff_last_10_per_game + 5) / 10.0, 0, 1) * 100

    # Luck factor: positive luck suggests regression, negative suggests improvement
    # Invert: unlucky teams (negative) get a boost
    luck_norm = np.clip((-f.luck_factor + 0.05) / 0.10, 0, 1) * 100

    return float(np.clip(
        l10_norm * 0.40
        + streak_norm * 0.25
        + rdiff_norm * 0.20
        + luck_norm * 0.15,
        0,
        100,
    ))


def calculate_defense_score(f: DefenseFeatures) -> float:
    """Calculate defense score (0-100).

    Components:
      OAA           30%
      DRS           25%
      UZR           20%
      Fielding %    15%
      Framing       10%
    """
    # OAA: range roughly -30 to +30
    oaa_norm = np.clip((f.oaa + 30) / 60.0, 0, 1) * 100

    # DRS: same range
    drs_norm = np.clip((f.drs + 30) / 60.0, 0, 1) * 100

    # UZR: range roughly -20 to +20
    uzr_norm = np.clip((f.uzr + 20) / 40.0, 0, 1) * 100

    # Fielding %: .975 to .990
    fpct_norm = np.clip((f.fielding_pct - 0.970) / 0.025, 0, 1) * 100

    # Framing: range roughly -15 to +15 runs
    frame_norm = np.clip((f.catcher_framing_runs + 15) / 30.0, 0, 1) * 100

    return float(np.clip(
        oaa_norm * 0.30
        + drs_norm * 0.25
        + uzr_norm * 0.20
        + fpct_norm * 0.15
        + frame_norm * 0.10,
        0,
        100,
    ))


def compute_momentum_features(
    game_results: list[dict],
    season_runs_scored: int,
    season_runs_allowed: int,
    games_played: int,
) -> MomentumFeatures:
    """Build MomentumFeatures from recent game results.

    Args:
        game_results: List of recent game dicts with 'won' (bool) and 'run_diff' (int),
                      ordered oldest first.
        season_runs_scored: Total runs scored this season.
        season_runs_allowed: Total runs allowed this season.
        games_played: Total games played.
    """
    gp = max(games_played, 1)

    def _record(n: int) -> tuple[int, int]:
        recent = game_results[-n:] if len(game_results) >= n else game_results
        wins = sum(1 for g in recent if g.get("won"))
        return wins, len(recent) - wins

    l5w, l5l = _record(5)
    l10w, l10l = _record(10)
    l20w, l20l = _record(20)

    # Current streak
    streak = 0
    for g in reversed(game_results):
        if g.get("won"):
            if streak >= 0:
                streak += 1
            else:
                break
        else:
            if streak <= 0:
                streak -= 1
            else:
                break

    # Longest streaks in last 30 games
    last_30 = game_results[-30:] if len(game_results) >= 30 else game_results
    max_win_streak = max_loss_streak = cur_w = cur_l = 0
    for g in last_30:
        if g.get("won"):
            cur_w += 1
            cur_l = 0
            max_win_streak = max(max_win_streak, cur_w)
        else:
            cur_l += 1
            cur_w = 0
            max_loss_streak = max(max_loss_streak, cur_l)

    # Run differentials
    rd_season = season_runs_scored - season_runs_allowed
    rd_l10 = sum(g.get("run_diff", 0) for g in game_results[-10:])

    # Pythagorean
    rs2 = season_runs_scored ** 2
    ra2 = season_runs_allowed ** 2
    pythag = rs2 / (rs2 + ra2) if (rs2 + ra2) > 0 else 0.5
    actual = (l10w + l20w) / max(l10w + l10l + l20w + l20l, 1)  # rough
    total_wins = sum(1 for g in game_results if g.get("won"))
    actual = total_wins / max(len(game_results), 1)

    return MomentumFeatures(
        last_5_wins=l5w,
        last_5_losses=l5l,
        last_10_wins=l10w,
        last_10_losses=l10l,
        last_20_wins=l20w,
        last_20_losses=l20l,
        current_streak=streak,
        longest_win_streak_30d=max_win_streak,
        longest_loss_streak_30d=max_loss_streak,
        run_diff_season=rd_season,
        run_diff_last_10=rd_l10,
        run_diff_per_game=rd_season / gp,
        run_diff_last_10_per_game=rd_l10 / min(len(game_results), 10) if game_results else 0,
        pythag_win_pct=pythag,
        actual_win_pct=actual,
        luck_factor=actual - pythag,
    )


def compute_matchup_features(
    h2h_games: list[dict],
    home_rest_days: int,
    away_rest_days: int,
    away_travel_miles: float,
    away_road_games: int,
    away_tz_change: int,
    is_day: bool,
    is_dh: bool,
    game_num: int,
    home_home_wpct: float,
    away_away_wpct: float,
    is_division: bool,
    is_interleague: bool,
) -> MatchupFeatures:
    """Build MatchupFeatures from context."""
    h2h_w = sum(1 for g in h2h_games if g.get("home_won"))
    h2h_l = len(h2h_games) - h2h_w
    h2h_rd = sum(g.get("run_diff", 0) for g in h2h_games)

    return MatchupFeatures(
        h2h_wins=h2h_w,
        h2h_losses=h2h_l,
        h2h_run_diff=h2h_rd,
        h2h_games=len(h2h_games),
        home_rest_days=home_rest_days,
        away_rest_days=away_rest_days,
        away_travel_distance_miles=away_travel_miles,
        away_games_in_row_on_road=away_road_games,
        away_timezone_change=away_tz_change,
        is_day_game=is_day,
        is_doubleheader=is_dh,
        game_number=game_num,
        home_team_home_win_pct=home_home_wpct,
        away_team_away_win_pct=away_away_wpct,
        is_division_game=is_division,
        is_interleague=is_interleague,
    )
