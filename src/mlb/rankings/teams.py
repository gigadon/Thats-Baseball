"""Team power rankings.

Generates daily team rankings using composite scores from the feature engine.
Supports ranking by overall power, offense, pitching, defense, bullpen, momentum,
and custom weighted combinations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

from mlb.features.bullpen import BullpenFeatures, calculate_bullpen_score
from mlb.features.momentum import (
    DefenseFeatures,
    MomentumFeatures,
    calculate_defense_score,
    calculate_momentum_score,
)
from mlb.features.offense import OffenseFeatures, calculate_offense_score
from mlb.features.pitching import PitchingFeatures, calculate_pitching_score

logger = logging.getLogger(__name__)


@dataclass
class TeamRanking:
    """A single team's ranking entry."""

    rank: int
    team_id: str
    team_name: str
    division: str
    league: str

    # Composite scores (0-100)
    power_score: float
    offense_score: float
    pitching_score: float
    defense_score: float
    bullpen_score: float
    momentum_score: float

    # Record
    wins: int
    losses: int
    win_pct: float
    run_diff: int
    pythag_win_pct: float

    # Trends
    last_10_record: str  # e.g. "7-3"
    streak: str  # e.g. "W5" or "L2"
    rank_change: int  # +/- from previous day (positive = improved)

    # Tier classification
    tier: str  # "elite", "contender", "average", "below_average", "rebuilding"


@dataclass
class RankingsSnapshot:
    """Complete rankings for a single date."""

    ranking_date: str
    rankings: list[TeamRanking]
    category: str  # "power", "offense", "pitching", etc.
    generated_at: str = ""

    @property
    def by_division(self) -> dict[str, list[TeamRanking]]:
        groups: dict[str, list[TeamRanking]] = {}
        for r in self.rankings:
            groups.setdefault(r.division, []).append(r)
        return groups

    @property
    def by_league(self) -> dict[str, list[TeamRanking]]:
        groups: dict[str, list[TeamRanking]] = {}
        for r in self.rankings:
            groups.setdefault(r.league, []).append(r)
        return groups

    @property
    def by_tier(self) -> dict[str, list[TeamRanking]]:
        groups: dict[str, list[TeamRanking]] = {}
        for r in self.rankings:
            groups.setdefault(r.tier, []).append(r)
        return groups


def _classify_tier(power_score: float) -> str:
    if power_score >= 70:
        return "elite"
    elif power_score >= 58:
        return "contender"
    elif power_score >= 45:
        return "average"
    elif power_score >= 35:
        return "below_average"
    else:
        return "rebuilding"


def _power_score(off: float, pit: float, dfn: float, bp: float, mom: float) -> float:
    return off * 0.30 + pit * 0.30 + dfn * 0.15 + bp * 0.15 + mom * 0.10


class TeamRankingService:
    """Generates and manages team power rankings."""

    def __init__(self):
        self._previous_rankings: dict[str, int] = {}  # team_id → previous rank

    def generate_rankings(
        self,
        teams: list[dict[str, Any]],
        category: str = "power",
        ranking_date: str | None = None,
    ) -> RankingsSnapshot:
        """Generate rankings for all teams.

        Args:
            teams: List of dicts, each containing:
                - team_id, team_name, division, league
                - offense: OffenseFeatures
                - pitching: PitchingFeatures
                - defense: DefenseFeatures
                - bullpen: BullpenFeatures
                - momentum: MomentumFeatures
                - wins, losses, run_diff
                - last_10_wins, last_10_losses, streak
            category: Sort key — "power", "offense", "pitching", "defense",
                      "bullpen", "momentum"
            ranking_date: Date string (defaults to today).
        """
        ranking_date = ranking_date or date.today().isoformat()
        entries: list[TeamRanking] = []

        for team in teams:
            off_score = calculate_offense_score(team["offense"])
            pit_score = calculate_pitching_score(team["pitching"])
            def_score = calculate_defense_score(team["defense"])
            bp_score = calculate_bullpen_score(team["bullpen"])
            mom_score = calculate_momentum_score(team["momentum"])
            power = _power_score(off_score, pit_score, def_score, bp_score, mom_score)

            wins = team.get("wins", 0)
            losses = team.get("losses", 0)
            gp = wins + losses or 1
            l10w = team.get("last_10_wins", 0)
            l10l = team.get("last_10_losses", 0)
            streak_val = team.get("streak", 0)
            streak_str = f"W{streak_val}" if streak_val > 0 else f"L{abs(streak_val)}"
            run_diff = team.get("run_diff", 0)

            rs = team.get("runs_scored", wins * 4 + run_diff // 2)
            ra = team.get("runs_allowed", losses * 4 - run_diff // 2)
            rs2, ra2 = max(rs, 1) ** 2, max(ra, 1) ** 2
            pythag = rs2 / (rs2 + ra2)

            entries.append(TeamRanking(
                rank=0,  # Assigned after sorting
                team_id=team["team_id"],
                team_name=team.get("team_name", team["team_id"]),
                division=team.get("division", ""),
                league=team.get("league", ""),
                power_score=round(power, 1),
                offense_score=round(off_score, 1),
                pitching_score=round(pit_score, 1),
                defense_score=round(def_score, 1),
                bullpen_score=round(bp_score, 1),
                momentum_score=round(mom_score, 1),
                wins=wins,
                losses=losses,
                win_pct=round(wins / gp, 3),
                run_diff=run_diff,
                pythag_win_pct=round(pythag, 3),
                last_10_record=f"{l10w}-{l10l}",
                streak=streak_str,
                rank_change=0,
                tier=_classify_tier(power),
            ))

        # Sort by selected category
        sort_key = {
            "power": lambda e: e.power_score,
            "offense": lambda e: e.offense_score,
            "pitching": lambda e: e.pitching_score,
            "defense": lambda e: e.defense_score,
            "bullpen": lambda e: e.bullpen_score,
            "momentum": lambda e: e.momentum_score,
        }.get(category, lambda e: e.power_score)

        entries.sort(key=sort_key, reverse=True)

        # Assign ranks and calculate rank changes
        for i, entry in enumerate(entries, 1):
            entry.rank = i
            prev = self._previous_rankings.get(entry.team_id)
            if prev is not None:
                entry.rank_change = prev - i  # Positive = moved up
            else:
                entry.rank_change = 0

        # Store current rankings for next comparison
        self._previous_rankings = {e.team_id: e.rank for e in entries}

        return RankingsSnapshot(
            ranking_date=ranking_date,
            rankings=entries,
            category=category,
        )

    def get_team_rank(self, snapshot: RankingsSnapshot, team_id: str) -> TeamRanking | None:
        """Get a specific team's ranking from a snapshot."""
        for r in snapshot.rankings:
            if r.team_id == team_id:
                return r
        return None

    def compare_teams(
        self, snapshot: RankingsSnapshot, team_a: str, team_b: str
    ) -> dict[str, Any]:
        """Compare two teams across all categories."""
        a = self.get_team_rank(snapshot, team_a)
        b = self.get_team_rank(snapshot, team_b)
        if not a or not b:
            return {}

        return {
            "team_a": team_a,
            "team_b": team_b,
            "power": {"a": a.power_score, "b": b.power_score, "diff": round(a.power_score - b.power_score, 1)},
            "offense": {"a": a.offense_score, "b": b.offense_score, "diff": round(a.offense_score - b.offense_score, 1)},
            "pitching": {"a": a.pitching_score, "b": b.pitching_score, "diff": round(a.pitching_score - b.pitching_score, 1)},
            "defense": {"a": a.defense_score, "b": b.defense_score, "diff": round(a.defense_score - b.defense_score, 1)},
            "bullpen": {"a": a.bullpen_score, "b": b.bullpen_score, "diff": round(a.bullpen_score - b.bullpen_score, 1)},
            "momentum": {"a": a.momentum_score, "b": b.momentum_score, "diff": round(a.momentum_score - b.momentum_score, 1)},
            "advantage": team_a if a.power_score > b.power_score else team_b,
        }
