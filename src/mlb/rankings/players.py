"""Player position rankings.

Ranks players within each position group using role-appropriate metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PlayerRanking:
    """A single player's ranking entry."""

    rank: int
    player_id: int
    player_name: str
    team_id: str
    position: str
    score: float  # Composite score (0-100)

    # Key stats (vary by position)
    key_stats: dict[str, float]

    # Context
    games_played: int
    rank_change: int
    tier: str  # "ace", "star", "starter", "average", "below_average"


@dataclass
class PositionRankings:
    """Rankings for a single position group."""

    position: str
    ranking_date: str
    rankings: list[PlayerRanking]


# ─── Scoring Functions ────────────────────────────────────────


def _score_sp(stats: dict) -> tuple[float, dict[str, float]]:
    """Score a starting pitcher."""
    era = float(stats.get("era", 4.50) or 4.50)
    fip = float(stats.get("fip", 4.50) or 4.50)
    k9 = float(stats.get("k_per_9", 8.0) or 8.0)
    bb9 = float(stats.get("bb_per_9", 3.5) or 3.5)
    whip = float(stats.get("whip", 1.30) or 1.30)
    ip = float(stats.get("innings_pitched", 0) or 0)
    war = float(stats.get("war", 0) or 0)

    fip_norm = np.clip((6.0 - fip) / 4.0, 0, 1) * 100
    k_norm = np.clip(k9 / 14.0, 0, 1) * 100
    bb_norm = np.clip((5.0 - bb9) / 5.0, 0, 1) * 100
    whip_norm = np.clip((2.0 - whip) / 1.0, 0, 1) * 100
    ip_norm = np.clip(ip / 200.0, 0, 1) * 100

    score = fip_norm * 0.30 + k_norm * 0.25 + bb_norm * 0.15 + whip_norm * 0.15 + ip_norm * 0.15
    key = {"ERA": era, "FIP": fip, "K/9": k9, "BB/9": bb9, "WHIP": whip, "IP": ip, "WAR": war}
    return float(np.clip(score, 0, 100)), key


def _score_rp(stats: dict) -> tuple[float, dict[str, float]]:
    """Score a relief pitcher."""
    era = float(stats.get("era", 4.00) or 4.00)
    fip = float(stats.get("fip", 4.00) or 4.00)
    k9 = float(stats.get("k_per_9", 9.0) or 9.0)
    bb9 = float(stats.get("bb_per_9", 3.5) or 3.5)
    saves = int(stats.get("saves", 0) or 0)
    holds = int(stats.get("holds", 0) or 0)

    fip_norm = np.clip((5.5 - fip) / 3.5, 0, 1) * 100
    k_norm = np.clip(k9 / 15.0, 0, 1) * 100
    bb_norm = np.clip((5.0 - bb9) / 5.0, 0, 1) * 100
    impact = np.clip((saves + holds) / 30.0, 0, 1) * 100

    score = fip_norm * 0.35 + k_norm * 0.30 + bb_norm * 0.15 + impact * 0.20
    key = {"ERA": era, "FIP": fip, "K/9": k9, "BB/9": bb9, "SV": saves, "HLD": holds}
    return float(np.clip(score, 0, 100)), key


def _score_hitter(stats: dict) -> tuple[float, dict[str, float]]:
    """Score a position player (hitter)."""
    avg = float(stats.get("batting_avg", .250) or .250)
    obp = float(stats.get("obp", .320) or .320)
    slg = float(stats.get("slg", .400) or .400)
    ops = float(stats.get("ops", .720) or .720)
    wrc_plus = float(stats.get("wrc_plus", 100) or 100)
    hr = int(stats.get("home_runs", 0) or 0)
    sb = int(stats.get("stolen_bases", 0) or 0)
    war = float(stats.get("war", 0) or 0)

    wrc_norm = np.clip((wrc_plus - 50) / 100, 0, 1) * 100
    obp_norm = np.clip((obp - 0.250) / 0.200, 0, 1) * 100
    slg_norm = np.clip((slg - 0.300) / 0.350, 0, 1) * 100
    power = np.clip(hr / 40.0, 0, 1) * 100
    speed = np.clip(sb / 30.0, 0, 1) * 100

    score = wrc_norm * 0.35 + obp_norm * 0.15 + slg_norm * 0.15 + power * 0.15 + speed * 0.10 + np.clip(war / 6.0, 0, 1) * 100 * 0.10
    key = {"AVG": avg, "OBP": obp, "SLG": slg, "OPS": ops, "wRC+": wrc_plus, "HR": hr, "SB": sb, "WAR": war}
    return float(np.clip(score, 0, 100)), key


SCORING_FN = {
    "SP": _score_sp,
    "RP": _score_rp,
    "C": _score_hitter,
    "1B": _score_hitter,
    "2B": _score_hitter,
    "3B": _score_hitter,
    "SS": _score_hitter,
    "OF": _score_hitter,
    "DH": _score_hitter,
}


def _classify_player_tier(score: float, position: str) -> str:
    if position == "SP":
        thresholds = [75, 60, 48, 35]
        labels = ["ace", "star", "starter", "average", "below_average"]
    elif position == "RP":
        thresholds = [75, 60, 45, 30]
        labels = ["elite_closer", "star", "solid", "average", "below_average"]
    else:
        thresholds = [72, 58, 45, 32]
        labels = ["mvp_candidate", "all_star", "starter", "average", "below_average"]

    for threshold, label in zip(thresholds, labels):
        if score >= threshold:
            return label
    return labels[-1]


class PlayerRankingService:
    """Generates position-specific player rankings."""

    def __init__(self):
        self._previous: dict[str, dict[int, int]] = {}  # position → {player_id: rank}

    def rank_position(
        self,
        players: list[dict[str, Any]],
        position: str,
        ranking_date: str | None = None,
        min_games: int = 10,
    ) -> PositionRankings:
        """Rank players at a given position.

        Args:
            players: List of dicts with player_id, player_name, team_id, stats dict.
            position: "SP", "RP", "C", "1B", "2B", "3B", "SS", "OF", "DH".
            ranking_date: Date string.
            min_games: Minimum games to qualify.
        """
        from datetime import date as dt_date

        ranking_date = ranking_date or dt_date.today().isoformat()
        scoring_fn = SCORING_FN.get(position, _score_hitter)

        entries: list[PlayerRanking] = []
        for p in players:
            gp = int(p.get("games_played", 0) or 0)
            if gp < min_games:
                continue

            score, key_stats = scoring_fn(p.get("stats", {}))
            entries.append(PlayerRanking(
                rank=0,
                player_id=p["player_id"],
                player_name=p.get("player_name", ""),
                team_id=p.get("team_id", ""),
                position=position,
                score=round(score, 1),
                key_stats=key_stats,
                games_played=gp,
                rank_change=0,
                tier=_classify_player_tier(score, position),
            ))

        entries.sort(key=lambda e: e.score, reverse=True)

        prev = self._previous.get(position, {})
        for i, entry in enumerate(entries, 1):
            entry.rank = i
            old = prev.get(entry.player_id)
            entry.rank_change = (old - i) if old is not None else 0

        self._previous[position] = {e.player_id: e.rank for e in entries}

        return PositionRankings(
            position=position,
            ranking_date=ranking_date,
            rankings=entries,
        )

    def rank_all_positions(
        self,
        players_by_position: dict[str, list[dict[str, Any]]],
        ranking_date: str | None = None,
    ) -> dict[str, PositionRankings]:
        """Rank all positions at once."""
        results = {}
        for pos, players in players_by_position.items():
            if pos in SCORING_FN:
                results[pos] = self.rank_position(players, pos, ranking_date)
        return results
