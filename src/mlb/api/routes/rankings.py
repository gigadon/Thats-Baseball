"""Rankings endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from mlb.api.schemas import (
    PlayerRankingsResponse,
    TeamComparisonResponse,
    TeamRankingsResponse,
)

router = APIRouter()

# In-memory store (replaced with DB in production)
_team_rankings_cache: dict[str, dict] = {}
_player_rankings_cache: dict[str, dict] = {}


@router.get("/teams", response_model=TeamRankingsResponse)
async def get_team_rankings(
    type: str = Query("power", description="Ranking type: power, offense, pitching, defense, bullpen, momentum"),
    date: str = Query(default=None, description="Date in YYYY-MM-DD format"),
):
    """Get team rankings by category."""
    from datetime import date as d

    target_date = date or d.today().isoformat()
    cache_key = f"{target_date}_{type}"
    cached = _team_rankings_cache.get(cache_key)

    if cached:
        return TeamRankingsResponse(**cached)

    return TeamRankingsResponse(
        ranking_date=target_date,
        category=type,
        rankings=[],
    )


@router.get("/players/{position}", response_model=PlayerRankingsResponse)
async def get_player_rankings(
    position: str,
    date: str = Query(default=None),
):
    """Get player rankings by position (SP, RP, C, 1B, 2B, 3B, SS, OF, DH)."""
    from datetime import date as d

    valid_positions = {"SP", "RP", "C", "1B", "2B", "3B", "SS", "OF", "DH"}
    pos = position.upper()
    if pos not in valid_positions:
        from fastapi import HTTPException
        raise HTTPException(400, f"Invalid position: {position}. Valid: {valid_positions}")

    target_date = date or d.today().isoformat()
    cache_key = f"{target_date}_{pos}"
    cached = _player_rankings_cache.get(cache_key)

    if cached:
        return PlayerRankingsResponse(**cached)

    return PlayerRankingsResponse(
        position=pos,
        ranking_date=target_date,
        rankings=[],
    )


@router.get("/teams/compare", response_model=TeamComparisonResponse)
async def compare_teams(
    team_a: str = Query(..., min_length=2, max_length=3),
    team_b: str = Query(..., min_length=2, max_length=3),
):
    """Compare two teams across all ranking categories."""
    # In production, pull from ranking service
    return TeamComparisonResponse(
        team_a=team_a.upper(),
        team_b=team_b.upper(),
        power={"a": 0, "b": 0, "diff": 0},
        offense={"a": 0, "b": 0, "diff": 0},
        pitching={"a": 0, "b": 0, "diff": 0},
        defense={"a": 0, "b": 0, "diff": 0},
        bullpen={"a": 0, "b": 0, "diff": 0},
        momentum={"a": 0, "b": 0, "diff": 0},
        advantage=team_a.upper(),
    )


def cache_team_rankings(date_str: str, category: str, data: dict):
    _team_rankings_cache[f"{date_str}_{category}"] = data


def cache_player_rankings(date_str: str, position: str, data: dict):
    _player_rankings_cache[f"{date_str}_{position}"] = data
