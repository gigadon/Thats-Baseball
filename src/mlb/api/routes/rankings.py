"""Rankings endpoints."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Query

from mlb.api.schemas import (
    PlayerRankingsResponse,
    TeamComparisonResponse,
    TeamRankingsResponse,
)

router = APIRouter()

# In-memory store + file-backed persistence
_team_rankings_cache: dict[str, dict] = {}
_player_rankings_cache: dict[str, dict] = {}
_CACHE_DIR = Path("data/predictions")


def _load_rankings_from_file(date_str: str, category: str) -> dict | None:
    from mlb.api.fileutil import safe_json_read

    cache_key = f"{date_str}_{category}"
    if cache_key in _team_rankings_cache:
        return _team_rankings_cache[cache_key]

    path = _CACHE_DIR / f"{date_str}.json"
    if path.exists():
        data = safe_json_read(path)
        rankings = data.get("rankings", {}).get(category)
        if rankings:
            _team_rankings_cache[cache_key] = rankings
            return rankings
    return None


@router.get("/teams", response_model=TeamRankingsResponse)
async def get_team_rankings(
    type: str = Query("power", description="Ranking type: power, offense, pitching, defense, bullpen, momentum"),
    date: str = Query(default=None, description="Date in YYYY-MM-DD format"),
):
    """Get team rankings by category."""
    from datetime import date as d

    target_date = date or d.today().isoformat()
    cached = _load_rankings_from_file(target_date, type)

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

    # Try loading from file
    from mlb.api.fileutil import safe_json_read

    path = _CACHE_DIR / f"{target_date}.json"
    if path.exists():
        data = safe_json_read(path)
        player_data = data.get("player_rankings", {}).get(pos)
        if player_data:
            _player_rankings_cache[cache_key] = player_data
            return PlayerRankingsResponse(**player_data)

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
    from datetime import date as d

    target_date = d.today().isoformat()
    a_upper = team_a.upper()
    b_upper = team_b.upper()

    # Load power rankings to get all team scores
    power_data = _load_rankings_from_file(target_date, "power")
    if not power_data or "rankings" not in power_data:
        return TeamComparisonResponse(
            team_a=a_upper, team_b=b_upper,
            power={"a": 0, "b": 0, "diff": 0},
            offense={"a": 0, "b": 0, "diff": 0},
            pitching={"a": 0, "b": 0, "diff": 0},
            defense={"a": 0, "b": 0, "diff": 0},
            bullpen={"a": 0, "b": 0, "diff": 0},
            momentum={"a": 0, "b": 0, "diff": 0},
            advantage=a_upper,
        )

    # Build lookup from rankings
    teams_by_id = {t["team_id"]: t for t in power_data["rankings"]}
    ta = teams_by_id.get(a_upper, {})
    tb = teams_by_id.get(b_upper, {})

    categories = ["power_score", "offense_score", "pitching_score", "defense_score", "bullpen_score", "momentum_score"]
    cat_names = ["power", "offense", "pitching", "defense", "bullpen", "momentum"]
    result = {}
    a_advantages = 0

    for cat_name, score_key in zip(cat_names, categories):
        a_val = ta.get(score_key, 0)
        b_val = tb.get(score_key, 0)
        result[cat_name] = {"a": a_val, "b": b_val, "diff": round(a_val - b_val, 1)}
        if a_val > b_val:
            a_advantages += 1

    advantage = a_upper if a_advantages >= 4 else b_upper if a_advantages <= 2 else "Even"

    return TeamComparisonResponse(
        team_a=a_upper,
        team_b=b_upper,
        advantage=advantage,
        **result,
    )


def cache_team_rankings(date_str: str, category: str, data: dict):
    from mlb.api.fileutil import safe_json_read, safe_json_merge, safe_json_write

    _team_rankings_cache[f"{date_str}_{category}"] = data

    # Persist — atomic read-modify-write with locking
    path = _CACHE_DIR / f"{date_str}.json"
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            existing = safe_json_read(path)
            rankings = existing.get("rankings", {})
            rankings[category] = data
            existing["rankings"] = rankings
            safe_json_write(path, existing)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def cache_player_rankings(date_str: str, position: str, data: dict):
    from mlb.api.fileutil import safe_json_read, safe_json_write

    _player_rankings_cache[f"{date_str}_{position}"] = data

    # Persist — atomic read-modify-write with locking
    path = _CACHE_DIR / f"{date_str}.json"
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            existing = safe_json_read(path)
            player_rankings = existing.get("player_rankings", {})
            player_rankings[position] = data
            existing["player_rankings"] = player_rankings
            safe_json_write(path, existing)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
