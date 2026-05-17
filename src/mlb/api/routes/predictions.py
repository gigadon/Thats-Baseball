"""Prediction endpoints."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from mlb.api.schemas import (
    DailyPredictionsResponse,
    GamePredictionResponse,
    PredictionRequest,
)
from mlb.data.line_movement import get_all_line_movements

router = APIRouter()

# In-memory store + file-backed persistence
_predictions_cache: dict[str, list[dict]] = {}
_CACHE_DIR = Path("data/predictions")


def _load_from_file(date_str: str) -> list[dict]:
    """Load predictions from JSON file if not in memory."""
    from mlb.api.fileutil import safe_json_read

    if date_str in _predictions_cache:
        return _predictions_cache[date_str]

    path = _CACHE_DIR / f"{date_str}.json"
    if path.exists():
        data = safe_json_read(path)
        preds = data.get("predictions", [])
        _predictions_cache[date_str] = preds
        return preds

    return []


@router.get("/daily", response_model=DailyPredictionsResponse)
async def get_daily_predictions(
    date: str = Query(default=None, description="Date in YYYY-MM-DD format"),
):
    """Get all game predictions for a given date."""
    target_date = date or _today()
    games = _load_from_file(target_date)

    # Enrich with betting data from the same file
    games = _enrich_with_betting(target_date, games)

    # Enrich with line movement data
    games = _enrich_with_line_movement(target_date, games)

    return DailyPredictionsResponse(
        date=target_date,
        games=[GamePredictionResponse(**g) for g in games],
        total_games=len(games),
    )


@router.post("/game", response_model=GamePredictionResponse)
async def predict_game(req: PredictionRequest):
    """Generate a prediction for a specific matchup."""
    # In production, this would:
    # 1. Load features for both teams
    # 2. Run through the prediction service
    # 3. Return the prediction
    #
    # For now, return a placeholder that shows the API shape.
    return GamePredictionResponse(
        game_id=f"{req.home_team}_{req.away_team}_{req.date}",
        game_date=req.date,
        home_team=req.home_team,
        away_team=req.away_team,
        home_win_prob=0.0,
        away_win_prob=0.0,
        predicted_home_runs=0.0,
        predicted_away_runs=0.0,
        predicted_total=0.0,
        confidence=0.0,
        model_agreement=0.0,
        predicted_winner=req.home_team,
        model_predictions={},
        top_factors=[],
    )


@router.get("/game/{game_id}", response_model=GamePredictionResponse)
async def get_prediction(game_id: str):
    """Get a prediction by game ID."""
    # Search cache
    for games in _predictions_cache.values():
        for g in games:
            if g.get("game_id") == game_id:
                return GamePredictionResponse(**g)

    raise HTTPException(status_code=404, detail=f"Prediction not found: {game_id}")


@router.get("/scores")
async def get_live_scores(
    date: str = Query(default=None, description="Date in YYYY-MM-DD format"),
):
    """Fetch live scores for today's games from MLB API."""
    from datetime import date as d

    target_date = date or d.today().isoformat()
    scores = await _fetch_scores(target_date)
    return {"date": target_date, "scores": scores}


# Scores cache: refreshed every 60s
_scores_cache: dict[str, dict] = {}  # date_str -> {timestamp, scores}


async def _fetch_scores(date_str: str) -> list[dict]:
    """Fetch scores from MLB API with 60s cache."""
    import time
    from datetime import date as d

    cached = _scores_cache.get(date_str)
    if cached and (time.time() - cached["timestamp"]) < 60:
        return cached["scores"]

    try:
        from mlb.data.mlb_api import MLBApiClient
        client = MLBApiClient()
        games = await client.get_schedule(d.fromisoformat(date_str))
        scores = []
        for g in games:
            scores.append({
                "game_id": g["game_id"],
                "status": g["status"],
                "home_score": g.get("home_score"),
                "away_score": g.get("away_score"),
                "inning": g.get("innings"),
            })
        _scores_cache[date_str] = {"timestamp": time.time(), "scores": scores}
        return scores
    except Exception:
        return _scores_cache.get(date_str, {}).get("scores", [])


def _today() -> str:
    from datetime import date as d
    return d.today().isoformat()


def _enrich_with_betting(date_str: str, games: list[dict]) -> list[dict]:
    """Merge betting slip data into prediction game dicts."""
    from mlb.api.fileutil import safe_json_read

    path = _CACHE_DIR / f"{date_str}.json"
    if not path.exists():
        return games

    data = safe_json_read(path)
    slip = data.get("betting_slip")
    if not slip or not slip.get("bets"):
        return games

    # Index bets by game_id
    bets_by_game: dict[str, list[dict]] = {}
    for b in slip["bets"]:
        gid = b.get("game_id", "")
        bets_by_game.setdefault(gid, []).append(b)

    enriched = []
    for g in games:
        g = dict(g)  # shallow copy
        game_bets = bets_by_game.get(g.get("game_id"), [])
        if game_bets:
            # Use the highest-edge bet for display
            best = max(game_bets, key=lambda b: b.get("edge_pct", 0))
            g["edge_pct"] = best.get("edge_pct", 0)
            g["market_implied"] = best.get("implied_prob", 0)
            g["bet_status"] = {
                "side": f"{best.get('selection', '')}",
                "odds": f"{'+' if best.get('odds', 0) > 0 else ''}{best.get('odds', 0):.0f}",
                "stake": best.get("recommended_stake", 0),
            }
        enriched.append(g)
    return enriched


def _enrich_with_line_movement(date_str: str, games: list[dict]) -> list[dict]:
    """Add line movement sparkline data to each game."""
    movements = get_all_line_movements(date_str)
    if not movements:
        return games

    enriched = []
    for g in games:
        g = dict(g)
        key = (g.get("home_team", ""), g.get("away_team", ""))
        probs = movements.get(key, [])
        if probs:
            g["line_movement"] = probs
        enriched.append(g)
    return enriched


@router.get("/line-movement")
async def get_line_movement_data(
    date: str = Query(default=None),
):
    """Get line movement data for all games on a date."""
    target_date = date or _today()
    movements = get_all_line_movements(target_date)

    result = []
    for (home, away), probs in movements.items():
        result.append({
            "home_team": home,
            "away_team": away,
            "snapshots": probs,
            "open": probs[0] if probs else None,
            "current": probs[-1] if probs else None,
            "movement": round(probs[-1] - probs[0], 4) if len(probs) >= 2 else 0,
        })

    return {"date": target_date, "games": result}


def cache_predictions(date_str: str, predictions: list[dict]):
    """Store predictions in memory and write to JSON file.

    Merges by game_id so re-running mid-day doesn't drop finished games.
    """
    import fcntl
    from mlb.api.fileutil import safe_json_read, safe_json_write

    path = _CACHE_DIR / f"{date_str}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = path.with_suffix(".lock")
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            existing = safe_json_read(path)
            # Merge: keep existing games, update/add from new predictions
            existing_preds = {g["game_id"]: g for g in existing.get("predictions", [])}
            for g in predictions:
                existing_preds[g["game_id"]] = g
            merged = list(existing_preds.values())
            existing["date"] = date_str
            existing["predictions"] = merged
            safe_json_write(path, existing)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    _predictions_cache[date_str] = merged
