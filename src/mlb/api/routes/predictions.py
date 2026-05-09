"""Prediction endpoints."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException, Query

from mlb.api.schemas import (
    DailyPredictionsResponse,
    GamePredictionResponse,
    PredictionRequest,
)

router = APIRouter()

# In-memory store (replaced with DB in production)
_predictions_cache: dict[str, list[dict]] = {}


@router.get("/daily", response_model=DailyPredictionsResponse)
async def get_daily_predictions(
    date: str = Query(default=None, description="Date in YYYY-MM-DD format"),
):
    """Get all game predictions for a given date."""
    target_date = date or _today()
    games = _predictions_cache.get(target_date, [])

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


def _today() -> str:
    from datetime import date as d
    return d.today().isoformat()


def cache_predictions(date_str: str, predictions: list[dict]):
    """Store predictions in memory (called by the pipeline)."""
    _predictions_cache[date_str] = predictions
