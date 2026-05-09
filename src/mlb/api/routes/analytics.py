"""Analytics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from mlb.api.schemas import (
    BacktestRequest,
    BacktestResponse,
    PerformanceResponse,
    PnLResponse,
    StadiumAnalysisResponse,
)
from mlb.features.stadium import STADIUM_INFO

router = APIRouter()


@router.get("/performance", response_model=PerformanceResponse)
async def get_performance(
    start_date: str = Query(default=None),
    end_date: str = Query(default=None),
):
    """Get model performance metrics for a period."""
    return PerformanceResponse(
        period=f"{start_date or 'season_start'} to {end_date or 'today'}",
        total_games=0,
        accuracy=0.0,
        brier_score=0.0,
        auc_roc=0.0,
        roi_flat=0.0,
        roi_kelly=0.0,
        high_confidence_accuracy=0.0,
        calibration_error=0.0,
    )


@router.get("/pnl", response_model=list[PnLResponse])
async def get_pnl_history(
    start_date: str = Query(default=None),
    end_date: str = Query(default=None),
):
    """Get daily P&L history."""
    return []


@router.get("/stadium/{team_id}", response_model=StadiumAnalysisResponse)
async def get_stadium_analysis(team_id: str):
    """Get stadium factors and analysis for a team."""
    tid = team_id.upper()
    info = STADIUM_INFO.get(tid)

    if not info:
        from fastapi import HTTPException
        raise HTTPException(404, f"Unknown team: {tid}")

    return StadiumAnalysisResponse(
        team_id=tid,
        stadium_name=f"{tid} Stadium",
        overall_pf=100,
        runs_pf=100,
        hr_pf=100,
        altitude_ft=info.get("altitude", 0),
        surface=info.get("surface", "grass"),
        roof=info.get("roof", "open"),
        dimension_avg=round(
            (info.get("lf", 330) + info.get("cf", 400) + info.get("rf", 330)) / 3, 1
        ),
    )


@router.get("/stadiums", response_model=list[StadiumAnalysisResponse])
async def get_all_stadiums():
    """Get stadium analysis for all 30 parks."""
    results = []
    for tid, info in sorted(STADIUM_INFO.items()):
        results.append(StadiumAnalysisResponse(
            team_id=tid,
            stadium_name=f"{tid} Stadium",
            overall_pf=100,
            runs_pf=100,
            hr_pf=100,
            altitude_ft=info.get("altitude", 0),
            surface=info.get("surface", "grass"),
            roof=info.get("roof", "open"),
            dimension_avg=round(
                (info.get("lf", 330) + info.get("cf", 400) + info.get("rf", 330)) / 3, 1
            ),
        ))
    return results


@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(req: BacktestRequest):
    """Run a historical backtest with specified parameters."""
    # In production, this would load historical predictions and run the backtester
    return BacktestResponse(
        start_date=req.start_date,
        end_date=req.end_date,
        total_games=0,
        total_bets=0,
        win_rate=0.0,
        roi_flat=0.0,
        roi_kelly=0.0,
        max_drawdown=0.0,
        sharpe_ratio=0.0,
        monthly_results=[],
    )
