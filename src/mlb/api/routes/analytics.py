"""Analytics endpoints."""

from __future__ import annotations

from pathlib import Path

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
    """Get aggregate betting performance stats from settlement data."""
    from mlb.betting.settlement import load_all_settlements

    settlements = load_all_settlements(Path("data"))
    if start_date:
        settlements = [s for s in settlements if s["date"] >= start_date]
    if end_date:
        settlements = [s for s in settlements if s["date"] <= end_date]

    all_bets = [b for s in settlements for b in s.get("bets", [])]
    total = len(all_bets)
    wins = sum(1 for b in all_bets if b.get("result") == "win")
    losses = sum(1 for b in all_bets if b.get("result") == "loss")
    pushes = sum(1 for b in all_bets if b.get("result") == "push")
    staked = sum(b.get("recommended_stake", 0) for b in all_bets)
    pnl = sum(b.get("pnl", 0) for b in all_bets)
    win_rate = round(wins / total, 4) if total > 0 else 0.0
    roi = round(pnl / staked, 4) if staked > 0 else 0.0

    # Compute max drawdown from cumulative P&L across days
    max_dd = 0.0
    if settlements:
        peak = 0.0
        for s in settlements:
            cp = s.get("summary", {}).get("cumulative_pnl", 0.0)
            peak = max(peak, cp)
            dd = (peak - cp) / max(peak, 1) if peak > 0 else 0
            max_dd = max(max_dd, dd)

    return PerformanceResponse(
        period=f"{start_date or 'season_start'} to {end_date or 'today'}",
        total_bets=total,
        wins=wins,
        losses=losses,
        pushes=pushes,
        win_rate=win_rate,
        total_staked=round(staked, 2),
        total_pnl=round(pnl, 2),
        roi=roi,
        max_drawdown=round(max_dd, 4),
        total_games=total,
        accuracy=win_rate,
        roi_flat=roi,
        roi_kelly=roi,
    )


@router.get("/pnl", response_model=list[PnLResponse])
async def get_pnl_history(
    start_date: str = Query(default=None),
    end_date: str = Query(default=None),
):
    """Get daily P&L history from settlement files."""
    from mlb.betting.settlement import load_all_settlements

    settlements = load_all_settlements(Path("data"))
    if start_date:
        settlements = [s for s in settlements if s["date"] >= start_date]
    if end_date:
        settlements = [s for s in settlements if s["date"] <= end_date]

    return [
        PnLResponse(
            date=s["date"],
            daily_pnl=s["summary"]["daily_pnl"],
            cumulative_pnl=s["summary"]["cumulative_pnl"],
            bets_placed=s["summary"]["bets_placed"],
            bets_won=s["summary"]["bets_won"],
            roi=s["summary"]["roi"],
            max_drawdown=s["summary"]["max_drawdown"],
        )
        for s in settlements
    ]


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
