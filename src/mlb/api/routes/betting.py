"""Betting endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from mlb.api.schemas import BetResponse, BettingConfigRequest, BettingSlipResponse

router = APIRouter()

_betting_cache: dict[str, dict] = {}


@router.get("/recommendations", response_model=BettingSlipResponse)
async def get_recommendations(
    date: str = Query(default=None),
    bankroll: float = Query(10000.0),
    risk: str = Query("moderate", description="conservative, moderate, aggressive"),
):
    """Get betting recommendations for a given date."""
    from datetime import date as d

    target_date = date or d.today().isoformat()
    cached = _betting_cache.get(target_date)

    if cached:
        # Rescale stakes for the requested bankroll
        slip = dict(cached)
        slip["bankroll"] = bankroll
        return BettingSlipResponse(**slip)

    return BettingSlipResponse(
        slip_date=target_date,
        bankroll=bankroll,
        total_stake=0,
        num_bets=0,
        bets=[],
        total_ev=0,
        max_exposure=0,
        risk_level=risk,
    )


@router.get("/value", response_model=list[BetResponse])
async def get_value_bets(
    min_edge: float = Query(0.02, description="Minimum edge percentage"),
    date: str = Query(default=None),
):
    """Get current value bets above the edge threshold."""
    from datetime import date as d

    target_date = date or d.today().isoformat()
    cached = _betting_cache.get(target_date)

    if cached and cached.get("bets"):
        return [
            BetResponse(**b) for b in cached["bets"]
            if b.get("edge", 0) >= min_edge
        ]

    return []


@router.get("/history")
async def get_betting_history(
    start_date: str = Query(default=None),
    end_date: str = Query(default=None),
    limit: int = Query(30, ge=1, le=365),
):
    """Get betting P&L history."""
    # In production, pull from betting_results table
    return {
        "period": f"{start_date or 'start'} to {end_date or 'now'}",
        "total_bets": 0,
        "total_wins": 0,
        "total_pnl": 0.0,
        "roi": 0.0,
        "records": [],
    }


def cache_betting_slip(date_str: str, data: dict):
    _betting_cache[date_str] = data
