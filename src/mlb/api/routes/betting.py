"""Betting endpoints."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Query

from mlb.api.schemas import BetResponse, BettingConfigRequest, BettingSlipResponse

router = APIRouter()

_betting_cache: dict[str, dict] = {}
_CACHE_DIR = Path("data/predictions")


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

    # Try loading from file if not in memory
    if not cached:
        path = _CACHE_DIR / f"{target_date}.json"
        if path.exists():
            data = json.loads(path.read_text())
            cached = data.get("betting_slip")
            if cached:
                _betting_cache[target_date] = cached

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
    """Get settled bet history with P&L."""
    from mlb.betting.settlement import load_all_settlements

    settlements = load_all_settlements(Path("data"))

    if start_date:
        settlements = [s for s in settlements if s["date"] >= start_date]
    if end_date:
        settlements = [s for s in settlements if s["date"] <= end_date]

    settlements = settlements[-limit:]

    all_bets = []
    for s in settlements:
        for bet in s.get("bets", []):
            bet["date"] = s["date"]
            all_bets.append(bet)

    total_won = sum(1 for b in all_bets if b.get("result") == "win")
    total_staked = sum(b.get("recommended_stake", 0) for b in all_bets)
    total_pnl = sum(b.get("pnl", 0) for b in all_bets)

    return {
        "period": f"{start_date or 'start'} to {end_date or 'now'}",
        "total_bets": len(all_bets),
        "total_wins": total_won,
        "total_pnl": round(total_pnl, 2),
        "roi": round(total_pnl / total_staked, 4) if total_staked > 0 else 0.0,
        "records": settlements,
    }


def cache_betting_slip(date_str: str, data: dict):
    _betting_cache[date_str] = data

    # Persist — merge into existing predictions file
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{date_str}.json"
    existing = {}
    if path.exists():
        existing = json.loads(path.read_text())
    existing["betting_slip"] = data
    path.write_text(json.dumps(existing, default=str))
