"""Analytics endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Query

from mlb.api.schemas import (
    AccuracyResponse,
    AccuracyTierResponse,
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


@router.get("/accuracy", response_model=AccuracyResponse)
async def get_accuracy():
    """Get prediction accuracy stats from tracked results."""
    from datetime import date, timedelta

    from mlb.models.accuracy import load_all_accuracy, get_rolling_accuracy

    records = load_all_accuracy(Path("data"))
    if not records:
        return AccuracyResponse()

    # Overall stats
    all_results = [r for rec in records for r in rec.get("results", [])]
    total = len(all_results)
    correct = sum(1 for r in all_results if r.get("correct"))
    brier_vals = [rec["summary"]["brier_score"] for rec in records if "brier_score" in rec.get("summary", {})]

    # Last 7d / 30d
    today = date.today()
    last_7 = [rec for rec in records if (today - date.fromisoformat(rec["date"])).days <= 7]
    last_30 = [rec for rec in records if (today - date.fromisoformat(rec["date"])).days <= 30]

    def _acc(recs):
        rs = [r for rec in recs for r in rec.get("results", [])]
        t = len(rs)
        c = sum(1 for r in rs if r.get("correct"))
        return (round(c / t, 4) if t > 0 else 0.0), t

    l7_acc, l7_g = _acc(last_7)
    l30_acc, l30_g = _acc(last_30)

    # Accuracy by confidence tier
    tiers = [
        ("Low (< 40)", lambda r: r.get("confidence", 50) < 40),
        ("Medium (40-60)", lambda r: 40 <= r.get("confidence", 50) < 60),
        ("High (60+)", lambda r: r.get("confidence", 50) >= 60),
    ]
    by_confidence = []
    for label, pred_fn in tiers:
        tier_results = [r for r in all_results if pred_fn(r)]
        t = len(tier_results)
        c = sum(1 for r in tier_results if r.get("correct"))
        by_confidence.append(AccuracyTierResponse(
            tier=label,
            accuracy=round(c / t, 4) if t > 0 else 0.0,
            games=t,
        ))

    # Calibration: bin predicted prob into deciles and compute actual win rate
    calibration = []
    bins = [(0.40, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.60),
            (0.60, 0.65), (0.65, 0.70), (0.70, 0.80)]
    for lo, hi in bins:
        bin_results = [r for r in all_results if lo <= r.get("home_win_prob", 0.5) < hi]
        if bin_results:
            actual = sum(1 for r in bin_results if r.get("actual_winner") == r.get("home_team")) / len(bin_results)
            calibration.append({
                "bin": f"{lo:.0%}-{hi:.0%}",
                "predicted": round((lo + hi) / 2, 3),
                "actual": round(actual, 4),
                "count": len(bin_results),
            })

    # Daily series for chart
    rolling = get_rolling_accuracy(window=7, data_dir=Path("data"))

    return AccuracyResponse(
        overall_accuracy=round(correct / total, 4) if total > 0 else 0.0,
        overall_games=total,
        last_7d_accuracy=l7_acc,
        last_7d_games=l7_g,
        last_30d_accuracy=l30_acc,
        last_30d_games=l30_g,
        brier_score=round(sum(brier_vals) / len(brier_vals), 4) if brier_vals else 0.0,
        by_confidence=by_confidence,
        daily_series=rolling,
        calibration=calibration,
    )


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
