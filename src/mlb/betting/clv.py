"""Closing Line Value (CLV) tracking.

Compares the model's predicted probability at bet time vs. the closing
line (last snapshot before game time) to measure whether the model
consistently beats the market.

Positive CLV over time is the strongest indicator of a profitable model.

Usage:
    python -m mlb.betting.clv --days 30
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def compute_daily_clv(
    target_date: str, data_dir: Path = Path("data")
) -> list[dict]:
    """Compute CLV for each bet placed on target_date.

    Compares model prob at bet time to the closing implied prob
    from the last line movement snapshot.

    Returns list of {game_id, selection, model_prob, closing_prob, clv}.
    """
    # The card as sent (mlb.etl.slate_record) — CLV on bets that were never
    # alerted is meaningless. Pre-lock days fall back to the prediction cache.
    from mlb.etl.slate_record import load_slate

    slate = load_slate(date.fromisoformat(target_date), data_dir)
    if slate is not None:
        slip = slate.get("betting_slip")
    else:
        pred_path = data_dir / "predictions" / f"{target_date}.json"
        if not pred_path.exists():
            return []
        slip = json.loads(pred_path.read_text()).get("betting_slip")

    if not slip or not slip.get("bets"):
        return []

    # Load closing lines from line movement
    lm_path = data_dir / "line_movement" / f"{target_date}.json"
    closing_probs: dict[tuple[str, str], float] = {}
    if lm_path.exists():
        lm_data = json.loads(lm_path.read_text())
        snapshots = lm_data.get("snapshots", [])
        if snapshots:
            # Last snapshot = closing line
            last_snap = snapshots[-1]
            for g in last_snap.get("games", []):
                key = (g["home_team"], g["away_team"])
                closing_probs[key] = g["home_prob"]

    results = []
    for bet in slip["bets"]:
        home = bet.get("home_team", "")
        away = bet.get("away_team", "")
        key = (home, away)

        model_prob = bet.get("model_prob", 0.5)
        closing_home_prob = closing_probs.get(key)

        if closing_home_prob is None:
            # No line movement data — use implied prob from odds as fallback
            closing_home_prob = bet.get("implied_prob")

        if closing_home_prob is None:
            continue

        # Determine what the model bet on and compute CLV
        selection = bet.get("selection", "")
        bet_type = bet.get("bet_type", "moneyline")

        if bet_type == "moneyline":
            # For moneyline, CLV = model_prob - closing_prob for the same side
            if selection == home:
                model_side_prob = model_prob
                closing_side_prob = closing_home_prob
            else:
                model_side_prob = 1 - model_prob
                closing_side_prob = 1 - closing_home_prob

            clv = model_side_prob - closing_side_prob
        else:
            # For totals, CLV is harder — skip for now
            continue

        results.append({
            "game_id": bet.get("game_id", ""),
            "home_team": home,
            "away_team": away,
            "selection": selection,
            "bet_type": bet_type,
            "model_prob": round(model_prob, 4),
            "closing_prob": round(closing_side_prob, 4),
            "clv": round(clv, 4),
        })

    return results


def compute_clv_summary(
    days: int = 30, data_dir: Path = Path("data")
) -> dict[str, Any]:
    """Compute aggregate CLV stats over the last N days.

    Returns {avg_clv, total_bets, positive_clv_pct, daily_series}.
    """
    today = date.today()
    all_bets: list[dict] = []
    daily_series: list[dict] = []

    for i in range(days):
        d = (today - timedelta(days=i)).isoformat()
        day_clv = compute_daily_clv(d, data_dir)
        if day_clv:
            avg = sum(b["clv"] for b in day_clv) / len(day_clv)
            daily_series.append({
                "date": d,
                "avg_clv": round(avg, 4),
                "num_bets": len(day_clv),
            })
            all_bets.extend(day_clv)

    daily_series.reverse()  # Chronological order

    if not all_bets:
        return {
            "avg_clv": 0.0,
            "total_bets": 0,
            "positive_clv_pct": 0.0,
            "daily_series": [],
        }

    avg_clv = sum(b["clv"] for b in all_bets) / len(all_bets)
    positive = sum(1 for b in all_bets if b["clv"] > 0)

    return {
        "avg_clv": round(avg_clv, 4),
        "total_bets": len(all_bets),
        "positive_clv_pct": round(positive / len(all_bets), 4),
        "daily_series": daily_series,
    }


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="CLV tracking")
    parser.add_argument("--days", type=int, default=30, help="Days to analyze")
    parser.add_argument("--date", help="Compute CLV for a specific date")
    args = parser.parse_args()

    if args.date:
        results = compute_daily_clv(args.date)
        for r in results:
            print(
                f"  {r['selection']:4s} {r['home_team']}v{r['away_team']}  "
                f"Model: {r['model_prob']:.1%}  Close: {r['closing_prob']:.1%}  "
                f"CLV: {r['clv']:+.1%}"
            )
    else:
        summary = compute_clv_summary(days=args.days)
        print(f"CLV Summary (last {args.days} days)")
        print(f"  Avg CLV: {summary['avg_clv']:+.2%}")
        print(f"  Total bets: {summary['total_bets']}")
        print(f"  Positive CLV: {summary['positive_clv_pct']:.0%}")


if __name__ == "__main__":
    main()
