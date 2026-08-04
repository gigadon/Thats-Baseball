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


class _ClosingLines:
    """Closing home-win probabilities for a date, addressable by game.

    The Odds API only knows the team pair, so the snapshot files this reads used
    to collapse a doubleheader into one series and hand both bets the same
    closing line. Snapshots now carry the MLB game_id (mlb.data.line_movement);
    where they don't — files written before that — an ambiguous pair yields
    nothing rather than the wrong leg's line.
    """

    def __init__(self, games: list[dict]):
        self._by_id: dict[str, float] = {}
        self._by_pair: dict[tuple[str, str], list[float]] = {}
        for g in games:
            prob = g.get("home_prob")
            if prob is None:
                continue
            game_id = str(g.get("game_id") or "")
            if game_id:
                self._by_id[game_id] = prob
            self._by_pair.setdefault(
                (g.get("home_team", ""), g.get("away_team", "")), []
            ).append(prob)

    def lookup(self, game_id: Any, home: str, away: str) -> float | None:
        gid = str(game_id or "")
        if gid and gid in self._by_id:
            return self._by_id[gid]

        candidates = self._by_pair.get((home, away), [])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            logger.warning(
                "Ambiguous closing line for %s@%s (%d games) — skipping CLV",
                away, home, len(candidates),
            )
        return None


def _closing_probs(target_date: str, data_dir: Path) -> _ClosingLines:
    """The last snapshot of the day — the closing line."""
    lm_path = data_dir / "line_movement" / f"{target_date}.json"
    if not lm_path.exists():
        return _ClosingLines([])

    snapshots = json.loads(lm_path.read_text()).get("snapshots", [])
    if not snapshots:
        return _ClosingLines([])
    return _ClosingLines(snapshots[-1].get("games", []))


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

    closing = _closing_probs(target_date, data_dir)

    results = []
    for bet in slip["bets"]:
        home = bet.get("home_team", "")
        away = bet.get("away_team", "")

        model_prob = bet.get("model_prob", 0.5)
        closing_home_prob = closing.lookup(bet.get("game_id"), home, away)

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
