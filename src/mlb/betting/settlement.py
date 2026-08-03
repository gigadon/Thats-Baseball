"""Bet settlement — match the card that was sent against actual scores, track P&L.

Loads a day's betting slip from the locked slate (mlb.etl.slate_record), fetches
final scores via mlb.data.scores, settles each bet, and persists results to
data/betting/{date}.json.

Usage:
    python -m mlb.betting.settlement                    # Settle today
    python -m mlb.betting.settlement --date 2026-05-11  # Settle specific date
    python -m mlb.betting.settlement --backfill 7       # Settle last 7 days
    python -m mlb.betting.settlement --date X --force   # Re-settle a settled day
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

from mlb.betting.engine import american_to_decimal

logger = logging.getLogger(__name__)


def _today_et() -> date:
    """Today in US Eastern — naive date.today() is tomorrow on UTC hosts after 8 PM ET."""
    return datetime.now(ZoneInfo("America/New_York")).date()


# ── Core settlement ──────────────────────────────────────────


async def settle_day(
    target_date: date,
    data_dir: Path = Path("data"),
) -> dict | None:
    """Settle all bets for a given date.

    Settles the card that was actually sent — the slip locked when the main card
    went out (mlb.etl.slate_record). Days from before the lock existed fall back to
    the prediction cache.

    Returns the settlement dict, or None if there's nothing to settle.
    """
    from mlb.data.scores import load_scorebook
    from mlb.etl.slate_record import load_slate

    date_str = target_date.isoformat()
    slate = load_slate(target_date, data_dir)

    if slate is not None:
        slip = slate.get("betting_slip")
    else:
        pred_file = data_dir / "predictions" / f"{date_str}.json"
        if not pred_file.exists():
            logger.info("No prediction file for %s", date_str)
            return None

        with open(pred_file) as f:
            pred_data = json.load(f)
        slip = pred_data.get("betting_slip")

    if not slip or not slip.get("bets"):
        logger.info("No bets to settle for %s", date_str)
        return None

    # Final scores, keyed by game_id so a doubleheader settles leg by leg.
    book = await load_scorebook(target_date, data_dir)
    if not len(book):
        logger.warning("No scores available for %s", date_str)
        return None

    # Settle each bet
    settled_bets = []
    for bet in slip["bets"]:
        game_score = book.lookup(
            bet.get("game_id"), bet["home_team"], bet["away_team"]
        )

        if game_score is None:
            logger.warning(
                "No score found for %s vs %s — skipping",
                bet["away_team"], bet["home_team"],
            )
            continue

        settled = _settle_single_bet(
            bet, game_score.home_score, game_score.away_score
        )
        settled_bets.append(settled)

    if not settled_bets:
        logger.info("No bets could be matched to scores for %s", date_str)
        return None

    # Compute summary
    summary = _compute_summary(settled_bets, data_dir, target_date)
    summary["bets_on_card"] = len(slip["bets"])

    result = {
        "date": date_str,
        "settled_at": datetime.now().isoformat(timespec="seconds"),
        "bankroll": slip.get("bankroll", 10000.0),
        "bets": settled_bets,
        "summary": summary,
    }

    # Persist
    betting_dir = data_dir / "betting"
    betting_dir.mkdir(parents=True, exist_ok=True)
    out_file = betting_dir / f"{date_str}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    # Settling a gap day invalidates the running total on every later file, so
    # re-walk the chain. Settling the newest day (the normal case) is a no-op.
    rewritten = rechain_settlements(data_dir)
    if date_str in rewritten:
        result["summary"] = rewritten[date_str]

    logger.info(
        "Settled %d of %d bets for %s — P&L: $%.2f",
        len(settled_bets), summary["bets_on_card"], date_str, summary["daily_pnl"],
    )
    return result


def _settle_single_bet(
    bet: dict[str, Any], home_score: int, away_score: int
) -> dict[str, Any]:
    """Settle one bet against actual scores. Returns augmented bet dict."""
    settled = {**bet}
    settled["actual_home_score"] = home_score
    settled["actual_away_score"] = away_score

    result = _determine_result(bet, home_score, away_score)
    settled["result"] = result

    stake = bet.get("recommended_stake", 0)
    dec_odds = bet.get("decimal_odds") or american_to_decimal(bet["odds"])

    if result == "win":
        settled["pnl"] = round(stake * (dec_odds - 1), 2)
    elif result == "loss":
        settled["pnl"] = round(-stake, 2)
    else:  # push
        settled["pnl"] = 0.0

    return settled


def _determine_result(
    bet: dict, home_score: int, away_score: int
) -> str:
    """Determine bet outcome: 'win', 'loss', or 'push'."""
    bet_type = bet.get("bet_type", "moneyline")
    selection = bet.get("selection", "")

    if bet_type == "moneyline":
        if home_score == away_score:
            return "push"
        if selection == "home":
            return "win" if home_score > away_score else "loss"
        return "win" if away_score > home_score else "loss"

    elif bet_type == "total":
        total = home_score + away_score
        line = bet.get("total_line") or 8.5
        if total == line:
            return "push"
        if selection == "over":
            return "win" if total > line else "loss"
        return "win" if total < line else "loss"

    return "push"


def _compute_summary(
    settled_bets: list[dict], data_dir: Path, target_date: date
) -> dict[str, Any]:
    """Compute daily and cumulative P&L summary."""
    bets_won = sum(1 for b in settled_bets if b["result"] == "win")
    bets_lost = sum(1 for b in settled_bets if b["result"] == "loss")
    bets_pushed = sum(1 for b in settled_bets if b["result"] == "push")
    total_staked = sum(b.get("recommended_stake", 0) for b in settled_bets)
    daily_pnl = sum(b.get("pnl", 0) for b in settled_bets)

    # Chain off the last settlement *before* this date, not the newest file on
    # disk — backfilling a gap day would otherwise pick up a later day's total.
    prior = [
        s for s in load_all_settlements(data_dir)
        if s["date"] < target_date.isoformat()
    ]
    prior_cumulative = prior[-1]["summary"]["cumulative_pnl"] if prior else 0.0
    cumulative_pnl = round(prior_cumulative + daily_pnl, 2)

    # Max drawdown: track peak cumulative P&L
    peak = 0.0
    max_dd = 0.0
    for s in prior:
        cp = s["summary"]["cumulative_pnl"]
        peak = max(peak, cp)
        dd = (peak - cp) / max(peak, 1) if peak > 0 else 0
        max_dd = max(max_dd, dd)
    # Include today
    peak = max(peak, cumulative_pnl)
    dd = (peak - cumulative_pnl) / max(peak, 1) if peak > 0 else 0
    max_dd = max(max_dd, dd)

    return {
        "bets_placed": len(settled_bets),
        "bets_won": bets_won,
        "bets_lost": bets_lost,
        "bets_pushed": bets_pushed,
        "total_staked": round(total_staked, 2),
        "daily_pnl": round(daily_pnl, 2),
        "roi": round(daily_pnl / total_staked, 4) if total_staked > 0 else 0.0,
        "cumulative_pnl": cumulative_pnl,
        "max_drawdown": round(max_dd, 4),
    }


def rechain_settlements(data_dir: Path = Path("data")) -> dict[str, dict]:
    """Re-walk cumulative P&L and drawdown across every settlement, in date order.

    Daily P&L belongs to its day alone, but the running total doesn't: settling a
    gap day after later days already exist leaves every file after it wrong.
    Rewrites only the files whose numbers actually change, and returns
    {date: summary} for those.
    """
    rewritten: dict[str, dict] = {}
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0

    for settlement in load_all_settlements(data_dir):
        summary = settlement["summary"]
        cumulative = round(cumulative + summary["daily_pnl"], 2)
        peak = max(peak, cumulative)
        dd = (peak - cumulative) / max(peak, 1) if peak > 0 else 0
        max_dd = max(max_dd, dd)

        if (
            summary.get("cumulative_pnl") != cumulative
            or summary.get("max_drawdown") != round(max_dd, 4)
        ):
            summary["cumulative_pnl"] = cumulative
            summary["max_drawdown"] = round(max_dd, 4)
            path = data_dir / "betting" / f"{settlement['date']}.json"
            with open(path, "w") as f:
                json.dump(settlement, f, indent=2)
            rewritten[settlement["date"]] = summary
            logger.info(
                "Rechained %s: cumulative $%.2f", settlement["date"], cumulative
            )

    return rewritten


# ── File I/O ─────────────────────────────────────────────────


def load_settlement(
    target_date: date, data_dir: Path = Path("data")
) -> dict | None:
    """Load settlement results for a specific date."""
    path = data_dir / "betting" / f"{target_date.isoformat()}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_all_settlements(data_dir: Path = Path("data")) -> list[dict]:
    """Load all settlement files, sorted by date."""
    betting_dir = data_dir / "betting"
    if not betting_dir.exists():
        return []

    settlements = []
    for path in sorted(betting_dir.glob("*.json")):
        with open(path) as f:
            settlements.append(json.load(f))
    return settlements


# ── CLI ──────────────────────────────────────────────────────


def main():
    import argparse
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Settle MLB bets against final scores")
    parser.add_argument("--date", type=str, help="Date to settle (YYYY-MM-DD)")
    parser.add_argument("--backfill", type=int, help="Settle last N days")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-settle days that already have a settlement file",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    async def run():
        from mlb.etl.slate_record import slate_path

        if args.backfill:
            today = _today_et()
            for i in range(args.backfill, 0, -1):
                d = today - timedelta(days=i)
                settlement_file = data_dir / "betting" / f"{d.isoformat()}.json"
                pred_file = data_dir / "predictions" / f"{d.isoformat()}.json"
                if settlement_file.exists() and not args.force:
                    logger.info("Already settled %s — skipping", d)
                    continue
                if not slate_path(d, data_dir).exists() and not pred_file.exists():
                    continue
                result = await settle_day(d, data_dir)
                if result:
                    s = result["summary"]
                    print(
                        f"  {d}: {s['bets_won']}W-{s['bets_lost']}L  "
                        f"P&L: ${s['daily_pnl']:+.2f}  "
                        f"Cumulative: ${s['cumulative_pnl']:+.2f}"
                    )
        else:
            target = date.fromisoformat(args.date) if args.date else _today_et()
            result = await settle_day(target, data_dir)
            if result:
                s = result["summary"]
                print(f"\nSettlement for {target}:")
                print(f"  Record: {s['bets_won']}W-{s['bets_lost']}L-{s['bets_pushed']}P")
                print(f"  Staked: ${s['total_staked']:.2f}")
                print(f"  Daily P&L: ${s['daily_pnl']:+.2f}")
                print(f"  ROI: {s['roi']:.1%}")
                print(f"  Cumulative P&L: ${s['cumulative_pnl']:+.2f}")
                print(f"  Max Drawdown: {s['max_drawdown']:.1%}")

                print(f"\n  Bets:")
                for b in result["bets"]:
                    emoji = "W" if b["result"] == "win" else ("L" if b["result"] == "loss" else "P")
                    print(
                        f"    [{emoji}] {b['bet_type'].upper()} {b['selection']}  "
                        f"{b['away_team']}@{b['home_team']}  "
                        f"Score: {b['actual_away_score']}-{b['actual_home_score']}  "
                        f"P&L: ${b['pnl']:+.2f}"
                    )
            else:
                print(f"Nothing to settle for {target}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
