"""Grade a freshly trained model against the one currently being served.

A retrain is only worth promoting if it is not worse than what it replaces, and
"not worse" has to be measured on the *same* out-of-sample window — the two
models have different train cutoffs, so letting each pick its own would compare
different games and prove nothing. This pins both to the later cutoff (the
staging model's), which is out-of-sample for both.

Usage:
    PYTHONPATH=src python3 -m mlb.models.retrain_report \
        --staging-win models/win_model_new --served-win models/win_model \
        --staging-runs models/runs_model_new --served-runs models/runs_model

Exits non-zero if a gate fails, so CI goes red without anyone reading the log.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib

from mlb.models.backtest import Backtester
from mlb.models.backtest_totals import TotalsBacktester

logger = logging.getLogger(__name__)

# How much worse than the served model the candidate may be before the gate
# fails. Both windows carry a few thousand games, so run-to-run noise moves AUC
# by a few thousandths; anything past that is a real regression, not variance.
AUC_TOLERANCE = 0.005
CALIB_TOLERANCE = 0.010


@dataclass
class Gate:
    name: str
    staging: float
    served: float
    tolerance: float
    higher_is_better: bool

    @property
    def delta(self) -> float:
        return self.staging - self.served

    @property
    def passed(self) -> bool:
        if self.higher_is_better:
            return self.staging >= self.served - self.tolerance
        return self.staging <= self.served + self.tolerance

    def line(self) -> str:
        arrow = "+" if self.delta >= 0 else ""
        mark = ":white_check_mark:" if self.passed else ":x:"
        return (
            f"{mark} {self.name}: {self.staging:.4f} vs {self.served:.4f} served "
            f"({arrow}{self.delta:.4f})"
        )


def _train_cutoff(model_dir: Path) -> str | None:
    meta_path = model_dir / "pipeline_meta.joblib"
    if not meta_path.exists():
        return None
    try:
        return joblib.load(meta_path).get("train_cutoff_date")
    except Exception:
        logger.warning("Could not read train cutoff from %s", meta_path)
        return None


def _common_oos_start(staging: Path, served: Path) -> str | None:
    """The later of the two cutoffs — out-of-sample for both models."""
    cutoffs = [c for c in (_train_cutoff(staging), _train_cutoff(served)) if c]
    return max(cutoffs) if cutoffs else None


def run_win_gates(
    staging: Path, served: Path, data_dir: Path
) -> tuple[list[Gate], list[str]]:
    oos_start = _common_oos_start(staging, served)
    logger.info("Win backtest window starts %s", oos_start or "(derived 80/20)")

    def bt(model_dir: Path):
        return Backtester(
            data_dir=data_dir, model_dir=model_dir, oos_start=oos_start
        ).run()

    new, old = bt(staging), bt(served)

    gates = [
        Gate("Win AUC", new.model_auc, old.model_auc, AUC_TOLERANCE, True),
        Gate("Win Brier", new.model_brier, old.model_brier, AUC_TOLERANCE, False),
    ]
    detail = [
        f"OOS from {new.oos_start} · {new.total_games} games "
        f"({new.games_with_odds} with pregame odds)",
        f"Market AUC {new.market_auc:.4f} · model accuracy {new.model_accuracy:.1%}",
        f"Headline edge {new.headline_edge:.0%}: {new.headline_bets} bets, "
        f"flat ROI {new.headline_flat_roi:+.1%} (t={new.headline_t_stat:.2f})",
    ]
    return gates, detail


def run_totals_gates(
    staging: Path, served: Path, data_dir: Path
) -> tuple[list[Gate], list[str]]:
    # A win-model-only retrain leaves models/runs_model_new absent. Skipping is
    # right here: there is no candidate to grade, so there is nothing to gate.
    # Never treat a missing *served* runs model the same way — that would drop a
    # real gate silently.
    if not (staging / "runs_regressor.joblib").exists():
        logger.info("No staging runs model at %s — skipping totals gates", staging)
        return [], [f"skipped: no staging runs model at {staging}"]

    def bt(model_dir: Path):
        return TotalsBacktester(data_dir=data_dir, model_dir=model_dir).run()

    new, old = bt(staging), bt(served)

    gates = [
        Gate(
            "Totals calibration gap",
            new.mean_abs_calib_gap, old.mean_abs_calib_gap, CALIB_TOLERANCE, False,
        ),
        Gate("Totals Brier", new.brier_over, old.brier_over, AUC_TOLERANCE, False),
    ]
    detail = [
        f"OOS from {new.oos_start} · {new.total_games} games ({new.mode})",
        f"P(over) {new.pred_over_rate:.3f} predicted vs {new.actual_over_rate:.3f} actual",
    ]
    return gates, detail


def format_report(
    win_gates: list[Gate], win_detail: list[str],
    totals_gates: list[Gate], totals_detail: list[str],
    passed: bool,
) -> str:
    verdict = (
        ":white_check_mark: *Gates passed — safe to promote*" if passed
        else ":x: *Gates failed — do not promote*"
    )
    lines = [
        "*Weekly Retrain — staging candidate*",
        verdict,
        "",
        "*Win model*",
        *[g.line() for g in win_gates],
        *[f"_{d}_" for d in win_detail],
        "",
        "*Runs model*",
        *[g.line() for g in totals_gates],
        *[f"_{d}_" for d in totals_detail],
        "",
        "_Staging models are attached as workflow artifacts; promotion is manual._",
    ]
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Grade a staging retrain against the served model")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--staging-win", type=Path, default=Path("models/win_model_new"))
    p.add_argument("--served-win", type=Path, default=Path("models/win_model"))
    p.add_argument("--staging-runs", type=Path, default=Path("models/runs_model_new"))
    p.add_argument("--served-runs", type=Path, default=Path("models/runs_model"))
    p.add_argument("--slack", action="store_true", help="Post the report to Slack")
    args = p.parse_args()

    win_gates, win_detail = run_win_gates(args.staging_win, args.served_win, args.data_dir)
    totals_gates, totals_detail = run_totals_gates(
        args.staging_runs, args.served_runs, args.data_dir
    )

    all_gates = win_gates + totals_gates
    passed = all(g.passed for g in all_gates)
    report = format_report(win_gates, win_detail, totals_gates, totals_detail, passed)

    print()
    print(report.replace(":white_check_mark:", "PASS").replace(":x:", "FAIL"))
    print()

    if args.slack:
        from mlb.alerts import AlertService

        asyncio.run(AlertService().send_alert(report))

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
