"""Honest over/under (totals) backtester — evaluates the runs model vs the market.

The moneyline analog lives in ``backtest.py``; this is its totals counterpart and
follows the same discipline:

  * **Out-of-sample only.** Games are scored on/after the runs model's train/test
    cutoff (an 80/20 time split, matching ``train_runs.py``), so the model is
    never graded on games it trained on.
  * **Real line, standard price.** The over/under line is ``market_total`` (the
    real posted total, carried in the training parquet). ``odds_history.csv``
    stores only the line, not the over/under juice, so bets are settled at the
    market-standard **−110 both sides** — which is exactly why the breakeven is
    **52.38%**.
  * **Same probability model as production.** ``P(over)`` comes from
    ``runs_calibration.over_under_probabilities`` (empirical residual CDF, else
    Normal), so a change that helps here helps the live engine too.
  * Edge is reported with a **bootstrap CI and t-stat**, plus a bet-every-game
    baseline (win rate ≈ 50%, ROI ≈ −vig) that confirms the mechanics.

Usage:
    PYTHONPATH=src python -m mlb.models.backtest_totals
    PYTHONPATH=src python -m mlb.models.backtest_totals --oos-start 2024-08-25 --min-edge 0.03
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mlb.models.backtest import _american_to_decimal
from mlb.models.runs_calibration import DEFAULT_RESIDUAL_STD, over_under_probabilities

logger = logging.getLogger(__name__)

# Over/under juice is assumed standard −110 on both sides (see module docstring).
STD_OU_PRICE = -110.0
BREAKEVEN = abs(STD_OU_PRICE) / (abs(STD_OU_PRICE) + 100.0)  # 0.5238 at −110

# Runs model's default train/test split (mirrors train_runs.train_runs_model).
_DEFAULT_TEST_SIZE = 0.20


@dataclass
class TotalsBacktestResult:
    oos_start: str
    total_games: int          # games actually scored (the evaluation slice)
    breakeven: float          # win rate needed to profit at −110
    mode: str                 # calibration mode (walk-forward vs in-sample)
    calib_games: int          # games used only to estimate the residual CDF

    # Calibration of P(over) vs the observed over-rate
    pred_over_rate: float     # mean model P(over)
    actual_over_rate: float   # fraction of games that went over
    brier_over: float         # Brier score of P(over) vs the over outcome
    mean_abs_calib_gap: float

    # Edge-threshold sweep (one dict per min_edge)
    thresholds: list[dict] = field(default_factory=list)

    # Significance at the headline edge
    headline_edge: float = 0.0
    headline_bets: int = 0
    headline_win_rate: float = 0.0
    headline_flat_roi: float = 0.0
    headline_t_stat: float = 0.0
    headline_ci: tuple[float, float] = (0.0, 0.0)
    headline_p_positive: float = 0.0

    # Sanity: bet the model's side on every game (edge ≥ 0)
    bet_all_win_rate: float = 0.0
    bet_all_roi: float = 0.0

    calibration: list[dict] = field(default_factory=list)


class TotalsBacktester:
    """Walk-forward-style backtest of runs-model over/under picks vs the market."""

    def __init__(
        self,
        data_dir: Path = Path("data"),
        model_dir: Path = Path("models/runs_model"),
        min_edge: float = 0.03,
        flat_bet_size: float = 100.0,
        oos_start: str | None = None,
        bootstrap_iters: int = 5000,
        seed: int = 0,
        calib_frac: float = 0.5,
        use_model_residuals: bool = False,
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.min_edge = min_edge
        self.flat_bet_size = flat_bet_size
        self.oos_start = oos_start
        self.bootstrap_iters = bootstrap_iters
        self.seed = seed
        # Walk-forward calibration: the earlier `calib_frac` of the OOS window
        # supplies the residuals used to build P(over) for the LATER games it is
        # then scored on. This keeps calibration strictly out-of-sample. Setting
        # use_model_residuals reverts to the model's own test residuals (which
        # overlap the eval window → in-sample; kept only for comparison).
        self.calib_frac = calib_frac
        self.use_model_residuals = use_model_residuals

    # ── Model ─────────────────────────────────────────────────
    def _load_model(self):
        regressor = joblib.load(self.model_dir / "runs_regressor.joblib")
        feature_names = joblib.load(self.model_dir / "runs_feature_names.joblib")

        residual_std = getattr(regressor, "residual_std", None)
        residuals = getattr(regressor, "residuals", None)
        if residual_std is None or residuals is None:
            metrics_path = self.model_dir / "runs_metrics.joblib"
            if metrics_path.exists():
                m = joblib.load(metrics_path)
                residual_std = residual_std if residual_std is not None else m.get("residual_std")
                residuals = residuals if residuals is not None else m.get("residuals")
        residual_std = float(residual_std) if residual_std else DEFAULT_RESIDUAL_STD
        residuals = np.asarray(residuals, dtype=float) if residuals is not None else None
        return regressor, feature_names, residual_std, residuals

    # ── OOS cutoff ────────────────────────────────────────────
    def _resolve_oos_start(self, df_all: pd.DataFrame) -> pd.Timestamp:
        if self.oos_start:
            return pd.Timestamp(self.oos_start)
        dates = df_all["game_date"].sort_values().reset_index(drop=True)
        split = int(len(dates) * (1 - _DEFAULT_TEST_SIZE))
        cutoff = dates.iloc[split]
        logger.warning(
            "Runs model stores no train cutoff; deriving %.0f/%.0f split → OOS starts %s.",
            (1 - _DEFAULT_TEST_SIZE) * 100, _DEFAULT_TEST_SIZE * 100, cutoff.date(),
        )
        return cutoff

    def run(self, seasons: list[int] | None = None) -> TotalsBacktestResult:
        from sklearn.metrics import brier_score_loss

        df_all = pd.read_parquet(self.data_dir / "training_data.parquet")
        df_all["game_date"] = pd.to_datetime(df_all["game_date"])
        df_all = df_all.sort_values("game_date").reset_index(drop=True)

        regressor, feature_names, residual_std, residuals = self._load_model()

        oos_start = self._resolve_oos_start(df_all)
        df = df_all[df_all["game_date"] >= oos_start].reset_index(drop=True)
        if seasons:
            df = df[df["game_date"].dt.year.isin(seasons)].reset_index(drop=True)

        # Need a real posted line and an actual total to settle against.
        df = df[df["market_total"].notna() & df["total_runs"].notna()].reset_index(drop=True)
        # Drop pushes (actual total exactly on the line) — no bet is settled.
        df = df[df["total_runs"] != df["market_total"]].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No settleable OOS totals games on/after {oos_start.date()}")

        logger.info(
            "Out-of-sample totals backtest: %d games (%s → %s)",
            len(df), df["game_date"].min().date(), df["game_date"].max().date(),
        )

        # Predict totals (verbatim runs-model output, clamped like inference).
        X = df[feature_names].to_numpy(dtype=float)
        pred_total = np.clip(np.asarray(regressor.predict(X), dtype=float), 3.0, 30.0)
        line = df["market_total"].to_numpy(dtype=float)
        actual_total = df["total_runs"].to_numpy(dtype=float)
        actual_over = actual_total > line  # pushes already removed

        # Pick the residuals that build P(over) and the slice of games to score.
        # Walk-forward (default): the earlier calib_frac supplies residuals for
        # the later games it is scored on — calibration stays out-of-sample.
        if self.use_model_residuals:
            calib_resid = residuals            # may be None → Normal(residual_std)
            ev = slice(0, len(df))
            mode, calib_games = "in-sample (model test residuals)", 0
        else:
            k = int(len(df) * self.calib_frac)
            if k < 50 or len(df) - k < 50:
                raise ValueError(
                    f"OOS window too small to split (n={len(df)}, calib={k}); "
                    "pass use_model_residuals=True or widen the window."
                )
            calib_resid = actual_total[:k] - pred_total[:k]
            ev = slice(k, len(df))
            mode, calib_games = "walk-forward", k

        eval_over = actual_over[ev]
        records, p_over_all = self._build_records(
            pred_total[ev], line[ev], eval_over, calib_resid, residual_std
        )

        thresholds = [self._simulate(records, me) for me in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08)]
        rets = self._per_bet_returns(records, self.min_edge)
        sig = self._significance(rets)
        base = self._simulate(records, 0.0)

        return TotalsBacktestResult(
            oos_start=str(oos_start.date()),
            total_games=len(records),
            breakeven=round(BREAKEVEN, 4),
            mode=mode,
            calib_games=calib_games,
            pred_over_rate=round(float(p_over_all.mean()), 4),
            actual_over_rate=round(float(eval_over.mean()), 4),
            brier_over=round(float(brier_score_loss(eval_over.astype(int), p_over_all)), 4),
            mean_abs_calib_gap=round(
                float(np.mean([b["gap"] for b in self._calibration(p_over_all, eval_over)]) or 0.0), 4
            ),
            thresholds=thresholds,
            headline_edge=self.min_edge,
            headline_bets=len(rets),
            headline_win_rate=base["win_rate"] if self.min_edge == 0 else self._simulate(records, self.min_edge)["win_rate"],
            headline_flat_roi=round(sig["roi"], 4),
            headline_t_stat=round(sig["t"], 3),
            headline_ci=(round(sig["ci_low"], 4), round(sig["ci_high"], 4)),
            headline_p_positive=round(sig["p_positive"], 4),
            bet_all_win_rate=base["win_rate"],
            bet_all_roi=base["flat_roi"],
            calibration=self._calibration(p_over_all, eval_over),
        )

    # ── Internals ─────────────────────────────────────────────
    def _build_records(self, pred_total, line, actual_over, residuals, residual_std):
        """One record per game: the side the model backs (over/under), its signed
        edge vs the no-vig fair 0.50, the −110 price, and whether it won."""
        recs = []
        p_over_all = np.empty(len(pred_total), dtype=float)
        for i in range(len(pred_total)):
            p_over, p_under = over_under_probabilities(
                float(pred_total[i]), float(line[i]),
                residuals=residuals, residual_std=residual_std,
            )
            p_over_all[i] = p_over
            if p_over >= p_under:
                side, p = "over", p_over
                won = bool(actual_over[i])
            else:
                side, p = "under", p_under
                won = not bool(actual_over[i])
            # No-vig fair prob is 0.50 per side at −110/−110.
            recs.append({"edge": p - 0.5, "side": side, "price": STD_OU_PRICE, "p": p, "won": won})
        return recs, p_over_all

    def _simulate(self, records, min_edge: float) -> dict:
        b = _american_to_decimal(STD_OU_PRICE) - 1.0
        flat_pnl = 0.0
        w = l = 0
        for r in records:
            if r["edge"] < min_edge:
                continue
            if r["won"]:
                flat_pnl += self.flat_bet_size * b
                w += 1
            else:
                flat_pnl -= self.flat_bet_size
                l += 1
        nb = w + l
        return {
            "min_edge": min_edge,
            "bets": nb,
            "win_rate": round(w / nb, 4) if nb else 0.0,
            "flat_roi": round(flat_pnl / (nb * self.flat_bet_size), 4) if nb else 0.0,
            "flat_pnl": round(flat_pnl, 2),
        }

    def _per_bet_returns(self, records, min_edge: float) -> np.ndarray:
        b = _american_to_decimal(STD_OU_PRICE) - 1.0
        return np.array(
            [b if r["won"] else -1.0 for r in records if r["edge"] >= min_edge],
            dtype=float,
        )

    def _significance(self, rets: np.ndarray) -> dict:
        if len(rets) < 2:
            return {"roi": 0.0, "t": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_positive": 0.0}
        roi = float(rets.mean())
        se = float(rets.std(ddof=1) / np.sqrt(len(rets)))
        rng = np.random.default_rng(self.seed)
        boot = np.array([
            rng.choice(rets, len(rets), replace=True).mean()
            for _ in range(self.bootstrap_iters)
        ])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {
            "roi": roi,
            "t": roi / se if se > 0 else 0.0,
            "ci_low": float(lo),
            "ci_high": float(hi),
            "p_positive": float((boot > 0).mean()),
        }

    def _calibration(self, p_over, actual_over, n_bins: int = 10) -> list[dict]:
        actual = np.asarray(actual_over, dtype=float)
        edges = np.linspace(0, 1, n_bins + 1)
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (p_over >= lo) & (p_over < hi)
            count = int(mask.sum())
            if count == 0:
                continue
            bins.append({
                "bin": f"{lo:.1f}-{hi:.1f}",
                "predicted_avg": round(float(p_over[mask].mean()), 4),
                "observed_avg": round(float(actual[mask].mean()), 4),
                "count": count,
                "gap": round(abs(float(p_over[mask].mean()) - float(actual[mask].mean())), 4),
            })
        return bins


# ── CLI ───────────────────────────────────────────────────────


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Honest out-of-sample MLB over/under backtest")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-dir", type=str, default="models/runs_model")
    parser.add_argument("--oos-start", type=str, default=None,
                        help="First out-of-sample date YYYY-MM-DD (default: derive 80/20 cutoff)")
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--calib-frac", type=float, default=0.5,
                        help="Fraction of the OOS window used only to estimate the residual CDF")
    parser.add_argument("--use-model-residuals", action="store_true",
                        help="Use the model's own test residuals (in-sample; for comparison only)")
    args = parser.parse_args()

    bt = TotalsBacktester(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
        min_edge=args.min_edge,
        oos_start=args.oos_start,
        calib_frac=args.calib_frac,
        use_model_residuals=args.use_model_residuals,
    )
    r = bt.run(args.seasons)

    print(f"\n{'='*64}")
    print(f"  HONEST OUT-OF-SAMPLE OVER/UNDER BACKTEST  (OOS from {r.oos_start})")
    print(f"{'='*64}")
    cal = f"in-sample" if r.calib_games == 0 else f"walk-forward: {r.calib_games} calib → {r.total_games} eval"
    print(f"  Games scored: {r.total_games}   Breakeven @ −110: {r.breakeven:.2%}   [{cal}]")
    print()
    print(f"  P(OVER) CALIBRATION")
    print(f"  Predicted over-rate {r.pred_over_rate:.3f}  |  Actual over-rate {r.actual_over_rate:.3f}")
    print(f"  Brier {r.brier_over:.4f}  |  mean |pred−obs| across bins {r.mean_abs_calib_gap:.4f}")
    print()
    print(f"  BETTING @ −110  (edge = model P − 0.50)")
    print(f"  {'min_edge':>8}{'bets':>7}{'win%':>8}{'flat_ROI':>10}{'flat_P&L':>11}")
    for t in r.thresholds:
        print(f"  {t['min_edge']:>8.2f}{t['bets']:>7}{t['win_rate']:>7.1%}"
              f"{t['flat_roi']:>+9.2%} ${t['flat_pnl']:>+9,.0f}")
    print()
    print(f"  SIGNIFICANCE @ edge {r.headline_edge:.2f}  (n={r.headline_bets} bets)")
    print(f"  Win {r.headline_win_rate:.1%} vs {r.breakeven:.1%} breakeven | "
          f"ROI {r.headline_flat_roi:+.2%} | t={r.headline_t_stat:.2f} | "
          f"95% CI [{r.headline_ci[0]:+.2%}, {r.headline_ci[1]:+.2%}] | P(ROI>0)={r.headline_p_positive:.1%}")
    verdict = "SIGNIFICANT EDGE" if r.headline_t_stat >= 2 else "NOT significant — consistent with no totals edge"
    print(f"  Verdict: {verdict}")
    print()
    print(f"  SANITY: bet model side every game → win {r.bet_all_win_rate:.1%}, "
          f"ROI {r.bet_all_roi:+.2%}  (should ≈ 50% / −vig with no edge)")
    print()


if __name__ == "__main__":
    main()
