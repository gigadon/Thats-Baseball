"""Honest historical backtester — evaluates the model against the REAL market.

Critical methodology (vs the old version, which was invalid):
  * Bets are measured against the **real, de-vigged market line**
    (``market_home_prob``) and settled at the **real moneyline prices** from
    ``data/odds_history.csv`` — NOT against a synthetic line fabricated from
    the model's own prediction.
  * Only **out-of-sample** games are scored — those after the model's
    train/test cutoff — so we never grade the model on games it trained on.
  * Edge is reported with a **bootstrap confidence interval and t-stat**, and
    a favorite-betting baseline (which should ≈ −vig) confirms the mechanics.

Usage:
    PYTHONPATH=src python -m mlb.models.backtest
    PYTHONPATH=src python -m mlb.models.backtest --oos-start 2024-08-25 --min-edge 0.03
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mlb.models.pipeline import TrainingPipeline

logger = logging.getLogger(__name__)

# Columns that are not features (mirror of pipeline.NON_FEATURE_COLS).
_META_COLS = [
    "game_id", "game_date", "home_team", "away_team",
    "home_score", "away_score", "home_win", "total_runs", "season",
]


@dataclass
class BacktestResult:
    """Summary of an honest out-of-sample backtest."""

    oos_start: str
    total_games: int          # out-of-sample games scored
    games_with_odds: int      # games matched to a real moneyline

    # Discrimination/calibration — model vs the de-vigged market line
    model_auc: float
    model_brier: float
    model_accuracy: float
    market_auc: float
    market_brier: float
    market_accuracy: float
    mean_prob_gap: float      # mean |model − market| probability

    # Edge-threshold sweep: one dict per min_edge
    thresholds: list[dict] = field(default_factory=list)

    # Significance at the headline edge (bootstrap over per-bet returns)
    headline_edge: float = 0.0
    headline_bets: int = 0
    headline_flat_roi: float = 0.0
    headline_t_stat: float = 0.0
    headline_ci: tuple[float, float] = (0.0, 0.0)
    headline_p_positive: float = 0.0

    # Sanity baseline: flat-bet the market favorite every game (≈ −vig)
    favorite_baseline_roi: float = 0.0

    calibration: list[dict] = field(default_factory=list)
    monthly: list[dict] = field(default_factory=list)


def _american_to_decimal(odds: float) -> float:
    """American moneyline → decimal odds."""
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))


class Backtester:
    """Walk-forward-style backtest of model predictions against the real market."""

    def __init__(
        self,
        data_dir: Path = Path("data"),
        model_dir: Path = Path("models/win_model"),
        bankroll: float = 10000.0,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.03,
        flat_bet_size: float = 100.0,
        oos_start: str | None = None,
        bootstrap_iters: int = 5000,
        seed: int = 0,
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.initial_bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.flat_bet_size = flat_bet_size
        self.oos_start = oos_start
        self.bootstrap_iters = bootstrap_iters
        self.seed = seed

    # ── Out-of-sample cutoff ──────────────────────────────────
    def _resolve_oos_start(self, df_all: pd.DataFrame, pipeline: TrainingPipeline) -> pd.Timestamp:
        """Determine the first out-of-sample date.

        Priority: explicit --oos-start → cutoff stored on the model →
        recompute the pipeline's time-based train/test split.
        """
        if self.oos_start:
            return pd.Timestamp(self.oos_start)

        stored = getattr(pipeline, "train_cutoff_", None)
        if stored is not None:
            logger.info("Using train cutoff stored on the model: %s", stored)
            return pd.Timestamp(stored)

        # Recompute the 80/20 time split the pipeline uses by default.
        test_size = pipeline.config.test_size
        dates = df_all["game_date"].sort_values().reset_index(drop=True)
        split = int(len(dates) * (1 - test_size))
        cutoff = dates.iloc[split]
        logger.warning(
            "Model has no stored train cutoff; deriving 80/20 split → OOS starts %s. "
            "Retrain to persist train_cutoff_date for exactness.", cutoff.date(),
        )
        return cutoff

    # ── Odds ──────────────────────────────────────────────────
    def _load_odds(self) -> dict[tuple, tuple[float, float]]:
        """Real moneylines keyed by (game_date, home_team, away_team)."""
        path = self.data_dir / "odds_history.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — the honest backtest needs real moneylines."
            )
        odds = pd.read_csv(path)
        odds["game_date"] = pd.to_datetime(odds["game_date"])
        return {
            (r.game_date, r.home_team, r.away_team): (float(r.home_moneyline), float(r.away_moneyline))
            for r in odds.itertuples()
            if pd.notna(r.home_moneyline) and pd.notna(r.away_moneyline)
        }

    def run(self, seasons: list[int] | None = None) -> BacktestResult:
        from sklearn.metrics import roc_auc_score, brier_score_loss

        df_all = pd.read_parquet(self.data_dir / "training_data.parquet")
        df_all["game_date"] = pd.to_datetime(df_all["game_date"])
        df_all = df_all.sort_values("game_date").reset_index(drop=True)

        pipeline = TrainingPipeline()
        pipeline.load(self.model_dir)

        oos_start = self._resolve_oos_start(df_all, pipeline)
        df = df_all[df_all["game_date"] >= oos_start].reset_index(drop=True)
        if seasons:
            df = df[df["game_date"].dt.year.isin(seasons)].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No out-of-sample games on/after {oos_start.date()}")

        logger.info(
            "Out-of-sample backtest: %d games (%s → %s)",
            len(df), df["game_date"].min().date(), df["game_date"].max().date(),
        )

        # Predict
        feature_cols = [c for c in df.columns if c not in _META_COLS]
        p_model = np.asarray(pipeline.predict(df[feature_cols]))
        p_mkt = df["market_home_prob"].values.astype(float)
        y = df["home_win"].values.astype(int)

        oddmap = self._load_odds()

        # Discrimination / calibration vs market
        model_auc = float(roc_auc_score(y, p_model))
        market_auc = float(roc_auc_score(y, p_mkt))

        # Per-game betting records (computed once at edge 0, filtered per threshold)
        records = self._build_records(df, p_model, p_mkt, y, oddmap)
        games_with_odds = len(records)

        thresholds = [
            self._simulate(records, me) for me in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08)
        ]

        # Significance at the headline edge
        rets = self._per_bet_returns(records, self.min_edge)
        sig = self._significance(rets)

        # Baseline: flat-bet the market favorite (should ≈ −vig)
        fav = self._favorite_baseline(df, p_mkt, y, oddmap)

        return BacktestResult(
            oos_start=str(oos_start.date()),
            total_games=len(df),
            games_with_odds=games_with_odds,
            model_auc=round(model_auc, 4),
            model_brier=round(float(brier_score_loss(y, p_model)), 4),
            model_accuracy=round(float(((p_model >= 0.5) == y).mean()), 4),
            market_auc=round(market_auc, 4),
            market_brier=round(float(brier_score_loss(y, p_mkt)), 4),
            market_accuracy=round(float(((p_mkt >= 0.5) == y).mean()), 4),
            mean_prob_gap=round(float(np.abs(p_model - p_mkt).mean()), 4),
            thresholds=thresholds,
            headline_edge=self.min_edge,
            headline_bets=len(rets),
            headline_flat_roi=round(sig["roi"], 4),
            headline_t_stat=round(sig["t"], 3),
            headline_ci=(round(sig["ci_low"], 4), round(sig["ci_high"], 4)),
            headline_p_positive=round(sig["p_positive"], 4),
            favorite_baseline_roi=round(fav, 4),
            calibration=self._compute_calibration(p_model, y),
            monthly=self._monthly(df, p_model, p_mkt, y, oddmap),
        )

    # ── Internals ─────────────────────────────────────────────
    def _build_records(self, df, p_model, p_mkt, y, oddmap) -> list[dict]:
        """One record per game that has a real moneyline, with the side the
        model would back and its signed edge vs the de-vigged market line."""
        recs = []
        for i in range(len(df)):
            key = (df["game_date"].iloc[i], df["home_team"].iloc[i], df["away_team"].iloc[i])
            if key not in oddmap:
                continue
            hml, aml = oddmap[key]
            pm, mk, hw = float(p_model[i]), float(p_mkt[i]), int(y[i])
            edge_home = pm - mk  # >0 → model likes home more than market
            if edge_home >= 0:
                side, price, p, edge = "home", hml, pm, edge_home
            else:
                side, price, p, edge = "away", aml, 1.0 - pm, -edge_home
            won = (side == "home" and hw == 1) or (side == "away" and hw == 0)
            recs.append({"edge": edge, "side": side, "price": price, "p": p, "won": won})
        return recs

    def _simulate(self, records, min_edge: float) -> dict:
        """Flat + fractional-Kelly P&L over records with edge ≥ min_edge."""
        bk = self.initial_bankroll
        wagered = 0.0
        flat_pnl = 0.0
        w = l = 0
        for r in records:
            if r["edge"] < min_edge:
                continue
            d = _american_to_decimal(r["price"])
            b = d - 1.0
            p = r["p"]
            kelly = max(0.0, (b * p - (1 - p)) / b) * self.kelly_fraction
            stake = min(bk * kelly, bk * 0.05)
            if stake < 1:
                continue
            if r["won"]:
                bk += stake * b
                flat_pnl += self.flat_bet_size * b
                w += 1
            else:
                bk -= stake
                flat_pnl -= self.flat_bet_size
                l += 1
            wagered += stake
        nb = w + l
        return {
            "min_edge": min_edge,
            "bets": nb,
            "win_rate": round(w / nb, 4) if nb else 0.0,
            "flat_roi": round(flat_pnl / (nb * self.flat_bet_size), 4) if nb else 0.0,
            "kelly_roi": round((bk - self.initial_bankroll) / wagered, 4) if wagered > 0 else 0.0,
            "flat_pnl": round(flat_pnl, 2),
            "end_bankroll": round(bk, 2),
        }

    def _per_bet_returns(self, records, min_edge: float) -> np.ndarray:
        """Per-bet unit returns (decimal−1 if won else −1) for flat bets ≥ min_edge."""
        out = []
        for r in records:
            if r["edge"] < min_edge:
                continue
            b = _american_to_decimal(r["price"]) - 1.0
            out.append(b if r["won"] else -1.0)
        return np.array(out, dtype=float)

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

    def _favorite_baseline(self, df, p_mkt, y, oddmap) -> float:
        rets = []
        for i in range(len(df)):
            key = (df["game_date"].iloc[i], df["home_team"].iloc[i], df["away_team"].iloc[i])
            if key not in oddmap:
                continue
            hml, aml = oddmap[key]
            hw = int(y[i])
            if p_mkt[i] >= 0.5:
                price, won = hml, hw == 1
            else:
                price, won = aml, hw == 0
            b = _american_to_decimal(price) - 1.0
            rets.append(b if won else -1.0)
        return float(np.mean(rets)) if rets else 0.0

    def _monthly(self, df, p_model, p_mkt, y, oddmap) -> list[dict]:
        recs = self._build_records(df, p_model, p_mkt, y, oddmap)
        # Attach month by re-walking (records align to odds-matched games)
        months: dict[str, dict] = {}
        ri = 0
        for i in range(len(df)):
            key = (df["game_date"].iloc[i], df["home_team"].iloc[i], df["away_team"].iloc[i])
            if key not in oddmap:
                continue
            mk = df["game_date"].iloc[i].strftime("%Y-%m")
            m = months.setdefault(mk, {"month": mk, "games": 0, "bets": 0, "wins": 0, "flat_pnl": 0.0})
            m["games"] += 1
            r = recs[ri]; ri += 1
            if r["edge"] >= self.min_edge:
                b = _american_to_decimal(r["price"]) - 1.0
                m["bets"] += 1
                if r["won"]:
                    m["wins"] += 1
                    m["flat_pnl"] += self.flat_bet_size * b
                else:
                    m["flat_pnl"] -= self.flat_bet_size
        for m in months.values():
            m["flat_pnl"] = round(m["flat_pnl"], 2)
        return sorted(months.values(), key=lambda x: x["month"])

    def _compute_calibration(self, predictions, actuals, n_bins: int = 10) -> list[dict]:
        edges = np.linspace(0, 1, n_bins + 1)
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (predictions >= lo) & (predictions < hi)
            count = int(mask.sum())
            if count == 0:
                continue
            bins.append({
                "bin": f"{lo:.1f}-{hi:.1f}",
                "predicted_avg": round(float(predictions[mask].mean()), 4),
                "observed_avg": round(float(actuals[mask].mean()), 4),
                "count": count,
                "gap": round(abs(float(predictions[mask].mean()) - float(actuals[mask].mean())), 4),
            })
        return bins


# ── CLI ───────────────────────────────────────────────────────


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Honest out-of-sample MLB betting backtest")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-dir", type=str, default="models/win_model")
    parser.add_argument("--oos-start", type=str, default=None,
                        help="First out-of-sample date YYYY-MM-DD (default: derive train/test cutoff)")
    parser.add_argument("--seasons", type=int, nargs="+", default=None,
                        help="Optional: restrict OOS games to these seasons")
    parser.add_argument("--bankroll", type=float, default=10000)
    parser.add_argument("--kelly", type=float, default=0.25)
    parser.add_argument("--min-edge", type=float, default=0.03)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if model_dir.name == "models" and (model_dir / "win_model").exists():
        model_dir = model_dir / "win_model"

    bt = Backtester(
        data_dir=Path(args.data_dir),
        model_dir=model_dir,
        bankroll=args.bankroll,
        kelly_fraction=args.kelly,
        min_edge=args.min_edge,
        oos_start=args.oos_start,
    )
    r = bt.run(args.seasons)

    print(f"\n{'='*64}")
    print(f"  HONEST OUT-OF-SAMPLE BACKTEST  (OOS from {r.oos_start})")
    print(f"{'='*64}")
    print(f"  Games scored:      {r.total_games}  (with real odds: {r.games_with_odds})")
    print()
    print(f"  PREDICTION vs MARKET")
    print(f"  {'':14}{'AUC':>8}{'Brier':>9}{'Acc':>8}")
    print(f"  {'Model':14}{r.model_auc:>8.4f}{r.model_brier:>9.4f}{r.model_accuracy:>8.3f}")
    print(f"  {'Market':14}{r.market_auc:>8.4f}{r.market_brier:>9.4f}{r.market_accuracy:>8.3f}")
    print(f"  Mean |model−market| prob gap: {r.mean_prob_gap:.4f}")
    print()
    print(f"  BETTING vs REAL NO-VIG LINE, REAL PRICES")
    print(f"  {'min_edge':>8}{'bets':>7}{'win%':>8}{'flat_ROI':>10}{'kelly_ROI':>11}{'flat_P&L':>11}")
    for t in r.thresholds:
        print(f"  {t['min_edge']:>8.2f}{t['bets']:>7}{t['win_rate']:>7.1%}"
              f"{t['flat_roi']:>+9.2%}{t['kelly_roi']:>+10.2%} ${t['flat_pnl']:>+9,.0f}")
    print()
    print(f"  SIGNIFICANCE @ edge {r.headline_edge:.2f}  (n={r.headline_bets} bets)")
    print(f"  Flat ROI {r.headline_flat_roi:+.2%} | t={r.headline_t_stat:.2f} | "
          f"95% CI [{r.headline_ci[0]:+.2%}, {r.headline_ci[1]:+.2%}] | P(ROI>0)={r.headline_p_positive:.1%}")
    verdict = "SIGNIFICANT EDGE" if r.headline_t_stat >= 2 else "NOT significant — consistent with no edge"
    print(f"  Verdict: {verdict}")
    print()
    print(f"  SANITY: flat-bet market favorite ROI = {r.favorite_baseline_roi:+.2%}  (should ≈ −vig)")
    print()

    if r.monthly:
        print(f"  MONTHLY (flat bets @ edge {r.headline_edge:.2f})")
        print(f"  {'Month':<9}{'Games':>7}{'Bets':>6}{'Wins':>6}{'Flat P&L':>12}")
        for m in r.monthly:
            print(f"  {m['month']:<9}{m['games']:>7}{m['bets']:>6}{m['wins']:>6} ${m['flat_pnl']:>+9,.0f}")
        print()


if __name__ == "__main__":
    main()
