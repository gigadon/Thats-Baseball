"""Feature-parity regression tests.

The June 2026 home-bias incident was caused by the live pipeline drifting from
the training pipeline (different formulas, stale defaults). These tests pin the
shared implementations (mlb.features.formulas / mlb.features.defaults) to the
artifacts they must stay consistent with, so drift fails CI instead of skewing
live predictions.
"""

from pathlib import Path

import pytest

from mlb.features.defaults import (
    CONVENTION_DEFAULTS,
    MEDIAN_FEATURES,
    load_defaults,
)
from mlb.features.formulas import (
    american_implied,
    bp_freshness_from_ip,
    compute_interaction_features,
    devig_home_prob,
    lineup_obp,
)

TRAINING_PARQUET = Path("data/training_data.parquet")
ODDS_HISTORY = Path("data/odds_history.csv")

needs_parquet = pytest.mark.skipif(
    not TRAINING_PARQUET.exists(), reason="training parquet not available"
)
needs_odds = pytest.mark.skipif(
    not ODDS_HISTORY.exists(), reason="odds history not available"
)


class TestFormulas:
    def test_american_implied(self):
        assert american_implied(-150) == pytest.approx(0.6)
        assert american_implied(150) == pytest.approx(0.4)
        assert american_implied(100) == pytest.approx(0.5)

    def test_devig_is_normalized(self):
        h = devig_home_prob(-130, 110)
        a = devig_home_prob(110, -130)
        assert h + a == pytest.approx(1.0)
        assert 0.5 < h < 0.6

    @needs_odds
    def test_devig_matches_odds_history(self):
        """odds_history.csv is training's market_home_prob source — the shared
        de-vig transform must reproduce it exactly (to CSV rounding)."""
        import pandas as pd

        odds = pd.read_csv(ODDS_HISTORY).dropna(
            subset=["home_moneyline", "away_moneyline", "market_home_prob"]
        )
        sample = odds.sample(n=min(200, len(odds)), random_state=42)
        for r in sample.itertuples():
            expected = devig_home_prob(r.home_moneyline, r.away_moneyline)
            assert expected == pytest.approx(r.market_home_prob, abs=1e-3), (
                f"devig mismatch for {r.home_team} vs {r.away_team} on {r.game_date}"
            )

    def test_lineup_obp_training_formula(self):
        # (H+BB)/(AB+BB), not the API OBP (no HBP/SF terms)
        assert lineup_obp(hits=150, walks=50, at_bats=500, default=0.3) == pytest.approx(200 / 550)
        assert lineup_obp(0, 0, 0, default=0.3046) == 0.3046

    def test_bp_freshness_stays_in_training_range(self):
        # Training bp_freshness spans ~0.91-0.96; the live mapping must never
        # leave [0.90, 0.96] (the old 1-ip/15 formula produced ~0.39 → home bias)
        for ip in (0.0, 3.0, 6.0, 10.0, 15.0, 30.0):
            assert 0.90 <= bp_freshness_from_ip(ip) <= 0.96

    @needs_parquet
    def test_interactions_reproduce_training_columns(self):
        """The shared interaction function must reproduce the interact_*
        columns stored in the training parquet (verbatim-extraction check)."""
        import pandas as pd

        df = pd.read_parquet(TRAINING_PARQUET)
        sample = df.sample(n=min(300, len(df)), random_state=42)
        interact_cols = [c for c in df.columns if c.startswith("interact_")]
        assert interact_cols, "no interact_* columns in training data"

        for _, row in sample.iterrows():
            feat = row.to_dict()
            out = compute_interaction_features(feat, row["park_runs_factor"])
            for col in interact_cols:
                assert out[col] == pytest.approx(row[col], abs=1e-9), (
                    f"{col} drifted from training for game {row['game_id']}"
                )


class TestDefaultsRegistry:
    def test_registry_covers_all_declared_features(self):
        d = load_defaults()
        for name in MEDIAN_FEATURES + list(CONVENTION_DEFAULTS):
            assert name in d, f"missing default for {name}"

    def test_conventions_are_exact(self):
        d = load_defaults()
        for name, value in CONVENTION_DEFAULTS.items():
            assert d[name] == value

    @needs_parquet
    def test_median_defaults_match_training(self):
        """Catches a stale feature_defaults.json after the training data
        changes: every median default must match a fresh recomputation."""
        import pandas as pd

        df = pd.read_parquet(TRAINING_PARQUET)
        d = load_defaults()
        for base in MEDIAN_FEATURES:
            cols = [c for c in (f"h_{base}", f"a_{base}") if c in df.columns]
            if not cols and base in df.columns:
                cols = [base]
            if not cols:
                continue  # not derivable from this parquet; frozen value governs
            fresh = float(pd.concat([df[c] for c in cols]).median())
            std = float(pd.concat([df[c] for c in cols]).std()) or 1.0
            assert abs(d[base] - fresh) <= max(0.25 * std, 1e-3), (
                f"default {base}={d[base]} is stale vs training median {fresh:.4f} "
                f"— rerun: PYTHONPATH=src python -m mlb.features.defaults"
            )
