"""Shared feature-default registry — one source of truth for both pipelines.

The live daily runner needs fallback values whenever pre-game data is missing
(lineups not posted, SP unannounced, odds unmatched). Those defaults MUST sit
in the training distribution: stale hardcoded defaults (e.g. lineup_obp 0.320
vs training 0.304, sp_season_era 4.50 vs 3.86) were out-of-distribution and
were a root cause of the live home-pick bias fixed in June 2026.

Two kinds of entries:

* Computed defaults — the training MEDIAN of the pooled h_/a_ columns
  (median, not mean: features like sp_rest_days are skewed by offseason gaps).
  Regenerate with ``generate_defaults()`` after rebuilding training data.
* Convention defaults — deliberately fixed values, never recomputed:
  - weather sentinels (72°F / 5 mph / 0.50) match build_training_data's
    dome/missing-weather convention;
  - venue win pcts stay symmetric 0.500/0.500 (commit 4ecec9f fixed a home
    bias caused by asymmetric venue defaults — do NOT switch these to medians,
    which are ~0.55/0.45 and would re-tilt picks home);
  - elo 1500 is the definitional new-team rating;
  - sp_season_ip 0.0 signals "no data" to confidence scoring;
  - bvp_ops has no training column (live-only injection), league avg 0.750.

Usage:
    from mlb.features.defaults import load_defaults
    D = load_defaults()
    era = sp_stats.get("era", D["sp_season_era"])

Regenerate after retraining data:
    PYTHONPATH=src python -m mlb.features.defaults
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULTS_PATH = Path("data/feature_defaults.json")

# Base feature names (h_/a_ prefixes stripped) whose default is the pooled
# training median. Keep in sync with the fallback sites in daily_runner.py
# and build_training_data.py.
MEDIAN_FEATURES: list[str] = [
    "lineup_ops", "lineup_obp", "lineup_slg", "lineup_wt_ops", "lineup_top4_ops",
    "lineup_ops_7d", "lineup_hot_pct",
    "platoon_adv", "platoon_wt_adv",
    "sp_season_era", "sp_season_whip", "sp_season_k9", "sp_season_bb9",
    "sp_recent_era", "sp_recent_whip", "sp_recent_k9",
    "bp_relievers_used_3d", "bp_freshness",
    "venue_home_rs_per_game", "venue_away_rs_per_game",
    "market_home_prob", "market_total",
]

# Deliberately fixed values — see module docstring for the reason each exists.
CONVENTION_DEFAULTS: dict[str, float] = {
    "temperature": 72.0,
    "wind_speed": 5.0,
    "humidity": 0.50,
    "venue_home_win_pct": 0.500,
    "venue_away_win_pct": 0.500,
    "elo": 1500.0,
    "sp_season_ip": 0.0,
    "bvp_ops": 0.750,
    # Missing-data sentinels shared with build_training_data (a missing value
    # here means "season start / no prior game", where training also inserts 5
    # — the median of NON-missing games would be the wrong reference).
    "rest_days": 5.0,
    "sp_rest_days": 5.0,
}

# Frozen snapshot of the generated values (training medians as of the
# 2021-2025 parquet) so live prediction never breaks if the JSON is absent.
_FROZEN_DEFAULTS: dict[str, float] = {
    "lineup_ops": 0.7051, "lineup_obp": 0.3046, "lineup_slg": 0.4,
    "lineup_wt_ops": 0.7052, "lineup_top4_ops": 0.707,
    "lineup_ops_7d": 0.72, "lineup_hot_pct": 0.4,
    "platoon_adv": 0.5455, "platoon_wt_adv": 0.539,
    "sp_season_era": 3.8571, "sp_season_whip": 1.2414,
    "sp_season_k9": 8.0426, "sp_season_bb9": 2.8845,
    "sp_recent_era": 3.78, "sp_recent_whip": 1.24, "sp_recent_k9": 8.1,
    "bp_relievers_used_3d": 7.0, "bp_freshness": 0.9341,
    "venue_home_rs_per_game": 4.4, "venue_away_rs_per_game": 4.4,
    "market_home_prob": 0.537, "market_total": 8.5,
    **CONVENTION_DEFAULTS,
}

_cached: dict[str, float] | None = None


def generate_defaults(
    parquet_path: str | Path = "data/training_data.parquet",
    output_path: str | Path = DEFAULTS_PATH,
) -> dict[str, float]:
    """Recompute median defaults from the training parquet and write the JSON."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    out: dict[str, float] = {}
    for base in MEDIAN_FEATURES:
        cols = [c for c in (f"h_{base}", f"a_{base}") if c in df.columns]
        if not cols and base in df.columns:
            cols = [base]  # unprefixed game-level features (market_*)
        if not cols:
            logger.warning("No training column for default %r — keeping frozen value", base)
            out[base] = _FROZEN_DEFAULTS[base]
            continue
        out[base] = round(float(pd.concat([df[c] for c in cols]).median()), 4)

    out.update(CONVENTION_DEFAULTS)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    logger.info("Wrote %d feature defaults to %s", len(out), output_path)
    return out


def load_defaults(path: str | Path = DEFAULTS_PATH) -> dict[str, float]:
    """Load the defaults registry (JSON if present, frozen fallback otherwise)."""
    global _cached
    if _cached is not None:
        return _cached
    p = Path(path)
    values = dict(_FROZEN_DEFAULTS)
    if p.exists():
        try:
            with open(p) as f:
                values.update(json.load(f))
        except Exception as e:  # malformed file must never break live predictions
            logger.warning("Failed to read %s (%s) — using frozen defaults", p, e)
    _cached = values
    return values


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generated = generate_defaults()
    print(f"{len(generated)} defaults written to {DEFAULTS_PATH}:")
    for k, v in sorted(generated.items()):
        print(f"  {k:26} {v}")
