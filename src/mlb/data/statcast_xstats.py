"""Baseball Savant expected-stats (xwOBA) ingestion.

Fetches season-level expected-statistics leaderboards (batter + pitcher) from
Baseball Savant via plain httpx — no pybaseball dependency. MLBAM ``player_id``
joins directly to our batting_/pitching_ CSVs.

Tier-3 finding (2026-06-16): xwOBA-based lineup/SP quality carries a REAL,
leakage-free signal on game *totals* (prior-season xwOBA → OOS residual t=2.23)
but NONE on the moneyline (prior-season t=0.15 — the apparent ML signal was
pure current-season leakage). However the totals signal is economically
sub-vig: in an honest O/U backtest it does not beat the −110 line
(directional accuracy ~51.5% vs 52.4% breakeven). It is therefore NOT wired
into the model features. This module is kept as reusable infrastructure for
revisiting with point-in-time current-season xwOBA (the only path with a
borderline shot at a totals edge).

Usage:
    PYTHONPATH=src python -m mlb.data.statcast_xstats --years 2021 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
import io
import logging
import time
from pathlib import Path

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

LEADERBOARD_URL = "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
CACHE_DIR = Path("data/savant")

# League-average est_woba fallbacks for players absent from a season leaderboard.
LEAGUE_AVG_BATTER_XWOBA = 0.310
LEAGUE_AVG_PITCHER_XWOBA = 0.320


def fetch_expected_stats(kind: str, year: int, *, min_pa: str = "1", refresh: bool = False) -> pd.DataFrame:
    """Fetch (and cache) the Savant expected-statistics leaderboard.

    Args:
        kind: "batter" or "pitcher".
        year: season.
        min_pa: Savant ``min`` filter ("q" for qualified, or a number).
        refresh: re-fetch even if cached.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"xstats_{kind}_{year}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)

    resp = httpx.get(
        LEADERBOARD_URL,
        params={
            "type": kind, "year": str(year),
            "position": "", "team": "", "min": min_pa, "csv": "true",
        },
        timeout=40.0,
    )
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.to_csv(path, index=False)
    logger.info("Fetched %s %d: %d players", kind, year, len(df))
    time.sleep(0.4)  # rate-limit courtesy
    return df


def load_xwoba_maps(years: list[int]) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """Return ((player_id, year) -> est_woba) maps for batters and pitchers."""
    batter: dict[tuple[int, int], float] = {}
    pitcher: dict[tuple[int, int], float] = {}
    for year in years:
        for r in fetch_expected_stats("batter", year).itertuples():
            batter[(int(r.player_id), year)] = float(r.est_woba)
        for r in fetch_expected_stats("pitcher", year).itertuples():
            pitcher[(int(r.player_id), year)] = float(r.est_woba)
    return batter, pitcher


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fetch Baseball Savant expected-stats leaderboards")
    parser.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    for year in args.years:
        for kind in ("batter", "pitcher"):
            d = fetch_expected_stats(kind, year, refresh=args.refresh)
            print(f"{kind} {year}: {len(d)} players")


if __name__ == "__main__":
    main()
