"""Build training dataset from backfilled CSV files.

Reads games, batting, and pitching CSVs and computes rolling team-level
features for each game to produce a model-ready training dataset.

Usage:
    python -m mlb.etl.build_training_data --data-dir data --seasons 2023 2024 2025
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrainingDataBuilder:
    """Builds feature vectors from historical CSV files."""

    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = data_dir

    def build(self, seasons: list[int], output: str = "training_data.parquet") -> pd.DataFrame:
        """Build training data for the given seasons.

        Returns a DataFrame where each row is one game with:
          - Feature columns for home and away teams
          - Target column: home_win (1/0)
        """
        # Load raw data
        games_df = self._load_games(seasons)
        batting_df = self._load_batting(seasons)
        pitching_df = self._load_pitching(seasons)

        logger.info(
            "Loaded %d games, %d batting rows, %d pitching rows",
            len(games_df), len(batting_df), len(pitching_df),
        )

        # Build team rolling stats
        team_stats = self._compute_team_rolling_stats(games_df, batting_df, pitching_df)

        # Generate feature rows for each game
        feature_rows = []
        games_sorted = games_df.sort_values("game_date").reset_index(drop=True)

        for idx, game in games_sorted.iterrows():
            if game["status"] != "Final":
                continue
            if game["home_score"] is None or game["away_score"] is None:
                continue

            game_date = game["game_date"]
            home = game["home_team_id"]
            away = game["away_team_id"]

            home_stats = team_stats.get(home)
            away_stats = team_stats.get(away)
            if not home_stats or not away_stats:
                continue

            # Get most recent stats BEFORE this game
            home_feat = self._get_team_features(home_stats, game_date)
            away_feat = self._get_team_features(away_stats, game_date)

            if home_feat is None or away_feat is None:
                continue  # Not enough history yet

            # Build feature row
            row = {"game_id": game["game_id"], "game_date": game_date}
            row["home_team"] = home
            row["away_team"] = away
            row["home_score"] = game["home_score"]
            row["away_score"] = game["away_score"]
            row["home_win"] = 1 if game["home_score"] > game["away_score"] else 0

            # Home features
            for k, v in home_feat.items():
                row[f"h_{k}"] = v

            # Away features
            for k, v in away_feat.items():
                row[f"a_{k}"] = v

            # Differentials
            for k in home_feat:
                row[f"diff_{k}"] = home_feat[k] - away_feat[k]

            feature_rows.append(row)

            if (idx + 1) % 500 == 0:
                logger.info("Processed %d / %d games", idx + 1, len(games_sorted))

        df = pd.DataFrame(feature_rows)
        logger.info("Built %d training samples with %d features", len(df), len(df.columns) - 6)

        # Save
        out_path = self.data_dir / output
        df.to_parquet(out_path, index=False)
        logger.info("Saved training data to %s", out_path)

        return df

    # ── Data Loading ──────────────────────────────────────

    def _load_games(self, seasons: list[int]) -> pd.DataFrame:
        frames = []
        for s in seasons:
            path = self.data_dir / f"games_{s}.csv"
            if path.exists():
                df = pd.read_csv(path)
                frames.append(df)
                logger.info("Loaded %d games from %s", len(df), path)
            else:
                logger.warning("Games file not found: %s", path)
        if not frames:
            raise FileNotFoundError("No games CSVs found")
        df = pd.concat(frames, ignore_index=True)
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
        df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
        return df

    def _load_batting(self, seasons: list[int]) -> pd.DataFrame:
        frames = []
        for s in seasons:
            path = self.data_dir / f"batting_{s}.csv"
            if path.exists():
                frames.append(pd.read_csv(path))
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        return df

    def _load_pitching(self, seasons: list[int]) -> pd.DataFrame:
        frames = []
        for s in seasons:
            path = self.data_dir / f"pitching_{s}.csv"
            if path.exists():
                frames.append(pd.read_csv(path))
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        return df

    # ── Rolling Stats ─────────────────────────────────────

    def _compute_team_rolling_stats(
        self,
        games_df: pd.DataFrame,
        batting_df: pd.DataFrame,
        pitching_df: pd.DataFrame,
    ) -> dict[str, list[dict]]:
        """Compute game-by-game rolling stats for each team.

        Returns {team_id: [{"date": ..., "stat": ...}, ...]} sorted by date.
        """
        teams = set(games_df["home_team_id"].unique()) | set(games_df["away_team_id"].unique())
        team_stats: dict[str, list[dict]] = {}

        for team_id in sorted(teams):
            # Get all games for this team (home or away)
            team_games = games_df[
                ((games_df["home_team_id"] == team_id) | (games_df["away_team_id"] == team_id))
                & (games_df["status"] == "Final")
            ].sort_values("game_date").reset_index(drop=True)

            if len(team_games) < 10:
                continue

            # Batting stats for this team
            team_batting = batting_df[batting_df["team_id"] == team_id] if len(batting_df) else pd.DataFrame()
            team_pitching = pitching_df[pitching_df["team_id"] == team_id] if len(pitching_df) else pd.DataFrame()

            stats_list = []
            wins = losses = total_rs = total_ra = 0
            results_buffer: list[dict] = []  # For rolling calculations

            for _, game in team_games.iterrows():
                is_home = game["home_team_id"] == team_id
                rs = game["home_score"] if is_home else game["away_score"]
                ra = game["away_score"] if is_home else game["home_score"]

                if pd.isna(rs) or pd.isna(ra):
                    continue

                rs, ra = int(rs), int(ra)
                won = rs > ra
                if won:
                    wins += 1
                else:
                    losses += 1
                total_rs += rs
                total_ra += ra

                results_buffer.append({
                    "date": game["game_date"],
                    "won": won,
                    "rs": rs,
                    "ra": ra,
                    "run_diff": rs - ra,
                    "is_home": is_home,
                })

                gp = wins + losses

                # Game-level batting aggregates for this game
                game_bat = team_batting[team_batting["game_id"] == game["game_id"]] if len(team_batting) else pd.DataFrame()
                g_ab = int(game_bat["at_bats"].sum()) if len(game_bat) else 0
                g_h = int(game_bat["hits"].sum()) if len(game_bat) else 0
                g_hr = int(game_bat["home_runs"].sum()) if len(game_bat) else 0
                g_bb = int(game_bat["walks"].sum()) if len(game_bat) else 0
                g_so = int(game_bat["strikeouts"].sum()) if len(game_bat) else 0
                g_2b = int(game_bat["doubles"].sum()) if len(game_bat) else 0
                g_3b = int(game_bat["triples"].sum()) if len(game_bat) else 0
                g_sb = int(game_bat["stolen_bases"].sum()) if len(game_bat) else 0

                # Game-level pitching aggregates
                game_pit = team_pitching[team_pitching["game_id"] == game["game_id"]] if len(team_pitching) else pd.DataFrame()
                g_ip = float(game_pit["innings_pitched"].sum()) if len(game_pit) else 0
                g_er = int(game_pit["earned_runs"].sum()) if len(game_pit) else 0
                g_ha = int(game_pit["hits_allowed"].sum()) if len(game_pit) else 0
                g_bba = int(game_pit["walks_allowed"].sum()) if len(game_pit) else 0
                g_ka = int(game_pit["strikeouts_recorded"].sum()) if len(game_pit) else 0

                # Rolling windows
                last_10 = results_buffer[-10:]
                last_20 = results_buffer[-20:]
                l10_w = sum(1 for r in last_10 if r["won"])
                l20_w = sum(1 for r in last_20 if r["won"])
                l10_rd = sum(r["run_diff"] for r in last_10)

                # Streak
                streak = 0
                for r in reversed(results_buffer):
                    if r["won"]:
                        if streak >= 0:
                            streak += 1
                        else:
                            break
                    else:
                        if streak <= 0:
                            streak -= 1
                        else:
                            break

                # Pythagorean
                rs2 = max(total_rs, 1) ** 2
                ra2 = max(total_ra, 1) ** 2
                pythag = rs2 / (rs2 + ra2)

                # Season batting averages
                # Compute cumulative from results_buffer length
                avg = g_h / max(g_ab, 1)
                obp = (g_h + g_bb) / max(g_ab + g_bb, 1)
                slg_num = g_h + g_2b + 2 * g_3b + 3 * g_hr
                slg = slg_num / max(g_ab, 1)

                stats_list.append({
                    "date": game["game_date"],
                    "gp": gp,
                    "wins": wins,
                    "losses": losses,
                    "win_pct": wins / max(gp, 1),
                    "rs_per_game": total_rs / max(gp, 1),
                    "ra_per_game": total_ra / max(gp, 1),
                    "run_diff": total_rs - total_ra,
                    "run_diff_per_game": (total_rs - total_ra) / max(gp, 1),
                    "pythag_wpct": pythag,
                    "luck": (wins / max(gp, 1)) - pythag,
                    # Recent
                    "l10_wins": l10_w,
                    "l10_losses": len(last_10) - l10_w,
                    "l10_run_diff": l10_rd,
                    "l20_wins": l20_w,
                    "l20_losses": len(last_20) - l20_w,
                    "streak": streak,
                    "rs_last_10": sum(r["rs"] for r in last_10) / max(len(last_10), 1),
                    "ra_last_10": sum(r["ra"] for r in last_10) / max(len(last_10), 1),
                    # Game-level offense
                    "g_avg": avg,
                    "g_obp": obp,
                    "g_slg": slg,
                    "g_ops": obp + slg,
                    "g_hr": g_hr,
                    "g_bb": g_bb,
                    "g_so": g_so,
                    "g_sb": g_sb,
                    # Game-level pitching
                    "g_ip": g_ip,
                    "g_er": g_er,
                    "g_ha": g_ha,
                    "g_bba": g_bba,
                    "g_ka": g_ka,
                    "g_era": g_er * 9 / max(g_ip, 0.1),
                    "g_whip": (g_ha + g_bba) / max(g_ip, 0.1),
                    "g_k9": g_ka * 9 / max(g_ip, 0.1),
                    # Home/away
                    "home_wpct": (
                        sum(1 for r in results_buffer if r["is_home"] and r["won"])
                        / max(sum(1 for r in results_buffer if r["is_home"]), 1)
                    ),
                    "away_wpct": (
                        sum(1 for r in results_buffer if not r["is_home"] and r["won"])
                        / max(sum(1 for r in results_buffer if not r["is_home"]), 1)
                    ),
                })

            team_stats[team_id] = stats_list

        logger.info("Computed rolling stats for %d teams", len(team_stats))
        return team_stats

    def _get_team_features(
        self, stats_list: list[dict], game_date: date
    ) -> dict[str, float] | None:
        """Get the most recent team stats BEFORE game_date.

        Requires at least 15 games of history.
        """
        # Find the latest entry before game_date
        candidates = [s for s in stats_list if s["date"] < game_date]
        if len(candidates) < 15:
            return None

        latest = candidates[-1]

        # Also compute rolling averages from last N entries
        last_7 = candidates[-7:]
        last_14 = candidates[-14:]

        def _roll_avg(entries: list[dict], field: str) -> float:
            vals = [e.get(field, 0) for e in entries]
            return sum(vals) / len(vals) if vals else 0

        features = {
            # Season
            "win_pct": latest["win_pct"],
            "rs_per_game": latest["rs_per_game"],
            "ra_per_game": latest["ra_per_game"],
            "run_diff_pg": latest["run_diff_per_game"],
            "pythag": latest["pythag_wpct"],
            "luck": latest["luck"],
            # Recent form
            "l10_wpct": latest["l10_wins"] / max(latest["l10_wins"] + latest["l10_losses"], 1),
            "l10_rd": latest["l10_run_diff"],
            "l20_wpct": latest["l20_wins"] / max(latest["l20_wins"] + latest["l20_losses"], 1),
            "streak": latest["streak"],
            "rs_last_10": latest["rs_last_10"],
            "ra_last_10": latest["ra_last_10"],
            # Rolling batting (7-game)
            "avg_7": _roll_avg(last_7, "g_avg"),
            "obp_7": _roll_avg(last_7, "g_obp"),
            "slg_7": _roll_avg(last_7, "g_slg"),
            "ops_7": _roll_avg(last_7, "g_ops"),
            "hr_7": _roll_avg(last_7, "g_hr"),
            "bb_7": _roll_avg(last_7, "g_bb"),
            "so_7": _roll_avg(last_7, "g_so"),
            # Rolling batting (14-game)
            "avg_14": _roll_avg(last_14, "g_avg"),
            "obp_14": _roll_avg(last_14, "g_obp"),
            "slg_14": _roll_avg(last_14, "g_slg"),
            "ops_14": _roll_avg(last_14, "g_ops"),
            # Rolling pitching (7-game)
            "era_7": _roll_avg(last_7, "g_era"),
            "whip_7": _roll_avg(last_7, "g_whip"),
            "k9_7": _roll_avg(last_7, "g_k9"),
            # Rolling pitching (14-game)
            "era_14": _roll_avg(last_14, "g_era"),
            "whip_14": _roll_avg(last_14, "g_whip"),
            "k9_14": _roll_avg(last_14, "g_k9"),
            # Home/away
            "home_wpct": latest["home_wpct"],
            "away_wpct": latest["away_wpct"],
            # Games played (proxy for sample stability)
            "gp": latest["gp"],
        }

        return features


# ── CLI ───────────────────────────────────────────────────────


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build training data from backfilled CSVs")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=str, default="training_data.parquet")
    args = parser.parse_args()

    builder = TrainingDataBuilder(data_dir=Path(args.data_dir))
    df = builder.build(args.seasons, args.output)
    print(f"\nTraining data: {len(df)} samples, {len(df.columns)} columns")
    print(f"Home win rate: {df['home_win'].mean():.3f}")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")


if __name__ == "__main__":
    main()
