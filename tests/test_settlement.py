"""Tests for bet settlement — score matching, the P&L chain, and the daily review."""

import json
from datetime import date
from pathlib import Path

import pytest

from mlb.betting.settlement import load_settlement, rechain_settlements, settle_day
from mlb.models.accuracy import build_daily_review


class _NoFinalsClient:
    """MLB API stub with nothing final — pushes settle_day down its fallback chain."""

    def __init__(self, games=None):
        self._games = games or []

    async def get_schedule(self, target_date):
        return self._games

    async def close(self):
        pass


@pytest.fixture
def no_mlb_finals(monkeypatch):
    """The MLB API is the primary score source now; keep tests off the network."""
    import mlb.data.mlb_api as api

    monkeypatch.setattr(api, "MLBApiClient", lambda *a, **k: _NoFinalsClient())


def _write_slip(tmp_path: Path, date_str: str, bets: list[dict]) -> None:
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    slip = {"bankroll": 10000.0, "bets": bets}
    (pred_dir / f"{date_str}.json").write_text(json.dumps({"betting_slip": slip}))


def _bet(game_id, home, away, selection="home", stake=100.0):
    return {
        "game_id": game_id, "game_date": "2026-07-22",
        "home_team": home, "away_team": away,
        "bet_type": "moneyline", "selection": selection,
        "odds": -110, "recommended_stake": stake,
    }


async def test_settlement_ignores_adjacent_series_game(
    tmp_path, monkeypatch, no_mlb_finals
):
    """A same-matchup game from the next day must not settle today's slip.

    Regression: a multi-day scores window keyed only on (home, away) let the
    later series game overwrite and mis-grade the earlier date.
    """
    _write_slip(
        tmp_path,
        "2026-07-07",
        [{
            "game_id": "1", "game_date": "2026-07-07",
            "home_team": "TEX", "away_team": "LAA",
            "bet_type": "moneyline", "selection": "home",
            "odds": -110, "recommended_stake": 100.0,
        }],
    )

    # Odds window returns BOTH days' TEX/LAA games: TEX won 8-3 on the 7th
    # (bet wins) but lost 1-13 on the 8th. Only the 7th should count.
    async def fake_scores(self, days_from=1):
        return [
            {"commence_time": "2026-07-07T23:05:00Z", "home_team": "TEX",
             "away_team": "LAA", "home_score": 8, "away_score": 3, "completed": True},
            {"commence_time": "2026-07-08T23:05:00Z", "home_team": "TEX",
             "away_team": "LAA", "home_score": 1, "away_score": 13, "completed": True},
        ]

    monkeypatch.setattr("mlb.data.odds_api.OddsApiClient.get_scores", fake_scores)
    # Keep the lookup on the Odds-API path (recent date) regardless of wall clock.
    monkeypatch.setattr("mlb.data.scores._today_et", lambda: date(2026, 7, 8))

    result = await settle_day(date(2026, 7, 7), data_dir=tmp_path)

    assert result is not None
    assert result["summary"]["bets_won"] == 1
    assert result["summary"]["bets_lost"] == 0
    assert result["bets"][0]["actual_home_score"] == 8  # the 7th, not the 8th


class TestDoubleheaders:
    async def test_each_leg_settles_against_its_own_score(self, tmp_path, monkeypatch):
        """Two bets on one matchup must not both take the same score.

        Regression: keying scores by (home, away) collapsed a twin bill into one
        entry, so both bets graded off whichever leg landed in the map last —
        2W or 2L, never the truth. 2026-07-22 really had two of these.
        """
        import mlb.data.mlb_api as api

        monkeypatch.setattr(api, "MLBApiClient", lambda *a, **k: _NoFinalsClient([
            {"game_id": "824735", "status": "Final", "home_team_id": "BOS",
             "away_team_id": "BAL", "home_score": 7, "away_score": 2},   # home wins
            {"game_id": "824732", "status": "Final", "home_team_id": "BOS",
             "away_team_id": "BAL", "home_score": 1, "away_score": 9},   # home loses
        ]))
        _write_slip(tmp_path, "2026-07-22", [
            _bet("824735", "BOS", "BAL"),
            _bet("824732", "BOS", "BAL"),
        ])

        result = await settle_day(date(2026, 7, 22), data_dir=tmp_path)

        assert result["summary"]["bets_won"] == 1
        assert result["summary"]["bets_lost"] == 1

    async def test_a_leg_without_an_id_is_skipped_not_guessed(
        self, tmp_path, monkeypatch
    ):
        import mlb.data.mlb_api as api

        monkeypatch.setattr(api, "MLBApiClient", lambda *a, **k: _NoFinalsClient([
            {"game_id": "824735", "status": "Final", "home_team_id": "BOS",
             "away_team_id": "BAL", "home_score": 7, "away_score": 2},
            {"game_id": "824732", "status": "Final", "home_team_id": "BOS",
             "away_team_id": "BAL", "home_score": 1, "away_score": 9},
        ]))
        _write_slip(tmp_path, "2026-07-22", [
            _bet("824735", "BOS", "BAL"),
            _bet(None, "BOS", "BAL"),          # ambiguous — which leg?
        ])

        result = await settle_day(date(2026, 7, 22), data_dir=tmp_path)

        assert result["summary"]["bets_placed"] == 1
        assert result["summary"]["bets_on_card"] == 2   # the shortfall is recorded


def _write_settlement(tmp_path, date_str, daily_pnl, cumulative, staked=100.0):
    betting = tmp_path / "betting"
    betting.mkdir(parents=True, exist_ok=True)
    (betting / f"{date_str}.json").write_text(json.dumps({
        "date": date_str,
        "bets": [],
        "summary": {
            "bets_placed": 1, "bets_won": 1, "bets_lost": 0, "bets_pushed": 0,
            "total_staked": staked, "daily_pnl": daily_pnl, "roi": 0.0,
            "cumulative_pnl": cumulative, "max_drawdown": 0.0,
        },
    }))


class TestCumulativeChain:
    async def test_gap_day_chains_off_the_prior_date_not_the_newest_file(
        self, tmp_path, monkeypatch
    ):
        """Backfilling a gap day must not inherit a later day's running total.

        Regression: the baseline was load_all_settlements()[-1] — newest file by
        name — so settling an old day chained off the future.
        """
        import mlb.data.mlb_api as api

        monkeypatch.setattr(api, "MLBApiClient", lambda *a, **k: _NoFinalsClient([
            {"game_id": "824735", "status": "Final", "home_team_id": "BOS",
             "away_team_id": "BAL", "home_score": 7, "away_score": 2},
        ]))
        _write_settlement(tmp_path, "2026-07-20", daily_pnl=50.0, cumulative=50.0)
        _write_settlement(tmp_path, "2026-07-27", daily_pnl=900.0, cumulative=950.0)
        _write_slip(tmp_path, "2026-07-22", [_bet("824735", "BOS", "BAL")])

        result = await settle_day(date(2026, 7, 22), data_dir=tmp_path)

        # 7/22 chains off 7/20 ($50), not off 7/27 ($950).
        daily = result["summary"]["daily_pnl"]
        assert result["summary"]["cumulative_pnl"] == round(50.0 + daily, 2)
        # ...and 7/27, which now sits downstream of a new day, is repaired too.
        assert load_settlement(date(2026, 7, 27), tmp_path)["summary"][
            "cumulative_pnl"
        ] == round(50.0 + daily + 900.0, 2)

    def test_rechain_is_a_no_op_on_a_consistent_chain(self, tmp_path):
        _write_settlement(tmp_path, "2026-07-20", daily_pnl=50.0, cumulative=50.0)
        _write_settlement(tmp_path, "2026-07-21", daily_pnl=25.0, cumulative=75.0)

        assert rechain_settlements(tmp_path) == {}

    def test_rechain_repairs_a_broken_chain_and_then_settles(self, tmp_path):
        _write_settlement(tmp_path, "2026-07-20", daily_pnl=50.0, cumulative=50.0)
        _write_settlement(tmp_path, "2026-07-21", daily_pnl=25.0, cumulative=999.0)

        rewritten = rechain_settlements(tmp_path)

        assert set(rewritten) == {"2026-07-21"}
        assert rewritten["2026-07-21"]["cumulative_pnl"] == 75.0
        assert rechain_settlements(tmp_path) == {}   # idempotent


class TestDailyReview:
    def test_high_conf_split_and_window(self, tmp_path):
        acc_dir = tmp_path / "accuracy"
        acc_dir.mkdir(parents=True)
        # Yesterday (7/8): high-conf (>=65) loss, high-conf win, boundary win, below.
        (acc_dir / "2026-07-08.json").write_text(json.dumps({
            "date": "2026-07-08",
            "results": [
                {"confidence": 80, "correct": False},
                {"confidence": 70, "correct": True},
                {"confidence": 65, "correct": True},  # exactly 65 → included (>=65)
                {"confidence": 50, "correct": True},
            ],
        }))
        # Another in-window day (7/7): one high-conf win, one low-conf game.
        (acc_dir / "2026-07-07.json").write_text(json.dumps({
            "date": "2026-07-07",
            "results": [
                {"confidence": 88, "correct": True},
                {"confidence": 40, "correct": False},
            ],
        }))

        review = build_daily_review(date(2026, 7, 9), data_dir=tmp_path)

        assert review["full"] == {"correct": 3, "total": 4, "incorrect": 1}
        # Yesterday high-conf: confidence 80 (L), 70 (W), 65 (W); 50 excluded.
        assert review["high_conf"] == {"correct": 2, "total": 3, "incorrect": 1}
        # Window sums high-conf across 7/8 (2/3) and 7/7 (1/1) = 3/4.
        assert review["high_conf_window"]["correct"] == 3
        assert review["high_conf_window"]["total"] == 4
        assert review["card"] is None  # no settlement file written
        # Legacy records carry no coverage block; the review must not invent one.
        assert review["coverage"] is None
