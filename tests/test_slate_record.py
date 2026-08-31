"""Tests for the slate of record: what the main card locks is what gets graded."""

from datetime import date

import pytest

from mlb.etl.slate_record import (
    load_slate,
    lock_slate,
    locked_betting_slip,
    locked_predictions,
    resolve_send_mode,
    slate_path,
)

DAY = date(2026, 6, 1)  # well in the past: settlement skips the Odds API


def _slip(game_id="g1", stake=100.0):
    return {
        "slip_date": DAY.isoformat(),
        "bankroll": 10000.0,
        "num_bets": 1,
        "total_stake": stake,
        "total_ev": 10.0,
        "max_exposure": 0.0,
        "risk_level": "moderate",
        "bets": [{
            "game_id": game_id,
            "home_team": "NYY",
            "away_team": "BOS",
            "bet_type": "moneyline",
            "selection": "home",
            "odds": -120,
            "decimal_odds": 1.833,
            "recommended_stake": stake,
            "total_line": None,
        }],
    }


def _preds(prob=0.62):
    return [{
        "game_id": "g1",
        "home_team": "NYY",
        "away_team": "BOS",
        "home_win_prob": prob,
        "confidence": 70,
        "predicted_winner": "NYY" if prob > 0.5 else "BOS",
    }]


class TestLockAndLoad:
    def test_roundtrip(self, tmp_path):
        lock_slate(DAY, _preds(), _slip(), tmp_path)
        slate = load_slate(DAY, tmp_path)
        assert slate["date"] == DAY.isoformat()
        assert slate["locked_at"]
        assert locked_predictions(DAY, tmp_path) == _preds()
        assert locked_betting_slip(DAY, tmp_path)["bets"][0]["game_id"] == "g1"

    def test_unlocked_day_is_none_not_empty(self, tmp_path):
        # None (never locked) must stay distinguishable from [] / no bets, so
        # callers can fall back to the pre-lock prediction cache.
        assert load_slate(DAY, tmp_path) is None
        assert locked_predictions(DAY, tmp_path) is None
        assert locked_betting_slip(DAY, tmp_path) is None

    def test_main_card_with_no_bets_locks_none(self, tmp_path):
        lock_slate(DAY, _preds(), None, tmp_path)
        assert locked_predictions(DAY, tmp_path) == _preds()
        assert locked_betting_slip(DAY, tmp_path) is None

    def test_corrupt_file_is_survivable(self, tmp_path):
        path = slate_path(DAY, tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("{not json")
        assert load_slate(DAY, tmp_path) is None


# ── Runner wiring ────────────────────────────────────────────────────


def _runner(tmp_path):
    from mlb.etl.daily_runner import DailyRunner

    r = DailyRunner.__new__(DailyRunner)  # skip __init__ (no clients/models needed)
    r.data_dir = tmp_path
    return r


class TestApplySlateRecord:
    def test_preview_locks_the_day(self, tmp_path):
        r = _runner(tmp_path)
        result = {"predictions": _preds(), "betting_slip": _slip()}

        r._apply_slate_record(DAY, result, "preview")

        assert locked_betting_slip(DAY, tmp_path)["bets"][0]["odds"] == -120

    def test_wave_adopts_the_locked_card_and_drops_its_own(self, tmp_path):
        r = _runner(tmp_path)
        lock_slate(DAY, _preds(), _slip(stake=100.0), tmp_path)

        # A later run recomputes a slip from fresher odds — never sent, so never kept.
        fresh = {"predictions": _preds(0.71), "betting_slip": _slip(stake=250.0)}
        r._apply_slate_record(DAY, fresh, "wave")

        assert fresh["betting_slip"]["total_stake"] == 100.0
        assert locked_betting_slip(DAY, tmp_path)["total_stake"] == 100.0

    def test_wave_without_a_locked_day_sends_no_bets(self, tmp_path):
        r = _runner(tmp_path)
        result = {"predictions": _preds(), "betting_slip": _slip()}

        r._apply_slate_record(DAY, result, "wave")

        assert result["betting_slip"] is None
        assert load_slate(DAY, tmp_path) is None

    def test_manual_full_run_fills_a_gap_but_never_overwrites(self, tmp_path):
        r = _runner(tmp_path)
        r._apply_slate_record(DAY, {"predictions": _preds(), "betting_slip": _slip()}, "full")
        assert locked_betting_slip(DAY, tmp_path)["total_stake"] == 100.0

        later = {"predictions": _preds(), "betting_slip": _slip(stake=999.0)}
        r._apply_slate_record(DAY, later, "full")

        assert locked_betting_slip(DAY, tmp_path)["total_stake"] == 100.0
        assert later["betting_slip"]["total_stake"] == 100.0

    def test_a_real_run_upgrades_a_reconstructed_slate(self, tmp_path):
        """A slate rebuilt from git history is a stand-in for a run that never
        locked; an actual run replaces it rather than deferring to it."""
        r = _runner(tmp_path)
        lock_slate(DAY, _preds(), _slip(stake=100.0), tmp_path,
                   reconstructed_from={"commit": "abc123"})

        real = {"predictions": _preds(), "betting_slip": _slip(stake=250.0)}
        r._apply_slate_record(DAY, real, "full")

        assert locked_betting_slip(DAY, tmp_path)["total_stake"] == 250.0
        assert "reconstructed" not in load_slate(DAY, tmp_path)


# ── Grading reads the record, not the live cache ─────────────────────


def _write_live_cache(tmp_path, predictions, slip):
    import json

    path = tmp_path / "predictions" / f"{DAY.isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "date": DAY.isoformat(), "predictions": predictions, "betting_slip": slip,
    }))


def _write_scores_csv(tmp_path, home_score, away_score):
    import pandas as pd

    pd.DataFrame([{
        "game_date": DAY.isoformat(), "status": "Final",
        "home_team_id": "NYY", "away_team_id": "BOS",
        "home_score": home_score, "away_score": away_score,
    }]).to_csv(tmp_path / f"games_{DAY.year}.csv", index=False)


class TestSettlementUsesTheRecord:
    @pytest.fixture(autouse=True)
    def _no_mlb_finals(self, monkeypatch):
        """Scores now come from the MLB API first; these tests exercise the CSV
        fallback, so hand the API back an empty day (and stay off the network)."""
        import mlb.data.mlb_api as api

        monkeypatch.setattr(api, "MLBApiClient", lambda *a, **k: _FakeMLBClient(games=[]))

    async def test_settles_locked_card_not_the_refreshed_one(self, tmp_path):
        from mlb.betting.settlement import settle_day

        # Sent card: home -120 @ $100. The live cache holds a later, never-sent
        # recompute that flipped to the away side at a bigger stake.
        lock_slate(DAY, _preds(), _slip(stake=100.0), tmp_path)
        never_sent = _slip(stake=500.0)
        never_sent["bets"][0]["selection"] = "away"
        _write_live_cache(tmp_path, _preds(), never_sent)
        _write_scores_csv(tmp_path, home_score=5, away_score=2)  # home won

        settlement = await settle_day(DAY, tmp_path)

        assert settlement["summary"]["bets_won"] == 1        # not the away flip
        assert settlement["summary"]["total_staked"] == 100.0

    async def test_falls_back_to_cache_for_pre_lock_days(self, tmp_path):
        from mlb.betting.settlement import settle_day

        _write_live_cache(tmp_path, _preds(), _slip(stake=100.0))
        _write_scores_csv(tmp_path, home_score=5, away_score=2)

        settlement = await settle_day(DAY, tmp_path)

        assert settlement["summary"]["bets_placed"] == 1

    async def test_locked_day_with_no_bets_settles_nothing(self, tmp_path):
        from mlb.betting.settlement import settle_day

        # The main card went out with no bets; a later run's slip must not sneak in.
        lock_slate(DAY, _preds(), None, tmp_path)
        _write_live_cache(tmp_path, _preds(), _slip())
        _write_scores_csv(tmp_path, home_score=5, away_score=2)

        assert await settle_day(DAY, tmp_path) is None


class _FakeMLBClient:
    """Stands in for MLBApiClient — grading only needs final scores.

    Pass `games` to return an explicit schedule (an empty list stands in for a
    day the API has nothing final for).
    """

    def __init__(self, home_score=2, away_score=5, games=None):
        self._scores = (home_score, away_score)
        self._games = games

    async def get_schedule(self, target_date):
        if self._games is not None:
            return self._games
        home, away = self._scores
        return [{
            "game_id": "g1",
            "status": "Final", "home_team_id": "NYY", "away_team_id": "BOS",
            "home_score": home, "away_score": away,
        }]

    async def close(self):
        pass


class TestAccuracyUsesTheRecord:
    @pytest.fixture(autouse=True)
    def _fake_api(self, monkeypatch):
        import mlb.data.mlb_api as api

        monkeypatch.setattr(api, "MLBApiClient", _FakeMLBClient)

    async def test_grades_the_sent_pick_not_the_late_flip(self, tmp_path):
        from mlb.models.accuracy import track_accuracy

        # Sent at the main card: NYY (0.62). A later run flipped to BOS, which is
        # what actually happened — grading that would flatter a call never made.
        lock_slate(DAY, _preds(prob=0.62), None, tmp_path)
        _write_live_cache(tmp_path, _preds(prob=0.30), None)

        record = await track_accuracy(DAY, tmp_path)

        assert record["graded_from"] == "slate"
        assert record["summary"]["correct"] == 0
        assert record["results"][0]["home_win_prob"] == 0.62

    async def test_falls_back_to_cache_for_pre_lock_days(self, tmp_path):
        from mlb.models.accuracy import track_accuracy

        _write_live_cache(tmp_path, _preds(prob=0.30), None)

        record = await track_accuracy(DAY, tmp_path)

        assert record["graded_from"] == "cache"
        assert record["summary"]["correct"] == 1  # BOS pick, BOS won


class TestResolveSendMode:
    """Which run takes the main card is decided by state, not by cron identity."""

    def test_unlocked_day_gets_the_main_card(self, tmp_path):
        assert resolve_send_mode(DAY, tmp_path) == "preview"

    def test_once_locked_every_later_run_is_a_wave(self, tmp_path):
        lock_slate(DAY, _preds(), _slip(), tmp_path)
        assert resolve_send_mode(DAY, tmp_path) == "wave"

    def test_a_dropped_cron_does_not_cost_the_day(self, tmp_path):
        """The 2026-08-27 case: the 16:30 cron never fired, so the next run
        must still be able to send the main card."""
        assert resolve_send_mode(DAY, tmp_path) == "preview"
        lock_slate(DAY, _preds(), _slip(), tmp_path)   # backstop run takes it
        assert resolve_send_mode(DAY, tmp_path) == "wave"
