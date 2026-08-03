"""Grading: what gets counted, what re-grading is allowed to overwrite, coverage."""

import json
from datetime import date

import pytest

from mlb.alerts import AlertService
from mlb.etl.slate_record import lock_slate
from mlb.models.accuracy import build_daily_review, load_accuracy, track_accuracy

DAY = date(2026, 7, 22)


class _FakeMLBClient:
    def __init__(self, games):
        self._games = games

    async def get_schedule(self, target_date):
        return self._games

    async def close(self):
        pass


def _final(game_id, home, away, home_score, away_score):
    return {
        "game_id": game_id, "status": "Final",
        "home_team_id": home, "away_team_id": away,
        "home_score": home_score, "away_score": away_score,
    }


def _pred(game_id, home, away, prob=0.62, confidence=70):
    return {
        "game_id": game_id, "home_team": home, "away_team": away,
        "home_win_prob": prob, "confidence": confidence,
        "predicted_winner": home if prob > 0.5 else away,
    }


def _stub(game_id, home, away):
    return {
        "game_id": game_id, "home_team": home, "away_team": away,
        "home_win_prob": 0.5, "confidence": 0.0, "predicted_winner": home,
    }


@pytest.fixture
def mlb(monkeypatch):
    """Install a fake schedule; returns a setter so each test picks its own."""
    import mlb.data.mlb_api as api

    def _install(games):
        monkeypatch.setattr(api, "MLBApiClient", lambda *a, **k: _FakeMLBClient(games))

    return _install


def _write_cache(tmp_path, predictions, day=DAY):
    path = tmp_path / "predictions" / f"{day.isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"date": day.isoformat(), "predictions": predictions}))


class TestGradesTheWholeSlate:
    async def test_every_real_prediction_with_a_final_is_graded(self, tmp_path, mlb):
        mlb([
            _final("1", "BOS", "BAL", 7, 2),
            _final("2", "NYY", "TOR", 1, 4),
            _final("3", "SEA", "LAA", 5, 3),
        ])
        lock_slate(DAY, [
            _pred("1", "BOS", "BAL", prob=0.62),   # BOS won  → correct
            _pred("2", "NYY", "TOR", prob=0.58),   # NYY lost → wrong
            _pred("3", "SEA", "LAA", prob=0.55),   # SEA won  → correct
        ], None, tmp_path)

        record = await track_accuracy(DAY, tmp_path)

        assert record["summary"]["total_games"] == 3
        assert record["summary"]["correct"] == 2
        assert record["summary"]["graded_games"] == record["summary"]["final_games"]

    async def test_doubleheader_legs_grade_separately(self, tmp_path, mlb):
        mlb([
            _final("824735", "BOS", "BAL", 7, 2),   # home won
            _final("824732", "BOS", "BAL", 1, 9),   # home lost
        ])
        lock_slate(DAY, [
            _pred("824735", "BOS", "BAL", prob=0.62),   # correct
            _pred("824732", "BOS", "BAL", prob=0.62),   # wrong
        ], None, tmp_path)

        record = await track_accuracy(DAY, tmp_path)

        assert record["summary"]["total_games"] == 2
        assert record["summary"]["correct"] == 1

    async def test_stubs_are_reported_as_ungraded_not_counted(self, tmp_path, mlb):
        mlb([_final("1", "BOS", "BAL", 7, 2), _final("2", "NYY", "TOR", 1, 4)])
        lock_slate(DAY, [_pred("1", "BOS", "BAL"), _stub("2", "NYY", "TOR")],
                   None, tmp_path)

        record = await track_accuracy(DAY, tmp_path)

        assert record["summary"]["total_games"] == 1
        assert record["summary"]["slate_games"] == 2
        assert record["summary"]["final_games"] == 2
        assert record["ungraded"] == [
            {"game_id": "2", "home_team": "NYY", "away_team": "TOR", "reason": "stub"}
        ]

    async def test_a_game_with_no_final_is_reported_as_ungraded(self, tmp_path, mlb):
        mlb([_final("1", "BOS", "BAL", 7, 2)])
        lock_slate(DAY, [_pred("1", "BOS", "BAL"), _pred("2", "NYY", "TOR")],
                   None, tmp_path)

        record = await track_accuracy(DAY, tmp_path)

        assert [u["reason"] for u in record["ungraded"]] == ["no_score"]


class TestRegradeQualityGate:
    async def test_a_cache_regrade_cannot_clobber_a_slate_record(self, tmp_path, mlb):
        """The daily backfill re-grades a two-week window; a degraded cache read
        must never replace the record built from the slate that was sent."""
        mlb([_final("1", "BOS", "BAL", 7, 2), _final("2", "NYY", "TOR", 1, 4)])
        lock_slate(DAY, [_pred("1", "BOS", "BAL"), _pred("2", "NYY", "TOR")],
                   None, tmp_path)
        good = await track_accuracy(DAY, tmp_path)
        assert good["summary"]["total_games"] == 2

        # The slate disappears (as it does for every pre-lock day) and the cache
        # holds one real prediction plus a stub.
        (tmp_path / "slates" / f"{DAY.isoformat()}.json").unlink()
        _write_cache(tmp_path, [_pred("1", "BOS", "BAL"), _stub("2", "NYY", "TOR")])

        kept = await track_accuracy(DAY, tmp_path)

        assert kept["graded_from"] == "slate"
        assert kept["summary"]["total_games"] == 2
        assert load_accuracy(DAY, tmp_path)["summary"]["total_games"] == 2

    async def test_more_coverage_from_the_same_source_does_write(self, tmp_path, mlb):
        # First pass: only one game has gone final.
        mlb([_final("1", "BOS", "BAL", 7, 2)])
        _write_cache(tmp_path, [_pred("1", "BOS", "BAL"), _pred("2", "NYY", "TOR")])
        first = await track_accuracy(DAY, tmp_path)
        assert first["summary"]["total_games"] == 1

        # Later that night the second game finishes too.
        mlb([_final("1", "BOS", "BAL", 7, 2), _final("2", "NYY", "TOR", 1, 4)])
        second = await track_accuracy(DAY, tmp_path)

        assert second["summary"]["total_games"] == 2

    async def test_a_real_lock_beats_a_reconstruction(self, tmp_path, mlb):
        mlb([_final("1", "BOS", "BAL", 7, 2)])
        lock_slate(DAY, [_pred("1", "BOS", "BAL")], None, tmp_path,
                   reconstructed_from={"commit": "abc123"})
        rebuilt = await track_accuracy(DAY, tmp_path)
        assert rebuilt["graded_from"] == "slate:reconstructed"

        # A genuine run later locks the same day for real.
        lock_slate(DAY, [_pred("1", "BOS", "BAL")], None, tmp_path)
        upgraded = await track_accuracy(DAY, tmp_path)

        assert upgraded["graded_from"] == "slate"

    async def test_force_overrides_the_gate(self, tmp_path, mlb):
        mlb([_final("1", "BOS", "BAL", 7, 2), _final("2", "NYY", "TOR", 1, 4)])
        lock_slate(DAY, [_pred("1", "BOS", "BAL"), _pred("2", "NYY", "TOR")],
                   None, tmp_path)
        await track_accuracy(DAY, tmp_path)

        (tmp_path / "slates" / f"{DAY.isoformat()}.json").unlink()
        _write_cache(tmp_path, [_pred("1", "BOS", "BAL")])

        forced = await track_accuracy(DAY, tmp_path, force=True)

        assert forced["graded_from"] == "cache"
        assert forced["summary"]["total_games"] == 1


class TestCoverageReachesSlack:
    """A 4-of-11 day reads exactly like a complete one unless we say otherwise."""

    def _write_accuracy(self, tmp_path, summary):
        acc = tmp_path / "accuracy"
        acc.mkdir(parents=True, exist_ok=True)
        (acc / "2026-07-22.json").write_text(json.dumps({
            "date": "2026-07-22",
            "results": [{"confidence": 70, "correct": True}],
            "summary": summary,
        }))

    def _slate_line(self, tmp_path):
        review = build_daily_review(date(2026, 7, 23), data_dir=tmp_path)
        rendered = AlertService()._format_review_text(review)
        line = next(ln for ln in rendered.splitlines() if "Full slate" in ln)
        return review, line

    def test_partial_day_is_called_out(self, tmp_path):
        self._write_accuracy(tmp_path, {
            "total_games": 4, "correct": 3, "incorrect": 1,
            "graded_games": 4, "final_games": 11, "slate_games": 12,
        })

        review, line = self._slate_line(tmp_path)

        assert review["coverage"] == {"graded": 4, "final": 11, "slate": 12}
        assert "only 4 of 11 graded" in line

    def test_complete_day_has_no_suffix(self, tmp_path):
        self._write_accuracy(tmp_path, {
            "total_games": 11, "correct": 7, "incorrect": 4,
            "graded_games": 11, "final_games": 11, "slate_games": 11,
        })

        _, line = self._slate_line(tmp_path)

        assert "graded" not in line

    def test_legacy_record_without_coverage_renders_cleanly(self, tmp_path):
        # Every accuracy file written before this change looks like this.
        self._write_accuracy(tmp_path, {
            "total_games": 4, "correct": 3, "incorrect": 1,
            "accuracy": 0.75, "brier_score": 0.2,
        })

        review, line = self._slate_line(tmp_path)

        assert review["coverage"] is None
        assert " of " not in line
