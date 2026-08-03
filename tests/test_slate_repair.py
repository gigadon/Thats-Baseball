"""Picking the right commit when rebuilding a slate from git history.

`choose_main_run_blob` takes its candidates as an argument precisely so the
selection rule can be tested without a repo.
"""

from datetime import date, datetime, timezone

import pytest

from mlb.etl.slate_record import load_slate, lock_slate
from mlb.etl.slate_repair import Candidate, choose_main_run_blob, reconstruct_slate

DAY = date(2026, 7, 27)


def _at(hour, minute=0, day=27):
    return datetime(2026, 7, day, hour, minute, tzinfo=timezone.utc)


def _candidate(sha, when, real, total=12, bets=0):
    preds = [
        {"game_id": str(i), "home_team": "BOS", "away_team": "BAL",
         "home_win_prob": 0.62, "confidence": 70}
        for i in range(real)
    ] + [
        {"game_id": f"s{i}", "home_team": "NYY", "away_team": "TOR",
         "home_win_prob": 0.5, "confidence": 0.0}
        for i in range(total - real)
    ]
    slip = {"bets": [{"game_id": str(i)} for i in range(bets)]} if bets else None
    return Candidate(
        sha=sha, committed_at=when,
        blob={"predictions": preds, "betting_slip": slip},
        real_predictions=real,
    )


class TestSelectionRule:
    def test_picks_the_first_post_cron_commit_over_a_midnight_rollover(self):
        """The real 2026-07-27 shape.

        The day's *earliest* commit (04:13Z) belongs to the previous evening's
        late wave. It carries the full slate but only 7 bets, against the main
        card's 13 — so "earliest commit" is the wrong rule.
        """
        rollover = _candidate("aaa", _at(4, 13), real=12, bets=7)
        main_run = _candidate("bbb", _at(18, 4), real=12, bets=13)
        evening = _candidate("ccc", _at(19, 21), real=12, bets=5)

        chosen, confidence = choose_main_run_blob(
            DAY, [rollover, main_run, evening]
        )

        assert chosen.sha == "bbb"
        assert chosen.num_bets == 13
        assert confidence == "high"

    def test_coverage_beats_earliness(self):
        thin = _candidate("aaa", _at(17), real=4)
        full = _candidate("bbb", _at(20), real=14, total=14)

        chosen, _ = choose_main_run_blob(DAY, [thin, full])

        assert chosen.sha == "bbb"

    def test_late_degraded_commits_are_ignored(self):
        main_run = _candidate("aaa", _at(17, 51), real=17, total=17)
        clobbered = _candidate("bbb", _at(23, 34), real=9, total=17)
        wiped = _candidate("ccc", _at(3, 54, day=28), real=0, total=17)

        chosen, _ = choose_main_run_blob(DAY, [main_run, clobbered, wiped])

        assert chosen.sha == "aaa"

    def test_a_day_with_no_real_predictions_is_skipped(self):
        # e.g. 2026-07-14, the All-Star break: a row, but nothing predicted.
        assert choose_main_run_blob(DAY, [_candidate("aaa", _at(17), real=0)]) is None

    def test_no_candidates_at_all(self):
        assert choose_main_run_blob(DAY, []) is None

    def test_only_pre_cron_commits_yields_low_confidence(self):
        early = _candidate("aaa", _at(4, 13), real=12)

        chosen, confidence = choose_main_run_blob(DAY, [early])

        assert chosen.sha == "aaa"
        assert confidence == "low"


class TestReconstruct:
    @pytest.fixture
    def one_commit(self, monkeypatch):
        monkeypatch.setattr(
            "mlb.etl.slate_repair.candidate_commits",
            lambda d, repo=None: [_candidate("abc123def", _at(18, 4), real=12, bets=13)],
        )

    def test_writes_a_marked_slate(self, tmp_path, one_commit):
        summary = reconstruct_slate(DAY, tmp_path)

        slate = load_slate(DAY, tmp_path)
        assert summary["commit"] == "abc123de"          # short sha for the report
        assert slate["reconstructed"]["commit"] == "abc123def"   # full sha on record
        assert slate["reconstructed"]["confidence"] == "high"
        assert len(slate["predictions"]) == 12
        assert len(slate["betting_slip"]["bets"]) == 13

    def test_dry_run_writes_nothing(self, tmp_path, one_commit):
        summary = reconstruct_slate(DAY, tmp_path, dry_run=True)

        assert summary["real"] == 12
        assert load_slate(DAY, tmp_path) is None

    def test_a_genuine_lock_is_never_overwritten(self, tmp_path, one_commit):
        lock_slate(DAY, [{"game_id": "real"}], None, tmp_path)

        assert reconstruct_slate(DAY, tmp_path) is None
        assert load_slate(DAY, tmp_path)["predictions"] == [{"game_id": "real"}]

    def test_an_earlier_reconstruction_can_be_redone(self, tmp_path, one_commit):
        lock_slate(DAY, [{"game_id": "old"}], None, tmp_path,
                   reconstructed_from={"commit": "older"})

        assert reconstruct_slate(DAY, tmp_path) is not None
        assert load_slate(DAY, tmp_path)["reconstructed"]["commit"] == "abc123def"
