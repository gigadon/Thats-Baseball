"""The prediction cache must not lose a real call to a later run's placeholder.

The pipeline runs four times a day and only predicts games that haven't finished.
Every finished game comes back as a stub (prob 0.5, confidence 0), so the evening
runs used to overwrite the morning's real predictions — which is what left whole
days ungraded (2026-08-02 graded 1 of 15 games).
"""

import json

import pytest

from mlb.api.routes.predictions import cache_predictions

DATE = "2026-07-22"


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    import mlb.api.routes.predictions as preds

    monkeypatch.setattr(preds, "_CACHE_DIR", tmp_path / "predictions")
    monkeypatch.setattr(preds, "_predictions_cache", {})
    return tmp_path / "predictions"


def _read(cache_dir):
    return {
        p["game_id"]: p
        for p in json.loads((cache_dir / f"{DATE}.json").read_text())["predictions"]
    }


def _real(game_id="1", prob=0.63, confidence=72):
    return {
        "game_id": game_id, "home_team": "BOS", "away_team": "BAL",
        "home_win_prob": prob, "away_win_prob": round(1 - prob, 2),
        "confidence": confidence, "predicted_winner": "BOS",
        "status": "Scheduled", "home_moneyline": -140,
    }


def _stub(game_id="1", status="Final"):
    return {
        "game_id": game_id, "home_team": "BOS", "away_team": "BAL",
        "home_win_prob": 0.5, "away_win_prob": 0.5,
        "confidence": 0.0, "predicted_winner": "BOS",
        "status": status, "home_moneyline": -155,
    }


def test_stub_never_replaces_a_real_prediction(cache_dir):
    cache_predictions(DATE, [_real()])
    cache_predictions(DATE, [_stub()])

    kept = _read(cache_dir)["1"]

    assert kept["home_win_prob"] == 0.63
    assert kept["confidence"] == 72


def test_the_stub_still_refreshes_live_display_fields(cache_dir):
    """The dashboard reason stubs exist for — status and closing odds — survives."""
    cache_predictions(DATE, [_real()])
    cache_predictions(DATE, [_stub()])

    kept = _read(cache_dir)["1"]

    assert kept["status"] == "Final"
    assert kept["home_moneyline"] == -155


def test_stub_is_stored_when_there_is_no_prior_prediction(cache_dir):
    cache_predictions(DATE, [_stub()])

    assert _read(cache_dir)["1"]["confidence"] == 0.0


def test_a_real_prediction_replaces_a_stub(cache_dir):
    # Starters get announced mid-morning; the first real call must win.
    cache_predictions(DATE, [_stub(status="Scheduled")])
    cache_predictions(DATE, [_real()])

    assert _read(cache_dir)["1"]["confidence"] == 72


def test_a_newer_real_prediction_still_replaces_an_older_one(cache_dir):
    cache_predictions(DATE, [_real(prob=0.63)])
    cache_predictions(DATE, [_real(prob=0.71, confidence=80)])

    assert _read(cache_dir)["1"]["home_win_prob"] == 0.71


def test_other_games_are_untouched(cache_dir):
    cache_predictions(DATE, [_real("1"), _real("2", prob=0.58)])
    cache_predictions(DATE, [_stub("1")])

    cached = _read(cache_dir)

    assert cached["2"]["home_win_prob"] == 0.58
    assert len(cached) == 2
