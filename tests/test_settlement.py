"""Tests for bet settlement — score matching and the daily review builder."""

import json
from datetime import date
from pathlib import Path

from mlb.betting.settlement import _score_on_date, settle_day
from mlb.models.accuracy import build_daily_review


class TestScoreOnDate:
    def test_matches_evening_game_in_et(self):
        # 7:05 PM ET first pitch → stored as the next UTC day, still July 7 in ET.
        score = {"commence_time": "2026-07-07T23:05:00Z"}
        assert _score_on_date(score, date(2026, 7, 7))
        assert not _score_on_date(score, date(2026, 7, 8))

    def test_after_midnight_utc_still_prior_et_date(self):
        score = {"commence_time": "2026-07-08T01:10:00Z"}  # 9:10 PM ET on the 7th
        assert _score_on_date(score, date(2026, 7, 7))

    def test_missing_or_bad_commence_time_is_lenient(self):
        assert _score_on_date({}, date(2026, 7, 7))
        assert _score_on_date({"commence_time": "not-a-date"}, date(2026, 7, 7))


def _write_slip(tmp_path: Path, date_str: str, bets: list[dict]) -> None:
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    slip = {"bankroll": 10000.0, "bets": bets}
    (pred_dir / f"{date_str}.json").write_text(json.dumps({"betting_slip": slip}))


async def test_settlement_ignores_adjacent_series_game(tmp_path, monkeypatch):
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

    monkeypatch.setattr("mlb.betting.settlement.OddsApiClient.get_scores", fake_scores)
    # Keep settle_day on the Odds-API path (recent date) regardless of wall clock.
    monkeypatch.setattr("mlb.betting.settlement._today_et", lambda: date(2026, 7, 8))

    result = await settle_day(date(2026, 7, 7), data_dir=tmp_path)

    assert result is not None
    assert result["summary"]["bets_won"] == 1
    assert result["summary"]["bets_lost"] == 0
    assert result["bets"][0]["actual_home_score"] == 8  # the 7th, not the 8th


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
