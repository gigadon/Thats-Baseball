"""Keeping in-play prices out of odds_history.csv — the training odds source.

The file is read as the market's pregame opinion in two places: market_home_prob
is a model feature, and mlb.models.backtest grades the model against the same
rows. A quote taken after first pitch is neither — it encodes the score, so it
"predicts" the outcome ~92% of the time and teaches the model to trust an input
it will never see live.
"""

import csv
from datetime import datetime, timedelta, timezone

import pytest

from mlb.etl.daily_runner import DailyRunner
from mlb.features.formulas import is_pregame_line


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


NOW = datetime.now(timezone.utc)
LATER = _iso(NOW + timedelta(hours=3))
EARLIER = _iso(NOW - timedelta(hours=3))


def _game(game_id, home, away, time, status="Scheduled", game_date="2026-08-04"):
    return {
        "game_id": game_id, "home_team_id": home, "away_team_id": away,
        "game_time": time, "status": status, "game_date": game_date,
    }


def _odds(home_ml=-140, away_ml=120, total=8.5):
    return {
        "home_moneyline": home_ml, "away_moneyline": away_ml, "total_line": total,
    }


class TestIsPregameLine:
    def test_accepts_an_ordinary_line(self):
        assert is_pregame_line(-120, 110, 8.5)

    def test_accepts_a_big_but_real_favorite_at_coors(self):
        assert is_pregame_line(-350, 280, 13.5)

    def test_accepts_a_moneyline_with_no_total(self):
        assert is_pregame_line(-120, 110, None)
        assert is_pregame_line(-120, 110, "")

    def test_rejects_an_in_play_moneyline(self):
        # SF@MIL, 2026-07-28, captured with the game already decided.
        assert not is_pregame_line(3500, -50000, 6.5)

    def test_rejects_an_in_play_total(self):
        # LAA@MIL, 2026-08-02: 4.5 is runs *remaining*, not the posted total.
        assert not is_pregame_line(-280, 240, 4.5)

    def test_rejects_missing_or_unparseable_moneylines(self):
        assert not is_pregame_line(None, 110, 8.5)
        assert not is_pregame_line("", 110, 8.5)
        assert not is_pregame_line("abc", 110, 8.5)

    def test_keeps_a_row_whose_total_is_unparseable(self):
        # The moneyline is still usable; don't discard it over a bad total field.
        assert is_pregame_line(-120, 110, "n/a")


class TestSaveOddsHistory:
    @pytest.fixture
    def runner(self, tmp_path):
        return DailyRunner(data_dir=tmp_path, model_dir=tmp_path)

    def _rows(self, data_dir):
        path = data_dir / "odds_history.csv"
        if not path.exists():
            return []
        return list(csv.DictReader(path.open()))

    def test_writes_a_pregame_game(self, runner, tmp_path):
        games = [_game("824700", "NYY", "TOR", LATER)]
        runner._save_odds_history({"824700": _odds()}, games)

        rows = self._rows(tmp_path)
        assert len(rows) == 1
        assert rows[0]["home_team"] == "NYY"
        assert rows[0]["away_team"] == "TOR"
        assert rows[0]["game_date"] == "2026-08-04"

    def test_skips_a_game_already_underway(self, runner, tmp_path):
        games = [_game("824700", "NYY", "TOR", EARLIER, status="In Progress")]
        runner._save_odds_history({"824700": _odds(-5000, 2200, 3.5)}, games)

        assert self._rows(tmp_path) == []

    def test_skips_on_the_clock_even_if_status_is_stale(self, runner, tmp_path):
        # Status can lag the feed; first pitch having passed is enough.
        games = [_game("824700", "NYY", "TOR", EARLIER, status="Scheduled")]
        runner._save_odds_history({"824700": _odds()}, games)

        assert self._rows(tmp_path) == []

    def test_skips_on_status_even_without_a_start_time(self, runner, tmp_path):
        games = [_game("824700", "NYY", "TOR", None, status="Final")]
        runner._save_odds_history({"824700": _odds()}, games)

        assert self._rows(tmp_path) == []

    def test_dates_the_row_from_the_game_not_the_run(self, runner, tmp_path):
        # The late cron runs after midnight UTC; the row belongs to the game's
        # own official date, not to whatever "today" the process thinks it is.
        games = [_game("824700", "NYY", "TOR", LATER, game_date="2026-08-03")]
        runner._save_odds_history({"824700": _odds()}, games)

        assert self._rows(tmp_path)[0]["game_date"] == "2026-08-03"

    def test_ignores_odds_for_a_game_not_on_the_schedule(self, runner, tmp_path):
        # The feed spans several dates; only games we resolved get persisted.
        games = [_game("824700", "NYY", "TOR", LATER)]
        runner._save_odds_history({"999999": _odds()}, games)

        assert self._rows(tmp_path) == []

    def test_skips_an_event_missing_a_moneyline(self, runner, tmp_path):
        games = [_game("824700", "NYY", "TOR", LATER)]
        runner._save_odds_history({"824700": _odds(home_ml=None)}, games)

        assert self._rows(tmp_path) == []

    def test_devigs_the_home_probability(self, runner, tmp_path):
        games = [_game("824700", "NYY", "TOR", LATER)]
        runner._save_odds_history({"824700": _odds(-140, 120)}, games)

        # -140/+120 de-vigged proportionally, matching the live feature transform.
        assert float(self._rows(tmp_path)[0]["market_home_prob"]) == pytest.approx(
            0.5620, abs=1e-4
        )

    def test_appends_without_losing_earlier_rows(self, runner, tmp_path):
        games = [_game("824700", "NYY", "TOR", LATER)]
        runner._save_odds_history({"824700": _odds(-140, 120)}, games)
        runner._save_odds_history({"824700": _odds(-150, 130)}, games)

        rows = self._rows(tmp_path)
        assert len(rows) == 2
        # Readers take the last row for a game — the price closest to first pitch.
        assert rows[-1]["home_moneyline"] == "-150"
