"""Tests for the shared score lookup — game_id keying and source fallback."""

from datetime import date

import pandas as pd
import pytest

from mlb.data.scores import (
    FinalScore,
    ScoreBook,
    load_scorebook,
    score_on_date,
    scorebook_from_csv,
)

DAY = date(2026, 7, 22)  # a real doubleheader date: BOS/BAL and NYY/PIT


def _score(game_id, home, away, hs, as_, source="mlb_api"):
    return FinalScore(
        game_id=game_id, home_team=home, away_team=away,
        home_score=hs, away_score=as_, source=source,
    )


class TestScoreOnDate:
    def test_matches_evening_game_in_et(self):
        # 7:05 PM ET first pitch → stored as the next UTC day, still July 7 in ET.
        score = {"commence_time": "2026-07-07T23:05:00Z"}
        assert score_on_date(score, date(2026, 7, 7))
        assert not score_on_date(score, date(2026, 7, 8))

    def test_after_midnight_utc_still_prior_et_date(self):
        score = {"commence_time": "2026-07-08T01:10:00Z"}  # 9:10 PM ET on the 7th
        assert score_on_date(score, date(2026, 7, 7))

    def test_missing_or_bad_commence_time_is_lenient(self):
        assert score_on_date({}, date(2026, 7, 7))
        assert score_on_date({"commence_time": "not-a-date"}, date(2026, 7, 7))


class TestDoubleheaders:
    """The reason this module exists: a team pair is not unique on a date."""

    @pytest.fixture
    def book(self):
        # 2026-07-22: BOS swept BAL in a twin bill, opposite results by leg.
        return ScoreBook(DAY, [
            _score("824735", "BOS", "BAL", 7, 2),
            _score("824732", "BOS", "BAL", 1, 9),
        ], "mlb_api")

    def test_game_id_picks_the_right_leg(self, book):
        assert book.lookup("824735", "BOS", "BAL").home_score == 7
        assert book.lookup("824732", "BOS", "BAL").home_score == 1

    def test_int_game_id_matches_a_string_key(self, book):
        # CSV reads give ints, the MLB API gives strings; they have to compare equal.
        assert book.lookup(824735, "BOS", "BAL").home_score == 7

    def test_ambiguous_pair_is_skipped_not_guessed(self, book):
        # Without an id there is no honest answer — returning either leg would be
        # a coin flip recorded as a result.
        assert book.lookup(None, "BOS", "BAL") is None
        assert book.lookup("", "BOS", "BAL") is None

    def test_unknown_game_id_falls_back_to_the_pair_and_stays_ambiguous(self, book):
        assert book.lookup("999999", "BOS", "BAL") is None

    def test_ambiguous_pairs_reported(self, book):
        assert book.ambiguous_pairs() == [("BOS", "BAL")]


class TestUnambiguousLookup:
    @pytest.fixture
    def book(self):
        return ScoreBook(DAY, [_score("824700", "NYY", "TOR", 4, 3)], "mlb_api")

    def test_pair_matches_when_game_id_is_missing(self, book):
        # Odds API entries carry no gamePk; a single-game matchup still resolves.
        assert book.lookup(None, "NYY", "TOR").home_score == 4

    def test_game_id_wins_when_present(self, book):
        assert book.lookup("824700", "NYY", "TOR").away_score == 3

    def test_unknown_matchup_is_none(self, book):
        assert book.lookup("1", "SEA", "LAA") is None

    def test_reversed_matchup_does_not_match(self, book):
        # Home and away are not interchangeable — TOR hosting NYY is another game.
        assert book.lookup(None, "TOR", "NYY") is None


class TestCsvSource:
    def _write(self, tmp_path, rows):
        pd.DataFrame(rows).to_csv(tmp_path / "games_2026.csv", index=False)

    def test_reads_finals_for_the_date_only(self, tmp_path):
        self._write(tmp_path, [
            {"game_id": 824735, "game_date": "2026-07-22", "status": "Final",
             "home_team_id": "BOS", "away_team_id": "BAL",
             "home_score": 7, "away_score": 2},
            {"game_id": 824800, "game_date": "2026-07-23", "status": "Final",
             "home_team_id": "BOS", "away_team_id": "BAL",
             "home_score": 3, "away_score": 1},
            {"game_id": 824901, "game_date": "2026-07-22", "status": "Scheduled",
             "home_team_id": "NYY", "away_team_id": "TOR",
             "home_score": 0, "away_score": 0},
        ])

        book = scorebook_from_csv(DAY, tmp_path)

        assert len(book) == 1
        assert book.lookup(824735, "BOS", "BAL").home_score == 7

    def test_csv_without_a_game_id_column_still_matches_on_pair(self, tmp_path):
        # Older fixtures/CSVs predate the game_id column.
        self._write(tmp_path, [
            {"game_date": "2026-07-22", "status": "Final",
             "home_team_id": "NYY", "away_team_id": "BOS",
             "home_score": 5, "away_score": 2},
        ])

        book = scorebook_from_csv(DAY, tmp_path)

        assert book.lookup(None, "NYY", "BOS").home_score == 5

    def test_missing_csv_is_empty_not_an_error(self, tmp_path):
        assert len(scorebook_from_csv(DAY, tmp_path)) == 0


class _FakeClient:
    def __init__(self, games=None, error=None):
        self._games = games or []
        self._error = error

    async def get_schedule(self, target_date):
        if self._error:
            raise self._error
        return self._games

    async def close(self):
        pass


class TestSourceFallback:
    @staticmethod
    def _patch_mlb(monkeypatch, client):
        import mlb.data.mlb_api as api

        monkeypatch.setattr(api, "MLBApiClient", lambda *a, **k: client)

    async def test_mlb_api_is_primary(self, monkeypatch, tmp_path):
        self._patch_mlb(monkeypatch, _FakeClient([{
            "game_id": "824735", "status": "Final",
            "home_team_id": "BOS", "away_team_id": "BAL",
            "home_score": 7, "away_score": 2,
        }]))

        book = await load_scorebook(DAY, tmp_path)

        assert book.source == "mlb_api"
        assert book.lookup("824735", "BOS", "BAL").home_score == 7

    async def test_non_final_games_are_excluded(self, monkeypatch, tmp_path):
        # A suspended game must not grade as if it had finished.
        self._patch_mlb(monkeypatch, _FakeClient([{
            "game_id": "1", "status": "Suspended",
            "home_team_id": "BOS", "away_team_id": "BAL",
            "home_score": 3, "away_score": 1,
        }]))

        book = await load_scorebook(DAY, tmp_path, allow_odds_api=False)

        assert len(book) == 0

    async def test_falls_back_to_csv_when_the_api_fails(self, monkeypatch, tmp_path):
        self._patch_mlb(monkeypatch, _FakeClient(error=RuntimeError("502")))
        pd.DataFrame([{
            "game_id": 824735, "game_date": "2026-07-22", "status": "Final",
            "home_team_id": "BOS", "away_team_id": "BAL",
            "home_score": 7, "away_score": 2,
        }]).to_csv(tmp_path / "games_2026.csv", index=False)

        book = await load_scorebook(DAY, tmp_path, allow_odds_api=False)

        assert book.source == "games_csv"
        assert len(book) == 1

    async def test_no_source_yields_an_empty_book(self, monkeypatch, tmp_path):
        self._patch_mlb(monkeypatch, _FakeClient([]))

        book = await load_scorebook(DAY, tmp_path, allow_odds_api=False)

        assert len(book) == 0
        assert book.source == "none"

    async def test_odds_api_is_last_resort_and_date_filtered(self, monkeypatch, tmp_path):
        self._patch_mlb(monkeypatch, _FakeClient([]))
        monkeypatch.setattr("mlb.data.scores._today_et", lambda: date(2026, 7, 23))

        async def fake_scores(self, days_from=1):
            return [
                {"commence_time": "2026-07-22T23:05:00Z", "home_team": "TEX",
                 "away_team": "LAA", "home_score": 8, "away_score": 3},
                {"commence_time": "2026-07-23T23:05:00Z", "home_team": "TEX",
                 "away_team": "LAA", "home_score": 1, "away_score": 13},
            ]

        monkeypatch.setattr("mlb.data.odds_api.OddsApiClient.get_scores", fake_scores)

        book = await load_scorebook(DAY, tmp_path)

        assert book.source == "odds_api"
        assert len(book) == 1  # the adjacent series game is filtered out
        assert book.lookup(None, "TEX", "LAA").home_score == 8
