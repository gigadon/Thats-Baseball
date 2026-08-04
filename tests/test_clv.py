"""CLV must grade a bet against its own game's closing line.

The Odds API knows only the team pair, so snapshots keyed on it collapsed a
doubleheader into one series and both bets took whichever leg was written last.
"""

import json
from datetime import date, datetime, timezone

import pytest

from mlb.betting.clv import compute_daily_clv
from mlb.data.line_movement import (
    get_all_line_movements,
    get_line_movement,
    record_snapshot,
)
from mlb.etl.slate_record import lock_slate

DAY = "2026-07-22"


def _snapshot_game(game_id, home, away, home_prob):
    return {
        "game_id": game_id, "home_team": home, "away_team": away,
        "home_prob": home_prob, "home_moneyline": -140, "away_moneyline": 120,
    }


def _write_snapshots(tmp_path, *snapshots):
    lm = tmp_path / "line_movement"
    lm.mkdir(parents=True, exist_ok=True)
    (lm / f"{DAY}.json").write_text(json.dumps({
        "date": DAY,
        "snapshots": [
            {"timestamp": f"2026-07-22T{10 + i}:00:00", "games": list(games)}
            for i, games in enumerate(snapshots)
        ],
    }))


def _bet(game_id, home, away, model_prob=0.62, selection=None):
    return {
        "game_id": game_id, "home_team": home, "away_team": away,
        "bet_type": "moneyline", "selection": selection or home,
        "odds": -140, "model_prob": model_prob, "implied_prob": 0.58,
        "recommended_stake": 100.0,
    }


def _lock(tmp_path, bets):
    lock_slate(
        date.fromisoformat(DAY), [],
        {"bankroll": 10000.0, "num_bets": len(bets), "bets": bets},
        tmp_path,
    )


class TestDoubleheaderClosingLines:
    def test_each_leg_grades_against_its_own_closing_line(self, tmp_path):
        # Game 1 closed at 0.55, game 2 at 0.70. The model had both at 0.62.
        _write_snapshots(tmp_path, [
            _snapshot_game("824735", "BOS", "BAL", 0.55),
            _snapshot_game("824732", "BOS", "BAL", 0.70),
        ])
        _lock(tmp_path, [
            _bet("824735", "BOS", "BAL", model_prob=0.62),
            _bet("824732", "BOS", "BAL", model_prob=0.62),
        ])

        results = compute_daily_clv(DAY, tmp_path)
        clv = {r["game_id"]: r["clv"] for r in results}

        # +7 points on the leg the market underrated, -8 on the one it overrated.
        assert clv["824735"] == pytest.approx(0.07)
        assert clv["824732"] == pytest.approx(-0.08)

    def test_a_legacy_snapshot_without_game_ids_skips_the_pair(self, tmp_path):
        """Files written before game_ids were recorded can't be separated —
        report nothing rather than grade against the wrong leg."""
        _write_snapshots(tmp_path, [
            _snapshot_game("", "BOS", "BAL", 0.55),
            _snapshot_game("", "BOS", "BAL", 0.70),
        ])
        _lock(tmp_path, [_bet("824735", "BOS", "BAL", model_prob=0.62)])

        results = compute_daily_clv(DAY, tmp_path)

        # Falls through to the bet's own implied prob, not a coin-flip leg.
        assert results[0]["closing_prob"] == pytest.approx(0.58)


class TestSingleGames:
    def test_matches_on_game_id(self, tmp_path):
        _write_snapshots(tmp_path, [_snapshot_game("824700", "NYY", "TOR", 0.55)])
        _lock(tmp_path, [_bet("824700", "NYY", "TOR", model_prob=0.62)])

        assert compute_daily_clv(DAY, tmp_path)[0]["clv"] == pytest.approx(0.07)

    def test_legacy_snapshot_still_matches_an_unambiguous_pair(self, tmp_path):
        _write_snapshots(tmp_path, [_snapshot_game("", "NYY", "TOR", 0.55)])
        _lock(tmp_path, [_bet("824700", "NYY", "TOR", model_prob=0.62)])

        assert compute_daily_clv(DAY, tmp_path)[0]["clv"] == pytest.approx(0.07)

    def test_away_side_bet_flips_the_comparison(self, tmp_path):
        _write_snapshots(tmp_path, [_snapshot_game("824700", "NYY", "TOR", 0.55)])
        _lock(tmp_path, [
            _bet("824700", "NYY", "TOR", model_prob=0.40, selection="TOR"),
        ])

        # Model has away at 0.60, market at 0.45 → +15 points.
        assert compute_daily_clv(DAY, tmp_path)[0]["clv"] == pytest.approx(0.15)

    def test_no_snapshots_falls_back_to_implied_prob(self, tmp_path):
        _lock(tmp_path, [_bet("824700", "NYY", "TOR", model_prob=0.62)])

        results = compute_daily_clv(DAY, tmp_path)

        assert results[0]["closing_prob"] == pytest.approx(0.58)


class TestClosingLineAcrossSnapshots:
    """The pipeline appends a snapshot per run; the close is the last pre-game one."""

    def test_uses_the_last_snapshot_the_game_appears_in(self, tmp_path):
        # An afternoon game is gone from the odds feed by the evening runs, so
        # the day's final snapshot doesn't contain it at all.
        _write_snapshots(
            tmp_path,
            [_snapshot_game("824700", "NYY", "TOR", 0.50),      # main card
             _snapshot_game("824900", "SD", "LAD", 0.60)],
            [_snapshot_game("824700", "NYY", "TOR", 0.55),      # last before 1:05pm
             _snapshot_game("824900", "SD", "LAD", 0.62)],
            [_snapshot_game("824900", "SD", "LAD", 0.68)],      # evening: NYY gone
        )
        _lock(tmp_path, [
            _bet("824700", "NYY", "TOR", model_prob=0.62),
            _bet("824900", "SD", "LAD", model_prob=0.62),
        ])

        clv = {r["game_id"]: r["closing_prob"] for r in compute_daily_clv(DAY, tmp_path)}

        assert clv["824700"] == pytest.approx(0.55)   # not the 0.50 open
        assert clv["824900"] == pytest.approx(0.68)

    def test_a_legacy_doubleheader_stays_ambiguous_across_snapshots(self, tmp_path):
        # Grouping per snapshot must not let dedup collapse two id-less legs.
        _write_snapshots(
            tmp_path,
            [_snapshot_game("", "BOS", "BAL", 0.55),
             _snapshot_game("", "BOS", "BAL", 0.70)],
            [_snapshot_game("", "BOS", "BAL", 0.57),
             _snapshot_game("", "BOS", "BAL", 0.72)],
        )
        _lock(tmp_path, [_bet("824735", "BOS", "BAL", model_prob=0.62)])

        results = compute_daily_clv(DAY, tmp_path)

        assert results[0]["closing_prob"] == pytest.approx(0.58)  # implied fallback


class TestRecordSnapshot:
    # Noon ET on game day: first pitch is still ahead.
    PREGAME = datetime(2026, 7, 22, 16, 0, tzinfo=timezone.utc)

    def _odds(self, **overrides):
        base = {
            "home_team": "NYY", "away_team": "TOR",
            "commence_time": "2026-07-22T23:05:00Z",
            "home_moneyline": -140, "away_moneyline": 120, "total_line": 8.5,
        }
        base.update(overrides)
        return {"824700": base}

    def _read(self, tmp_path):
        return json.loads((tmp_path / "line_movement" / f"{DAY}.json").read_text())

    def test_appends_one_snapshot_per_call(self, tmp_path):
        day = date.fromisoformat(DAY)

        assert record_snapshot(self._odds(), day, tmp_path, now=self.PREGAME) == 1
        record_snapshot(self._odds(), day, tmp_path, now=self.PREGAME)

        data = self._read(tmp_path)
        assert len(data["snapshots"]) == 2
        assert data["snapshots"][0]["games"][0]["game_id"] == "824700"

    def test_probability_is_devigged(self, tmp_path):
        # -110 both sides is a 50/50 market once the vig comes out.
        record_snapshot(
            self._odds(home_moneyline=-110, away_moneyline=-110),
            date.fromisoformat(DAY), tmp_path, now=self.PREGAME,
        )

        game = self._read(tmp_path)["snapshots"][0]["games"][0]
        assert game["home_prob"] == pytest.approx(0.5)

    def test_games_without_moneylines_are_skipped(self, tmp_path):
        n = record_snapshot(
            self._odds(home_moneyline=None, away_moneyline=None),
            date.fromisoformat(DAY), tmp_path, now=self.PREGAME,
        )

        assert n == 0
        assert not (tmp_path / "line_movement" / f"{DAY}.json").exists()

    def test_in_play_lines_are_not_recorded(self, tmp_path):
        """The feed keeps some games after first pitch and switches them to
        in-play prices; a 3% home side two hours in is not a closing line."""
        two_hours_in = datetime(2026, 7, 23, 1, 5, tzinfo=timezone.utc)

        n = record_snapshot(
            self._odds(home_moneyline=3300, away_moneyline=-8000),
            date.fromisoformat(DAY), tmp_path, now=two_hours_in,
        )

        assert n == 0

    def test_a_game_with_no_start_time_is_kept(self, tmp_path):
        n = record_snapshot(
            self._odds(commence_time=""),
            date.fromisoformat(DAY), tmp_path, now=self.PREGAME,
        )

        assert n == 1


class TestSummaryCountsOnlyRealClosingLines:
    @pytest.fixture(autouse=True)
    def _pin_today(self, monkeypatch):
        monkeypatch.setattr("mlb.betting.clv._today_et", lambda: date(2026, 7, 23))

    def test_fallback_bets_are_excluded_and_counted(self, tmp_path):
        from mlb.betting.clv import compute_clv_summary

        # 7/22 has a snapshot; 7/21 does not, so its bet only has the price it was
        # struck at — that is edge, not line value, and must not skew the average.
        _write_snapshots(tmp_path, [_snapshot_game("824700", "NYY", "TOR", 0.55)])
        _lock(tmp_path, [_bet("824700", "NYY", "TOR", model_prob=0.62)])
        lock_slate(
            date(2026, 7, 21), [],
            {"bankroll": 10000.0, "num_bets": 1,
             "bets": [_bet("824999", "SD", "LAD", model_prob=0.80)]},
            tmp_path,
        )

        summary = compute_clv_summary(3, tmp_path)

        assert summary["total_bets"] == 1
        assert summary["bets_without_closing_line"] == 1
        assert summary["avg_clv"] == pytest.approx(0.07)

    def test_results_mark_where_the_closing_price_came_from(self, tmp_path):
        _write_snapshots(tmp_path, [_snapshot_game("824700", "NYY", "TOR", 0.55)])
        _lock(tmp_path, [
            _bet("824700", "NYY", "TOR"),
            _bet("824999", "SD", "LAD"),      # no snapshot for this one
        ])

        sources = {r["game_id"]: r["closing_source"]
                   for r in compute_daily_clv(DAY, tmp_path)}

        assert sources == {"824700": "line_movement", "824999": "bet_implied"}


class TestLineMovementReaders:
    def test_movements_are_keyed_by_game(self, tmp_path):
        _write_snapshots(
            tmp_path,
            [_snapshot_game("824735", "BOS", "BAL", 0.55),
             _snapshot_game("824732", "BOS", "BAL", 0.70)],
            [_snapshot_game("824735", "BOS", "BAL", 0.58),
             _snapshot_game("824732", "BOS", "BAL", 0.66)],
        )

        movements = get_all_line_movements(DAY, tmp_path)

        assert movements["824735"]["probs"] == [0.55, 0.58]
        assert movements["824732"]["probs"] == [0.70, 0.66]

    def test_game_id_pins_a_doubleheader_leg(self, tmp_path):
        _write_snapshots(tmp_path, [
            _snapshot_game("824735", "BOS", "BAL", 0.55),
            _snapshot_game("824732", "BOS", "BAL", 0.70),
        ])

        assert get_line_movement(DAY, "BOS", "BAL", tmp_path, "824735") == [0.55]
        # Without one, the matchup alone is not an answer.
        assert get_line_movement(DAY, "BOS", "BAL", tmp_path) == []

    def test_unambiguous_matchup_needs_no_game_id(self, tmp_path):
        _write_snapshots(tmp_path, [_snapshot_game("824700", "NYY", "TOR", 0.55)])

        assert get_line_movement(DAY, "NYY", "TOR", tmp_path) == [0.55]
