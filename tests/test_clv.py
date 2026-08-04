"""CLV must grade a bet against its own game's closing line.

The Odds API knows only the team pair, so snapshots keyed on it collapsed a
doubleheader into one series and both bets took whichever leg was written last.
"""

import json
from datetime import date

import pytest

from mlb.betting.clv import compute_daily_clv
from mlb.data.line_movement import get_all_line_movements, get_line_movement
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
