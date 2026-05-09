"""Tests for the betting engine."""

import pytest

from mlb.betting.engine import (
    BettingConfig,
    BettingEngine,
    BettingSlip,
    american_to_decimal,
    american_to_implied,
    decimal_to_american,
    remove_vig,
)
from mlb.models.predict import GamePrediction


class TestOddsConversion:
    def test_favorite_to_decimal(self):
        assert american_to_decimal(-150) == pytest.approx(1.6667, abs=0.001)

    def test_underdog_to_decimal(self):
        assert american_to_decimal(150) == 2.5

    def test_even_money(self):
        assert american_to_decimal(100) == 2.0
        assert american_to_decimal(-100) == 2.0

    def test_decimal_to_american_favorite(self):
        assert decimal_to_american(1.5) == -200

    def test_decimal_to_american_underdog(self):
        assert decimal_to_american(3.0) == 200

    def test_implied_probability_favorite(self):
        # -150 implies 60% win probability
        assert american_to_implied(-150) == 0.6

    def test_implied_probability_underdog(self):
        # +200 implies 33.3% win probability
        assert american_to_implied(200) == pytest.approx(0.3333, abs=0.001)

    def test_implied_probabilities_sum_over_1(self):
        # With vig, implied probs should sum > 1
        h = american_to_implied(-150)
        a = american_to_implied(130)
        assert h + a > 1.0


class TestRemoveVig:
    def test_removes_vig(self):
        true_h, true_a = remove_vig(-150, 130)
        assert true_h + true_a == pytest.approx(1.0, abs=0.001)
        assert true_h > 0.5  # Favorite should still be >50%

    def test_even_odds(self):
        true_h, true_a = remove_vig(-110, -110)
        assert true_h == pytest.approx(0.5, abs=0.001)
        assert true_a == pytest.approx(0.5, abs=0.001)


def _make_prediction(**kwargs) -> GamePrediction:
    defaults = {
        "game_id": "123456",
        "game_date": "2026-05-09",
        "home_team_id": "NYY",
        "away_team_id": "BOS",
        "home_win_prob": 0.60,
        "away_win_prob": 0.40,
        "predicted_home_runs": 5.0,
        "predicted_away_runs": 3.5,
        "predicted_total": 8.5,
        "confidence": 65.0,
        "model_agreement": 0.85,
        "model_predictions": {"xgboost": 0.60},
        "top_factors": [("diff_whip_7", 0.05)],
        "home_power_score": 55.0,
        "away_power_score": 48.0,
    }
    defaults.update(kwargs)
    return GamePrediction(**defaults)


class TestBettingEngine:
    def setup_method(self):
        self.engine = BettingEngine()

    def test_find_value_bet_positive_edge(self):
        pred = _make_prediction(home_win_prob=0.65)
        odds = [{
            "game_id": "123456",
            "home_moneyline": -130,  # implies ~56.5%
            "away_moneyline": 110,
            "total_line": 8.5,
            "over_odds": -110,
            "under_odds": -110,
        }]

        slip = self.engine.find_value_bets([pred], odds)
        # Should find at least a home moneyline value bet (65% model vs ~56.5% implied)
        home_bets = [b for b in slip.bets if b.selection == "home" and b.bet_type == "moneyline"]
        assert len(home_bets) >= 1
        assert home_bets[0].edge > 0

    def test_no_value_when_aligned(self):
        pred = _make_prediction(home_win_prob=0.55)
        odds = [{
            "game_id": "123456",
            "home_moneyline": -130,
            "away_moneyline": 110,
            "total_line": 8.5,
            "over_odds": -110,
            "under_odds": -110,
        }]

        slip = self.engine.find_value_bets([pred], odds)
        # Edge is small — may not meet 2% threshold
        for bet in slip.bets:
            if bet.bet_type == "moneyline":
                assert bet.edge >= self.engine.config.min_edge

    def test_position_limits(self):
        config = BettingConfig(max_daily_exposure=0.10)
        engine = BettingEngine(config=config)

        # Create multiple high-edge bets
        predictions = []
        odds_data = []
        for i in range(10):
            pred = _make_prediction(
                game_id=f"game_{i}",
                home_win_prob=0.70,
            )
            predictions.append(pred)
            odds_data.append({
                "game_id": f"game_{i}",
                "home_moneyline": -110,
                "away_moneyline": -110,
                "total_line": None,
            })

        slip = engine.find_value_bets(predictions, odds_data, bankroll=10000)
        assert slip.total_stake <= 1000  # 10% of $10K

    def test_settle_winning_bet(self):
        pred = _make_prediction(home_win_prob=0.65)
        odds = [{
            "game_id": "123456",
            "home_moneyline": -130,
            "away_moneyline": 110,
            "total_line": None,
        }]

        slip = self.engine.find_value_bets([pred], odds)
        if slip.num_bets > 0:
            results = {"123456": {"home_score": 5, "away_score": 3}}
            pnl = self.engine.settle_bets(slip, results, 10000)
            assert pnl.bets_won >= 1 or pnl.bets_placed == 0

    def test_empty_odds(self):
        pred = _make_prediction()
        slip = self.engine.find_value_bets([pred], [])
        assert slip.num_bets == 0

    def test_odds_range_filtering(self):
        config = BettingConfig(min_odds=-150, max_odds=150)
        engine = BettingEngine(config=config)

        pred = _make_prediction(home_win_prob=0.80)
        odds = [{
            "game_id": "123456",
            "home_moneyline": -300,  # Outside range
            "away_moneyline": 250,
            "total_line": None,
        }]

        slip = engine.find_value_bets([pred], odds)
        ml_bets = [b for b in slip.bets if b.bet_type == "moneyline"]
        assert len(ml_bets) == 0
