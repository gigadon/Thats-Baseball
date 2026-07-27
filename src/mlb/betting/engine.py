"""Betting engine — value detection, Kelly sizing, portfolio optimization, P&L.

Core responsibilities:
  - Convert odds formats
  - Detect value bets (model edge > threshold)
  - Calculate optimal bet sizes via Kelly Criterion
  - Manage bankroll and position limits
  - Generate betting slips with risk controls
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from mlb.models.predict import GamePrediction
from mlb.models.runs_calibration import over_under_probabilities

logger = logging.getLogger(__name__)


# ─── Odds Conversion ─────────────────────────────────────────


def american_to_decimal(odds: float) -> float:
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


def decimal_to_american(dec: float) -> float:
    if dec >= 2.0:
        return round((dec - 1) * 100)
    return round(-100 / (dec - 1))


def american_to_implied(odds: float) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def remove_vig(home_odds: float, away_odds: float) -> tuple[float, float]:
    """Remove vig to get true implied probabilities."""
    h_imp = american_to_implied(home_odds)
    a_imp = american_to_implied(away_odds)
    total = h_imp + a_imp
    return h_imp / total, a_imp / total


# ─── Data Classes ─────────────────────────────────────────────


@dataclass
class BetOpportunity:
    """A single identified betting opportunity."""

    game_id: str
    game_date: str
    home_team: str
    away_team: str

    bet_type: str  # "moneyline", "total", "runline"
    selection: str  # "home", "away", "over", "under"
    odds: float  # American odds
    decimal_odds: float

    # Edge
    model_prob: float  # Our predicted probability
    implied_prob: float  # Market implied probability (vig-removed)
    raw_implied_prob: float  # Market implied (with vig)
    edge: float  # model_prob - implied_prob
    edge_pct: float  # edge as percentage

    # Sizing
    kelly_fraction: float  # Full Kelly fraction
    recommended_fraction: float  # Fractional Kelly (typically 0.25)
    recommended_stake: float  # Dollar amount
    max_stake: float  # Position limit

    # Confidence
    confidence: float  # Model confidence (0-100)
    model_agreement: float  # How much models agree

    # Expected value
    ev_per_dollar: float  # Expected value per $1 wagered

    # Total line (for total bets)
    total_line: float | None = None

    @property
    def is_value(self) -> bool:
        return self.edge > 0 and self.ev_per_dollar > 0


@dataclass
class BettingSlip:
    """A recommended set of bets for a given day."""

    slip_date: str
    bankroll: float
    total_stake: float
    num_bets: int
    bets: list[BetOpportunity]
    total_ev: float  # Sum of expected values
    max_exposure: float  # Maximum possible loss
    risk_level: str  # "conservative", "moderate", "aggressive"


@dataclass
class PnLRecord:
    """Profit & loss tracking record."""

    date: str
    starting_bankroll: float
    ending_bankroll: float
    daily_pnl: float
    bets_placed: int
    bets_won: int
    bets_lost: int
    roi: float
    cumulative_pnl: float
    max_drawdown: float


# ─── Betting Engine ───────────────────────────────────────────


@dataclass
class BettingConfig:
    """Betting engine configuration."""

    min_edge: float = 0.01  # Minimum 1% edge to bet
    # Real MLB edges vs the closing line are ~0-3%; anything above this cap is
    # a model/odds artifact (bad feature, stale line, mismatched game), not value.
    max_edge: float = 0.10
    kelly_fraction: float = 0.35  # ~Third Kelly
    max_bet_pct: float = 0.05  # Max 5% of bankroll per bet
    max_daily_exposure: float = 0.20  # Max 20% of bankroll at risk per day
    min_confidence: float = 50.0  # Minimum model confidence to bet
    min_odds: float = -200  # Won't bet on heavy favorites past -200
    max_odds: float = 250  # Won't bet on longshots past +250

    # Over/under calibration. The runs model's held-out residuals (preferred) or
    # its residual std drive P(over); these are populated from the loaded model at
    # runtime (see daily_runner). Defaults are used only if a model predates
    # residual persistence — the old code hardcoded 4.1, which understated the
    # true ~4.45-run uncertainty and systematically inflated totals edges.
    total_residual_std: float = 4.45
    total_residuals: list[float] | None = None


class BettingEngine:
    """Core betting engine for value detection and position sizing."""

    def __init__(self, config: BettingConfig | None = None):
        self.config = config or BettingConfig()
        self._pnl_history: list[PnLRecord] = []

    def find_value_bets(
        self,
        predictions: list[GamePrediction],
        odds_data: list[dict[str, Any]],
        bankroll: float = 10000.0,
        line_movement: dict[str, dict] | None = None,
    ) -> BettingSlip:
        """Identify all value bets from today's predictions.

        Args:
            predictions: Model predictions for each game.
            odds_data: Market odds, each dict has:
                game_id, home_moneyline, away_moneyline,
                total_line, over_odds, under_odds
            bankroll: Current bankroll.
            line_movement: Optional dict of game_id → {opening_home_prob, current_home_prob}.
                If a line has moved >3% against the model's side, the bet is skipped.

        Returns:
            BettingSlip with recommended bets.
        """
        odds_map = {o["game_id"]: o for o in odds_data}
        opportunities: list[BetOpportunity] = []

        for pred in predictions:
            odds = odds_map.get(pred.game_id)
            if not odds:
                continue

            # Moneyline bets
            ml_bets = self._evaluate_moneyline(pred, odds, bankroll)
            opportunities.extend(ml_bets)

            # Total (over/under) bets
            total_bets = self._evaluate_total(pred, odds, bankroll)
            opportunities.extend(total_bets)

        # Filter by minimum edge, confidence, and CLV
        value_bets = []
        for b in opportunities:
            if b.edge < self.config.min_edge:
                continue
            if b.edge > self.config.max_edge:
                logger.info(
                    "Suppressed implausible edge %.1f%% on %s %s (%s) — "
                    "max_edge cap; likely model/odds artifact, not value",
                    b.edge * 100, b.selection,
                    b.home_team if b.selection == "home" else b.away_team,
                    b.bet_type,
                )
                continue
            if b.confidence < self.config.min_confidence:
                continue
            if b.ev_per_dollar <= 0:
                continue
            # CLV filter: skip if line moved against our side
            lm = line_movement.get(b.game_id) if line_movement else None
            if not self._passes_clv_filter(b, lm):
                logger.info(
                    "CLV filter: skipping %s %s (%s) — line moved against",
                    b.selection, b.home_team if b.selection == "home" else b.away_team,
                    b.bet_type,
                )
                continue
            value_bets.append(b)

        # Sort by EV (best first)
        value_bets.sort(key=lambda b: b.ev_per_dollar, reverse=True)

        # Apply portfolio constraints
        value_bets = self._apply_position_limits(value_bets, bankroll)

        total_stake = sum(b.recommended_stake for b in value_bets)
        total_ev = sum(b.ev_per_dollar * b.recommended_stake for b in value_bets)
        max_exposure = sum(b.recommended_stake for b in value_bets)

        risk_level = "conservative"
        if total_stake / bankroll > 0.10:
            risk_level = "moderate"
        if total_stake / bankroll > 0.15:
            risk_level = "aggressive"

        return BettingSlip(
            slip_date=datetime.now().strftime("%Y-%m-%d"),
            bankroll=bankroll,
            total_stake=round(total_stake, 2),
            num_bets=len(value_bets),
            bets=value_bets,
            total_ev=round(total_ev, 2),
            max_exposure=round(max_exposure, 2),
            risk_level=risk_level,
        )

    def settle_bets(
        self,
        slip: BettingSlip,
        results: dict[str, dict],
        bankroll: float,
    ) -> PnLRecord:
        """Settle a day's bets against actual results.

        Args:
            slip: The betting slip with placed bets.
            results: Dict of game_id → {"home_score": int, "away_score": int}.
            bankroll: Starting bankroll for the day.
        """
        daily_pnl = 0.0
        wins = losses = 0

        for bet in slip.bets:
            result = results.get(bet.game_id)
            if not result:
                continue

            home_score = result["home_score"]
            away_score = result["away_score"]

            won = self._is_winner(bet, home_score, away_score)
            if won is None:
                continue  # Push

            if won:
                payout = bet.recommended_stake * (bet.decimal_odds - 1)
                daily_pnl += payout
                wins += 1
            else:
                daily_pnl -= bet.recommended_stake
                losses += 1

        ending = bankroll + daily_pnl
        cum_pnl = sum(r.daily_pnl for r in self._pnl_history) + daily_pnl

        # Max drawdown
        peak = bankroll
        max_dd = 0.0
        running = bankroll
        for r in self._pnl_history:
            running += r.daily_pnl
            peak = max(peak, running)
            dd = (peak - running) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        running += daily_pnl
        peak = max(peak, running)
        dd = (peak - running) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        record = PnLRecord(
            date=slip.slip_date,
            starting_bankroll=bankroll,
            ending_bankroll=round(ending, 2),
            daily_pnl=round(daily_pnl, 2),
            bets_placed=wins + losses,
            bets_won=wins,
            bets_lost=losses,
            roi=round(daily_pnl / slip.total_stake, 4) if slip.total_stake > 0 else 0.0,
            cumulative_pnl=round(cum_pnl, 2),
            max_drawdown=round(max_dd, 4),
        )
        self._pnl_history.append(record)
        return record

    @property
    def pnl_history(self) -> list[PnLRecord]:
        return self._pnl_history

    @property
    def total_pnl(self) -> float:
        return sum(r.daily_pnl for r in self._pnl_history)

    @property
    def total_roi(self) -> float:
        total_staked = sum(
            r.starting_bankroll * 0.1 for r in self._pnl_history  # rough estimate
        )
        return self.total_pnl / total_staked if total_staked > 0 else 0.0

    # ── Internal ──────────────────────────────────────────

    def _evaluate_moneyline(
        self,
        pred: GamePrediction,
        odds: dict,
        bankroll: float,
    ) -> list[BetOpportunity]:
        bets = []
        home_ml = odds.get("home_moneyline")
        away_ml = odds.get("away_moneyline")

        if home_ml is None or away_ml is None:
            return bets

        # Remove vig
        true_home, true_away = remove_vig(home_ml, away_ml)

        # Home bet
        home_edge = pred.home_win_prob - true_home
        if home_edge > 0 and self._odds_in_range(home_ml):
            bets.append(self._build_opportunity(
                pred, "moneyline", "home", home_ml,
                pred.home_win_prob, true_home, american_to_implied(home_ml),
                bankroll,
            ))

        # Away bet
        away_edge = pred.away_win_prob - true_away
        if away_edge > 0 and self._odds_in_range(away_ml):
            bets.append(self._build_opportunity(
                pred, "moneyline", "away", away_ml,
                pred.away_win_prob, true_away, american_to_implied(away_ml),
                bankroll,
            ))

        return bets

    def _evaluate_total(
        self,
        pred: GamePrediction,
        odds: dict,
        bankroll: float,
    ) -> list[BetOpportunity]:
        bets = []
        total_line = odds.get("total_line")
        over_odds = odds.get("over_odds", -110)
        under_odds = odds.get("under_odds", -110)

        if total_line is None:
            return bets

        # Estimate over/under probability from the predicted total using the
        # runs model's real held-out residual distribution (empirical CDF when
        # available, else Normal(residual_std)). Single-sourced in
        # runs_calibration so the live engine and the totals backtester agree.
        over_prob, under_prob = over_under_probabilities(
            pred.predicted_total,
            total_line,
            residuals=self.config.total_residuals,
            residual_std=self.config.total_residual_std,
        )

        true_over, true_under = remove_vig(over_odds, under_odds)

        # Over
        over_edge = over_prob - true_over
        if over_edge > self.config.min_edge:
            bets.append(self._build_opportunity(
                pred, "total", "over", over_odds,
                over_prob, true_over, american_to_implied(over_odds),
                bankroll, total_line=total_line,
            ))

        # Under
        under_edge = under_prob - true_under
        if under_edge > self.config.min_edge:
            bets.append(self._build_opportunity(
                pred, "total", "under", under_odds,
                under_prob, true_under, american_to_implied(under_odds),
                bankroll, total_line=total_line,
            ))

        return bets

    def _build_opportunity(
        self,
        pred: GamePrediction,
        bet_type: str,
        selection: str,
        odds: float,
        model_prob: float,
        implied_prob: float,
        raw_implied: float,
        bankroll: float,
        total_line: float | None = None,
    ) -> BetOpportunity:
        dec_odds = american_to_decimal(odds)
        edge = model_prob - implied_prob

        # Kelly Criterion: f* = (bp - q) / b where b = decimal_odds - 1
        b = dec_odds - 1
        q = 1 - model_prob
        kelly = (b * model_prob - q) / b if b > 0 else 0
        kelly = max(kelly, 0)

        frac_kelly = kelly * self.config.kelly_fraction
        stake = bankroll * frac_kelly
        max_stake = bankroll * self.config.max_bet_pct
        stake = min(stake, max_stake)

        ev = model_prob * b - q  # EV per $1

        return BetOpportunity(
            game_id=pred.game_id,
            game_date=pred.game_date,
            home_team=pred.home_team_id,
            away_team=pred.away_team_id,
            bet_type=bet_type,
            selection=selection,
            odds=odds,
            decimal_odds=round(dec_odds, 3),
            model_prob=round(model_prob, 4),
            implied_prob=round(implied_prob, 4),
            raw_implied_prob=round(raw_implied, 4),
            edge=round(edge, 4),
            edge_pct=round(edge * 100, 2),
            kelly_fraction=round(kelly, 4),
            recommended_fraction=round(frac_kelly, 4),
            recommended_stake=round(stake, 2),
            max_stake=round(max_stake, 2),
            confidence=pred.confidence,
            model_agreement=pred.model_agreement,
            ev_per_dollar=round(ev, 4),
            total_line=total_line,
        )

    def _apply_position_limits(
        self, bets: list[BetOpportunity], bankroll: float
    ) -> list[BetOpportunity]:
        """Enforce daily exposure and per-game limits."""
        max_daily = bankroll * self.config.max_daily_exposure
        total = 0.0
        accepted = []
        games_bet: set[str] = set()

        for bet in bets:
            # One bet per game per type
            key = f"{bet.game_id}_{bet.bet_type}"
            if key in games_bet:
                continue

            if total + bet.recommended_stake > max_daily:
                remaining = max_daily - total
                if remaining < 10:
                    break
                bet.recommended_stake = round(remaining, 2)

            accepted.append(bet)
            games_bet.add(key)
            total += bet.recommended_stake

        return accepted

    def _passes_clv_filter(
        self, bet: BetOpportunity, line_movement: dict | None
    ) -> bool:
        """Return True if the bet passes CLV filtering (should be placed).

        If the line has moved >3% against the model's side since open,
        the market is signaling information the model may not have.
        """
        if line_movement is None:
            return True

        opening = line_movement.get("opening_home_prob")
        current = line_movement.get("current_home_prob")
        if opening is None or current is None:
            return True

        # For away bets, invert to get the bet-side probability
        if bet.selection == "away":
            opening = 1 - opening
            current = 1 - current

        # If the line moved against our side by >3%, skip
        movement = current - opening
        if movement < -0.03:
            return False
        return True

    def _odds_in_range(self, odds: float) -> bool:
        return self.config.min_odds <= odds <= self.config.max_odds

    def _is_winner(
        self, bet: BetOpportunity, home_score: int, away_score: int
    ) -> bool | None:
        if bet.bet_type == "moneyline":
            if home_score == away_score:
                return None  # Push (shouldn't happen in MLB)
            if bet.selection == "home":
                return home_score > away_score
            return away_score > home_score

        elif bet.bet_type == "total":
            total = home_score + away_score
            line = bet.total_line or 8.5
            if total == line:
                return None  # Push
            if bet.selection == "over":
                return total > line
            return total < line

        return None
