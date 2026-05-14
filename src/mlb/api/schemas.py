"""Pydantic response/request schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ─── Predictions ──────────────────────────────────────────────


class GamePredictionResponse(BaseModel):
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    home_win_prob: float
    away_win_prob: float
    predicted_home_runs: float
    predicted_away_runs: float
    predicted_total: float
    confidence: float
    model_agreement: float
    predicted_winner: str
    model_predictions: dict[str, float] = {}
    top_factors: list[list] = []
    home_power_score: float = 0.0
    away_power_score: float = 0.0
    home_sp_name: str = "TBD"
    away_sp_name: str = "TBD"
    home_wins: int = 0
    home_losses: int = 0
    away_wins: int = 0
    away_losses: int = 0
    home_streak: str = ""
    away_streak: str = ""
    home_sp_era: float | None = None
    away_sp_era: float | None = None
    home_sp_wins: int | None = None
    home_sp_losses: int | None = None
    away_sp_wins: int | None = None
    away_sp_losses: int | None = None


class PredictionRequest(BaseModel):
    home_team: str = Field(..., min_length=2, max_length=3)
    away_team: str = Field(..., min_length=2, max_length=3)
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")


class DailyPredictionsResponse(BaseModel):
    date: str
    games: list[GamePredictionResponse]
    total_games: int


# ─── Rankings ─────────────────────────────────────────────────


class TeamRankingResponse(BaseModel):
    rank: int
    team_id: str
    team_name: str
    division: str
    league: str
    power_score: float
    offense_score: float
    pitching_score: float
    defense_score: float
    bullpen_score: float
    momentum_score: float
    wins: int
    losses: int
    win_pct: float
    run_diff: int
    last_10_record: str
    streak: str
    rank_change: int
    tier: str


class TeamRankingsResponse(BaseModel):
    ranking_date: str
    category: str
    rankings: list[TeamRankingResponse]


class PlayerRankingResponse(BaseModel):
    rank: int
    player_id: int
    player_name: str
    team_id: str
    position: str
    score: float
    key_stats: dict[str, float]
    games_played: int
    rank_change: int
    tier: str


class PlayerRankingsResponse(BaseModel):
    position: str
    ranking_date: str
    rankings: list[PlayerRankingResponse]


class TeamComparisonResponse(BaseModel):
    team_a: str
    team_b: str
    power: dict
    offense: dict
    pitching: dict
    defense: dict
    bullpen: dict
    momentum: dict
    advantage: str


# ─── Betting ──────────────────────────────────────────────────


class BetResponse(BaseModel):
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    bet_type: str
    selection: str
    odds: float
    model_prob: float
    implied_prob: float
    edge: float
    edge_pct: float
    kelly_fraction: float
    recommended_stake: float
    confidence: float
    ev_per_dollar: float
    decimal_odds: float = 0.0
    total_line: float | None = None


class BettingSlipResponse(BaseModel):
    slip_date: str
    bankroll: float
    total_stake: float
    num_bets: int
    bets: list[BetResponse]
    total_ev: float
    max_exposure: float
    risk_level: str


class BettingConfigRequest(BaseModel):
    bankroll: float = 10000.0
    risk: str = "moderate"  # conservative, moderate, aggressive
    min_edge: float = 0.02
    kelly_fraction: float = 0.25


# ─── Analytics ────────────────────────────────────────────────


class PerformanceResponse(BaseModel):
    period: str
    total_games: int
    accuracy: float
    brier_score: float
    auc_roc: float
    roi_flat: float
    roi_kelly: float
    high_confidence_accuracy: float
    calibration_error: float


class PnLResponse(BaseModel):
    date: str
    daily_pnl: float
    cumulative_pnl: float
    bets_placed: int
    bets_won: int
    roi: float
    max_drawdown: float


class StadiumAnalysisResponse(BaseModel):
    team_id: str
    stadium_name: str
    overall_pf: float
    runs_pf: float
    hr_pf: float
    altitude_ft: float
    surface: str
    roof: str
    dimension_avg: float


class BacktestRequest(BaseModel):
    start_date: str
    end_date: str
    strategy: str = "kelly_fractional"
    bankroll: float = 10000.0
    kelly_fraction: float = 0.25


class BacktestResponse(BaseModel):
    start_date: str
    end_date: str
    total_games: int
    total_bets: int
    win_rate: float
    roi_flat: float
    roi_kelly: float
    max_drawdown: float
    sharpe_ratio: float
    monthly_results: list[dict]
