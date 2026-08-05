"""Shared feature formulas — single implementations used by BOTH pipelines.

The training builder (mlb.etl.build_training_data) and the live daily runner
(mlb.etl.daily_runner) historically re-implemented the same feature math, and
the copies drifted (de-vigged market prob vs raw vig-inflated implied; bullpen
freshness on a different scale; lineup OBP with vs without HBP/SF). Each drift
fed the model out-of-distribution values live and biased picks. Every formula
here has exactly one implementation; both pipelines must import from this
module rather than re-deriving the math inline.

See also mlb.features.defaults for the shared missing-data defaults registry.
"""

from __future__ import annotations

from typing import Mapping


def american_implied(moneyline: float) -> float:
    """Raw implied win probability of an American moneyline (includes vig)."""
    ml = float(moneyline)
    return abs(ml) / (abs(ml) + 100.0) if ml < 0 else 100.0 / (ml + 100.0)


def devig_home_prob(home_ml: float, away_ml: float) -> float:
    """No-vig (proportionally normalized) home win probability from moneylines.

    This is the exact transform used to build odds_history.csv, which is the
    training source of ``market_home_prob`` — the live feature MUST use the
    same transform. (Feeding the raw vig-inflated home implied prob instead
    was a cause of live home bias.)
    """
    h_imp = american_implied(home_ml)
    a_imp = american_implied(away_ml)
    return h_imp / (h_imp + a_imp)


# Bounds on what a *pregame* MLB line can be. Both the original dataset import
# and (until the fix in daily_runner._save_odds_history) the live capture wrote
# in-play quotes into odds_history.csv: a game in the 8th with a big lead prices
# at -50000 with 3.5 runs left on the total. Those are not the market's opinion
# before first pitch, and they corrupt both sides at once — market_home_prob is a
# model feature, and the same file is the market baseline mlb.models.backtest
# grades the model against.
#
# Real pregame lines sit well inside these: the biggest MLB favorites land around
# -400, and totals run 6.5 to 12.5 (13.5 at Coors in thin air).
MAX_PREGAME_MONEYLINE = 600
MIN_PREGAME_TOTAL = 5.5
MAX_PREGAME_TOTAL = 14.0


def is_pregame_line(
    home_ml: object, away_ml: object, total_line: object = None
) -> bool:
    """True if a moneyline/total triple is plausibly a line posted before first pitch.

    Shared by every odds_history.csv reader — the training builder, the backtest's
    market baseline — so they agree on what counts as a real market price.
    """
    try:
        h = float(home_ml)  # type: ignore[arg-type]
        a = float(away_ml)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    if h != h or a != a:  # NaN
        return False

    if abs(h) > MAX_PREGAME_MONEYLINE or abs(a) > MAX_PREGAME_MONEYLINE:
        return False

    if total_line not in (None, "", "nan", "None"):
        try:
            t = float(total_line)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return True  # unparseable total, moneyline still usable
        if t == t and not (MIN_PREGAME_TOTAL <= t <= MAX_PREGAME_TOTAL):
            return False

    return True


def lineup_obp(hits: float, walks: float, at_bats: float, default: float) -> float:
    """Training's OBP formula: (H+BB)/(AB+BB) — deliberately omits HBP/SF.

    The MLB API's real OBP includes HBP/SF and runs ~0.016 higher than this,
    which is ~0.8 train-std out of distribution; live code must use this
    formula, not the API's obp field.
    """
    denom = at_bats + walks
    return (hits + walks) / denom if denom > 0 else default


def bp_freshness_from_ip(ip_3d: float) -> float:
    """Map recent bullpen IP to the training bp_freshness scale.

    In training, bp_freshness is the fraction of the season's bullpen unused
    over the last 3 days: mean ~0.934, std ~0.019, range ~0.91–0.96 —
    effectively near-constant. The live pipeline only has reliever IP over
    3 days, so it reproduces the training scale: centered ~0.945 with a mild
    penalty for heavy recent usage, clamped to the training range. (The old
    live formula ``1 - ip/15`` produced ~0.39 — about 29 std out of
    distribution — and drove live picks toward home.)
    """
    return min(0.96, max(0.90, 0.945 - max(0.0, ip_3d - 6.0) * 0.005))


def compute_interaction_features(
    feat: Mapping[str, float],
    park_runs_factor: float,
    default_ops: float = 0.720,
    default_sp_era: float = 3.8571,
) -> dict[str, float]:
    """The ~10 ``interact_*`` mismatch-detection features.

    ``feat`` must already contain the diff/base features both pipelines build
    (``elo_diff``, ``diff_sp_season_era``, ``diff_bp_freshness``,
    ``diff_momentum``, ``diff_ewm_win_pct``, ``rest_diff``,
    ``h_sp_season_era``, ``a_sp_season_era``, ``h_ops_14``, ``a_ops_14``).
    Previously duplicated inline in build_training_data and daily_runner.
    """
    elo_gap = feat.get("elo_diff", 0.0) / 100.0  # ~±2 range
    sp_era_gap = feat.get("diff_sp_season_era", 0.0)  # negative = home SP better
    bp_fresh = feat.get("diff_bp_freshness", 0.0)
    mom_gap = feat.get("diff_momentum", 0.0)
    form_gap = feat.get("diff_ewm_win_pct", 0.0) * 10  # scale up

    out: dict[str, float] = {
        # Core interactions
        "interact_elo_x_sp": elo_gap * (-sp_era_gap),  # both favor home → positive
        "interact_elo_x_bp": elo_gap * bp_fresh,
        "interact_sp_x_bp": (-sp_era_gap) * bp_fresh,
        "interact_elo_x_momentum": elo_gap * mom_gap,
    }

    # SP quality × opposing offense (mismatch detector)
    h_off_ops = feat.get("h_ops_14", default_ops)
    a_off_ops = feat.get("a_ops_14", default_ops)
    out["interact_hsp_vs_aoff"] = (-sp_era_gap) * (a_off_ops - 0.720) * 10
    out["interact_asp_vs_hoff"] = sp_era_gap * (h_off_ops - 0.720) * 10

    # Rest × form (rested team on a hot streak)
    out["interact_rest_x_form"] = feat.get("rest_diff", 0.0) * form_gap

    # Park factor × SP quality (elite SP in hitter park)
    out["interact_park_x_sp"] = (park_runs_factor - 1.0) * 10 * (-sp_era_gap)

    # Bullpen fatigue × close-game likelihood: two good SPs → likely close game
    h_sp_era = feat.get("h_sp_season_era", default_sp_era)
    a_sp_era = feat.get("a_sp_season_era", default_sp_era)
    pitching_duel = max(0, (9.0 - h_sp_era - a_sp_era) / 4.0)
    out["interact_bp_x_duel"] = bp_fresh * pitching_duel

    return out
