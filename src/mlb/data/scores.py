"""Final scores for a single day, keyed by game_id — the one source both graders read.

Accuracy and settlement used to fetch scores separately, from different APIs, and
index them by ``(home_team, away_team)``. That had two consequences:

- **Doubleheaders graded wrong.** A team pair is not unique on a date. Both legs
  collapsed into one map entry, so whichever score landed last graded both games.
- **The two graders could disagree.** One went to the MLB API, the other to the
  Odds API, and their team-abbreviation maps drift independently. A drift showed
  up as bets silently skipped for "no score".

``ScoreBook`` fixes both: it indexes by ``game_id`` (which predictions and bets
both carry), falls back to the team pair only when that pair is unambiguous for
the date, and refuses to guess when it isn't.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Statuses that mean the game is over and the score is the final one. Kept short
# and explicit on purpose — a suspended game must not grade as if it finished.
FINAL_STATUSES = frozenset({"Final", "Game Over", "Completed Early"})


def _today_et() -> date:
    """Today in US Eastern — naive date.today() is tomorrow on UTC hosts after 8 PM ET."""
    return datetime.now(ZoneInfo("America/New_York")).date()


def _norm_id(value: Any) -> str:
    """Normalise a game id to a string. Missing/NaN becomes ''.

    CSV reads give ints (or floats when the column has holes), the MLB API gives
    strings; they have to compare equal.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        if value.is_integer():
            return str(int(value))
    return str(value).strip()


@dataclass(frozen=True)
class FinalScore:
    """One completed game.

    `game_id` is "" for sources that don't carry the MLB gamePk (the Odds API has
    only its own event ids). Such entries can still be matched on an unambiguous
    team pair, but never on a doubleheader.
    """

    game_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    source: str


class ScoreBook:
    """The final scores for one date, with game_id and team-pair indexes."""

    def __init__(
        self,
        target_date: date,
        scores: Sequence[FinalScore],
        source: str = "none",
    ) -> None:
        self.target_date = target_date
        self.source = source
        self._scores = list(scores)
        self._by_id: dict[str, FinalScore] = {}
        self._by_pair: dict[tuple[str, str], list[FinalScore]] = {}
        for s in self._scores:
            if s.game_id:
                self._by_id[s.game_id] = s
            self._by_pair.setdefault((s.home_team, s.away_team), []).append(s)

    def __len__(self) -> int:
        return len(self._scores)

    def __iter__(self):
        return iter(self._scores)

    def lookup(
        self, game_id: Any, home_team: str, away_team: str
    ) -> FinalScore | None:
        """The final score for a game, or None when it can't be identified.

        Prefers `game_id`. Falls back to the team pair only when exactly one game
        on this date had that matchup — on a doubleheader, returning either leg
        would be a coin flip dressed up as a result, so return nothing instead.
        """
        gid = _norm_id(game_id)
        if gid and gid in self._by_id:
            return self._by_id[gid]

        candidates = self._by_pair.get((home_team, away_team), [])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            logger.warning(
                "Ambiguous matchup %s@%s on %s (%d games, game_id %r not matched) "
                "— skipping rather than guessing a leg",
                away_team, home_team, self.target_date, len(candidates), gid,
            )
        return None

    def ambiguous_pairs(self) -> list[tuple[str, str]]:
        """Matchups that appear more than once on this date (doubleheaders)."""
        return [pair for pair, games in self._by_pair.items() if len(games) > 1]

    def game_ids(self) -> set[str]:
        return set(self._by_id)


def score_on_date(score: dict, target_date: date) -> bool:
    """True if an Odds API score's first pitch (ET) falls on target_date.

    Used to keep a multi-day scores window from settling a slip with an
    adjacent series game's result. When commence_time is absent, fall back to
    True so we don't silently drop otherwise-matchable scores.
    """
    commence = score.get("commence_time")
    if not commence:
        return True
    try:
        dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return True
    return dt.astimezone(ZoneInfo("America/New_York")).date() == target_date


# ── Sources, in preference order ─────────────────────────────


async def _from_mlb_api(target_date: date) -> ScoreBook:
    """Primary source: carries real gamePks and the abbrevs predictions use."""
    from mlb.data.mlb_api import MLBApiClient

    client = MLBApiClient()
    try:
        games = await client.get_schedule(target_date)
    finally:
        await client.close()

    scores = [
        FinalScore(
            game_id=_norm_id(g.get("game_id")),
            home_team=g["home_team_id"],
            away_team=g["away_team_id"],
            home_score=int(g["home_score"]),
            away_score=int(g["away_score"]),
            source="mlb_api",
        )
        for g in games
        if g.get("status") in FINAL_STATUSES and g.get("home_score") is not None
    ]
    return ScoreBook(target_date, scores, "mlb_api")


def scorebook_from_csv(
    target_date: date, data_dir: Path = Path("data")
) -> ScoreBook:
    """Fallback: the committed games CSV. Also game_id-keyed, so doubleheader-safe.

    The CSV has real holes (the backfill only covers a rolling window), so it is
    never the primary source.
    """
    import pandas as pd

    csv_path = data_dir / f"games_{target_date.year}.csv"
    if not csv_path.exists():
        return ScoreBook(target_date, [], "games_csv")

    df = pd.read_csv(csv_path)
    day_games = df[
        (df["game_date"] == target_date.isoformat()) & (df["status"].isin(FINAL_STATUSES))
    ]

    scores = [
        FinalScore(
            game_id=_norm_id(row.get("game_id")),
            home_team=row["home_team_id"],
            away_team=row["away_team_id"],
            home_score=int(row["home_score"]),
            away_score=int(row["away_score"]),
            source="games_csv",
        )
        for _, row in day_games.iterrows()
    ]
    if scores:
        logger.info("Loaded %d scores from CSV for %s", len(scores), target_date)
    return ScoreBook(target_date, scores, "games_csv")


async def _from_odds_api(target_date: date) -> ScoreBook:
    """Last resort. Has no gamePk, so doubleheaders stay ambiguous — correctly:
    the Odds API genuinely cannot tell the legs apart."""
    from mlb.data.odds_api import OddsApiClient

    days_ago = (_today_et() - target_date).days
    client = OddsApiClient()
    raw = await client.get_scores(days_from=days_ago + 1)

    scores = [
        FinalScore(
            game_id="",
            home_team=s["home_team"],
            away_team=s["away_team"],
            home_score=int(s["home_score"]),
            away_score=int(s["away_score"]),
            source="odds_api",
        )
        for s in raw
        if score_on_date(s, target_date)
    ]
    return ScoreBook(target_date, scores, "odds_api")


async def load_scorebook(
    target_date: date,
    data_dir: Path = Path("data"),
    *,
    allow_odds_api: bool = True,
) -> ScoreBook:
    """Final scores for `target_date`, from the best source that has them.

    MLB API → committed games CSV → Odds API. The first two carry game_ids and so
    can separate doubleheader legs; the Odds API can't, and is only reached when
    the others come back empty.
    """
    try:
        book = await _from_mlb_api(target_date)
        if len(book):
            return book
        logger.info("MLB API returned no finals for %s", target_date)
    except Exception as e:
        logger.warning("MLB API scores failed for %s: %s", target_date, e)

    book = scorebook_from_csv(target_date, data_dir)
    if len(book):
        return book

    # The Odds API scores endpoint only reaches ~3 days back.
    if allow_odds_api and (_today_et() - target_date).days <= 3:
        try:
            book = await _from_odds_api(target_date)
            if len(book):
                return book
        except Exception as e:
            logger.debug("Odds API scores failed for %s: %s", target_date, e)

    logger.warning("No scores available for %s from any source", target_date)
    return ScoreBook(target_date, [], "none")
