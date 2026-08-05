"""Tie Odds API events to MLB games.

The Odds API has no MLB gamePk — only its own event id and the two team names.
Everything downstream therefore keyed odds on ``(home_team, away_team)``, which
is not unique on a date. On a doubleheader that silently went wrong in three
places at once: both legs took the second event's line as a model feature, one
leg got priced (and bet) twice while the other was never priced at all, and the
closing line used for CLV belonged to whichever leg was written last.

The one field both sides share is first pitch. `resolve_odds_game_ids` matches on
the team pair and, when a pair covers more than one game, on start time — legs
are hours apart, so the ordering is unambiguous. Events it cannot place stay
unresolved rather than being attached to a guess.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

# Slack between an event's advertised start and the MLB schedule's. The odds feed
# runs a minute or so off. It has to stay well under the gap between doubleheader
# legs (~3.5h) and nowhere near a day, because the feed covers several dates at
# once and a finished game drops out of it — leaving only tomorrow's event for
# that matchup.
DEFAULT_TOLERANCE_MINUTES = 120


def parse_dt(value: Any) -> datetime | None:
    """Parse an ISO timestamp (the feeds use a trailing Z), or None if unusable."""
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def resolve_odds_game_ids(
    odds_data: Sequence[dict],
    games: Iterable[dict],
    *,
    tolerance_minutes: int = DEFAULT_TOLERANCE_MINUTES,
) -> dict[str, str]:
    """Map each odds event id to the MLB game_id it belongs to.

    `games` are MLB schedule dicts (`game_id`, `home_team_id`, `away_team_id`,
    `game_time`); `odds_data` are Odds API dicts (`odds_event_id`, `home_team`,
    `away_team`, `commence_time`).

    Events whose matchup isn't on the schedule, and doubleheader legs that can't
    be told apart on time, are simply absent from the result.
    """
    games_by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for g in games:
        games_by_pair[(g.get("home_team_id"), g.get("away_team_id"))].append(g)

    events_by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for o in odds_data:
        events_by_pair[(o.get("home_team"), o.get("away_team"))].append(o)

    resolved: dict[str, str] = {}
    for pair, events in events_by_pair.items():
        candidates = games_by_pair.get(pair, [])
        if not candidates:
            continue

        if len(candidates) == 1 and len(events) == 1:
            resolved.update(_resolve_one(pair, events[0], candidates[0], tolerance_minutes))
            continue

        resolved.update(_resolve_by_time(pair, events, candidates, tolerance_minutes))

    return resolved


def _resolve_one(
    pair: tuple[str, str], event: dict, game: dict, tolerance_minutes: int
) -> dict[str, str]:
    """One game, one event — still check the clock when both sides give a time.

    The odds feed spans several dates and drops a game once it starts, so a
    matchup can be left with only *tomorrow's* event. Matching on the pair alone
    would price today's game off tomorrow's line.
    """
    event_id = event.get("odds_event_id")
    if not event_id:
        return {}

    event_dt = parse_dt(event.get("commence_time"))
    game_dt = parse_dt(game.get("game_time"))
    if event_dt and game_dt:
        delta = abs((game_dt - event_dt).total_seconds())
        if delta > tolerance_minutes * 60:
            logger.debug(
                "Odds event %s (%s@%s, %s) is %.1fh from the scheduled game — "
                "different date, skipping",
                event_id, pair[1], pair[0], event.get("commence_time"), delta / 3600,
            )
            return {}

    return {event_id: game["game_id"]}


def _resolve_by_time(
    pair: tuple[str, str],
    events: list[dict],
    candidates: list[dict],
    tolerance_minutes: int,
) -> dict[str, str]:
    """Separate same-matchup games by first pitch."""
    timed_events = [
        (dt, e) for e in events
        if (dt := parse_dt(e.get("commence_time"))) is not None
        and e.get("odds_event_id")
    ]
    timed_games = [
        (dt, g) for g in candidates
        if (dt := parse_dt(g.get("game_time"))) is not None
    ]

    if not timed_events or not timed_games:
        # Only alarming when there is actually something to tell apart.
        log = logger.warning if len(candidates) > 1 else logger.debug
        log(
            "Cannot separate %s@%s (%d event(s), %d game(s)) — no start times, "
            "leaving unresolved", pair[1], pair[0], len(events), len(candidates),
        )
        return {}

    timed_events.sort(key=lambda x: x[0])
    timed_games.sort(key=lambda x: x[0])

    resolved: dict[str, str] = {}
    taken: set[str] = set()
    tolerance = tolerance_minutes * 60

    for event_dt, event in timed_events:
        best = None
        for game_dt, game in timed_games:
            if game["game_id"] in taken:
                continue
            delta = abs((game_dt - event_dt).total_seconds())
            if delta <= tolerance and (best is None or delta < best[0]):
                best = (delta, game)
        if best is None:
            # The feed covers several dates, so most misses are simply another
            # day's game — routine. A doubleheader leg going unplaced is not.
            log = logger.warning if len(candidates) > 1 else logger.debug
            log(
                "Odds event %s (%s@%s, %s) matches no game within %dm",
                event.get("odds_event_id"), pair[1], pair[0],
                event.get("commence_time"), tolerance_minutes,
            )
            continue
        resolved[event["odds_event_id"]] = best[1]["game_id"]
        taken.add(best[1]["game_id"])

    if len(candidates) > 1:
        logger.info(
            "Doubleheader %s@%s: matched %d of %d event(s) to legs by start time",
            pair[1], pair[0], len(resolved), len(events),
        )
    return resolved


def index_odds_by_game(
    odds_data: Sequence[dict],
    games: Iterable[dict],
    *,
    tolerance_minutes: int = DEFAULT_TOLERANCE_MINUTES,
) -> dict[str, dict]:
    """Odds keyed by MLB game_id — the lookup every caller actually wants."""
    games = list(games)
    resolved = resolve_odds_game_ids(
        odds_data, games, tolerance_minutes=tolerance_minutes
    )
    by_game: dict[str, dict] = {}
    for o in odds_data:
        game_id = resolved.get(o.get("odds_event_id"))
        if game_id:
            by_game[game_id] = o
    return by_game
