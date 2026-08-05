"""Tying Odds API events to MLB games — especially doubleheader legs."""

from mlb.data.odds_match import index_odds_by_game, resolve_odds_game_ids


def _game(game_id, home, away, time):
    return {
        "game_id": game_id, "home_team_id": home,
        "away_team_id": away, "game_time": time,
    }


def _event(event_id, home, away, time, home_ml=-140):
    return {
        "odds_event_id": event_id, "home_team": home, "away_team": away,
        "commence_time": time, "home_moneyline": home_ml, "away_moneyline": 120,
        "total_line": 8.5,
    }


class TestSingleGames:
    def test_matches_one_event_to_one_game(self):
        games = [_game("824700", "NYY", "TOR", "2026-07-22T23:05:00Z")]
        events = [_event("evt1", "NYY", "TOR", "2026-07-22T23:05:00Z")]

        assert resolve_odds_game_ids(events, games) == {"evt1": "824700"}

    def test_a_single_game_matches_even_without_times(self):
        # Only doubleheaders need the clock; don't make a clean day depend on it.
        games = [_game("824700", "NYY", "TOR", None)]
        events = [_event("evt1", "NYY", "TOR", None)]

        assert resolve_odds_game_ids(events, games) == {"evt1": "824700"}

    def test_matchup_not_on_the_schedule_is_dropped(self):
        games = [_game("824700", "NYY", "TOR", "2026-07-22T23:05:00Z")]
        events = [_event("evt1", "SEA", "LAA", "2026-07-22T23:05:00Z")]

        assert resolve_odds_game_ids(events, games) == {}

    def test_tomorrows_event_does_not_price_todays_game(self):
        """The feed spans several dates and drops a game once it starts, so a
        matchup can be left holding only the next day's event."""
        games = [_game("824700", "NYY", "STL", "2026-07-22T23:05:00Z")]
        events = [_event("evt_tomorrow", "NYY", "STL", "2026-07-23T23:06:00Z")]

        assert resolve_odds_game_ids(events, games) == {}

    def test_a_minute_of_feed_skew_is_tolerated(self):
        # The Odds API routinely posts first pitch a minute off the schedule.
        games = [_game("824700", "NYY", "STL", "2026-07-22T23:05:00Z")]
        events = [_event("evt1", "NYY", "STL", "2026-07-22T23:06:00Z")]

        assert resolve_odds_game_ids(events, games) == {"evt1": "824700"}

    def test_reversed_matchup_does_not_match(self):
        games = [_game("824700", "NYY", "TOR", "2026-07-22T23:05:00Z")]
        events = [_event("evt1", "TOR", "NYY", "2026-07-22T23:05:00Z")]

        assert resolve_odds_game_ids(events, games) == {}


class TestDoubleheaders:
    """2026-07-22 really had BOS/BAL and NYY/PIT twin bills."""

    GAMES = [
        _game("824735", "BOS", "BAL", "2026-07-22T17:10:00Z"),   # game 1
        _game("824732", "BOS", "BAL", "2026-07-22T23:10:00Z"),   # game 2
    ]

    def test_each_leg_gets_its_own_event(self):
        events = [
            _event("evt2", "BOS", "BAL", "2026-07-22T23:10:00Z", home_ml=105),
            _event("evt1", "BOS", "BAL", "2026-07-22T17:10:00Z", home_ml=-140),
        ]

        assert resolve_odds_game_ids(events, self.GAMES) == {
            "evt1": "824735", "evt2": "824732",
        }

    def test_each_leg_keeps_its_own_line(self):
        events = [
            _event("evt1", "BOS", "BAL", "2026-07-22T17:10:00Z", home_ml=-140),
            _event("evt2", "BOS", "BAL", "2026-07-22T23:10:00Z", home_ml=105),
        ]

        by_game = index_odds_by_game(events, self.GAMES)

        # Before this fix both legs took the last event's line, and one leg was
        # priced twice while the other was never priced at all.
        assert by_game["824735"]["home_moneyline"] == -140
        assert by_game["824732"]["home_moneyline"] == 105

    def test_small_clock_skew_still_matches(self):
        events = [
            _event("evt1", "BOS", "BAL", "2026-07-22T17:05:00Z"),
            _event("evt2", "BOS", "BAL", "2026-07-22T23:15:00Z"),
        ]

        assert resolve_odds_game_ids(events, self.GAMES) == {
            "evt1": "824735", "evt2": "824732",
        }

    def test_one_event_for_two_legs_takes_the_nearer_leg(self):
        events = [_event("evt2", "BOS", "BAL", "2026-07-22T23:10:00Z")]

        assert resolve_odds_game_ids(events, self.GAMES) == {"evt2": "824732"}

    def test_an_event_far_from_both_legs_is_left_unresolved(self):
        events = [_event("evt9", "BOS", "BAL", "2026-07-23T23:10:00Z")]

        assert resolve_odds_game_ids(events, self.GAMES) == {}

    def test_a_leg_is_never_assigned_twice(self):
        # Two events clustered on game 1: the second must not steal the same leg.
        events = [
            _event("evt1", "BOS", "BAL", "2026-07-22T17:10:00Z"),
            _event("evt1b", "BOS", "BAL", "2026-07-22T17:12:00Z"),
        ]

        resolved = resolve_odds_game_ids(events, self.GAMES)

        assert len(set(resolved.values())) == len(resolved)

    def test_missing_times_leave_the_pair_unresolved_rather_than_guessing(self):
        events = [
            _event("evt1", "BOS", "BAL", None),
            _event("evt2", "BOS", "BAL", None),
        ]

        assert resolve_odds_game_ids(events, self.GAMES) == {}


def test_mixed_slate_resolves_everything_it_can():
    games = [
        _game("824735", "BOS", "BAL", "2026-07-22T17:10:00Z"),
        _game("824732", "BOS", "BAL", "2026-07-22T23:10:00Z"),
        _game("824700", "NYY", "TOR", "2026-07-22T23:05:00Z"),
    ]
    events = [
        _event("evt1", "BOS", "BAL", "2026-07-22T17:10:00Z"),
        _event("evt2", "BOS", "BAL", "2026-07-22T23:10:00Z"),
        _event("evt3", "NYY", "TOR", "2026-07-22T23:05:00Z"),
        _event("evt4", "SEA", "LAA", "2026-07-22T02:10:00Z"),   # not scheduled
    ]

    resolved = resolve_odds_game_ids(events, games)

    assert resolved == {"evt1": "824735", "evt2": "824732", "evt3": "824700"}
