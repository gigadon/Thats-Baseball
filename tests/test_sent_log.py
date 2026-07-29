"""Tests for wave-send windowing, de-dup, and preview/wave card shaping."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

from mlb.etl.sent_log import (
    append_sent,
    due_game_ids,
    load_sent,
    parse_game_time,
)

ET = ZoneInfo("America/New_York")


class TestParseGameTime:
    def test_zulu_and_offset(self):
        assert parse_game_time("2026-07-21T23:41:00Z") is not None
        assert parse_game_time("2026-07-21T19:41:00-04:00") is not None

    def test_missing_or_bad(self):
        assert parse_game_time("") is None
        assert parse_game_time(None) is None
        assert parse_game_time("not-a-date") is None
        assert parse_game_time(12345) is None


class TestDueGameIds:
    def _preds(self):
        return [
            {"game_id": "in", "game_time": "2026-07-21T23:05:00Z"},   # 7:05 PM ET
            {"game_id": "far", "game_time": "2026-07-22T03:00:00Z"},  # 11:00 PM ET
            {"game_id": "past", "game_time": "2026-07-21T20:00:00Z"}, # 4:00 PM ET
            {"game_id": "notime", "game_time": ""},
        ]

    def test_window_selects_only_upcoming_within_lead(self):
        now = datetime(2026, 7, 21, 18, 0, tzinfo=ET)  # 6 PM ET
        due = due_game_ids(self._preds(), now, lead_hours=4)
        assert "in" in due        # 7:05 PM is within +4h
        assert "far" not in due   # 11 PM is beyond +4h
        assert "past" not in due  # 4 PM already started

    def test_missing_time_is_safety_net(self):
        now = datetime(2026, 7, 21, 18, 0, tzinfo=ET)
        assert "notime" in due_game_ids(self._preds(), now, lead_hours=4)

    def test_wider_lead_pulls_in_later_games(self):
        now = datetime(2026, 7, 21, 18, 0, tzinfo=ET)
        due = due_game_ids(self._preds(), now, lead_hours=6)  # horizon = midnight ET
        assert {"in", "far", "notime"} <= due


class TestSentLog:
    def test_empty_then_append_and_dedup(self, tmp_path):
        d = date(2026, 7, 21)
        assert load_sent(d, tmp_path) == []
        append_sent(d, ["1", "2"], tmp_path)
        assert load_sent(d, tmp_path) == ["1", "2"]
        # Re-appending an existing id plus a new one de-dups, preserves order.
        append_sent(d, ["2", "3"], tmp_path)
        assert load_sent(d, tmp_path) == ["1", "2", "3"]

    def test_ids_coerced_to_str(self, tmp_path):
        d = date(2026, 7, 21)
        append_sent(d, [1, 2], tmp_path)
        assert load_sent(d, tmp_path) == ["1", "2"]


# ── Card shaping (imports the heavy DailyRunner lazily) ──────────────


class _FakeAlert:
    def __init__(self):
        self.sent = []

    async def send_betting_alert(self, result):
        self.sent.append(result)


def _runner(tmp_path):
    from mlb.etl.daily_runner import DailyRunner

    r = DailyRunner.__new__(DailyRunner)  # skip __init__ (no clients/models needed)
    r.data_dir = tmp_path
    r.alert_service = _FakeAlert()
    return r


def _fix_now(monkeypatch, dt):
    from mlb.etl import daily_runner as dr

    class _FakeDT:
        @staticmethod
        def now(tz=None):
            return dt.astimezone(tz) if tz else dt

    monkeypatch.setattr(dr, "datetime", _FakeDT)


def _slip():
    return {
        "slip_date": "2026-07-21", "bankroll": 10000.0, "risk_level": "moderate",
        "max_exposure": 0.0, "num_bets": 2, "total_stake": 150.0, "total_ev": 20.0,
        "bets": [
            {"game_id": "in", "recommended_stake": 100.0, "ev_per_dollar": 0.10},
            {"game_id": "far", "recommended_stake": 50.0, "ev_per_dollar": 0.20},
        ],
    }


def _result():
    return {
        "date": "2026-07-21",
        "predictions": [
            {"game_id": "in", "game_time": "2026-07-21T23:05:00Z"},
            {"game_id": "far", "game_time": "2026-07-22T03:00:00Z"},
        ],
        "betting_slip": _slip(),
        "review": {"yesterday": "2026-07-20", "full": None},
    }


async def test_wave_sends_due_games_only_and_no_bets(tmp_path, monkeypatch):
    _fix_now(monkeypatch, datetime(2026, 7, 21, 18, 0, tzinfo=ET))  # 6 PM ET
    r = _runner(tmp_path)

    await r._send_cards(date(2026, 7, 21), _result(), "wave", 4)
    assert len(r.alert_service.sent) == 1
    sent = r.alert_service.sent[0]
    assert sent["kind"] == "wave"
    assert [p["game_id"] for p in sent["predictions"]] == ["in"]     # only due game
    assert "betting_slip" not in sent      # bets ride the main card only
    assert "review" not in sent            # waves omit the review
    assert load_sent(date(2026, 7, 21), tmp_path) == ["in"]

    # A second wave at the same time re-sends nothing (already recorded).
    await r._send_cards(date(2026, 7, 21), _result(), "wave", 4)
    assert len(r.alert_service.sent) == 1


async def test_preview_sends_full_slate_and_whole_card(tmp_path, monkeypatch):
    _fix_now(monkeypatch, datetime(2026, 7, 21, 8, 0, tzinfo=ET))  # 8 AM: nothing due yet
    r = _runner(tmp_path)

    await r._send_cards(date(2026, 7, 21), _result(), "preview", 4)
    assert len(r.alert_service.sent) == 1
    sent = r.alert_service.sent[0]
    assert sent["kind"] == "preview"
    assert {p["game_id"] for p in sent["predictions"]} == {"in", "far"}  # full slate
    assert sent["review"] == {"yesterday": "2026-07-20", "full": None}
    # Every bet goes out here, including the late game outside the 4h window.
    assert [b["game_id"] for b in sent["betting_slip"]["bets"]] == ["in", "far"]
    assert load_sent(date(2026, 7, 21), tmp_path) == []                  # nothing due yet


async def test_preview_marks_games_already_inside_the_window(tmp_path, monkeypatch):
    """A game due at main-card time isn't re-sent as a wave reminder later."""
    _fix_now(monkeypatch, datetime(2026, 7, 21, 18, 0, tzinfo=ET))  # 6 PM ET
    r = _runner(tmp_path)

    await r._send_cards(date(2026, 7, 21), _result(), "preview", 4)
    assert load_sent(date(2026, 7, 21), tmp_path) == ["in"]

    await r._send_cards(date(2026, 7, 21), _result(), "wave", 4)
    assert len(r.alert_service.sent) == 1  # nothing new to remind about
