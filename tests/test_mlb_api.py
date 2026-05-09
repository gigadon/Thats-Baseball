"""Tests for the MLB API client helpers."""

import pytest

from mlb.data.mlb_api import TEAM_ABBREVS, TEAM_IDS, _int, _ip


class TestTeamMappings:
    def test_30_teams(self):
        assert len(TEAM_ABBREVS) == 30

    def test_reverse_lookup(self):
        for api_id, abbrev in TEAM_ABBREVS.items():
            assert TEAM_IDS[abbrev] == api_id

    def test_known_teams(self):
        assert TEAM_ABBREVS[147] == "NYY"
        assert TEAM_ABBREVS[111] == "BOS"
        assert TEAM_ABBREVS[119] == "LAD"
        assert TEAM_IDS["NYY"] == 147


class TestIntHelper:
    def test_normal_int(self):
        assert _int(5) == 5

    def test_string_int(self):
        assert _int("3") == 3

    def test_none(self):
        assert _int(None) is None

    def test_invalid(self):
        assert _int("abc") is None


class TestIPHelper:
    def test_whole_innings(self):
        assert _ip("6") == 6.0

    def test_one_third(self):
        assert _ip("6.1") == pytest.approx(6.333, abs=0.01)

    def test_two_thirds(self):
        assert _ip("6.2") == pytest.approx(6.667, abs=0.01)

    def test_zero(self):
        assert _ip("0") == 0.0

    def test_none(self):
        assert _ip(None) is None
