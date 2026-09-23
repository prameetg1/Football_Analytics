"""Offline tests for the soccerdata schedule adapter (cached tables, no network)."""

import os

import pytest

from src.analytics.adapters import SoccerDataAdapter


def _data(*parts):
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..",
                                         "data", *parts))


@pytest.fixture
def understat():
    return SoccerDataAdapter("understat", _data("retrieval", "understat",
                                                "2021", "schedule.json"))


@pytest.fixture
def fbref():
    return SoccerDataAdapter("fbref", _data("retrieval", "fbref",
                                            "2021", "schedule.json"))


class TestSoccerDataAdapter:
    def test_is_lineups_and_events(self, understat):
        from src.analytics.adapters.base import EventsProvider, LineupsProvider
        assert isinstance(understat, LineupsProvider)
        assert isinstance(understat, EventsProvider)

    def test_list_matches_epi(self, understat, fbref):
        assert len(understat.list_matches()) == 380
        assert len(fbref.list_matches()) == 380

    def test_match(self, understat):
        gid = understat.list_matches()[0]
        m = understat.match(gid)
        assert m.home_team and m.away_team
        assert len(m.teams) == 2
        assert sorted(t.side for t in m.teams) == [0, 1]
        assert all(t.name for t in m.teams)

    def test_events_scoreline(self, fbref):
        # fbref has no shots.json -> events are synthesized scoreline goals
        fgid = fbref.list_matches()[0]
        evs = fbref.events(fgid)
        assert all(e.type == "Goal" for e in evs)
        assert all(e.team_side in (0, 1) for e in evs)
        assert 0 <= len(evs) <= 20

    def test_sofascore_score_columns(self):
        ad = SoccerDataAdapter("sofascore", _data("retrieval", "sofascore",
                                                  "2021", "schedule.json"))
        gid = ad.list_matches()[0]
        m = ad.match(gid)
        n = len(ad.events(gid))
        assert 0 <= n <= 20
        assert m.home_team and m.away_team

    def test_understat_shots(self, understat):
        gid = understat.list_matches()[0]
        shots = understat.shots(gid)
        assert len(shots) > 0
        for s in shots:
            assert s.type == "Shot"
            assert s.result in ("goal", "saved", "off_target", "blocked",
                                "post")
            assert s.team_side in (0, 1)
            assert 0.0 <= s.x <= 120.0
            assert 0.0 <= s.y <= 70.0
        # a goal must exist in the opened Fulham v Arsenal 0-3 game
        assert any(s.result == "goal" for s in shots)
        assert shots == sorted(shots, key=lambda s: s.timestamp_sec)

    def test_understat_events_falls_back_to_shots(self, understat):
        gid = understat.list_matches()[0]
        assert len(understat.events(gid)) == len(understat.shots(gid))

    def test_season_awareness(self):
        # 2021 dir -> 2021-22; 2025 dir -> 2025-26 (full completed season)
        for d, season, expect_shots in (("2021", "2021-22", "Fulham"),
                                        ("2025", "2025-26", "Liverpool")):
            ad = SoccerDataAdapter(
                "understat", _data("retrieval", "understat", d,
                                   "schedule.json"), season=season)
            gid = ad.list_matches()[0]
            m = ad.match(gid)
            assert m.season == season
            assert m.home_team == expect_shots  # league opener home side
            assert len(ad.shots(gid)) > 0