"""Tests for the retrieval layer, dense-tracking adapter, and resolver.

These use dependencies installed for this project and are fully offline:
* the SkillCorner match already cached under data/retrieval/skillcorner/,
* the cached StatsBomb matches under data/statsbomb/.

No network calls are made by this test module.
"""

from __future__ import annotations

import os

import pytest

from src.analytics.adapters import DenseTracking, SkillCornerAdapter
from src.analytics.adapters import IDSSEAdapter, MetricaAdapter
from src.analytics.adapters.resolver import Resolver, build_resolver


def _skillcorner_dir() -> str:
    p = os.path.join(
        os.path.dirname(__file__), "..", "data", "retrieval", "skillcorner", "1886347")
    return os.path.normpath(p)


def _idsse_dir() -> str:
    p = os.path.join(os.path.dirname(__file__), "..", "data",
                     "retrieval", "idsse", "J03WMX")
    return os.path.normpath(p)


def _metrica_dir(game: str = "sample_game_1") -> str:
    p = os.path.join(os.path.dirname(__file__), "..", "data",
                     "retrieval", "metrica", game)
    return os.path.normpath(p)


def _openfootball_dir(season: str = "2025-26") -> str:
    p = os.path.join(os.path.dirname(__file__), "..", "data",
                     "retrieval", "openfootball", season, "en.1")
    return os.path.normpath(p)


@pytest.fixture
def skillcorner_dir() -> str:
    return _skillcorner_dir()


@pytest.fixture
def sc_adapter(skillcorner_dir):
    return SkillCornerAdapter(skillcorner_dir)


class TestSkillCornerAdapter:
    """Offline smoke test against the cached A-League match."""

    def test_is_dense(self, sc_adapter):
        assert isinstance(sc_adapter, DenseTracking)

    def test_match(self, sc_adapter):
        m = sc_adapter.match(1886347)
        assert m.home_team == "Auckland FC"
        assert m.away_team == "Newcastle"
        assert len(m.teams) == 2
        assert len(m.players) >= 22
        assert all(0 <= t.side <= 1 for t in m.teams)

    def test_tracking_continuous(self, sc_adapter):
        frames = sc_adapter.tracking(1886347)
        assert len(frames) > 10000  # 10 fps full match
        # sorted timestamps, start ~0
        ts = [f.timestamp_sec for f in frames]
        assert ts == sorted(ts)
        assert ts[0] >= 0.0
        # dense frames carry per-player identity + canonical coordinates
        populated = [f for f in frames if f.players]
        assert populated
        for pid, side, x, y, keeper in populated[0].players:
            assert pid is not None
            assert side in (0, 1)
            assert 0.0 <= x <= 120.0
            assert 0.0 <= y <= 70.0
            assert isinstance(keeper, bool)

    def test_ball_present(self, sc_adapter):
        frames = sc_adapter.tracking(1886347)
        ball = [f.ball for f in frames if f.ball]
        assert len(ball) > 1000
        x, y = ball[0]
        assert 0.0 <= x <= 120.0 and 0.0 <= y <= 70.0

    def test_events(self, sc_adapter):
        evs = sc_adapter.events(1886347)
        assert len(evs) > 1000
        passes = sc_adapter.passes(1886347)
        shots = sc_adapter.shots(1886347)
        assert len(passes) > 100
        assert len(shots) > 0
        for p in passes[:5]:
            assert p.team_side in (0, 1)
            assert 0.0 <= p.x <= 120.0
            assert 0.0 <= p.y <= 70.0
        for s in shots[:5]:
            assert s.result in ("goal", "saved", "off_target", "blocked", "post",
                                "missed")


class TestMetricaAdapter:
    """Offline smoke test against the cached Metrica sample game 1."""

    @pytest.fixture
    def ad(self):
        return MetricaAdapter(_metrica_dir(), sample_every=50)

    def test_is_dense(self, ad):
        assert isinstance(ad, DenseTracking)

    def test_match(self, ad):
        m = ad.match(1)
        assert m.home_team == "Home"
        assert m.away_team == "Away"
        assert len(m.teams) == 2
        assert len(m.players) >= 22
        assert sorted(t.side for t in m.teams) == [0, 1]
        # goalkeeper detection by shirt number
        keeper_positions = {p.shirt_number: p.position for p in m.players}
        assert keeper_positions.get(1) == "Goalkeeper"
        assert keeper_positions.get(21) == "Goalkeeper"

    def test_events(self, ad):
        evs = ad.events(1)
        assert len(evs) > 1000
        passes = ad.passes(1)
        shots = ad.shots(1)
        assert len(passes) > 100
        assert len(shots) > 0
        for p in passes[:5]:
            assert p.team_side in (0, 1)
            assert 0.0 <= p.x <= 120.0
            assert 0.0 <= p.y <= 70.0
            assert p.outcome in ("complete", "incomplete")
        for s in shots[:5]:
            assert s.result in ("goal", "saved", "off_target", "blocked", "post")
        xs = [e.x for e in evs]
        ys = [e.y for e in evs]
        assert min(xs) >= 0.0 and max(xs) <= 120.0
        assert min(ys) >= 0.0 and max(ys) <= 70.0

    def test_tracking_continuous(self, ad):
        frames = ad.tracking(1)
        assert len(frames) > 1000  # 25 Hz sampled at /50 -> 0.5 Hz full match
        ts = [f.timestamp_sec for f in frames]
        assert ts == sorted(ts)
        assert ts[0] >= 0.0
        populated = [f for f in frames if f.players]
        assert populated
        for pid, side, x, y, keeper in populated[0].players:
            assert pid is not None
            assert side in (0, 1)
            assert 0.0 <= x <= 120.0
            assert 0.0 <= y <= 70.0
            assert isinstance(keeper, bool)

    def test_ball_present(self, ad):
        frames = ad.tracking(1)
        ball = [f.ball for f in frames if f.ball]
        assert len(ball) > 500
        x, y = ball[0]
        assert 0.0 <= x <= 120.0 and 0.0 <= y <= 70.0


class TestOpenFootballAdapter:
    """Offline test against the cached CC0 OpenFootball 2025-26 season."""

    @pytest.fixture
    def ad(self):
        from src.analytics.adapters.openfootball import OpenFootballAdapter
        return OpenFootballAdapter(_openfootball_dir("2025-26"))

    @pytest.fixture
    def ad24(self):
        from src.analytics.adapters.openfootball import OpenFootballAdapter
        return OpenFootballAdapter(_openfootball_dir("2024-25"))

    def test_is_lineups_and_events(self, ad):
        from src.analytics.adapters.base import EventsProvider, LineupsProvider
        assert isinstance(ad, LineupsProvider)
        assert isinstance(ad, EventsProvider)

    def test_match_metadata(self, ad):
        m = ad.match(0)
        assert m.home_team == "Liverpool FC"
        assert m.away_team == "AFC Bournemouth"
        assert m.season == "2025-26"
        assert [t.side for t in m.teams] == [0, 1]

    def test_goal_events_with_scorers(self, ad):
        goals = ad.events(0)
        assert len(goals) == 6  # Liverpool 4-2 Bournemouth
        for e in goals:
            assert e.type == "Goal"
            assert e.qualifiers["event_type"] in ("goal", "penalty", "own_goal")
        scorers = {e.qualifiers["scorer"] for e in goals}
        assert "Hugo EKITIKE" in scorers or any(
            "EKITIKE" in s for s in scorers)
        assert all(0.0 <= e.x <= 120.0 and 0.0 <= e.y <= 70.0 for e in goals)

    def test_passes_shots_empty(self, ad):
        assert ad.passes(0) == []
        assert ad.shots(0) == []

    def test_scoreline_fallback_24(self, ad24):
        m = ad24.match(0)
        g = ad24.events(0)
        assert m.home_team == "Manchester United FC"
        # 1-0 -> exactly one home goal (scorer unknown, but stream non-empty)
        assert len(g) == 1
        assert g[0].team_side == 0


class TestResolver:
    def test_cross_source_resolution(self, skillcorner_dir):
        """StatsBomb serves events/lineups; SkillCorner serves dense tracking."""
        rp = build_resolver(SkillCornerAdapter(skillcorner_dir)).resolve(
            3869685, needs={"events", "dense_tracking", "lineups"})
        assert rp.sources["events"] == "StatsBombAdapter"
        assert rp.sources["dense_tracking"] == "SkillCornerAdapter"
        assert rp.has("events")
        assert rp.has("dense_tracking")
        assert not rp.missing({"events", "dense_tracking"})

    def test_statsbomb_only_missing_dense(self):
        rp = build_resolver().resolve(3869685, needs={"events", "dense_tracking"})
        assert rp.has("events")
        assert not rp.has("dense_tracking")
        assert rp.missing({"dense_tracking"}) == {"dense_tracking"}

    def test_missing_reports(self, skillcorner_dir):
        from src.analytics.adapters import StatsBombAdapter
        rp = Resolver([StatsBombAdapter()]).resolve(
            3869685, needs={"events", "dense_tracking"})
        assert "dense_tracking" in rp.missing({"events", "dense_tracking"})

    def test_cross_source_metrica_integration(self):
        """End-to-end: StatsBomb events/lineups + Metrica dense tracking.

        Resolve a StatsBomb match id through the merged provider: StatsBomb
        serves events/lineups for match 3869685 (cached offline), Metrica
        supplies the dense-tracking frames for its sample game. The schema-
        driven models then consume `ResolvedProvider.tracking()`/`.events()`
        without knowing which source produced them.
        """
        mid = 3869685
        rp = build_resolver(MetricaAdapter(_metrica_dir(), sample_every=50)) \
            .resolve(mid, needs={"events", "lineups", "dense_tracking"})
        assert rp.sources["events"] == "StatsBombAdapter"
        assert rp.sources["dense_tracking"] == "MetricaAdapter"
        assert rp.sources["lineups"] == "StatsBombAdapter"

        # models over resolved dense tracking (provider-agnostic consumption)
        from src.analytics.models.flow import flow_field, team_formation
        frames = rp.tracking(mid)
        assert frames and frames[0].timestamp_sec >= 0.0
        fm = team_formation(frames, team_side=0)
        assert fm.n_samples > 0 and fm.team_side == 0
        ff = flow_field(frames, team_side=1)
        assert ff.team_side == 1

        from src.analytics.models.attacking import progressive_summary, shot_map
        passes = rp.passes(mid)
        shots = rp.shots(mid)
        assert passes
        ps = progressive_summary(passes)
        assert ps.total_completed.get(0, 0) + ps.total_completed.get(1, 0) > 0
        sm = shot_map(shots)
        assert sum(z.shots for z in sm.zones) == len(shots)

    def test_cross_source_skillcorner_metrics(self, skillcorner_dir):
        """SkillCorner dense tracking + StatsBomb events resolved together."""
        rp = build_resolver(SkillCornerAdapter(skillcorner_dir)).resolve(
            3869685, needs={"events", "dense_tracking"})
        frames = rp.tracking(3869685)
        from src.analytics.models.flow import team_formation
        assert team_formation(frames, team_side=0).n_samples > 0


class TestRetrievalRegistry:
    def test_registered_sources(self):
        from src.analytics.retrieval import available, get
        names = available()
        assert "statsbomb" in names
        assert "skillcorner" in names
        assert "http_file" in names
        assert "metrica" in names
        assert "openfootball" in names
        assert all(isinstance(n, str) for n in names)
        for n in ("statsbomb", "skillcorner", "http_file", "metrica",
                  "openfootball"):
            get(n)  # constructors resolve without network

    def test_statsbomb_retriever_cached(self):
        from src.analytics.retrieval import get
        r = get("statsbomb")
        ids = r.list_resources()
        assert len(ids) > 100


class TestWyscoutAdapter:
    """Offline test against the cached Huddersfield v Man City match."""

    @pytest.fixture
    def ad(self):
        p = os.path.join(os.path.dirname(__file__), "..", "data",
                         "retrieval", "wyscout", "matches", "2499841.json")
        from src.analytics.adapters import WyscoutAdapter
        return WyscoutAdapter(os.path.normpath(p))

    def test_match(self, ad):
        m = ad.match(2499841)
        assert m.home_team == "Huddersfield Town"
        assert m.away_team == "Manchester City"
        assert len(m.teams) == 2
        assert len(m.players) >= 22

    def test_events(self, ad):
        evs = ad.events(2499841)
        assert len(evs) > 1000
        passes = ad.passes(2499841)
        shots = ad.shots(2499841)
        assert len(passes) > 100
        assert len(shots) > 0
        for p in passes[:5]:
            assert p.team_side in (0, 1)
            assert 0.0 <= p.x <= 120.0
            assert 0.0 <= p.y <= 70.0
            assert p.outcome in ("complete", "incomplete")
        xs = [e.x for e in evs]
        ys = [e.y for e in evs]
        assert min(xs) >= 0.0 and max(xs) <= 120.0
        assert min(ys) >= 0.0 and max(ys) <= 70.0

    def test_wyscout_retriever_constructor(self):
        from src.analytics.retrieval import get
        get("wyscout")  # constructs without network


class TestIDSSEAdapter:
    """Offline smoke test against the cached Köln v Bayern match."""

    @pytest.fixture
    def ad(self):
        return IDSSEAdapter(_idsse_dir(), sample_every=25)

    def test_is_dense(self, ad):
        assert isinstance(ad, DenseTracking)

    def test_match(self, ad):
        m = ad.match(0)
        assert m.home_team == "1. FC Köln"
        assert m.away_team == "FC Bayern München"
        assert len(m.teams) == 2
        assert len(m.players) >= 22
        assert sorted(t.side for t in m.teams) == [0, 1]

    def test_events(self, ad):
        evs = ad.events(0)
        assert len(evs) > 1000
        passes = ad.passes(0)
        shots = ad.shots(0)
        assert len(passes) > 100
        assert len(shots) > 0
        # Köln 1:2 Bayern -> exactly 3 goals this match
        assert sum(1 for s in shots if s.result == "goal") == 3
        for p in passes[:5]:
            assert p.team_side in (0, 1)
            assert 0.0 <= p.x <= 120.0
            assert 0.0 <= p.y <= 70.0
            assert p.outcome in ("complete", "incomplete")
        xs = [e.x for e in evs]
        ys = [e.y for e in evs]
        assert min(xs) >= 0.0 and max(xs) <= 120.0
        assert min(ys) >= 0.0 and max(ys) <= 70.0

    def test_tracking_continuous(self, ad):
        frames = ad.tracking(0)
        assert len(frames) > 1000  # 25 Hz sampled at /25 -> ~1 Hz full match
        ts = [f.timestamp_sec for f in frames]
        assert ts == sorted(ts)
        assert ts[0] >= 0.0
        populated = [f for f in frames if f.players]
        assert populated
        # dense frames carry per-player identity + canonical coordinates
        for pid, side, x, y, keeper in populated[0].players:
            assert pid is not None
            assert side in (0, 1)
            assert 0.0 <= x <= 120.0
            assert 0.0 <= y <= 70.0
            assert isinstance(keeper, bool)

    def test_ball_present(self, ad):
        frames = ad.tracking(0)
        ball = [f.ball for f in frames if f.ball]
        assert len(ball) > 500
        x, y = ball[0]
        assert 0.0 <= x <= 120.0 and 0.0 <= y <= 70.0


class TestSoccerDataRetriever:
    """Source-capability correctness for the soccerdata-backed scrapers."""

    def test_source_registry_dropped_football_data(self):
        from src.analytics.retrieval import available
        names = available()
        assert "football-data" not in names
        assert "fbref" in names
        assert "espn" in names
        assert "sofascore" in names

    def test_corrected_class_and_league_keys(self):
        from src.analytics.retrieval.soccerdata import (
            SOCCERDATA_SOURCES,
            _SOURCE_LEAGUES,
        )
        assert "sofascore" in SOCCERDATA_SOURCES
        assert _SOURCE_LEAGUES["sofascore"] == "ENG-Premier League"
        assert _SOURCE_LEAGUES["espn"] == "ENG-Premier League"  # not ENG.1

    def test_available_resources(self):
        from src.analytics.retrieval import get
        r = get("fbref")
        res = r.available_resources()
        assert "schedule" in res
        assert "player_season" in res  # fbref supports it
        # match-level-only sources have no player-season
        assert "player_season" not in get("espn").available_resources()
        assert "player_season" not in get("sofascore").available_resources()

    def test_fetch_unknown_resource_reports(self):
        from src.analytics.retrieval import get
        from src.analytics.retrieval.base import RetrievalError
        with pytest.raises(RetrievalError):
            get("fbref").fetch("no_such_resource")


class TestHTTPFileRetriever:
    def test_lfs_resolution(self):
        from src.analytics.retrieval.http import HTTPFileRetriever
        h = HTTPFileRetriever()
        ptr = (
            b"version https://git-lfs.github.com/spec/v1\n"
            b"oid sha256:abc123\n"
            b"size 12345\n"
        )
        assert h._is_lfs_pointer(ptr)
        url = ("https://raw.githubusercontent.com/SkillCorner/opendata/"
               "master/data/matches/1886347/1886347_tracking_extrapolated.jsonl")
        media = h._lfs_media_url(url, ptr)
        assert "oid=abc123" in media
        assert "/1886347/1886347_tracking_extrapolated.jsonl" in media
        # repo-relative path, not doubled
        assert media.count("SkillCorner") == 1

    def test_not_pointer(self):
        from src.analytics.retrieval.http import HTTPFileRetriever
        assert HTTPFileRetriever()._is_lfs_pointer(b'{"frame": 0}') is False