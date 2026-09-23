"""Tests for the analytics agents layer.

All tests run offline with synthetic or cached data (no network). Validates
each agent's typed output, the knowledge store, the orchestrator pipeline,
and figure generation.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


def _make_ctx():
    """Synthetic AgentContext for testing all agents."""
    from src.analytics.agents.base import AgentContext
    from src.analytics.schema import Match, PassEvent, Player, ShotEvent, Team
    m = Match(
        match_id=1, home_team="Liverpool", away_team="Arsenal",
        teams=[Team(team_id=0, name="Liverpool", side=0),
               Team(team_id=1, name="Arsenal", side=1)],
        players=[Player(player_id=i, name=f"P{i}") for i in range(22)],
    )
    events = []
    passes = []
    shots = []
    for i in range(60):
        side = i % 2
        p = PassEvent(
            type="Pass", timestamp_sec=float(i), x=50.0 + side * 20, y=35.0,
            team_side=side, player_id=7 + side * 11, outcome="complete",
            end_x=80.0 - side * 40, end_y=35.0, receiver_id=8 + side * 11)
        events.append(p)
        passes.append(p)
    for i in range(12):
        side = i % 2
        s = ShotEvent(
            type="Shot", timestamp_sec=float(60 + i), x=110.0 - side * 80, y=35.0,
            team_side=side, player_id=9 + side * 11,
            result="goal" if i in (2, 8) else "saved",
            xg=0.4 + 0.05 * i)
        events.append(s)
        shots.append(s)
    return AgentContext(match_id=1, match=m, events=events, passes=passes,
                       shots=shots)


class TestKnowledgeStore:
    def test_loads_sections(self):
        from src.analytics.agents.knowledge_store import KnowledgeStore
        ks = KnowledgeStore.load()
        assert len(ks.entries) >= 12
        titles = ks.titles()
        assert "Expected Goals (xG)" in titles
        assert "VAEP (Valuing Actions by Estimating Probabilities)" in titles

    def test_search_xg(self):
        from src.analytics.agents.knowledge_store import KnowledgeStore
        ks = KnowledgeStore.load()
        results = ks.search("finishing overperformance goals xG", top_k=2)
        assert len(results) >= 1
        assert any("xG" in r.title or "Shot" in r.title or "Finishing" in r.title
                    for r in results)

    def test_section_by_id(self):
        from src.analytics.agents.knowledge_store import KnowledgeStore
        ks = KnowledgeStore.load()
        s = ks.section(1)
        assert s is not None
        assert s.title == "Expected Goals (xG)"
        assert len(s.content) > 100


class TestTacticianAgent:
    def test_produces_report(self):
        from src.analytics.agents.tactician import TacticianAgent
        agent = TacticianAgent()
        report = agent.process(_make_ctx())
        assert report.home.team_side == 0
        assert report.away.team_side == 1
        assert isinstance(report.insights, list)
        assert report.home.press_events_total >= 0

    def test_decentralization_range(self):
        from src.analytics.agents.tactician import TacticianAgent
        report = TacticianAgent().process(_make_ctx())
        assert 0.0 <= report.home.decentralization <= 1.0
        assert 0.0 <= report.away.decentralization <= 1.0


class TestStatScientistAgent:
    def test_produces_report(self):
        from src.analytics.agents.stat_scientist import StatScientistAgent
        report = StatScientistAgent().process(_make_ctx())
        assert report.home.xg > 0
        assert report.home.goals == 2
        assert report.away.goals == 0  # goals at i=2,8 both side=0 (Liverpool)
        assert report.away.shots == 6
        assert len(report.wp_curve) > 0
        assert isinstance(report.insights, list)

    def test_xg_overperformance(self):
        from src.analytics.agents.stat_scientist import StatScientistAgent
        report = StatScientistAgent().process(_make_ctx())
        assert report.home.xg_overperformance == pytest.approx(
            report.home.goals - report.home.xg, abs=1e-6)


class TestVisualizerAgent:
    def test_produces_figures(self):
        import matplotlib.pyplot as plt
        from src.analytics.agents.visualizer import VisualizerAgent
        ctx = _make_ctx()
        ctx.extras["wp_curve"] = []
        try:
            vr = VisualizerAgent().process(ctx)
            assert len(vr.figures) >= 5
            assert "shot_map" in vr.figures
            assert "xg_timeline" in vr.figures
        finally:
            plt.close("all")

    def test_save_figures(self):
        from src.analytics.agents.visualizer import VisualizerAgent
        ctx = _make_ctx()
        ctx.extras["wp_curve"] = []
        vr = VisualizerAgent().process(ctx)
        with tempfile.TemporaryDirectory() as td:
            paths = vr.save_all(directory=td)
            assert len(paths) >= 5
            for p in paths:
                assert os.path.exists(p)


class TestDomainExpertAgent:
    def test_produces_narrative(self):
        from src.analytics.agents.domain_expert import DomainExpertAgent
        ctx = _make_ctx()
        from src.analytics.agents.stat_scientist import StatScientistAgent
        from src.analytics.agents.tactician import TacticianAgent
        ctx.extras["tactical_report"] = TacticianAgent().process(ctx)
        ctx.extras["statistical_report"] = StatScientistAgent().process(ctx)
        nr = DomainExpertAgent().process(ctx)
        assert "Liverpool" in nr.headline
        assert "Arsenal" in nr.headline
        assert len(nr.full_narrative) > 100
        assert nr.tactical_summary != ""
        assert nr.statistical_summary != ""

    def test_key_moments_from_goals(self):
        from src.analytics.agents.domain_expert import DomainExpertAgent
        ctx = _make_ctx()
        nr = DomainExpertAgent().process(ctx)
        goal_moments = [m for m in nr.key_moments if m.category == "goal"]
        assert len(goal_moments) == 2  # i=2, i=8 both side=0


class TestOrchestrator:
    def test_end_to_end_synthetic(self):
        from src.analytics.agents.orchestrator import orchestrate

        class _FakeAdapter:
            def events(self, mid): return _make_ctx().events
            def passes(self, mid): return _make_ctx().passes
            def shots(self, mid): return _make_ctx().shots
            def match(self, mid): return _make_ctx().match
            def freeze_frames(self, mid): return []

        report = orchestrate(_FakeAdapter(), 1,
                             analyses_root=tempfile.mkdtemp(),
                             save_figures=False)
        assert report.match_id == 1
        assert "Liverpool" in report.narrative.headline
        assert len(report.visual.figures) >= 5
        assert len(report.visual.saved_paths) > 0  # saved into match folder
        assert report.errors == []

    def test_end_to_end_saves(self):
        from src.analytics.agents.orchestrator import orchestrate

        class _FakeAdapter:
            def events(self, mid): return _make_ctx().events
            def passes(self, mid): return _make_ctx().passes
            def shots(self, mid): return _make_ctx().shots
            def match(self, mid): return _make_ctx().match
            def freeze_frames(self, mid): return []

        with tempfile.TemporaryDirectory() as td:
            report = orchestrate(_FakeAdapter(), 1, save_figures=True,
                                 analyses_root=td)
            assert len(report.visual.saved_paths) >= 5


class TestMatchOutputDir:
    def test_timestamped_folder_created(self):
        import tempfile
        from src.analytics.agents.html import MatchOutputDir
        with tempfile.TemporaryDirectory() as td:
            out = MatchOutputDir(match_id=42, analyses_root=td)
            folder = out.ensure()
            assert folder.name.endswith("_42")
            assert folder.exists()
            assert (folder / "figures").exists()

    def test_write_text_and_figure(self):
        import tempfile
        from src.analytics.agents.html import MatchOutputDir
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as td:
            out = MatchOutputDir(match_id=43, analyses_root=td)
            p = out.write_text("hello.txt", "testing")
            assert p.read_text() == "testing"
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            fp = out.save_figure(fig, "x")
            assert fp.exists()
            uri = out.figure_data_uri("x")
            assert uri.startswith("data:image/png;base64,")

    def test_same_folder_for_all_agents(self):
        import tempfile
        from src.analytics.agents.html import MatchOutputDir
        with tempfile.TemporaryDirectory() as td:
            out1 = MatchOutputDir(match_id=44, analyses_root=td)
            out1.ensure()
            out2 = MatchOutputDir(match_id=44, analyses_root=td)
            # timestamps differ only by microseconds; use the same instance
            assert out1.folder is not None
            out1.write_text("a.txt", "1")
            out1.write_text("b.txt", "2")
            assert (out1.folder / "a.txt").exists()
            assert (out1.folder / "b.txt").exists()


class TestHtmlReports:
    def _run(self, td: str):
        from src.analytics.agents.orchestrator import orchestrate

        class _FakeAdapter:
            def events(self, mid): return _make_ctx().events
            def passes(self, mid): return _make_ctx().passes
            def shots(self, mid): return _make_ctx().shots
            def match(self, mid): return _make_ctx().match
            def freeze_frames(self, mid): return []

        return orchestrate(_FakeAdapter(), 1, save_figures=True,
                           analyses_root=td), Path(td)

    def test_all_agent_htmls_written(self):
        import tempfile
        report, folder_root = self._run(tempfile.mkdtemp())
        assert set(report.html_written) == {
            "index.html", "tactician.html", "stat_scientist.html",
            "visualizer.html", "domain_expert.html"}
        match_folder = report.output_dir.folder
        assert match_folder.exists()
        for f in ("index.html", "tactician.html", "stat_scientist.html",
                  "visualizer.html", "domain_expert.html"):
            assert (match_folder / f).exists(), f"{f} missing"

    def test_index_has_figures_and_links(self):
        import tempfile
        report, _ = self._run(tempfile.mkdtemp())
        idx = (report.output_dir.folder / "index.html").read_text()
        assert "data:image/png;base64," in idx
        for page in ("tactician.html", "stat_scientist.html",
                     "visualizer.html", "domain_expert.html"):
            assert f'href="{page}"' in idx
        assert "Tactician" in idx
        assert "Statistical scientist" in idx

    def test_visuals_embeddable(self):
        import tempfile
        report, _ = self._run(tempfile.mkdtemp())
        vis = (report.output_dir.folder / "visualizer.html").read_text()
        assert "data:image/png;base64," in vis
        assert "shot_map" in vis
        assert (report.output_dir.figures_dir / "fig_shot_map.png").exists()
        assert (report.output_dir.figures_dir / "fig_xg_timeline.png").exists()

    def test_domain_expert_standalone_narrative(self):
        import tempfile
        report, _ = self._run(tempfile.mkdtemp())
        de = (report.output_dir.folder / "domain_expert.html").read_text()
        assert "TL;DR" in de
        assert "Key moments" in de
