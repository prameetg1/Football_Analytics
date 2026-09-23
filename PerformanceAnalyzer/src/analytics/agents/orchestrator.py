"""Multi-agent orchestrator: chains all analytics agents into a match report.

Entry point: `orchestrate(adapter, match_id)` → `MatchInsightReport`.

Every analysed match is written to its own **timestamped analysis folder**:

    output/analyses/<YYYY-mm-dd_HH-MM-SS>_<match_id>/
        figures/fig_*.png            (all charts for this match)
        tactician.html
        stat_scientist.html
        visualizer.html
        domain_expert.html
        index.html

All agents write into the same `MatchOutputDir`, so each match is fully
self-contained: figures are inlined into the HTML as base64 data URIs and the
pages link to one another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from src.analytics.agents.base import AgentContext, AgentRegistry
from src.analytics.agents.domain_expert import DomainExpertAgent, NarrativeReport
from src.analytics.agents.html import MatchOutputDir
from src.analytics.agents.renderers import (
    render_domain_expert,
    render_index,
    render_stat_scientist,
    render_tactician,
    render_visualizer,
)
from src.analytics.agents.stat_scientist import StatScientistAgent, StatisticalReport
from src.analytics.agents.tactician import TacticianAgent, TacticalReport
from src.analytics.agents.visualizer import VisualizerAgent, VisualizationReport

HTML_FILES = ("index.html", "tactician.html", "stat_scientist.html",
              "visualizer.html", "domain_expert.html")


class _AdapterLike(Protocol):
    """Minimal adapter contract the orchestrator needs."""
    def events(self, match_id: int) -> list: ...
    def passes(self, match_id: int) -> list: ...
    def shots(self, match_id: int) -> list: ...
    def match(self, match_id: int) -> object: ...


def _build_frames(adapter: _AdapterLike, match_id: int) -> list:
    """Attempt to load tracking frames; return [] if not available."""
    try:
        return adapter.freeze_frames(match_id)
    except AttributeError:
        return []
    except Exception:
        return []


@dataclass
class MatchInsightReport:
    """Complete analytics insight for one match, produced by all agents."""
    match_id: int
    tactical: TacticalReport = field(default_factory=TacticalReport)
    statistical: StatisticalReport = field(default_factory=StatisticalReport)
    visual: VisualizationReport = field(default_factory=VisualizationReport)
    narrative: NarrativeReport = field(default_factory=NarrativeReport)
    errors: list[str] = field(default_factory=list)
    output_dir: MatchOutputDir | None = None
    html_written: list[str] = field(default_factory=list)


def orchestrate(
    adapter: _AdapterLike,
    match_id: int,
    with_spatial: bool = False,
    save_figures: bool = True,
    analyses_root: str | Path | None = None,
) -> MatchInsightReport:
    """Run the full agent pipeline on a match and emit HTML reports.

    1. Create the timestamped match folder (all agents share it).
    2. Load the data from the adapter.
    3. Run TacticianAgent → tactician.html
    4. Run StatScientistAgent → stat_scientist.html
    5. Run VisualizerAgent → visualizer.html (figures into figures/)
    6. Run DomainExpertAgent → domain_expert.html
    7. Assemble index.html and return the report.
    """
    out = MatchOutputDir(
        match_id,
        analyses_root=Path(analyses_root) if analyses_root else None)
    out.ensure()

    events = adapter.events(match_id)
    passes = adapter.passes(match_id)
    shots = adapter.shots(match_id)
    frames = _build_frames(adapter, match_id)
    match = adapter.match(match_id)

    ctx = AgentContext(match_id=match_id, match=match, events=events,
                       passes=passes, shots=shots, frames=frames)
    ctx.extras["output_dir"] = out
    report = MatchInsightReport(match_id=match_id, output_dir=out)

    # --- Tactician ---
    try:
        tac_agent = TacticianAgent()
        report.tactical = tac_agent.process(ctx)
        ctx.extras["tactical_report"] = report.tactical
        out.write_text("tactician.html",
                       render_tactician(report.tactical, out, tac_agent.name))
        report.html_written.append("tactician.html")
    except Exception as exc:
        report.errors.append(f"tactician: {exc}")

    # --- StatScientist ---
    try:
        stat_agent = StatScientistAgent()
        report.statistical = stat_agent.process(ctx)
        ctx.extras["statistical_report"] = report.statistical
        if report.statistical.wp_curve:
            ctx.extras["wp_curve"] = report.statistical.wp_curve
        out.write_text("stat_scientist.html",
                       render_stat_scientist(report.statistical, out,
                                             stat_agent.name))
        report.html_written.append("stat_scientist.html")
    except Exception as exc:
        report.errors.append(f"stat_scientist: {exc}")

    # --- Visualizer ---
    try:
        vis_agent = VisualizerAgent()
        report.visual = vis_agent.process(ctx)
        out.write_text("visualizer.html",
                       render_visualizer(report.visual, out, vis_agent.name))
        report.html_written.append("visualizer.html")
    except Exception as exc:
        report.errors.append(f"visualizer: {exc}")

    # --- DomainExpert ---
    try:
        domain_agent = DomainExpertAgent()
        report.narrative = domain_agent.process(ctx)
        out.write_text("domain_expert.html",
                       render_domain_expert(report.narrative, out,
                                            domain_agent.name))
        report.html_written.append("domain_expert.html")
    except Exception as exc:
        report.errors.append(f"domain_expert: {exc}")

    # --- Index (always, even if some agents failed) ---
    try:
        out.write_text(
            "index.html",
            render_index(out, tactical=report.tactical,
                         statistical=report.statistical,
                         visual=report.visual, domain=report.narrative))
        report.html_written.append("index.html")
    except Exception as exc:
        report.errors.append(f"index: {exc}")

    return report