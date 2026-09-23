"""Tactical analyst agent: formation, pressing, network centrality, shape.

Draws on the passing network, press maps, flow fields, team formation, and
shape metrics to produce a structured tactical assessment of a match.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.analytics.agents.base import AgentContext
from src.analytics.agents.knowledge_store import KnowledgeStore
from src.analytics.models.core import PassingNetwork, summarize_network
from src.analytics.models.flow import FlowField, TeamFormation, flow_field, team_formation
from src.analytics.models.pressure import PressMap, PressWindows, press_map, press_windows


@dataclass
class TacticalAssessment:
    """Structured tactical assessment for one side."""
    team_side: int
    team_name: str = ""
    formation_centroid: tuple[float, float] = (0.0, 0.0)
    formation_width: float = 0.0
    formation_length: float = 0.0
    decentralization: float = 0.0
    most_central_player: int | None = None
    press_events_total: int = 0
    press_final_third_share: float = 0.0
    deep_press_count: int = 0
    flow_magnitude_max: float = 0.0
    knowledge_refs: list[str] = field(default_factory=list)


@dataclass
class TacticalReport:
    """Full tactical report for a match."""
    home: TacticalAssessment = field(default_factory=lambda: TacticalAssessment(team_side=0))
    away: TacticalAssessment = field(default_factory=lambda: TacticalAssessment(team_side=1))
    insights: list[str] = field(default_factory=list)


class TacticianAgent:
    """Agent that produces a tactical assessment from tracking + event data."""

    name = "tactician"

    def __init__(self) -> None:
        self._kb = KnowledgeStore.load()

    def process(self, ctx: AgentContext) -> TacticalReport:
        report = TacticalReport()
        for side in (0, 1):
            ta = self._assess_side(ctx, side)
            if side == 0:
                report.home = ta
            else:
                report.away = ta
        report.insights = self._insights(report)
        return report

    def _assess_side(self, ctx: AgentContext, side: int) -> TacticalAssessment:
        ta = TacticalAssessment(team_side=side)
        ta.team_name = _team_name(ctx, side)

        # network centrality
        net = PassingNetwork()
        for p in ctx.passes:
            if p.outcome == "complete" and p.team_side == side \
                    and p.player_id is not None and p.receiver_id is not None:
                net.add(p.player_id, p.receiver_id)
        if net.edges:
            sn = summarize_network(net)
            ta.decentralization = sn.decentralization
            ta.most_central_player = sn.most_central

        # press maps
        pm = press_map(ctx.events, team_side=side)
        ta.press_events_total = pm.total_press_events
        ta.press_final_third_share = pm.share_in_final_third() or 0.0
        pw = press_windows(ctx.events, team_side=side)
        ta.deep_press_count = pw.deep_press_count

        # formation from frames (if available)
        frames = [f for f in ctx.frames if hasattr(f, "players")]
        if frames:
            fm = team_formation(frames, team_side=side)
            ta.formation_centroid = (fm.centroid_x, fm.centroid_y)
            ta.formation_width = fm.width
            ta.formation_length = fm.length
            fl = flow_field(frames, team_side=side)
            ta.flow_magnitude_max = float(fl.magnitude.max())

        # knowledge refs
        refs = self._kb.search("pressing formation network centrality", top_k=2)
        ta.knowledge_refs = [r.title for r in refs]
        return ta

    def _insights(self, report: TacticalReport) -> list[str]:
        out: list[str] = []
        h, a = report.home, report.away
        if h.decentralization < a.decentralization:
            out.append(
                f"{h.team_name} builds up more centrally "
                f"(decentralization {h.decentralization:.3f} vs {a.decentralization:.3f}); "
                "Soccermatics §3 links decentralized networks to ~8% higher scoring rates."
            )
        if h.press_final_third_share > 0.4:
            out.append(
                f"{h.team_name} concentrates pressing in the final third "
                f"({h.press_final_third_share:.0%} of defensive events)."
            )
        if a.press_final_third_share > 0.4:
            out.append(
                f"{a.team_name} concentrates pressing in the final third "
                f"({a.press_final_third_share:.0%} of defensive events)."
            )
        if h.flow_magnitude_max > 0 and a.flow_magnitude_max > 0:
            faster = h.team_name if h.flow_magnitude_max > a.flow_magnitude_max else a.team_name
            out.append(f"{faster} showed higher maximum team movement velocity.")
        return out


def _team_name(ctx: AgentContext, side: int) -> str:
    if side < len(ctx.match.teams):
        return ctx.match.teams[side].name
    return f"Team {side}"
