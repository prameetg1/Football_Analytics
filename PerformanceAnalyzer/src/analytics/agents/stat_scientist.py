"""Statistical scientist agent: xG, VAEP, win-prob, Dixon-Coles, expected points.

Draws on the statistical model layer to produce a structured assessment of
shot quality, on-ball value creation, over/under-performance, and win-probability
dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.analytics.agents.base import AgentContext
from src.analytics.agents.knowledge_store import KnowledgeStore
from src.analytics.models.core import (
    ExpectedPoints,
    Funnel,
    GoalLineValuation,
    expected_points_from_goals,
    funnel,
    possession_analysis,
)
from src.analytics.models.xg import ShotAggregates, aggregate_shots
from src.analytics.models.vaep import VaepSummary, vaep
from src.analytics.models.winprob import (
    GoalValueByMinute,
    ScoreCurve,
    goal_value_by_minute,
    score_curve,
)
from src.analytics.models.rating import DixonColesFit, fit_dixon_coles, predict_match


@dataclass
class StatisticalAssessment:
    """Structured statistical read-out for one side."""
    team_side: int
    team_name: str = ""
    xg: float = 0.0
    goals: int = 0
    xg_overperformance: float = 0.0
    shots: int = 0
    shots_on_target: int = 0
    vaep_value: float = 0.0
    vaep_actions: int = 0
    possession_share: float = 0.0
    passing_rate: float = 0.0
    funnel_possessions_to_goal: float = 0.0
    knowledge_refs: list[str] = field(default_factory=list)


@dataclass
class StatisticalReport:
    """Full statistical report for a match."""
    home: StatisticalAssessment = field(default_factory=lambda: StatisticalAssessment(team_side=0))
    away: StatisticalAssessment = field(default_factory=lambda: StatisticalAssessment(team_side=1))
    expected_points: ExpectedPoints | None = None
    goal_line: GoalLineValuation | None = None
    wp_curve: list[ScoreCurve] = field(default_factory=list)
    goal_value_curve: list[GoalValueByMinute] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)


class StatScientistAgent:
    """Agent that produces a statistical assessment from events + shots."""

    name = "stat_scientist"

    def __init__(self) -> None:
        self._kb = KnowledgeStore.load()

    def process(self, ctx: AgentContext) -> StatisticalReport:
        report = StatisticalReport()
        pos = possession_analysis(ctx.events, ctx.passes)
        shot_agg = aggregate_shots(ctx.shots)
        f = funnel(ctx.events, ctx.shots)
        valued, vaep_sum = vaep(ctx.passes, ctx.shots)

        for side in (0, 1):
            sa = self._assess_side(ctx, side, pos, shot_agg, f, vaep_sum)
            if side == 0:
                report.home = sa
            else:
                report.away = sa

        # expected points from scoring rates
        lf = {s: shot_agg.xg.get(s, 0.0) for s in (0, 1)}
        la = {s: shot_agg.xg.get(1 - s, 0.0) for s in (0, 1)}
        report.expected_points = expected_points_from_goals(lf, la)

        # goal line valuation
        try:
            report.goal_line = GoalLineValuation.from_rates(lf, la)
        except Exception:
            pass

        # win-prob score curve (balanced 1.4 per side as baseline)
        report.wp_curve = score_curve(1.4, 1.4, minute_step=5.0)
        report.goal_value_curve = goal_value_by_minute(1.4, 1.4, minute_step=10.0)
        ctx.extras["vaep_summary"] = vaep_sum

        report.insights = self._insights(report)
        return report

    def _assess_side(
        self, ctx: AgentContext, side: int, pos, shot_agg, f, vaep_sum,
    ) -> StatisticalAssessment:
        sa = StatisticalAssessment(team_side=side)
        sa.team_name = _team_name(ctx, side)
        sa.xg = round(shot_agg.xg.get(side, 0.0), 3)
        sa.goals = shot_agg.goals.get(side, 0)
        sa.xg_overperformance = round(sa.goals - sa.xg, 3)
        sa.shots = shot_agg.shots.get(side, 0)
        sa.shots_on_target = sum(
            1 for s in ctx.shots if s.team_side == side and s.result != "off_target"
        )
        sa.possession_share = round(pos.share(side), 3)
        sa.passing_rate = round(pos.passing_rate(side), 2)
        sa.funnel_possessions_to_goal = round(f.possessions_to_goal, 4)
        player_vals = vaep_sum.by_player
        sa.vaep_value = round(
            sum(v["value"] for pid, v in player_vals.items()
                if _player_side(pid, ctx) == side), 3)
        sa.vaep_actions = sum(
            v["actions"] for pid, v in player_vals.items()
            if _player_side(pid, ctx) == side)
        refs = self._kb.search("xG overperformance VAEP finishing", top_k=2)
        sa.knowledge_refs = [r.title for r in refs]
        return sa

    def _insights(self, report: StatisticalReport) -> list[str]:
        out: list[str] = []
        h, a = report.home, report.away
        diff = h.xg - a.xg
        if abs(diff) > 0.5:
            better = h.team_name if diff > 0 else a.team_name
            out.append(
                f"{better} created the better chances (xG {max(h.xg,a.xg):.2f} vs "
                f"{min(h.xg,a.xg):.2f})."
            )
        if abs(h.xg_overperformance) > 0.5:
            direction = "over" if h.xg_overperformance > 0 else "under"
            out.append(
                f"{h.team_name} {direction}-performed their xG by "
                f"{abs(h.xg_overperformance):.2f} ({h.goals} goals from {h.xg:.2f} xG)."
            )
        if abs(a.xg_overperformance) > 0.5:
            direction = "over" if a.xg_overperformance > 0 else "under"
            out.append(
                f"{a.team_name} {direction}-performed their xG by "
                f"{abs(a.xg_overperformance):.2f} ({a.goals} goals from {a.xg:.2f} xG)."
            )
        if h.vaep_value > 0 and a.vaep_value > 0:
            ratio = h.vaep_value / a.vaep_value
            if ratio > 1.2 or ratio < 0.8:
                more = h.team_name if ratio > 1 else a.team_name
                out.append(
                    f"{more} created significantly more on-ball value (VAEP "
                    f"{max(h.vaep_value,a.vaep_value):.2f} vs {min(h.vaep_value,a.vaep_value):.2f})."
                )
        return out


def _team_name(ctx: AgentContext, side: int) -> str:
    if side < len(ctx.match.teams):
        return ctx.match.teams[side].name
    return f"Team {side}"


def _player_side(player_id: int, ctx: AgentContext) -> int:
    """Guess player side from events (first event with that player_id)."""
    for e in ctx.events:
        if e.player_id == player_id and e.team_side is not None:
            return e.team_side
    return 0
