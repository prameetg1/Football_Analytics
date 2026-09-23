"""Domain expert agent: synthesizes all agent outputs into match narrative.

Takes the tactical, statistical, and visualization reports plus the knowledge
base to produce a human-readable match analysis with contextual insights,
key moments, and player highlights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.analytics.agents.base import AgentContext
from src.analytics.agents.knowledge_store import KnowledgeStore


@dataclass
class PlayerHighlight:
    """One notable player performance in the match."""
    player_id: int
    team_side: int
    role: str = ""
    summary: str = ""
    key_stat: str = ""


@dataclass
class MatchMoment:
    """One notable moment in the match."""
    minute: int
    description: str
    category: str = ""


@dataclass
class NarrativeReport:
    """Human-readable match narrative produced by the domain expert."""
    headline: str = ""
    tldr: str = ""
    tactical_summary: str = ""
    statistical_summary: str = ""
    key_moments: list[MatchMoment] = field(default_factory=list)
    player_highlights: list[PlayerHighlight] = field(default_factory=list)
    coaching_insights: list[str] = field(default_factory=list)
    full_narrative: str = ""


class DomainExpertAgent:
    """Agent that synthesizes all agent outputs into a coherent narrative."""

    name = "domain_expert"

    def __init__(self) -> None:
        self._kb = KnowledgeStore.load()

    def process(self, ctx: AgentContext) -> NarrativeReport:
        """Produce the narrative from the agent context (which must carry
        `tactical_report`, `statistical_report` in ctx.extras)."""
        tac = ctx.extras.get("tactical_report")
        stat = ctx.extras.get("statistical_report")
        vis = ctx.extras.get("visualization_report")
        nr = NarrativeReport()

        home = _team_name(ctx, 0)
        away = _team_name(ctx, 1)

        # headline
        h_goals = stat.home.goals if stat else 0
        a_goals = stat.away.goals if stat else 0
        nr.headline = f"{home} {h_goals}-{a_goals} {away}"

        # tldr
        if stat and abs(stat.home.xg - stat.away.xg) > 0.3:
            better_xg = home if stat.home.xg > stat.away.xg else away
            nr.tldr = (
                f"{better_xg} created the better chances despite the final score."
                if h_goals != a_goals else
                f"A balanced match where {better_xg} shaded the expected goals."
            )
        else:
            nr.tldr = "An even contest with chances shared between both sides."

        # tactical summary
        if tac:
            parts = []
            if tac.home.decentralization > 0 and tac.away.decentralization > 0:
                hd = tac.home.decentralization
                ad = tac.away.decentralization
                more = home if hd > ad else away
                parts.append(
                    f"Build-up style: {more} played a more distributed/passing-diverse "
                    f"network."
                )
            if tac.home.press_events_total or tac.away.press_events_total:
                parts.append(
                    f"Pressing: {home} {tac.home.press_events_total} defensive events, "
                    f"{away} {tac.away.press_events_total}."
                )
            if tac.home.formation_width > 0 or tac.away.formation_width > 0:
                wider = home if tac.home.formation_width > tac.away.formation_width else away
                parts.append(f"{wider} held a wider shape in possession.")
            if not parts:
                parts.append(
                    f"Tactical readout available (decentralization {home} "
                    f"{tac.home.decentralization:.3f}, {away} {tac.away.decentralization:.3f})."
                )
            nr.tactical_summary = " ".join(parts)

        # statistical summary
        if stat:
            nr.statistical_summary = (
                f"xG: {home} {stat.home.xg:.2f} ({stat.home.goals} goals), "
                f"{away} {stat.away.xg:.2f} ({stat.away.goals} goals). "
                f"Over/under-performance: {home} {stat.home.xg_overperformance:+.2f}, "
                f"{away} {stat.away.xg_overperformance:+.2f}."
            )

        # key moments: shot goals + high-xG misses
        for s in ctx.shots:
            xg = s.xg or 0.0
            if s.result == "goal":
                team = _team_name(ctx, s.team_side or 0)
                nr.key_moments.append(MatchMoment(
                    minute=s.minute,
                    description=f"Goal — {team} (xG {xg:.2f})",
                    category="goal"))
            elif xg > 0.4 and s.result == "off_target":
                team = _team_name(ctx, s.team_side or 0)
                nr.key_moments.append(MatchMoment(
                    minute=s.minute,
                    description=f"Big miss — {team} (xG {xg:.2f})",
                    category="miss"))
        nr.key_moments.sort(key=lambda m: m.minute)

        # player highlights: VAEP top performers
        vaep_summary = ctx.extras.get("vaep_summary")
        if vaep_summary:
            for pid, val in vaep_summary.top_players(3):
                side = _player_side(pid, ctx)
                nr.player_highlights.append(PlayerHighlight(
                    player_id=pid, team_side=side,
                    role="top value creator",
                    summary=f"Top VAEP contributor: +{val:.2f} value",
                    key_stat=f"VAEP +{val:.2f}"))

        # coaching insights from knowledge base
        refs = self._kb.search("possession pressing formation", top_k=2)
        if refs:
            nr.coaching_insights.append(
                f"Key reference: {refs[0].title} — "
                + refs[0].content[:120].replace("\n", " ") + "..."
            )

        # full narrative
        parts = [f"# {nr.headline}\n"]
        parts.append(f"**TL;DR:** {nr.tldr}\n")
        if nr.tactical_summary:
            parts.append(f"## Tactical Summary\n{nr.tactical_summary}\n")
        if nr.statistical_summary:
            parts.append(f"## Statistical Summary\n{nr.statistical_summary}\n")
        if nr.key_moments:
            parts.append("## Key Moments")
            for m in nr.key_moments:
                parts.append(f"- **{m.minute}'** — {m.description}")
            parts.append("")
        if nr.player_highlights:
            parts.append("## Player Highlights")
            for ph in nr.player_highlights:
                parts.append(f"- Player {ph.player_id}: {ph.summary}")
            parts.append("")
        if nr.coaching_insights:
            parts.append("## Coaching Insights")
            for ci in nr.coaching_insights:
                parts.append(f"- {ci}")
            parts.append("")
        nr.full_narrative = "\n".join(parts)
        return nr


def _team_name(ctx: AgentContext, side: int) -> str:
    if side < len(ctx.match.teams):
        return ctx.match.teams[side].name
    return f"Team {side}"


def _player_side(player_id: int, ctx: AgentContext) -> int:
    for e in ctx.events:
        if e.player_id == player_id and e.team_side is not None:
            return e.team_side
    return 0
