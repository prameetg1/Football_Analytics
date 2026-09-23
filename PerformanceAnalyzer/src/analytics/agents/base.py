"""Analytics agents layer: model+domain-knowledge-based agents for match analysis.

Each agent consumes `AgentContext` (populated by an adapter + `analyze_match`)
and produces typed assessment dataclasses. The agents are deterministic
model+knowledge pipelines; an optional natural-language enrichment step can be
added for summaries when a language model is provided, but none is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from src.analytics.schema import Event, Frame, FreezeFrame, Match, PassEvent, ShotEvent

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge_store"


@dataclass
class AgentContext:
    """Shared state passed to every agent for one match."""
    match_id: int
    match: Match = field(default_factory=lambda: Match(match_id=0))
    events: list[Event] = field(default_factory=list)
    passes: list[PassEvent] = field(default_factory=list)
    shots: list[ShotEvent] = field(default_factory=list)
    frames: list[Frame | FreezeFrame] = field(default_factory=list)
    spatial: object | None = None
    report: object | None = None
    extras: dict = field(default_factory=dict)


class Agent(Protocol):
    """Protocol every analytics agent must implement."""
    name: str

    def process(self, ctx: AgentContext) -> object: ...


@dataclass
class AgentRegistry:
    """Optional registry for named agents (useful for orchestrator dispatch)."""
    _agents: dict[str, Agent] = field(default_factory=dict)

    def register(self, agent: Agent) -> None:
        self._agents[agent.name] = agent

    def get(self, name: str) -> Agent | None:
        return self._agents.get(name)

    def names(self) -> list[str]:
        return list(self._agents)
