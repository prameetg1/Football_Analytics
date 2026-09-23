"""Analytics agents layer: model+domain-knowledge-based agents for match analysis.

Agents consume `AgentContext` (adapter data + model outputs) and produce
typed assessment dataclasses. The orchestrator chains them into a single
`MatchInsightReport`.
"""

from src.analytics.agents.base import AgentContext, AgentRegistry
from src.analytics.agents.domain_expert import DomainExpertAgent, NarrativeReport
from src.analytics.agents.html import MatchOutputDir
from src.analytics.agents.knowledge_store import KnowledgeStore
from src.analytics.agents.orchestrator import MatchInsightReport, orchestrate
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

__all__ = [
    "AgentContext", "AgentRegistry",
    "DomainExpertAgent", "NarrativeReport",
    "KnowledgeStore",
    "MatchInsightReport", "orchestrate", "MatchOutputDir",
    "render_domain_expert", "render_index",
    "render_stat_scientist", "render_tactician", "render_visualizer",
    "StatScientistAgent", "StatisticalReport",
    "TacticianAgent", "TacticalReport",
    "VisualizerAgent", "VisualizationReport",
]
