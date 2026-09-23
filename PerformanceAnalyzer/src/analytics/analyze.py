"""High-level analytics entrypoint: adapter -> MatchReport.

Bundles the statistical models into a single report object for one match, so a
product layer (or a report generator) can pull passing network, possession,
funnel, xG, shot map, pressing and goal-value numbers without touching adapters
or raw schema. The 360-heavy spatial models (flow field, team formation,
per-snapshot territory) are exposed separately via `analyze_spatial` because
they iterate the whole freeze-frame set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.analytics.adapters.base import ProviderAdapter
from src.analytics.models.attacking import (
    ProgressivePassSummary,
    ShotMap,
    progressive_summary,
    shot_map,
)
from src.analytics.models.core import (
    Funnel,
    GoalLineValuation,
    PassingNetwork,
    Possession,
    funnel,
    possession_analysis,
)
from src.analytics.models.pressure import (
    PressMap,
    PressWindows,
    press_map,
    press_windows,
)
from src.analytics.models.xg import ShotAggregates, aggregate_shots


@dataclass
class MatchReport:
    """Consolidated analytics for one match."""
    match_id: int
    home_team: str = ""
    away_team: str = ""
    passing_network: PassingNetwork = field(default_factory=PassingNetwork)
    possession: Possession = field(default_factory=Possession)
    funnel: Funnel = field(default_factory=Funnel)
    shots: ShotAggregates = field(default_factory=ShotAggregates)
    goal_value: GoalLineValuation | None = None
    shot_map: ShotMap | None = None
    passing_progress: ProgressivePassSummary | None = None
    press: dict[int, PressMap] = field(default_factory=dict)
    press_windows: dict[int, PressWindows] = field(default_factory=dict)
    spatial: "SpatialReport | None" = None
    # per-side aggregated numeric summary for quick display/exports
    summary: dict = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return bool(self.passing_network or self.possession.seconds or
                    self.funnel.shots or self.shots.shots)


def _build_passing_network(passes) -> PassingNetwork:
    net = PassingNetwork()
    for p in passes:
        if p.outcome == "complete" and p.player_id is not None \
                and p.receiver_id is not None:
            net.add(p.player_id, p.receiver_id)
    return net


def analyze_match(adapter: ProviderAdapter, match_id: int,
                  with_spatial: bool = False) -> MatchReport:
    """Run the model bundle over a match via any provider adapter.

    `with_spatial` additionally computes the 360-frame spatial models (pitch
    control / territory / formation / flow) over the full freeze-frame set —
    more expensive, so off by default.
    """
    events = adapter.events(match_id)
    passes = adapter.passes(match_id)
    shots = adapter.shots(match_id)

    rep = MatchReport(
        match_id=match_id,
        passing_network=_build_passing_network(passes),
        possession=possession_analysis(events, passes),
        funnel=funnel(events, shots),
        shots=aggregate_shots(shots),
        shot_map=shot_map(shots),
        passing_progress=progressive_summary(passes),
        press={s: press_map(events, s) for s in (0, 1)},
        press_windows={s: press_windows(events, s) for s in (0, 1)},
    )

    m = adapter.match(match_id)
    if m.teams:
        rep.home_team = m.teams[0].name if len(m.teams) > 0 else ""
        rep.away_team = m.teams[1].name if len(m.teams) > 1 else ""

    if with_spatial:
        rep.spatial = _analyze_spatial(adapter, match_id)

    rep.summary = {
        s: {
            "team": (m.teams[s].name if s < len(m.teams) else ""),
            "possession_share": round(rep.possession.share(s), 3),
            "passing_rate": round(rep.possession.passing_rate(s), 2),
            "shots": rep.shots.shots.get(s, 0),
            "xg": round(rep.shots.xg.get(s, 0.0), 3),
            "goals": rep.shots.goals.get(s, 0),
            "funnel_possessions_to_goal": round(
                rep.funnel.possessions_to_goal, 4),
            "progressive_share": round(
                rep.passing_progress.progressive_share(s)
                if rep.passing_progress else 0.0, 3),
            "press_events": rep.press.get(s).total_press_events
            if rep.press.get(s) else 0,
        }
        for s in (0, 1)
    }
    return rep


@dataclass
class SpatialReport:
    """Flow-field / formation / per-snapshot territory for one match."""
    match_id: int
    team_formation: dict[int, object] = field(default_factory=dict)
    flow: dict[int, object] = field(default_factory=dict)
    snapshots: list = field(default_factory=list)  # SnapshotSummary list


def _analyze_spatial(adapter: ProviderAdapter, match_id: int) -> SpatialReport:
    from src.analytics.models.flow import flow_field, team_formation
    from src.analytics.models.snapshot import territory_series

    frames = adapter.freeze_frames(match_id)
    if not frames:
        return SpatialReport(match_id=match_id)
    # Flow/formation are cheap-ish on the full set; territory samples a stride.
    return SpatialReport(
        match_id=match_id,
        team_formation={s: team_formation(frames, s) for s in (0, 1)},
        flow={s: flow_field(frames, s) for s in (0, 1)},
        snapshots=territory_series(frames[::20]),
    )

