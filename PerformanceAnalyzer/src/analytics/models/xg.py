"""xG aggregation + shot model integration over normalized shot events.

Reuses the fitted StatsBomb shot model (`src.analytics.adapters.statsbomb.shot_model`) for the
scoring probability at a shot's location, so we can attribute an xG to every
`ShotEvent` and aggregate per team / per player. Falls back to a geometric
value when no fitted model is cached.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.analytics.schema import ShotEvent
from src.analytics.adapters.statsbomb.shot_model import load_shot_model

PITCH_LENGTH = 120.0
PITCH_WIDTH = 70.0


@dataclass
class ShotAggregates:
    """xG / shot totals per side with their origin."""
    shots: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})
    xg: dict[int, float] = field(default_factory=lambda: {0: 0.0, 1: 0.0})
    goals: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})
    # per player: (shots, xg, goals)
    by_player: dict[int, tuple[int, float, int]] = field(default_factory=dict)

    def xg_per_shot(self, side: int) -> float | None:
        n = self.shots.get(side, 0)
        return (self.xg.get(side, 0.0) / n) if n else None

    def over_performance(self, side: int) -> float | None:
        """goals - xG for a side (>0 = overperformed, mostly luck)."""
        if side not in self.xg:
            return None
        return float(self.goals.get(side, 0) - self.xg.get(side, 0.0))


def _score_at(x: float, y: float) -> float:
    model = load_shot_model()
    if model is not None:
        return float(model.predict(np.array([x]), np.array([y]))[0])
    # geometric fallback: decay from near-goal baseline
    d = float(np.hypot(x - PITCH_LENGTH, y - PITCH_WIDTH / 2))
    return float(0.65 * np.exp(-0.06 * d))


def aggregate_shots(shots: list[ShotEvent]) -> ShotAggregates:
    """Aggregate xG/goals/shots per side and per player.

    Uses the provider-supplied xG when the event carries one (`ShotEvent.xg`),
    otherwise recomputes it from the shot's location via the fitted model.
    """
    agg = ShotAggregates()
    for s in shots:
        if s.team_side is None:
            continue
        xg = s.xg if s.xg is not None else _score_at(s.x, s.y)
        agg.shots[s.team_side] = agg.shots.get(s.team_side, 0) + 1
        agg.xg[s.team_side] = agg.xg.get(s.team_side, 0.0) + xg
        if s.result == "goal":
            agg.goals[s.team_side] = agg.goals.get(s.team_side, 0) + 1
        if s.player_id is not None:
            sh, x, g = agg.by_player.get(s.player_id, (0, 0.0, 0))
            agg.by_player[s.player_id] = (sh + 1, x + xg, g + (1 if s.result == "goal" else 0))
    return agg
