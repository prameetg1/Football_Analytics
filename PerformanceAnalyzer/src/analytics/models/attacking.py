"""Attacking models: shot maps and progressive/through-ball passing.

Complements `xg.py` (per-shot value) with *where* shots come from (shot map)
and *how* the ball advances (progressive passes, pass-to-space), so an analyst
can see not just total xG but the distribution and the build-up behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.analytics.schema import PassEvent, ShotEvent

PITCH_LENGTH = 120.0
PITCH_WIDTH = 70.0

# Progressivity: a pass is "progressive" if it moves the ball at least this far
# toward the opponent goal and does not move it backwards.
PROGRESSIVE_MIN_M = 10.0


def _progress_m(p: PassEvent) -> float:
    """Metres of longitudinal progress toward the (normalized) opponent goal.

    On the canonical fixed frame the defending goal of both teams is at x=0 and
    teams attack right->left (see schema docstring), but after `convert_pitch_xy`
    the attacking direction is consistent for whoever has the ball; we instead
    measure progress as |end_x - start_x| (advance downfield) regardless of side.
    """
    if p.end_x is None or p.end_y is None:
        return 0.0
    # positive when the ball ends further downfield than it started
    return float(p.end_x - p.x)


@dataclass
class ProgressivePassSummary:
    """Pass progressivity counts for a match (per team)."""
    progressive: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})
    backward: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})
    through_balls: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})
    total_completed: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})

    def progressive_share(self, side: int) -> float | None:
        tot = self.total_completed.get(side, 0)
        return (self.progressive.get(side, 0) / tot) if tot else None


def progressive_summary(passes: list[PassEvent]) -> ProgressivePassSummary:
    """Per-team progressive / backward / through-ball pass counts."""
    s = ProgressivePassSummary()
    for p in passes:
        if p.team_side is None:
            continue
        if p.outcome != "complete":
            continue
        s.total_completed[p.team_side] = s.total_completed.get(p.team_side, 0) + 1
        m = _progress_m(p)
        if m >= PROGRESSIVE_MIN_M:
            s.progressive[p.team_side] = s.progressive.get(p.team_side, 0) + 1
        elif m <= -3.0:
            s.backward[p.team_side] = s.backward.get(p.team_side, 0) + 1
        if p.through_ball:
            s.through_balls[p.team_side] = s.through_balls.get(p.team_side, 0) + 1
    return s


@dataclass
class ShotZone:
    """One pitch zone on the shot map."""
    xlo: float
    xhi: float
    ylo: float
    yhi: float
    shots: int = 0
    goals: int = 0
    xg: float = 0.0


@dataclass
class ShotMap:
    """Shots (attempts, goals, xG) bucketed into nx x ny pitch zones."""
    nx: int
    ny: int
    zones: list[ShotZone] = field(default_factory=list)

    def heatmap(self) -> np.ndarray:
        """(ny, nx) xG per zone."""
        h = np.zeros((self.ny, self.nx))
        for z in self.zones:
            i = int(z.xlo // (PITCH_LENGTH / self.nx))
            j = int(z.ylo // (PITCH_WIDTH / self.ny))
            if 0 <= i < self.nx and 0 <= j < self.ny:
                h[j, i] += z.xg
        return h


def shot_map(shots: list[ShotEvent], nx: int = 6, ny: int = 4) -> ShotMap:
    """Aggregate shots into an nx x ny shot map (xG-weighted heatmap).

    Uses the attacking-goal frame: buckets run from the defensive end (x=0) to
    the attacking goal (x=120). Only shots with a real xG or location are kept.
    """
    zones = []
    for i in range(nx):
        for j in range(ny):
            zones.append(ShotZone(
                xlo=i * PITCH_LENGTH / nx, xhi=(i + 1) * PITCH_LENGTH / nx,
                ylo=j * PITCH_WIDTH / ny, yhi=(j + 1) * PITCH_WIDTH / ny))
    sm = ShotMap(nx=nx, ny=ny, zones=zones)
    for s in shots:
        i = int(np.clip(s.x // (PITCH_LENGTH / nx), 0, nx - 1))
        j = int(np.clip(s.y // (PITCH_WIDTH / ny), 0, ny - 1))
        z = zones[i * ny + j]
        z.shots += 1
        z.xg += (s.xg if s.xg is not None else 0.0)
        if s.result == "goal":
            z.goals += 1
    return sm
