"""Team-level formations and flow fields from 360 freeze-frames.

Soccermatics Ch 12 visualises shape through average positions and flow. The
StatsBomb 360 freeze-frames carry positions **without player identity** (only
teammate/actor/keeper + location), so per-player tracking is impossible; the
data supports *team-level* aggregates only:

* `team_formation` — the side's average position (mean of all visible players),
  plus their spread, over a window of snapshots.
* `flow_field` — the side's movement binned to a pitch grid: every player sample
  contributes a velocity vector (derived between consecutive snapshots) to the
  cell nearest its position, averaged within each cell. This is a coherent
  team flow field even with unlabelled players.

Per-player formations/flow require true tracking data (the dormant VideoAdapter
or a premium provider), not StatsBomb 360.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.analytics.schema import Frame, FreezeFrame

PITCH_LENGTH = 120.0
PITCH_WIDTH = 70.0
_GRID_NX = 24
_GRID_NY = 16


@dataclass
class TeamFormation:
    """Average-position formation for one side over a window."""
    team_side: int
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    width: float = 0.0       # lateral spread (ptp of y)
    length: float = 0.0      # longitudinal spread (ptp of x)
    n_samples: int = 0       # total player-position samples observed


def _samples(frames: list[Frame | FreezeFrame], team_side: int
             ) -> list[tuple[float, float, float]]:
    """Flatten (t, x, y) player samples for a side across snapshots.

    Because players are unlabelled in StatsBomb 360, every entry is treated as
    an independent position sample (team-level aggregate).
    """
    out: list[tuple[float, float, float]] = []
    for f in frames:
        for _pid, side, x, y, _keeper in f.players:
            if side == team_side:
                out.append((f.timestamp_sec, x, y))
    return out


def team_formation(frames: list[Frame | FreezeFrame],
                   team_side: int) -> TeamFormation:
    """Average position + spread of one side over a window."""
    samples = _samples(frames, team_side)
    form = TeamFormation(team_side=team_side, n_samples=len(samples))
    if not samples:
        return form
    xs = np.array([s[1] for s in samples])
    ys = np.array([s[2] for s in samples])
    form.centroid_x = float(np.mean(xs))
    form.centroid_y = float(np.mean(ys))
    form.width = float(np.ptp(ys)) if xs.size > 1 else 0.0
    form.length = float(np.ptp(xs)) if xs.size > 1 else 0.0
    return form


@dataclass
class FlowField:
    """Team-aggregate flow field binned to an nx x ny pitch grid."""
    team_side: int
    grid_x: np.ndarray        # (ny, nx) cell centre x
    grid_y: np.ndarray
    vx: np.ndarray            # mean velocity x per cell
    vy: np.ndarray
    magnitude: np.ndarray
    counts: np.ndarray        # samples per cell


def flow_field(frames: list[Frame | FreezeFrame], team_side: int,
               nx: int = _GRID_NX, ny: int = _GRID_NY) -> FlowField:
    """Bin a side's movement into an nx x ny flow field.

    For each consecutive pair of snapshots we estimate each player's velocity
    from position change over the time gap (nan-safe), bin every sample by its
    location, and average velocity within each cell. Cells with no samples have
    zero velocity/counts.
    """
    ordered = sorted(frames, key=lambda f: f.timestamp_sec)
    if not ordered:
        return _empty_field(team_side, nx, ny)
    xs = (np.arange(nx) + 0.5) * PITCH_LENGTH / nx
    ys = (np.arange(ny) + 0.5) * PITCH_WIDTH / ny
    gx, gy = np.meshgrid(xs, ys)
    acc_vx = np.zeros((ny, nx))
    acc_vy = np.zeros((ny, nx))
    counts = np.zeros((ny, nx), dtype=int)

    # pair consecutive snapshots; velocity for a player = delta over the gap
    for a, b in zip(ordered[:-1], ordered[1:]):
        dt = b.timestamp_sec - a.timestamp_sec
        if dt <= 1e-6:
            continue
        pa = [p for p in a.players if p[1] == team_side]
        pb = [p for p in b.players if p[1] == team_side]
        if not pa or not pb:
            continue
        # match by nearest position (unlabelled players); greedy one-to-one
        used = [False] * len(pb)
        for p in pa:
            _, ax, ay = p[0], p[2], p[3]
            best = None
            bestd = np.inf
            for j, q in enumerate(pb):
                if used[j]:
                    continue
                d2 = (q[2] - ax) ** 2 + (q[3] - ay) ** 2
                if d2 < bestd:
                    bestd = d2
                    best = j
            if best is None:
                continue
            used[best] = True
            bx, by = pb[best][2], pb[best][3]
            vx = (bx - ax) / dt
            vy = (by - ay) / dt
            xi = int(np.clip(ax / PITCH_LENGTH * nx, 0, nx - 1))
            yi = int(np.clip(ay / PITCH_WIDTH * ny, 0, ny - 1))
            acc_vx[yi, xi] += vx
            acc_vy[yi, xi] += vy
            counts[yi, xi] += 1
    cnt = np.where(counts > 0, counts, 1)
    m_vx = np.where(counts > 0, acc_vx / cnt, 0.0)
    m_vy = np.where(counts > 0, acc_vy / cnt, 0.0)
    return FlowField(team_side=team_side, grid_x=gx, grid_y=gy,
                     vx=m_vx, vy=m_vy,
                     magnitude=np.hypot(m_vx, m_vy), counts=counts)


def _empty_field(team_side: int, nx: int, ny: int) -> FlowField:
    xs = (np.arange(nx) + 0.5) * PITCH_LENGTH / nx
    ys = (np.arange(ny) + 0.5) * PITCH_WIDTH / ny
    gx, gy = np.meshgrid(xs, ys)
    return FlowField(team_side=team_side, grid_x=gx, grid_y=gy,
                     vx=np.zeros((ny, nx)), vy=np.zeros((ny, nx)),
                     magnitude=np.zeros((ny, nx)), counts=np.zeros((ny, nx), dtype=int))
