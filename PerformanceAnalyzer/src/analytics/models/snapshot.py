"""Event-time spatial models over 360 freeze-frames.

The 360 freeze-frames give a position sample of the ~visible players every time
an event happens (~0.3 s apart). These snapshots are dense enough to build
per-moment spatial models — the same pitch control, possession and territory
that the (dormant) continuous tracking feed would give, but anchored to real
event times.

This module converts a `FreezeFrame`/`Frame` into a `TrackingFrame`-schema
DataFrame (so it can reuse `pitch_control_surface` / `field_tilt` unchanged) and
adds per-snapshot possession/territory summarises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analytics.common import TrackingFrame
from src.analytics.models.pitch_control import (
    field_tilt,
    pitch_control_surface,
)
from src.analytics.schema import Frame, FreezeFrame

# Who counts as a player (outfield + keeper) — same as the tracking metrics.
_PLAYER_CLASS = "player"
_KEEPER_CLASS = "goalkeeper"


@dataclass
class SnapshotSummary:
    """Per-snapshot spatial summary extracted from one 360 frame."""
    timestamp_sec: float
    possession_side: int | None
    frames: int = 0                  # number of players visible (both teams)
    control_team_0: float | None = None     # time-normalised local control side 0
    territory_team_0: float | None = None   # fraction of pitch controlled by side 0
    ball_zone_x: float | None = None        # ball x position [0,120]
    ball_possession_side: int | None = None


def to_tracking_df(players: list[tuple], timestamp_sec: float) -> pd.DataFrame:
    """360 `players` tuples -> one TrackingFrame row per player.

    `players` is (player_id, team_side, x, y, is_goalkeeper).
    """
    rows = []
    for pid, side, x, y, keeper in players:
        rows.append({
            "frame_idx": 0, "timestamp_sec": timestamp_sec,
            "track_id": pid if pid is not None else f"un-%d" % len(rows),
            "team_id": side, "class_name": _KEEPER_CLASS if keeper else _PLAYER_CLASS,
            "pitch_x": float(x), "pitch_y": float(y),
        })
    return pd.DataFrame(rows, columns=TrackingFrame.REQUIRED_COLUMNS)


def snapshot_summary(frame: Frame | FreezeFrame,
                     nx: int = 16, ny: int = 11) -> SnapshotSummary:
    """Pitch control + possession territory + ball zone for one snapshot.

    Pitch control reuses the racing model: for a single frame there is no
    velocity, so control is a static-reach race. `control_team_0` is the
    fraction of grid cells won by side 0 (mean local control).
    """
    df = to_tracking_df(frame.players, frame.timestamp_sec)
    if df.empty:
        return SnapshotSummary(frame.timestamp_sec, frame.possession_side,
                               frames=0)
    # territory: fraction of pitch cells where team 0 has >50% of the control
    control0 = pitch_control_surface(df, team_id=0, nx=nx, ny=ny)
    terr0 = float(np.mean(control0 > 0.5)) if control0.size else None
    control_mean = float(np.mean(control0)) if control0.size else None
    return SnapshotSummary(
        timestamp_sec=frame.timestamp_sec,
        possession_side=frame.possession_side,
        frames=len(df),
        control_team_0=control_mean,
        territory_team_0=terr0,
        ball_zone_x=(frame.ball[0] if frame.ball is not None else None),
        ball_possession_side=frame.possession_side,
    )


def territory_series(frames: list[Frame | FreezeFrame]) -> list[SnapshotSummary]:
    """Run `snapshot_summary` over every 360 frame, sorted by time."""
    ordered = list(frames)
    ordered.sort(key=lambda f: f.timestamp_sec)
    return [snapshot_summary(f) for f in ordered]
