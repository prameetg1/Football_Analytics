"""Shared helpers for tracking-frame metrics.

All metrics consume a DataFrame with the `TrackingFrame.REQUIRED_COLUMNS`
schema (see `src.analytics.common`). Positions are pitch metres
(pitch_x in [0, 120] along the length, pitch_y in [0, 70] across the width).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.common import PLAYER_CLASSES

# Above this speed (m/s) a per-frame jump is treated as a tracking/homography
# glitch rather than real motion (human sprints peak ~10 m/s).
MAX_PLAUSIBLE_SPEED_MS = 12.0

# Speed above which a distance segment counts as a sprint (m/s).
SPRINT_SPEED_MS = 7.0


def _valid(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["pitch_x", "pitch_y"])


def team_players(df: pd.DataFrame, team_id: int,
                 classes: tuple[str, ...] = PLAYER_CLASSES) -> pd.DataFrame:
    """Rows for a team's players (outfield + keeper by default) with valid pitch."""
    return _valid(df[(df["team_id"] == team_id)
                     & (df["class_name"].isin(classes))])


def track_paths(df: pd.DataFrame, team_id: int) -> dict[object, pd.DataFrame]:
    """Per-track chronological position table for a team.

    Returns {track_id: DataFrame[timestamp_sec, pitch_x, pitch_y]} sorted by
    time. Track ids are kept verbatim (ints from ByteTrack, or other keys).
    """
    sub = team_players(df, team_id)
    out: dict[object, pd.DataFrame] = {}
    for tid, grp in sub.groupby("track_id"):
        grp = grp.sort_values("timestamp_sec")
        out[tid] = grp[["timestamp_sec", "pitch_x", "pitch_y"]].reset_index(
            drop=True)
    return out


def segment_speeds(path: pd.DataFrame) -> np.ndarray:
    """Instantaneous speeds (m/s) between consecutive samples of a track."""
    if len(path) < 2:
        return np.asarray([])
    dx = np.diff(path["pitch_x"].to_numpy())
    dy = np.diff(path["pitch_y"].to_numpy())
    dt = np.diff(path["timestamp_sec"].to_numpy())
    dt = np.where(dt <= 0.0, np.nan, dt)
    speeds = np.hypot(dx, dy) / dt
    return speeds[np.isfinite(speeds)]


def plausible(speeds: np.ndarray) -> np.ndarray:
    """Drop implausibly fast per-frame jumps (glitches)."""
    return speeds[speeds <= MAX_PLAUSIBLE_SPEED_MS]
