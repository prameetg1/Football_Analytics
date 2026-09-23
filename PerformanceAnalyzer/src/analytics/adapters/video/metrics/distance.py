"""Distance metrics: total covered distance and sprint distance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.adapters.video.metrics.base import MetricBase
from src.analytics.adapters.video.metrics.util import SPRINT_SPEED_MS, plausible, segment_speeds, track_paths


class TotalDistanceMetric(MetricBase):
    """Total on-pitch distance (m) covered by a team across the clip."""

    def compute(self, df: pd.DataFrame, team_id: int) -> float | None:
        total = 0.0
        for path in track_paths(df, team_id).values():
            speeds = plausible(segment_speeds(path))
            dt = np.diff(path["timestamp_sec"].to_numpy()[:len(speeds) + 1])
            total += float(np.sum(speeds * np.where(dt <= 0, 0.0, dt)))
        if total <= 0:
            return None
        return round(total, 1)

    def description(self) -> str:
        return "Total distance covered (m)"


class SprintDistanceMetric(MetricBase):
    """Distance (m) covered while moving faster than `sprint_speed` m/s."""

    def __init__(self, sprint_speed: float = SPRINT_SPEED_MS):
        self.sprint_speed = sprint_speed

    def compute(self, df: pd.DataFrame, team_id: int) -> float | None:
        total = 0.0
        for path in track_paths(df, team_id).values():
            speeds = segment_speeds(path)
            dt = np.diff(path["timestamp_sec"].to_numpy()[:len(speeds) + 1])
            dt = np.where(dt <= 0, 0.0, dt)
            sprinting = (speeds > self.sprint_speed) & (speeds <= 12.0)
            total += float(np.sum(speeds[sprinting] * dt[sprinting]))
        if total <= 0:
            return None
        return round(total, 1)

    def description(self) -> str:
        return f"Sprint distance (m) at >{self.sprint_speed:g} m/s"
