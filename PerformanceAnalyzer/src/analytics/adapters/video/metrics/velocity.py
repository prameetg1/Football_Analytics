"""Velocity / speed metric."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.adapters.video.metrics.base import MetricBase
from src.analytics.adapters.video.metrics.util import plausible, segment_speeds, track_paths


class VelocityMetric(MetricBase):
    """Mean on-pitch speed (m/s) of a team's players across the clip."""

    def compute(self, df: pd.DataFrame, team_id: int) -> float | None:
        all_speeds: list[float] = []
        for path in track_paths(df, team_id).values():
            all_speeds.extend(plausible(segment_speeds(path)).tolist())
        if not all_speeds:
            return None
        return float(np.mean(all_speeds))

    def description(self) -> str:
        return "Mean player speed (m/s)"
