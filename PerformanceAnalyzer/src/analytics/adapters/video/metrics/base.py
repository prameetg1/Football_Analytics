"""Pluggable metric interface (mirrors DefenseAnalytics' MetricBase).

A metric consumes a per-frame tracking frame (or a rolling history) for a single
team and returns a scalar. Keeping this abstract makes metrics swappable without
touching the rest of the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.analytics.common import TrackingFrame  # re-exported for consumers


class MetricBase(ABC):
    @abstractmethod
    def compute(self, df: pd.DataFrame, team_id: int) -> float | None:
        """Compute a scalar metric for `team_id` from tracking/event frame."""
        ...

    @abstractmethod
    def description(self) -> str:
        ...

    @property
    def name(self) -> str:
        return type(self).__name__