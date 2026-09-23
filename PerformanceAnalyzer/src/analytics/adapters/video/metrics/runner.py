"""Runs a set of metrics over a tracking DataFrame and reports per-team values."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.analytics.adapters.video.metrics.base import MetricBase, TrackingFrame
from src.analytics.adapters.video.metrics.distance import SprintDistanceMetric, TotalDistanceMetric
from src.analytics.adapters.video.metrics.possession import PossessionMetric
from src.analytics.adapters.video.metrics.shape import (
    BlockPositionMetric,
    CompactnessMetric,
    FormationAngleMetric,
    LineHeightMetric,
    MomentOfInertiaMetric,
    SpatialEntropyMetric,
    SpreadMetric,
    StretchIndexMetric,
)
from src.analytics.adapters.video.metrics.velocity import VelocityMetric


def default_metrics(defend_x: float = 0.0) -> list[MetricBase]:
    """Canonical Stage-3 metric set + Stage-9 team-shape metrics.

    PPDA/turnover need ball-event data and are gated separately (Stage 9).
    """
    return [
        VelocityMetric(),
        TotalDistanceMetric(),
        SprintDistanceMetric(),
        PossessionMetric(),
        LineHeightMetric(defend_x=defend_x),
        SpreadMetric(),
        StretchIndexMetric(),
        CompactnessMetric(),
        SpatialEntropyMetric(),
        BlockPositionMetric(),
        MomentOfInertiaMetric(),
        FormationAngleMetric(),
    ]


@dataclass
class MetricsRunner:
    """Compute each metric per team and assemble a report table.

    Rows are metrics, columns are team ids, cells are the scalar value (NaN when
    the metric had no data for that team).

    Stage 7: an optional `gate` (MetricsGate) can refuse metric emission until
    a StatsBomb validation report passes. When no gate is provided the runner
    behaves exactly as before (ungated).
    """

    metrics: list[MetricBase] | None = None
    gate: "MetricsGate | None" = None

    def __post_init__(self) -> None:
        self.metrics = self.metrics if self.metrics is not None else default_metrics()

    def validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in TrackingFrame.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Tracking frame missing columns: {missing}")

    def teams(self, df: pd.DataFrame) -> list[int]:
        ids = sorted(int(t) for t in df["team_id"].dropna().unique())
        return ids

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        from src.analytics.adapters.video.metrics.gate import MetricsGate

        if isinstance(self.gate, MetricsGate):
            self.gate.check()          # raises MetricGateBlocked when gated
        self.validate(df)
        teams = self.teams(df)
        rows: dict[str, list[float]] = {}
        names: dict[str, str] = {}
        for metric in self.metrics:
            names[metric.name] = metric.description()
            rows[metric.name] = [
                metric.compute(df, team) for team in teams
            ]
        table = pd.DataFrame(rows, index=[f"team_{t}" for t in teams]).T
        table.insert(0, "description",
                     [names[c] for c in table.index])
        return table
