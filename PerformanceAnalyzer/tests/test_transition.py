"""Stage 9, Stage D: transition-shape analytics (D11, D12)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.adapters.video.events import Event
from src.analytics.adapters.video.metrics.base import TrackingFrame
from src.analytics.adapters.video.metrics.transition import (
    formation_recovery_frames,
    spread_series,
    transition_stretch,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=TrackingFrame.REQUIRED_COLUMNS)


def _recovery(losing_team: int, t: float) -> Event:
    return Event(kind="Ball Recovery", timestamp_sec=t, x=0.0, y=0.0,
                 team_id=1 - losing_team, qualifiers={"losing_team": losing_team})


def _two_team_frames(n_frames: int,
                     compact: bool = True) -> pd.DataFrame:
    """Team 0 clustered (or spread) at x=50; team 1 far away at x=100."""
    rows = []
    if compact:
        t0 = [(50.0, 30.0), (55.0, 40.0), (45.0, 35.0)]
    else:
        t0 = [(30.0, 10.0), (70.0, 20.0), (50.0, 60.0)]
    for i in range(n_frames):
        rows.append({"frame_idx": i, "timestamp_sec": float(i),
                     "track_id": "ball", "team_id": np.nan,
                     "pitch_x": 50.0, "pitch_y": 35.0, "class_name": "ball"})
        for j, (px, py) in enumerate(t0):
            rows.append({"frame_idx": i, "timestamp_sec": float(i),
                         "track_id": f"t0{j}", "team_id": 0,
                         "pitch_x": px, "pitch_y": py, "class_name": "player"})
        for j, dy in enumerate((-5.0, 0.0, 5.0)):
            rows.append({"frame_idx": i, "timestamp_sec": float(i),
                         "track_id": f"t1{j}", "team_id": 1,
                         "pitch_x": 100.0, "pitch_y": 35.0 + dy,
                         "class_name": "player"})
    return _frame(rows)


class TestSpreadSeries:
    def test_compact_team_smaller_spread(self):
        compact = spread_series(_two_team_frames(5, compact=True), 0)
        spread = spread_series(_two_team_frames(5, compact=False), 0)
        assert compact.mean() < spread.mean()

    def test_empty_team(self):
        assert spread_series(_two_team_frames(3), 7).empty


class TestTransitionStretch:
    def test_static_shape_no_delta(self):
        df = _two_team_frames(20)
        res = transition_stretch(df, [_recovery(0, 10.0)], 0)
        assert res["n"] == 1
        assert res["delta_m"] == pytest.approx(0.0)
        assert res["post_m"] > 0.0

    def test_ignores_turnovers_won(self):
        df = _two_team_frames(20)
        # losing_team == 1 -> team 0 won the ball, not lost it
        res = transition_stretch(df, [_recovery(1, 10.0)], 0)
        assert res["n"] == 0


class TestFormationRecovery:
    def test_static_shape_recovers_immediately(self):
        df = _two_team_frames(20)
        res = formation_recovery_frames(df, [_recovery(0, 10.0)], 0)
        assert res["n"] == 1
        assert res["recovered_share"] == pytest.approx(1.0)
        assert res["mean_frames"] == pytest.approx(1.0)

    def test_ignores_events_when_recovering(self):
        df = _two_team_frames(20)
        res = formation_recovery_frames(df, [_recovery(1, 10.0)], 0)
        assert res["n"] == 0
