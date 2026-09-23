"""Stage 9, Stage C: pitch control, EPV, threat and territory."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.adapters.video.metrics.base import TrackingFrame
from src.analytics.models.pitch_control import (
    _spearman_surface,
    _track_velocities,
    epv_grid,
    expected_threat_grid,
    field_tilt,
    pitch_control_surface,
    space_creation_runs,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=TrackingFrame.REQUIRED_COLUMNS)


def _two_team_frame() -> pd.DataFrame:
    """Two clusters: team 0 deep (x=20), team 1 advanced (x=100)."""
    rows = []
    for i in range(2):  # two frames
        for team, x in ((0, 20.0), (1, 100.0)):
            for j, dy in enumerate((-10.0, 0.0, 10.0)):
                rows.append({"frame_idx": i, "timestamp_sec": float(i),
                             "track_id": f"t{team}p{j}", "team_id": team,
                             "pitch_x": x, "pitch_y": 35.0 + dy,
                             "class_name": "player"})
        rows.append({"frame_idx": i, "timestamp_sec": float(i),
                     "track_id": "ball", "team_id": np.nan,
                     "pitch_x": 60.0, "pitch_y": 35.0, "class_name": "ball"})
    return _frame(rows)


class TestPitchControlSurface:
    def test_shape_and_range(self):
        df = _two_team_frame()
        grid = pitch_control_surface(df, 0)
        assert grid.shape == (16, 24)
        assert grid.min() >= 0.0 and grid.max() <= 1.0

    def test_team0_controls_own_half(self):
        df = _two_team_frame()
        grid = pitch_control_surface(df, 0)
        # near team 0's cluster (x=20) it should own the ball
        assert grid[8, 4] > 0.5          # y~35, x~20
        # near team 1's cluster (x=100) team 0 should not
        assert grid[8, 20] < 0.5

    def test_empty_team_returns_nan(self):
        df = _two_team_frame()
        grid = pitch_control_surface(df, 7)
        assert np.isnan(grid).all()


class TestEPV:
    def test_epv_shape(self):
        v = epv_grid()
        assert v.shape == (16, 24)

    def test_epv_increases_toward_goal(self):
        v = epv_grid()
        # value near the attacking goal (x=120) > value at the halfway line
        assert v[8, 23] > v[8, 12]

    def test_epv_bounded(self):
        v = epv_grid()
        assert 0.0 <= v.max() <= 1.0


class TestThreat:
    def test_threat_grid_shape(self):
        t = expected_threat_grid()
        assert t.shape == (16, 24)
        assert t.min() >= 0.0

    def test_threat_higher_in_final_third(self):
        t = expected_threat_grid()
        final = t[8, 20:]                # x > 100
        middle = t[8, 8:12]              # x ~ 40-60
        assert final.mean() > middle.mean()


class TestFieldTilt:
    def test_mean_ball_x_when_owned(self):
        df = _two_team_frame()
        # team 0 owns ball only when a team-0 player is nearest; here ball at
        # x=60 is far from both clusters, so owner is undefined -> 0.0
        assert field_tilt(df, 0) == pytest.approx(0.0)

    def test_tilt_reflects_territory(self):
        rows = []
        # team 0 owns the ball deep in the middle third at x=60
        for i, t in enumerate((0.0, 1.0)):
            rows.append({"frame_idx": i, "timestamp_sec": t,
                         "track_id": "ball", "team_id": np.nan,
                         "pitch_x": 60.0, "pitch_y": 35.0,
                         "class_name": "ball"})
            rows.append({"frame_idx": i, "timestamp_sec": t,
                         "track_id": "p0", "team_id": 0,
                         "pitch_x": 59.0, "pitch_y": 35.0,
                         "class_name": "player"})
            rows.append({"frame_idx": i, "timestamp_sec": t,
                         "track_id": "q0", "team_id": 1,
                         "pitch_x": 100.0, "pitch_y": 35.0,
                         "class_name": "player"})
        assert field_tilt(_frame(rows), 0) == pytest.approx(60.0)


def _space_frame() -> pd.DataFrame:
    """Team 0 has a runner sweeping across open space; team 1 clumped left."""
    rows = []
    runner = [(20.0, 30.0), (45.0, 30.0), (70.0, 30.0), (105.0, 30.0)]
    for i, (rx, ry) in enumerate(runner):
        rows.append({"frame_idx": i, "timestamp_sec": float(i),
                     "track_id": "ball", "team_id": np.nan,
                     "pitch_x": 50.0, "pitch_y": 35.0, "class_name": "ball"})
        rows.append({"frame_idx": i, "timestamp_sec": float(i),
                     "track_id": "r", "team_id": 0,
                     "pitch_x": rx, "pitch_y": ry, "class_name": "player"})
        # team 1 is clumped near its goal (x=5), leaving the space empty
        for j in range(4):
            rows.append({"frame_idx": i, "timestamp_sec": float(i),
                         "track_id": f"o{j}", "team_id": 1,
                         "pitch_x": 5.0 + j, "pitch_y": 30.0 + j,
                         "class_name": "player"})
    return _frame(rows)


class TestSpaceCreation:
    def test_runner_across_open_space_detected(self):
        runs = space_creation_runs(_space_frame(), 0)
        assert len(runs) == 1
        r = runs[0]
        assert r["distance_m"] == pytest.approx(85.0, abs=1.0)
        assert r["threatening"] is True      # ends at x=105 in the final third
        assert r["max_opp_control"] <= 0.3
    def test_runner_through_occupied_zone_not_detected(self):
        rows = []
        for i, x in enumerate((20.0, 40.0, 60.0)):
            rows.append({"frame_idx": i, "timestamp_sec": float(i),
                         "track_id": "ball", "team_id": np.nan,
                         "pitch_x": x, "pitch_y": 35.0, "class_name": "ball"})
            rows.append({"frame_idx": i, "timestamp_sec": float(i),
                         "track_id": "r", "team_id": 0,
                         "pitch_x": x, "pitch_y": 35.0, "class_name": "player"})
            for j in range(5):
                rows.append({"frame_idx": i, "timestamp_sec": float(i),
                             "track_id": f"o{j}", "team_id": 1,
                             "pitch_x": x - 2.0 + j, "pitch_y": 35.0,
                             "class_name": "player"})
        # team 1 hugs the runner the whole way -> no uncovered space
        assert space_creation_runs(_frame(rows), 0) == []

    def test_no_opponent_returns_empty(self):
        rows = []
        for i in range(2):
            rows.append({"frame_idx": i, "timestamp_sec": float(i),
                         "track_id": "ball", "team_id": np.nan,
                         "pitch_x": 50.0, "pitch_y": 35.0, "class_name": "ball"})
            rows.append({"frame_idx": i, "timestamp_sec": float(i),
                         "track_id": "r", "team_id": 0,
                         "pitch_x": 20.0 + 10 * i, "pitch_y": 35.0,
                         "class_name": "player"})
        assert space_creation_runs(_frame(rows), 0) == []


class TestSpearmanRace:
    """C1 physics: a sprinting player wins a race a static one loses."""

    def _vel_frame(self, vx: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        # team 0 at x=30 (moving, vx given), team 1 static at x=48
        rows = []
        for i in range(2):
            rows.append({"frame_idx": i, "timestamp_sec": float(i),
                         "track_id": "t0", "team_id": 0,
                         "pitch_x": 20.0 + 10.0 * i, "pitch_y": 35.0,
                         "class_name": "player"})
            rows.append({"frame_idx": i, "timestamp_sec": float(i),
                         "track_id": "t1", "team_id": 1,
                         "pitch_x": 48.0, "pitch_y": 35.0,
                         "class_name": "player"})
        df = _frame(rows)
        f1 = df[df["frame_idx"] == 1]
        own = f1[f1["team_id"] == 0]
        opp = f1[f1["team_id"] == 1]
        return own, opp

    def test_static_runner_loses_cell_closer_to_opponent(self):
        own, opp = self._vel_frame(0.0)
        vel = {}                     # no velocities -> both static
        surf = _spearman_surface(own, opp, vel, nx=24, ny=16)
        # cell centre x=44 (cell 8), y=35 (row 8): nearer team 1 (x=48)
        assert surf[8, 8] < 0.5

    def test_sprinting_runner_wins_the_same_cell(self):
        own, opp = self._vel_frame(10.0)
        vel = _track_velocities(_frame([
            {"frame_idx": 0, "timestamp_sec": 0.0, "track_id": "t0",
             "team_id": 0, "pitch_x": 20.0, "pitch_y": 35.0,
             "class_name": "player"},
            {"frame_idx": 1, "timestamp_sec": 1.0, "track_id": "t0",
             "team_id": 0, "pitch_x": 30.0, "pitch_y": 35.0,
             "class_name": "player"},
            {"frame_idx": 0, "timestamp_sec": 0.0, "track_id": "t1",
             "team_id": 1, "pitch_x": 48.0, "pitch_y": 35.0,
             "class_name": "player"},
            {"frame_idx": 1, "timestamp_sec": 1.0, "track_id": "t1",
             "team_id": 1, "pitch_x": 48.0, "pitch_y": 35.0,
             "class_name": "player"},
        ]))
        surf = _spearman_surface(own, opp, vel, nx=24, ny=16)
        # momentum (10 m/s toward x=44) flips the cell: now team 0 wins
        assert surf[8, 8] > 0.5
