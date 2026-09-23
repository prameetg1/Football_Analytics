"""Tests for the Stage-3 metric layer using synthetic tracking frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.adapters.video.metrics import (
    LineHeightMetric,
    MetricsRunner,
    PossessionMetric,
    SprintDistanceMetric,
    TotalDistanceMetric,
    VelocityMetric,
)

FPS = 25.0


def make_frame(n: int) -> pd.DataFrame:
    """n frames; one ball; team 0 = 3 outfield + 1 keeper, team 1 = 2 outfield."""
    tracks = {0: [("player", 0), ("player", 1), ("player", 2), ("goalkeeper", 3)],
              1: [("player", 3), ("player", 4)]}
    rows = []
    for i in range(n):
        t = i / FPS
        for tid, members in tracks.items():
            for tidx, (cls, j) in enumerate(members):
                rows.append({
                    "frame_idx": i, "timestamp_sec": t,
                    "track_id": f"{tid}_{cls}_{tidx}",
                    "team_id": tid, "class_name": cls,
                    "pitch_x": 10.0 + i * 0.04, "pitch_y": tid * 20.0 + j * 0.1,
                })
        rows.append({
            "frame_idx": i, "timestamp_sec": t, "track_id": "ball",
            "team_id": None, "class_name": "ball",
            "pitch_x": 10.0 + i * 0.04, "pitch_y": 0.5,
        })
    return pd.DataFrame(rows)


def test_velocity_metric():
    df = make_frame(50)
    val = VelocityMetric().compute(df, 0)
    assert val is not None
    assert val == pytest.approx(1.0, abs=0.05)


def test_total_distance_metric():
    df = make_frame(26)  # 1 second of motion
    val = TotalDistanceMetric().compute(df, 1)
    assert val is not None
    assert val == pytest.approx(2.0, abs=0.05)  # 2 players x 1 m/s x 1 s


def test_sprint_distance_metric_zero_when_slow():
    df = make_frame(26)
    val = SprintDistanceMetric().compute(df, 0)
    assert val is None  # nothing above 7 m/s


def test_sprint_distance_metric_counts_fast_segments():
    t = np.arange(10) / FPS
    rows = [{"frame_idx": i, "timestamp_sec": ti, "track_id": 1, "team_id": 0,
             "class_name": "player", "pitch_x": 8.0 * ti, "pitch_y": 0.0}
            for i, ti in enumerate(t)]
    df = pd.DataFrame(rows)
    val = SprintDistanceMetric().compute(df, 0)
    assert val is not None
    assert val == pytest.approx(8.0 * 9 / FPS, abs=0.1)  # 9 segments x 0.32 m


def test_possession_metric_assigns_near_team():
    df = make_frame(30)
    val = PossessionMetric().compute(df, 0)
    assert val is not None
    assert val > 0.9  # ball hugs team-0 track (y=0.5 vs y=20)


def test_possession_metric_hysteresis_blocks_flicker():
    df = make_frame(30)
    # Ball flickers to team-1's side (y=20) for only 3 frames -> no switch commits.
    mask = (df["class_name"] == "ball") & (df["frame_idx"].between(10, 12))
    df.loc[mask, "pitch_y"] = 20.0
    val0 = PossessionMetric(hysteresis_frames=5).compute(df, 0)
    val1 = PossessionMetric(hysteresis_frames=5).compute(df, 1)
    assert val0 > 0.9  # 3-frame dip never commits a switch
    assert val1 < 0.1


def test_possession_metric_switches_after_hysteresis():
    df = make_frame(30)
    # Ball on team-1's side for 10 consecutive frames -> switch commits ~half.
    mask = (df["class_name"] == "ball") & (df["frame_idx"].between(10, 19))
    df.loc[mask, "pitch_y"] = 20.0
    val1 = PossessionMetric(hysteresis_frames=5).compute(df, 1)
    assert val1 is not None
    assert 0.1 < val1 < 0.7  # 10-frame stint mostly counts (5+ commit)


def test_possession_metric_no_ball_returns_none():
    df = make_frame(5)
    df = df[df["class_name"] != "ball"]
    assert PossessionMetric().compute(df, 0) is None


def test_line_height_metric():
    df = make_frame(10)
    # team 0 defends x=0 -> its line sits ~10m; team 1 defending x=0 is deeper.
    val0 = LineHeightMetric(defend_x=0.0).compute(df, 0)
    assert val0 is not None
    assert val0 == pytest.approx(10.18, abs=0.1)  # mean of 3 deepest x = 10 + 0.04*mean(i)


def test_line_height_requires_enough_players():
    df = make_frame(5)
    df = df[df["team_id"] != 1]
    assert LineHeightMetric(n_defenders=3).compute(df, 0) is not None
    assert LineHeightMetric(n_defenders=10).compute(df, 0) is None


def test_line_height_excludes_goalkeeper():
    """Keeper always sits deepest; including him would collapse the line."""
    df = make_frame(10)
    # push team-0's keeper far back, outfield players stay at x=10
    gk = df[(df["class_name"] == "goalkeeper") & (df["team_id"] == 0)]
    df.loc[gk.index, "pitch_x"] = 1.0
    outfield = LineHeightMetric(defend_x=0.0, n_defenders=3).compute(df, 0)
    keeper_incl = LineHeightMetric(defend_x=0.0, n_defenders=3,
                                   include_goalkeeper=True).compute(df, 0)
    assert outfield is not None
    assert keeper_incl is not None
    # without keeper the line is ~10.18; with keeper it drops toward 1.0
    assert outfield == pytest.approx(10.18, abs=0.1)
    assert keeper_incl < outfield


def test_velocity_includes_goalkeeper():
    df = make_frame(20)
    assert VelocityMetric().compute(df, 0) == pytest.approx(1.0, abs=0.05)


def test_runner_report_schema():
    df = make_frame(20)
    report = MetricsRunner().compute(df)
    assert list(report.columns) == ["description", "team_0", "team_1"]
    assert "VelocityMetric" in report.index
    assert report.loc["PossessionMetric", "team_0"] is not None
    assert report.loc["VelocityMetric", "team_0"] == pytest.approx(
        report.loc["VelocityMetric", "team_1"])


def test_runner_validates_schema():
    df = make_frame(5).drop(columns=["team_id"])
    with pytest.raises(ValueError, match="team_id"):
        MetricsRunner().compute(df)
