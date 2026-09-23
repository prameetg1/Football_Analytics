"""Stage 9, Stage D: team-shape analytics — spread, stretch, compactness,
entropy, and the runner's default metric set.

Synthetic frames with known geometry (a square vs a spread line) prove each
shape metric's numeric behaviour; the runner test confirms the new metrics ship
by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.adapters.video.metrics import (
    BlockPositionMetric,
    CompactnessMetric,
    FormationAngleMetric,
    MetricsRunner,
    MomentOfInertiaMetric,
    SpatialEntropyMetric,
    SpreadMetric,
    StretchIndexMetric,
)


def make_shape_frame(team_x: list[float], team_y: list[float],
                     n_frames: int = 3) -> pd.DataFrame:
    """n_frames where one team sits at the given (x, y) positions."""
    rows = []
    for i in range(n_frames):
        for j, (x, y) in enumerate(zip(team_x, team_y)):
            rows.append({
                "frame_idx": i, "timestamp_sec": i / 5.0,
                "track_id": f"p{j}", "team_id": 0, "class_name": "player",
                "pitch_x": x, "pitch_y": y,
            })
    if not rows:
        return pd.DataFrame(columns=["frame_idx", "timestamp_sec", "track_id",
                                     "team_id", "class_name", "pitch_x",
                                     "pitch_y"])
    return pd.DataFrame(rows)


class TestSpread:
    def test_zero_spread_for_stacked_players(self):
        df = make_shape_frame([10.0] * 4, [35.0] * 4)
        assert SpreadMetric().compute(df, 0) == pytest.approx(0.0)

    def test_spread_scales_with_distance(self):
        # players at x = 10 and 30, y = 35 -> centroid x = 20, spread = 10
        df = make_shape_frame([10.0, 30.0, 10.0, 30.0],
                              [35.0, 35.0, 35.0, 35.0])
        assert SpreadMetric().compute(df, 0) == pytest.approx(10.0)

    def test_none_when_no_data(self):
        assert SpreadMetric().compute(make_shape_frame([], []), 0) is None


class TestStretchIndex:
    def test_square(self):
        # x extent 10 (10..20), y extent 10 (30..40) -> 10 * 10 = 100
        df = make_shape_frame([10.0, 20.0, 10.0, 20.0],
                              [30.0, 30.0, 40.0, 40.0])
        assert StretchIndexMetric().compute(df, 0) == pytest.approx(100.0)

    def test_line_has_zero_area(self):
        df = make_shape_frame([10.0, 20.0, 30.0, 40.0],
                              [35.0, 35.0, 35.0, 35.0])
        assert StretchIndexMetric().compute(df, 0) == pytest.approx(0.0)


class TestCompactness:
    def test_compactness_zero_for_duplicates(self):
        df = make_shape_frame([10.0] * 3, [35.0] * 3)
        assert CompactnessMetric().compute(df, 0) == pytest.approx(0.0)

    def test_pairwise_compactness(self):
        # pairwise distances (6 pairs):
        #   (10,35)-(20,35)=10, (10,35)-(10,40)=5, (10,35)-(20,40)=sqrt(125)
        #   (20,35)-(10,40)=sqrt(125), (20,35)-(20,40)=5, (10,40)-(20,40)=10
        df = make_shape_frame([10.0, 20.0, 10.0, 20.0],
                              [35.0, 35.0, 40.0, 40.0])
        expected = (10 + 5 + np.sqrt(125) + np.sqrt(125) + 5 + 10) / 6
        assert CompactnessMetric().compute(df, 0) == pytest.approx(expected, abs=0.02)

    def test_more_compact_smaller_value(self):
        tight = make_shape_frame([10.0, 11.0, 10.0, 11.0],
                                 [35.0, 35.0, 36.0, 36.0])
        loose = make_shape_frame([10.0, 20.0, 10.0, 20.0],
                                 [35.0, 35.0, 40.0, 40.0])
        assert (CompactnessMetric().compute(tight, 0)
                < CompactnessMetric().compute(loose, 0))


class TestSpatialEntropy:
    def test_entropy_zero_for_stacked_players(self):
        df = make_shape_frame([10.0] * 4, [35.0] * 4)
        assert SpatialEntropyMetric().compute(df, 0) == pytest.approx(0.0)

    def test_more_dispersion_higher_entropy(self):
        spread = make_shape_frame([10.0, 100.0, 30.0, 70.0],
                                  [20.0, 50.0, 30.0, 60.0])
        clustered = make_shape_frame([10.0, 12.0, 11.0, 10.5],
                                     [35.0, 35.5, 34.5, 35.0])
        assert (SpatialEntropyMetric().compute(spread, 0)
                > SpatialEntropyMetric().compute(clustered, 0))

    def test_entropy_bounded_by_one(self):
        df = make_shape_frame([10.0, 40.0, 80.0, 115.0],
                              [10.0, 30.0, 50.0, 60.0])
        val = SpatialEntropyMetric().compute(df, 0)
        assert 0.0 <= val <= 1.0


def test_runner_defaults_include_shape_metrics():
    runner = MetricsRunner()
    names = {m.name for m in runner.metrics}
    assert {"SpreadMetric", "StretchIndexMetric",
            "CompactnessMetric", "SpatialEntropyMetric",
            "BlockPositionMetric", "MomentOfInertiaMetric",
            "FormationAngleMetric"} <= names
    df = make_shape_frame([10.0, 30.0], [35.0, 35.0])
    table = runner.compute(df)
    for name in ("SpreadMetric", "StretchIndexMetric",
                 "CompactnessMetric", "SpatialEntropyMetric",
                 "BlockPositionMetric", "MomentOfInertiaMetric",
                 "FormationAngleMetric"):
        assert name in table.index


class TestBlockPosition:
    def test_midfield_block_between_lines(self):
        # back line x=10..12, midfield x=40/50, front line x=80..95
        df = make_shape_frame([10.0, 12.0, 11.0, 40.0, 50.0, 80.0, 90.0, 95.0],
                              [35.0] * 8)
        assert BlockPositionMetric().compute(df, 0) == pytest.approx(45.0)

    def test_none_when_too_few_players(self):
        df = make_shape_frame([10.0, 20.0, 30.0], [35.0] * 3)
        assert BlockPositionMetric().compute(df, 0) is None


class TestMomentOfInertia:
    def test_zero_for_stacked_players(self):
        df = make_shape_frame([10.0] * 4, [35.0] * 4)
        assert MomentOfInertiaMetric().compute(df, 0) == pytest.approx(0.0)

    def test_radius_of_gyration(self):
        # two players at x=10 and x=30 -> centroid 20, r2 = 100 each, mean 100
        df = make_shape_frame([10.0, 30.0], [35.0, 35.0])
        assert MomentOfInertiaMetric().compute(df, 0) == pytest.approx(10.0)


class TestFormationAngle:
    def test_vertical_line_is_90_degrees(self):
        df = make_shape_frame([20.0, 20.0, 20.0], [20.0, 35.0, 50.0])
        assert FormationAngleMetric().compute(df, 0) == pytest.approx(90.0)

    def test_horizontal_line_is_0_degrees(self):
        df = make_shape_frame([10.0, 40.0, 70.0], [35.0, 35.0, 35.0])
        assert FormationAngleMetric().compute(df, 0) == pytest.approx(0.0)
