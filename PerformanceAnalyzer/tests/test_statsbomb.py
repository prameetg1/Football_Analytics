"""Stage 7: StatsBomb ground-truth gate — alignment, validation, gating.

Driven by synthetic StatsBomb-shaped GT (no network), proving the CV-vs-GT
comparator and the MetricsGate refuse-to-emit behaviour.
"""

import numpy as np
import pandas as pd
import pytest

from src.analytics.adapters.video.metrics import MetricsGate, MetricsRunner, MetricGateBlocked
from src.analytics.adapters.video.metrics.base import TrackingFrame
from src.analytics.adapters.statsbomb.alignment import convert_pitch_xy, seconds_from_timestamp
from src.analytics.adapters.statsbomb.validate import (
    ValidationReport,
    detect_passes,
    nearest_assignment_mean_error,
    pass_match,
    position_error,
    validate_cv_against_gt,
)


def _frame(df, ts, rows):
    return df[df["timestamp_sec"] == ts]


def _cv_df():
    """Synthetic CV tracking frame: 4 frames, 2 teams x 2 players + ball."""
    rows = []
    ts = [0.0, 0.2, 0.4, 0.6]
    for i, t in enumerate(ts):
        for team, x in ((0, 10.0), (1, 60.0)):
            for j in (0, 1):
                rows.append({"frame_idx": i, "timestamp_sec": t,
                             "track_id": f"t{team}p{j}", "team_id": team,
                             "pitch_x": x + j * 5, "pitch_y": 30.0 + j * 10,
                             "class_name": "player"})
        rows.append({"frame_idx": i, "timestamp_sec": t,
                     "track_id": "ball", "team_id": np.nan,
                     "pitch_x": 20.0 + i * 8, "pitch_y": 35.0,
                     "class_name": "ball"})
    return pd.DataFrame(rows, columns=TrackingFrame.REQUIRED_COLUMNS)


def _gt_df():
    """The GT frame used across position-error tests (== _cv_df)."""
    return _cv_df()


def _gt_shift(cv: pd.DataFrame, dx: float = 0.5, dy: float = 0.0) -> pd.DataFrame:
    """A GT copy offset by (dx, dy) — simulates near-perfect CV-vs-GT."""
    g = cv.copy()
    g["pitch_x"] = g["pitch_x"] + dx
    g["pitch_y"] = g["pitch_y"] + dy
    return g


class TestAlignment:
    def test_seconds_from_timestamp(self):
        assert seconds_from_timestamp("00:00:00.000") == 0.0
        assert seconds_from_timestamp("00:02:11.500") == 13.5
        assert seconds_from_timestamp("01:02:03.500") == 65.5

    def test_convert_pitch_y_rescaled_to_70(self):
        x, y = convert_pitch_xy(60.0, 40.0, attacking=False)
        assert abs(y - 35.0) < 1e-9            # 40 * 70/80
        assert x == pytest.approx(60.0)

    def test_mirror_when_attacking(self):
        x, _ = convert_pitch_xy(90.0, 40.0, attacking=True)
        assert x == pytest.approx(30.0)        # 120 - 90


class TestPositionError:
    def test_perfect_alignment_zero_error(self):
        cv = _gt_df()
        assert position_error(cv, _gt_shift(cv, 0.0, 0.0)) == pytest.approx(0.0)

    def test_small_shift_reported(self):
        gt = _gt_df()
        cv = _gt_shift(gt, 2.0, 3.0)
        err = position_error(cv, gt)
        assert err == pytest.approx(np.hypot(2.0, 3.0), abs=1e-6)

    def test_nearest_assignment_ignores_unmatched(self):
        cv = np.array([[0.0, 0.0], [5.0, 0.0]])
        gt = np.array([[0.0, 0.0]])
        assert nearest_assignment_mean_error(cv, gt) == pytest.approx(0.0)


class TestPassMatch:
    def test_recall_precision_on_identical(self):
        passes = [(1.0, 20.0, 30.0), (5.0, 40.0, 30.0)]
        p, r = _pass_scores(passes, passes)
        assert p == pytest.approx(1.0) and r == pytest.approx(1.0)

    def test_missing_pass_hurts_recall(self):
        gt = [(1.0, 20.0, 30.0), (5.0, 40.0, 30.0)]
        cv = [(1.0, 20.0, 30.0)]                # missed the 5s pass
        p, r = _pass_scores(cv, gt)
        assert r == pytest.approx(0.5)
        assert p == pytest.approx(1.0)

    def test_false_positive_hurts_precision(self):
        gt = [(1.0, 20.0, 30.0)]
        cv = [(1.0, 20.0, 30.0), (100.0, 50.0, 50.0)]  # errant extra pass
        p, r = _pass_scores(cv, gt)
        assert r == pytest.approx(1.0)
        assert p == pytest.approx(0.5)


def _pass_scores(cv_passes, gt_passes):
    from src.analytics.adapters.statsbomb.validate import pass_match
    return pass_match(cv_passes, gt_passes)


class TestMetricsGate:
    def test_disabled_gate_allows(self):
        gate = MetricsGate(enabled=False)
        assert bool(gate) is True

    def test_enabled_without_report_blocks(self):
        gate = MetricsGate(enabled=True)        # report None
        assert bool(gate) is False
        with pytest.raises(MetricGateBlocked):
            gate.check()

    def test_pass_report_allows(self):
        gate = MetricsGate(enabled=True)
        gate.set_report(ValidationReport(
            position_error_m=2.0, pass_precision=0.8, pass_recall=0.8,
            n_gt_passes=5, n_cv_passes=5))
        assert bool(gate) is True
        gate.check()                            # no raise

    def test_fail_report_blocks(self):
        report = ValidationReport(
            position_error_m=20.0, pass_precision=0.1, pass_recall=0.05,
            n_gt_passes=20, n_cv_passes=5)
        gate = MetricsGate(enabled=True)
        gate.set_report(report)
        assert report.passes is False
        with pytest.raises(MetricGateBlocked):
            gate.check()

    def test_runner_emits_making_metric(self):
        cv = _gt_df()
        runner = MetricsRunner(gate=MetricsGate(enabled=True))
        runner.gate.set_report(ValidationReport(
            position_error_m=1.0, pass_precision=0.9, pass_recall=0.9,
            n_gt_passes=5, n_cv_passes=5))
        out = runner.compute(cv)
        assert "VelocityMetric" in out.index
        assert list(out.columns) == ["description", "team_0", "team_1"]


def test_full_validate_returns_report():
    gt = _gt_df()
    cv = _gt_shift(gt, 0.5, 0.0)
    report = validate_cv_against_gt(cv, gt)
    assert report.position_error_m == pytest.approx(0.5, abs=1e-6)
    # ball moves >1m each frame -> both detect the same passes
    assert report.pass_recall == pytest.approx(1.0)
    assert report.pass_precision == pytest.approx(1.0)