"""Stage 4 — ball-hypothesis Kalman tracker + export routing."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.adapters.video.tracking.ball_tracker import BallTracker


def _move(pts, fps=25.0):
    """Feed straight-line measurements; return a BallTracker with them tracked."""
    t = BallTracker(dt=1.0 / fps)
    for p in pts:
        t.step(p)
    return t


class TestInitAndTrack:
    def test_tracks_constant_velocity_line(self):
        t = BallTracker(dt=1.0 / 25.0)
        line = [(float(i), 0.0) for i in range(60)]
        pos = None
        for p in line:
            pos, valid = t.step(np.array(p))
            assert valid
        # Filtered position tracks the last measurement closely after warm-up.
        assert abs(pos[0] - 59.0) < 1.0
        assert abs(pos[1]) < 0.3
        # Velocity converges toward the true speed (1 m / frame * 25 = 25 m/s).
        vx = t.velocity[0]
        assert 22.0 < vx < 28.0

    def test_no_measurement_yet_means_no_hypothesis(self):
        t = BallTracker()
        assert t.step(None) == (None, False)
        assert t.position is None


class TestPredictionThroughOcclusion:
    def _seeded(self):
        t = BallTracker(dt=1.0 / 25.0, max_predict_frames=5)
        for p in [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]:
            t.step(np.array(p))
        return t

    def test_predicts_valid_while_within_limit(self):
        t = self._seeded()
        prev = None
        for _ in range(5):
            pos, valid = t.step(None)
            assert valid, "hypothesis should stay valid while predicting"
            assert pos is not None
            if prev is not None:
                assert pos[0] > prev - 1e-6  # extrapolates forward, not back
            prev = pos[0]

    def test_declares_lost_after_limit(self):
        t = self._seeded()
        for _ in range(5):
            t.step(None)
        pos, valid = t.step(None)  # 6th prediction-only frame
        assert valid is False and pos is None


class TestAssociationGate:
    def test_single_glitch_frame_is_ignored(self):
        t = BallTracker(dt=1.0 / 25.0, association_gate_m=8.0)
        for _ in range(5):
            t.step(np.array([0.0, 0.0]))
        # A teleporting/mis-projected measurement (H glitch) must NOT jump the
        # track when it has been tracking steadily.
        pos, valid = t.step(np.array([50.0, 50.0]))
        assert pos[0] < 2.0 and pos[1] < 2.0

    def test_reappearing_ball_after_miss_reanchors(self):
        t = BallTracker(dt=1.0 / 25.0, association_gate_m=8.0)
        for p in [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]:
            t.step(np.array(p))
        t.step(None)
        t.step(None)  # two missed frames -> next far measurement is a new ball
        pos, valid = t.step(np.array([30.0, 40.0]))
        assert valid
        assert pos[0] == pytest.approx(30.0, abs=1e-9)
        assert pos[1] == pytest.approx(40.0, abs=1e-9)
        # velocity reset by re-anchor
        assert t.velocity[0] < 1e-6

    def test_tight_gate_rejects_close_dual_consistently(self):
        t = BallTracker(dt=1.0 / 25.0, association_gate_m=1.0)
        for _ in range(3):
            t.step(np.array([0.0, 0.0]))
        pos, valid = t.step(np.array([5.0, 0.0]))
        assert pos[0] < 3.0  # outside tiny gate, ignored as glitch first frame


class _FakeDetector:
    def __init__(self, weights=None):
        pass

    def infer(self, frame):
        from src.analytics.adapters.video.detection.detection import Detection
        return [
            Detection(class_id=2, class_name="player", confidence=0.9,
                      xyxy=(10, 10, 30, 60)),
            Detection(class_id=0, class_name="ball", confidence=0.7,
                      xyxy=(20, 20, 28, 28)),
        ]


class _FakeKP:
    def detect(self, frame):
        return None


class _FakeMapper:
    current = None
    config = None

    def __init__(self, config=None):
        self.config = config

    def filter_keypoints(self, kp):
        return kp


def _write_clip(path, n=6, fps=25.0):
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                        fps, (64, 64))
    for _ in range(n):
        w.write(np.zeros((64, 64, 3), dtype=np.uint8))
    w.release()


class TestBallExportRouting:
    def test_hypothesis_row_exported_with_tracker_id(self, tmp_path, monkeypatch):
        """With a valid hypothesis the ball row carries pitch + BALL_TRACKER_ID."""
        from src.config import BALL_TRACKER_ID
        from src.analytics.adapters.video.pipeline import Pipeline, TrackedFrame

        video = tmp_path / "clip.mp4"
        _write_clip(video)

        monkeypatch.setattr("src.analytics.adapters.video.detection.detector.Detector", _FakeDetector)
        monkeypatch.setattr("src.analytics.adapters.video.homography.keypoint_detector.PitchKeypointDetector",
                            _FakeKP)
        monkeypatch.setattr("src.analytics.adapters.video.homography.homography.StablePitchMapper",
                            _FakeMapper)
        monkeypatch.setattr(Pipeline, "_project",
                            lambda self, dets, frame: {})
        monkeypatch.setattr(Pipeline, "_track_ball",
                            lambda self, dets, frame, dt: (np.array([10.0, 20.0]), True))

        pipe = Pipeline(detector_weights="fake", track=False,
                        use_homography=True, assign_teams=False)
        df = pipe.to_dataframe(str(video), analysis_hz=0)

        balls = df[df["class_name"] == "ball"]
        assert len(balls) == 6                      # one per frame
        assert (balls["track_id"] == BALL_TRACKER_ID).all()
        assert (balls["pitch_x"] == 10.0).all()
        assert (balls["pitch_y"] == 20.0).all()
        # players + ball = 12 rows
        assert len(df) == 12

    def test_no_hypothesis_falls_back_to_raw_ball_row(self, tmp_path, monkeypatch):
        """Without homography/ball-tracking the raw ball row stays (pitch NaN)."""
        from src.analytics.adapters.video.pipeline import Pipeline

        video = tmp_path / "clip.mp4"
        _write_clip(video)

        monkeypatch.setattr("src.analytics.adapters.video.detection.detector.Detector", _FakeDetector)
        pipe = Pipeline(detector_weights="fake", track=False,
                        use_homography=False, assign_teams=False)
        df = pipe.to_dataframe(str(video), analysis_hz=0)

        assert len(df) == 12                        # player + raw ball rows
        balls = df[df["class_name"] == "ball"]
        assert len(balls) == 6
        assert balls["pitch_x"].isna().all()        # no homography -> NaN