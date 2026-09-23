"""Tests for pipeline-stage adaptations adopted from the reference.

Covers: ball box padding, class-agnostic NMS before ByteTrack, pitch-position
outlier rejection, and the one-pass TeamAssigner bootstrap.
"""

import numpy as np
import pytest

from src.analytics.adapters.video.detection.detection import Detection
from src.analytics.adapters.video.pipeline import Pipeline, is_position_outlier
from src.analytics.adapters.video.teams.assigner import TeamAssigner


def _det(cls: int, name: str, xyxy, conf: float = 0.9) -> Detection:
    return Detection(class_id=cls, class_name=name, confidence=conf, xyxy=xyxy)


class TestBallPadding:
    def test_padded_expands_box_and_keeps_fields(self):
        d = _det(0, "ball", (100, 100, 120, 120), conf=0.8)
        p = d.padded(10)
        assert p.xyxy == (90, 90, 130, 130)
        assert p.class_name == "ball" and p.tracker_id is None
        # original untouched
        assert d.xyxy == (100, 100, 120, 120)

    def test_padding_clamped_at_origin(self):
        d = _det(0, "ball", (2, 3, 20, 20))
        assert d.padded(10).xyxy == (0.0, 0.0, 30.0, 30.0)


class TestOutlierRejection:
    def test_small_step_is_fine(self):
        assert not is_position_outlier(np.array([50.0, 30.0]),
                                       np.array([50.5, 30.2]))

    def test_teleport_is_outlier(self):
        assert is_position_outlier(np.array([50.0, 30.0]),
                                   np.array([60.0, 30.0]))

    def test_custom_threshold(self):
        assert is_position_outlier(np.array([0.0, 0.0]),
                                   np.array([3.0, 0.0]), max_step=5.0) is False
        assert is_position_outlier(np.array([0.0, 0.0]),
                                   np.array([6.0, 0.0]), max_step=5.0) is True


class TestClassAgnosticNMS:
    def test_overlapping_detections_are_suppressed(self):
        from src.analytics.adapters.video.tracking.tracker import ByteTrackTracker

        tr = ByteTrackTracker()
        # A player and a goalkeeper on top of each other -> NMS keeps one.
        dets = [
            _det(2, "player", (100, 100, 200, 300), conf=0.9),
            _det(1, "goalkeeper", (105, 105, 195, 295), conf=0.85),
        ]
        out = tr.update(dets)
        ids = [d.tracker_id for d in out if d.tracker_id is not None]
        assert len(ids) == 1

    def test_non_overlapping_all_kept(self):
        from src.analytics.adapters.video.tracking.tracker import ByteTrackTracker

        tr = ByteTrackTracker()
        dets = [
            _det(2, "player", (100, 100, 200, 300), conf=0.9),
            _det(2, "player", (500, 100, 600, 300), conf=0.85),
        ]
        out = tr.update(dets)
        ids = [d.tracker_id for d in out if d.tracker_id is not None]
        assert len(ids) == 2


class _FakeClassifier:
    def __init__(self):
        self.called = 0

    def fit(self, crops):
        self.called += 1
        return self

    @property
    def fitted(self):
        return True

    def predict(self, crops):
        # alternate teams by crop index
        return np.array([i % 2 for i in range(len(crops))])


class TestTeamAssigner:
    def test_bootstrap_fits_once_and_assigns_teams(self, monkeypatch):
        monkeypatch.setattr("src.analytics.adapters.video.teams.assigner.TeamClassifier", _FakeClassifier)
        a = TeamAssigner(bootstrap_crops=10)
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, size=(360, 640, 3), dtype=np.uint8)

        dets = []
        for i in range(12):  # 12 players -> bootstrap at 10
            dets.append(_det(2, "player", (10 + i, 10, 60 + i, 120)))

        a.assign(frame, dets[:6])
        assert not a.fitted                      # 6 crops < 10
        assert len(a._crops) == 6

        a.assign(frame, dets[6:])                # reaches 12 >= 10 -> fits
        assert a.fitted
        assert a._classifier.called == 1

        a.assign(frame, dets[6:])                # now predict
        assert all(d.team_id in (0, 1) for d in dets[6:])

    def test_goalkeeper_gets_nearest_team(self, monkeypatch):
        monkeypatch.setattr("src.analytics.adapters.video.teams.assigner.TeamClassifier", _FakeClassifier)
        a = TeamAssigner(bootstrap_crops=4)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        players = [
            _det(2, "player", (100, 100, 120, 200)),
            _det(2, "player", (500, 100, 520, 200)),
            _det(2, "player", (110, 100, 130, 200)),
            _det(2, "player", (490, 100, 510, 200)),
        ]
        a.assign(frame, players)  # collect 4 -> fit
        gk = _det(1, "goalkeeper", (90, 90, 110, 190))
        dets = players + [gk]
        a.assign(frame, dets)
        # GK sits next to the team-0 players (x<200) -> team 0.
        assert gk.team_id == 0


class TestTrackingExport:
    """Stage 2: Pipeline.to_dataframe() emits the TrackingFrame schema."""

    @staticmethod
    def _write_clip(path, n: int = 6, fps: float = 25.0):
        import cv2
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                            fps, (64, 64))
        for _ in range(n):
            w.write(np.zeros((64, 64, 3), dtype=np.uint8))
        w.release()

    def test_schema_rows_and_timestamps(self, tmp_path, monkeypatch):
        from src.analytics.adapters.video.metrics.base import TrackingFrame

        video = tmp_path / "clip.mp4"
        self._write_clip(video)
        calls = {"n": 0}

        class FakeDetector:
            def __init__(self, weights=None):
                pass

            def infer(self, frame):
                calls["n"] += 1
                return [
                    _det(2, "player", (10, 10, 30, 60)),
                    _det(0, "ball", (20, 20, 28, 28), conf=0.7),
                ]

        monkeypatch.setattr("src.analytics.adapters.video.detection.detector.Detector", FakeDetector)
        pipe = Pipeline(detector_weights="fake", track=False,
                        use_homography=False, assign_teams=False)
        df = pipe.to_dataframe(str(video), analysis_hz=0)  # no downsampling

        assert list(df.columns) == TrackingFrame.REQUIRED_COLUMNS
        assert len(df) == 12                    # 2 detections x 6 frames
        assert calls["n"] == 6                  # every frame processed
        assert df["track_id"].isna().all()      # no tracker wired
        assert df["pitch_x"].isna().all()       # no homography wired
        # timestamps advance by 1/fps per frame: rows 0-1 at 0.0, rows 2-3 at 0.04
        assert df["timestamp_sec"].tolist()[0] == 0.0
        assert df["timestamp_sec"].tolist()[2] == pytest.approx(1 / 25)

    def test_csv_export(self, tmp_path, monkeypatch):
        video = tmp_path / "clip.mp4"
        self._write_clip(video, n=2)

        class FakeDetector:
            def __init__(self, weights=None):
                pass

            def infer(self, frame):
                return [_det(2, "player", (10, 10, 30, 60))]

        monkeypatch.setattr("src.analytics.adapters.video.detection.detector.Detector", FakeDetector)
        pipe = Pipeline(detector_weights="fake", track=False,
                        use_homography=False, assign_teams=False)
        out = tmp_path / "track.csv"
        df = pipe.to_dataframe(str(video), out_path=str(out), analysis_hz=0)
        assert out.is_file() and out.stat().st_size > 0
        assert len(df) == 2

