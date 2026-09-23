"""Stage 5 efficiency: adaptive frame-rate export and optional detector ROI."""

import numpy as np
import pytest

from src.analytics.adapters.video.detection.detection import Detection
from src.analytics.adapters.video.detection.detector import Detector
from src.analytics.adapters.video.pipeline import Pipeline


def _det(cls: int, name: str, xyxy, conf: float = 0.9) -> Detection:
    return Detection(class_id=cls, class_name=name, confidence=conf, xyxy=xyxy)


class _ROIFake:
    """Detector stub that records the frame it saw and returns a box at origin."""

    def __init__(self, weights=None):
        self.saw_shapes = []

    def infer(self, frame):
        self.saw_shapes.append(frame.shape)
        # box at the top-left of whatever frame was passed (crop or full)
        return [_det(2, "player", (0.0, 0.0, 50.0, 100.0))]


class TestDetectorROI:
    @staticmethod
    def _bare_detector():
        det = Detector.__new__(Detector)         # skip model load in __init__
        det.weights = "x.pt"
        det.confidence = 0.3
        det.imgsz = 1280
        det.roi = None
        det._hosted = False
        det._api_key = ""
        return det

    def test_roi_crops_then_shifts_back(self, monkeypatch):
        det = self._bare_detector()
        fake = _ROIFake()
        monkeypatch.setattr(det, "_infer_local", lambda f: fake.infer(f))
        det.roi = (0.0, 0.0, 0.5, 0.5)          # top-left quarter
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        dets = det.infer(frame)
        # detector ran on the cropped 100x200 image, box shifted by ROI offset
        assert fake.saw_shapes[-1] == (100, 200, 3)
        assert dets[0].xyxy == (0.0, 0.0, 50.0, 100.0)

    def test_roi_ignored_when_none(self, monkeypatch):
        det = self._bare_detector()
        fake = _ROIFake()
        monkeypatch.setattr(det, "_infer_local", lambda f: fake.infer(f))
        det.roi = None
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        det.infer(frame)
        assert fake.saw_shapes[-1] == (200, 400, 3)


def _write_clip(path, n: int = 6, fps: float = 25.0):
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                        fps, (64, 64))
    for _ in range(n):
        w.write(np.zeros((64, 64, 3), dtype=np.uint8))
    w.release()


class TestAdaptiveFrameRate:
    @staticmethod
    def _pipe(monkeypatch):
        class FakeDetector:
            def __init__(self, weights=None):
                pass

            def infer(self, frame):
                return [_det(2, "player", (10, 10, 30, 60))]
        monkeypatch.setattr("src.analytics.adapters.video.detection.detector.Detector", FakeDetector)
        return Pipeline(detector_weights="fake", track=False,
                        use_homography=False, assign_teams=False)

    def test_analysis_hz_subsamples_rows(self, tmp_path, monkeypatch):
        video = tmp_path / "clip.mp4"
        _write_clip(video, n=25, fps=25)
        pipe = self._pipe(monkeypatch)
        full = pipe.to_dataframe(str(video), analysis_hz=0)      # 25 rows
        sampled = pipe.to_dataframe(str(video), analysis_hz=5.0)  # every 5th
        assert len(full) == 25
        assert len(sampled) == 5                 # frames 0,5,10,15,20
        idxs = sampled["frame_idx"].tolist()
        assert idxs == [0, 5, 10, 15, 20]

    def test_detection_still_runs_full_fps(self, tmp_path, monkeypatch):
        video = tmp_path / "clip.mp4"
        _write_clip(video, n=25, fps=25)
        calls = {"n": 0}

        class FakeDetector:
            def __init__(self, weights=None):
                pass

            def infer(self, frame):
                calls["n"] += 1
                return [_det(2, "player", (10, 10, 30, 60))]
        monkeypatch.setattr("src.analytics.adapters.video.detection.detector.Detector", FakeDetector)
        pipe = Pipeline(detector_weights="fake", track=False,
                        use_homography=False, assign_teams=False)
        pipe.to_dataframe(str(video), analysis_hz=5.0)
        assert calls["n"] == 25                  # tracked every source frame

    def test_zero_hz_disables_downsampling(self, tmp_path, monkeypatch):
        video = tmp_path / "clip.mp4"
        _write_clip(video, n=6, fps=25)
        pipe = self._pipe(monkeypatch)
        assert len(pipe.to_dataframe(str(video), analysis_hz=0)) == 6