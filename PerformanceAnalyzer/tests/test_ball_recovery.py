"""Stage 4b: ball-recovery tests (NMS, region/tiled inference, pipeline hook)."""

import numpy as np

from src.analytics.adapters.video.detection.detection import Detection
from src.analytics.adapters.video.detection.detector import Detector
from src.analytics.adapters.video.pipeline import Pipeline


def _det(cls: int, name: str, xyxy, conf: float = 0.9) -> Detection:
    return Detection(class_id=cls, class_name=name, confidence=conf, xyxy=xyxy)


class _RegionFake:
    """Detector stub that records infer calls and can return a ball on demand."""

    def __init__(self, ball_xyxy=None, ball_conf=0.2):
        self.ball_xyxy = ball_xyxy
        self.ball_conf = ball_conf
        self.region_calls: list[tuple[float, float, float]] = []

    def infer(self, frame):
        return []

    def infer_region(self, frame, center_xy, half, imgsz=None, conf=None):
        self.region_calls.append((center_xy[0], center_xy[1], half))
        if self.ball_xyxy is not None:
            return [_det(0, "ball", self.ball_xyxy, conf=self.ball_conf)]
        return []


def test_nms_keeps_one_per_overlap():
    dets = [
        _det(0, "ball", (100, 100, 120, 120), conf=0.9),
        _det(0, "ball", (102, 102, 122, 122), conf=0.85),   # same ball, lower conf
        _det(2, "player", (300, 300, 400, 500), conf=0.8),
    ]
    keep = Detector.nms(dets)
    assert len(keep) == 2                      # one ball + one player
    assert all(k.class_id != 0 or k.confidence == 0.9 for k in keep)


def test_nms_keeps_distinct_same_class():
    dets = [
        _det(0, "ball", (100, 100, 110, 110), conf=0.9),
        _det(0, "ball", (500, 500, 510, 510), conf=0.8),   # far apart -> distinct
    ]
    assert len(Detector.nms(dets)) == 2


def test_infer_region_shifts_boxes_back():
    det = Detector.__new__(Detector)           # skip model load
    det._hosted = False
    det.imgsz = 1280
    det.confidence = 0.3
    det._infer_local = lambda frame, imgsz=None, conf=None: [
        _det(0, "ball", (10.0, 10.0, 30.0, 30.0), conf=0.9)]

    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    dets = det.infer_region(frame, (300, 200), half=100)
    assert len(dets) == 1
    # box shifted back to full-frame coords (300-100, 200-100) + local offset
    assert dets[0].xyxy == (210.0, 110.0, 230.0, 130.0)


def test_infer_region_small_window_returns_empty():
    det = Detector.__new__(Detector)
    det._hosted = False
    det.imgsz = 1280
    det.confidence = 0.3
    det._infer_local = lambda frame, imgsz=None, conf=None: []
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    assert det.infer_region(frame, (25, 25), half=8) == []   # crop < 32px


class _BallTrackerFake:
    def __init__(self):
        self.position = None


class _MapperFake:
    def __init__(self):
        self._current = np.eye(3)

    @property
    def current(self):
        return self._current

    def transform_to_pixels(self, pts):
        return np.array([[100.0, 100.0]])


def test_pipeline_recover_adds_ball_when_missing():
    pipe = Pipeline.__new__(Pipeline)
    pipe._ball_recovery_enabled = True
    pipe.ball_tracker = _BallTrackerFake()
    pipe.ball_tracker.position = np.array([50.0, 50.0])
    pipe.pitch_mapper = _MapperFake()
    det = _RegionFake(ball_xyxy=(90, 90, 110, 110), ball_conf=0.3)
    pipe.detector = det
    pipe.tracker = None
    pipe.team_assigner = None

    out = pipe._recover_ball([_det(2, "player", (0, 0, 50, 100))], np.zeros((200, 200, 3), dtype=np.uint8))
    balls = [d for d in out if d.class_name == "ball"]
    assert len(balls) == 1
    assert balls[0].confidence == 0.3
    # recovery window centred on the back-projected prediction (100, 100)
    assert len(det.region_calls) == 1
    cx, cy, half = det.region_calls[0]
    assert (round(cx), round(cy)) == (100, 100)


def test_pipeline_recover_skips_when_no_prior():
    pipe = Pipeline.__new__(Pipeline)
    pipe._ball_recovery_enabled = True
    pipe.ball_tracker = _BallTrackerFake()      # position is None
    pipe.pitch_mapper = _MapperFake()
    det = _RegionFake(ball_xyxy=(90, 90, 110, 110))
    pipe.detector = det
    pipe.tracker = None
    pipe.team_assigner = None

    out = pipe._recover_ball([_det(2, "player", (0, 0, 50, 100))], np.zeros((200, 200, 3), dtype=np.uint8))
    assert det.region_calls == []
    assert all(d.class_name != "ball" for d in out)


def test_pipeline_recover_skips_when_strong_ball_present():
    pipe = Pipeline.__new__(Pipeline)
    pipe._ball_recovery_enabled = True
    pipe.ball_tracker = _BallTrackerFake()
    pipe.ball_tracker.position = np.array([50.0, 50.0])
    pipe.pitch_mapper = _MapperFake()
    det = _RegionFake()
    pipe.detector = det
    pipe.tracker = None
    pipe.team_assigner = None

    out = pipe._recover_ball([_det(0, "ball", (90, 90, 110, 110), conf=0.9)],
                             np.zeros((200, 200, 3), dtype=np.uint8))
    assert det.region_calls == []               # strong ball -> no recovery
    assert any(d.class_name == "ball" for d in out)
