"""Pitch keypoint detection — the "average defensive line" anchor.

Wraps a YOLO pose/keypoint model fine-tuned on the `football-field-detection`
dataset (see `References/train_pitch_keypoint_detector.py`). Each keypoint
carries a confidence; we expose a confidence-gated subset per frame. The model
is optional for Sprint 0 — callers may inject a stub that yields synthetic
keypoints so the homography module is testable without downloaded weights.
"""

from __future__ import annotations

import numpy as np

from src.config import KEYPOINT_CONFIDENCE_MIN


class PitchKeypointDetector:
    """Wraps an Ultralytics pose/keypoint model."""

    def __init__(self, weights: str | None = None,
                 confidence: float = 0.3,
                 min_keypoint_conf: float = KEYPOINT_CONFIDENCE_MIN,
                 imgsz: int = 640):
        self.confidence = confidence
        self.min_keypoint_conf = min_keypoint_conf
        self.imgsz = imgsz
        if weights is not None:
            from ultralytics import YOLO
            self._model = YOLO(weights)
        else:
            self._model = None

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Detect keypoints.

        Returns:
            (xy, confidence): pixel coordinates (N, 2) and per-point confidence
            (N,). Points already filtered to those with confidence >= threshold.
        """
        if self._model is None:
            raise RuntimeError(
                "PitchKeypointDetector has no weights; supply weights= or "
                "inject via set_keypoints_override for testing."
            )
        results = self._model(frame, conf=self.confidence, imgsz=self.imgsz,
                              verbose=False)
        kpt = results[0].keypoints
        if kpt is None:
            return np.empty((0, 2)), np.empty((0,))
        xy = kpt.xy[0].cpu().numpy()          # (K, 2)
        conf = kpt.conf[0].cpu().numpy()      # (K,)
        keep = conf >= self.min_keypoint_conf
        return xy[keep], conf[keep]