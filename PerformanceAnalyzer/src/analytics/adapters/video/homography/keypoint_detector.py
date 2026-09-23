"""Local pitch-keypoint detector (fine-tuned YOLO pose model).

Wraps an Ultralytics pose model trained on the `football-field-detection`
dataset (32 keypoints, SoccerPitchConfiguration order) and exposes the
confident frame keypoints plus their canonical (pitch) indices so the
homography stage can build index-aligned correspondences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import (
    PITCH_KEYPOINT_CONFIDENCE,
    PITCH_KEYPOINT_IMGSZ,
    PITCH_KEYPOINT_MODEL_WEIGHTS,
)


@dataclass
class PitchKeypoints:
    """Keypoints for one frame (all, unthresholded).

    `xy` and `indices` are index-aligned: `indices[i]` is the position of
    `xy[i]` in the model's 32-keypoint (SoccerPitchConfiguration) ordering.
    Confidence filtering is left to the consumer (homography stage) so the
    gate lives in one place (config.py) and can be tuned independently.
    """
    xy: np.ndarray          # (M, 2) pixel coords
    confidence: np.ndarray  # (M,) per-keypoint confidences
    indices: np.ndarray     # (M,) int indices into the canonical 32-keypoint set

    @property
    def valid(self) -> bool:
        return len(self.xy) >= 4

    def filtered(self, min_conf: float) -> "PitchKeypoints":
        """Return a new PitchKeypoints with confidence >= min_conf."""
        keep = self.confidence >= min_conf
        return PitchKeypoints(
            xy=self.xy[keep],
            confidence=self.confidence[keep],
            indices=self.indices[keep],
        )


class PitchKeypointDetector:
    """Ultralytics pose wrapper for the 32-keypoint field detector."""

    def __init__(self, weights: str | None = None,
                 confidence: float | None = None,
                 imgsz: int | None = None):
        from ultralytics import YOLO

        self.weights = weights or PITCH_KEYPOINT_MODEL_WEIGHTS
        self.confidence = confidence if confidence is not None else PITCH_KEYPOINT_CONFIDENCE
        self.imgsz = imgsz or PITCH_KEYPOINT_IMGSZ
        self._model = YOLO(self.weights)

    def detect(self, frame: np.ndarray) -> PitchKeypoints:
        results = self._model(
            frame, conf=self.confidence, imgsz=self.imgsz, verbose=False,
        )
        kps = results[0].keypoints if results else None
        if (kps is None or kps.xy is None
                or len(kps.xy) == 0):
            return PitchKeypoints(np.empty((0, 2)), np.empty((0,)), np.empty((0,), dtype=int))
        if kps.conf is None or kps.xy is None:
            return PitchKeypoints(np.empty((0, 2)), np.empty((0,)), np.empty((0,), dtype=int))

        xy_all = kps.xy[0].cpu().numpy()      # (K, 2) px
        conf_all = kps.conf[0].cpu().numpy()  # (K,)
        idx = np.arange(len(conf_all))
        return PitchKeypoints(
            xy=np.asarray(xy_all, dtype=np.float64),
            confidence=np.asarray(conf_all, dtype=np.float64),
            indices=idx.astype(int),
        )
