"""Homography + temporal smoothing — pixel (pitch) -> canonical pitch (x, y).

Computes the homography `H` from detected pitch keypoints to the canonical
`PitchConfig.vertices`, then keeps the last `maxlen` matrices in a deque and
returns their MEAN. The mean matrix is the key robustness feature: it averages
per-frame pan/zoom noise and bridges frames where the keypoint detector briefly
fails (matching the reference's deque-mean approach).

Stage 6 additions:
  * per-keypoint exponential smoothing of the pixel coordinates BEFORE the fit
    (reduces keypoint jitter that would otherwise poison RANSAC);
  * a shot-change detector: each freshly-fitted H is compared to the smoothed
    state at the canonical pitch points; if the mean displacement exceeds
    `cut_threshold_px` this is a camera cut, so the stateful prior (H deque and
    keypoint smoother) is reset — we never smooth across a replay cut.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import cv2

from src.config import (
    HOMOGRAPHY_MAXLEN,
    KEYPOINT_CONFIDENCE_MIN,
    KEYPOINT_SMOOTH_ALPHA,
    SHOT_CUT_THRESHOLD_PX,
)
from src.analytics.adapters.video.homography.pitch_model import PitchConfig

_DEFAULT_CONFIG = PitchConfig()


class ViewTransformer:
    """Estimate a homography and project points between two planes."""

    def __init__(self, source: np.ndarray, target: np.ndarray):
        if len(source) < 4 or len(target) < 4:
            raise ValueError(
                f"Homography needs >=4 correspondences, got {len(source)}"
            )
        # source: frame/pixel keypoints -> target: canonical pitch points.
        self.m, _ = cv2.findHomography(source, target, cv2.RANSAC)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.empty((0, 2))
        out = cv2.perspectiveTransform(pts, self.m)
        return out.reshape(-1, 2)


def _to_pixels(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map canonical pitch points through the INVERSE homography to pixels."""
    points = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, np.linalg.inv(H)).reshape(-1, 2)


class StablePitchMapper:
    """Per-frame homography with temporal smoothing + shot-cut reset."""

    def __init__(self, config: PitchConfig | None = None,
                 maxlen: int = HOMOGRAPHY_MAXLEN,
                 min_conf: float = KEYPOINT_CONFIDENCE_MIN,
                 kp_alpha: float = KEYPOINT_SMOOTH_ALPHA,
                 cut_threshold_px: float = SHOT_CUT_THRESHOLD_PX):
        self.config = config or _DEFAULT_CONFIG
        self.maxlen = maxlen
        self.min_conf = min_conf
        self.kp_alpha = kp_alpha
        self.cut_threshold_px = cut_threshold_px
        self._deque: deque[np.ndarray] = deque(maxlen=maxlen)
        self._current: np.ndarray | None = None  # (3, 3)
        self._kp_prior: dict[int, np.ndarray] = {}
        self.cut_detected_last = False

    @property
    def current(self) -> np.ndarray | None:
        return self._current

    def reset(self) -> None:
        """Drop all stateful history (used on shot change and by callers)."""
        self._deque.clear()
        self._current = None
        self._kp_prior.clear()
        self.cut_detected_last = False

    def filter_keypoints(self, kp) -> "PitchKeypoints":
        """Apply the confidence gate to a PitchKeypoints, returning a new one.

        Accepts any object exposing `.filtered(min_conf)` (the PitchKeypoints
        dataclass) so the homography stage owns the gate in one place.
        """
        return kp.filtered(self.min_conf)

    def smooth_keypoints(self, kp) -> np.ndarray:
        """Exponentially smooth pixel keypoints by canonical index.

        Returns an (M, 2) array aligned with `kp.xy` / `kp.indices`. A keypoint
        seen in the previous frame is blended with its previous smoothed value;
        first-seen (or newly re-id'd, e.g. after a cut) points pass through.
        """
        xy = np.asarray(kp.xy, dtype=np.float64).copy()
        if self.kp_alpha >= 1.0 or len(xy) == 0:
            return xy
        for i, idx in enumerate(kp.indices):
            prev = self._kp_prior.get(int(idx))
            if prev is not None:
                xy[i] = (1.0 - self.kp_alpha) * prev + self.kp_alpha * xy[i]
        for i, idx in enumerate(kp.indices):
            self._kp_prior[int(idx)] = xy[i].copy()
        return xy

    def _is_cut(self, new_h: np.ndarray) -> bool:
        """True if `new_h` maps canonical points far from the smoothed state."""
        if self._current is None:
            return False
        try:
            old_px = _to_pixels(self._current, self.config.vertices)
            new_px = _to_pixels(new_h, self.config.vertices)
        except np.linalg.LinAlgError:
            return True
        displacement = float(np.mean(np.linalg.norm(new_px - old_px, axis=1)))
        return displacement > self.cut_threshold_px

    def update(self, frame_keypoints: np.ndarray,
               pitch_keypoints: np.ndarray) -> np.ndarray:
        """Fit from pixel->pitch correspondences, append, return smoothed H."""
        vt = ViewTransformer(frame_keypoints, pitch_keypoints)
        new_h = vt.m
        self.cut_detected_last = self._current is not None and self._is_cut(new_h)
        if self.cut_detected_last:
            self._deque.clear()
            self._kp_prior.clear()
        self._deque.append(new_h)
        self._current = np.mean(np.asarray(self._deque), axis=0)
        return self._current

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if self._current is None:
            return np.empty((0, 2))
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.empty((0, 2))
        out = cv2.perspectiveTransform(pts, self._current)
        return out.reshape(-1, 2)

    def transform_to_pixels(self, points: np.ndarray) -> np.ndarray:
        """Map canonical pitch points back to frame pixels (inverse H)."""
        if self._current is None:
            return np.empty((0, 2))
        return _to_pixels(self._current, points)