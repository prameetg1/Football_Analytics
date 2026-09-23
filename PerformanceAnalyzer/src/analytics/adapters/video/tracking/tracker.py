"""Multi-object tracking — stable IDs across frames.

Wraps supervision's `ByteTrack` (occlusion-resilient, best-practice for sports).
Keeping the ball OUT of the tracker is the caller's responsibility: ball boxes
are routed to a separate stream (constructed in the orchestration layer) per
the reference architecture.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src.config import TRACK_CLASS_AGNOSTIC_NMS, TRACK_NMS_THRESHOLD
from src.analytics.adapters.video.detection.detection import Detection

try:
    import supervision as sv
    _SV = True
except Exception:  # pragma: no cover - optional dep
    sv = None
    _SV = False


def have_supervision() -> bool:
    """True if the supervision package is importable."""
    return _SV


class ByteTrackTracker:
    """Associates detections to stable IDs across frames using ByteTrack."""

    def __init__(self, track_activation_threshold: float = 0.3,
                 minimum_matching_threshold: float = 0.8,
                 lost_track_buffer: int = 30):
        if not _SV:
            raise RuntimeError("supervision is required for tracking")
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            minimum_matching_threshold=minimum_matching_threshold,
            lost_track_buffer=lost_track_buffer,
        )
        self._tracker.reset()

    def reset(self) -> None:
        self._tracker.reset()

    def update(self, detections: Iterable[Detection]) -> list[Detection]:
        """Assign tracker_ids to a batch of detections.

        Detections without a tracker_id (e.g. ball) are passed through intact.
        Class-agnostic NMS is applied before association (matches the
        reference), removing redundant overlapping boxes across classes so a
        single player isn't double-tracked as player + goalkeeper.
        """
        items = list(detections)
        to_track = [d for d in items if d is not None and d.class_name != "ball"]
        untouched = [d for d in items if d is not None and d.class_name == "ball"]

        if not to_track:
            return untouched

        xyxy = np.asarray([d.xyxy for d in to_track], dtype=np.float32)
        conf = np.asarray([d.confidence for d in to_track], dtype=np.float32)
        cls = np.asarray([d.class_id for d in to_track], dtype=int)

        sv_det = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=cls,
        )
        if TRACK_CLASS_AGNOSTIC_NMS:
            sv_det = sv_det.with_nms(
                threshold=TRACK_NMS_THRESHOLD, class_agnostic=True)
        sv_det = self._tracker.update_with_detections(sv_det)

        # Tracked boxes are a float-identical subset of the input boxes
        # (NMS never modifies them), so we can map tracker_ids back by box.
        tid_by_box = {
            tuple(float(v) for v in row): int(tid)
            for row, tid in zip(sv_det.xyxy, sv_det.tracker_id)
        }
        survivors = []
        for d in to_track:
            tid = tid_by_box.get(tuple(float(v) for v in d.xyxy))
            if tid is not None:
                d.tracker_id = tid
                survivors.append(d)
        return survivors + untouched