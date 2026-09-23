"""Supervision overlay builders — one per pipeline stage, mirroring the Roboflow
reference (`References/football_ai.py`) annotators so each stage's outcome can be
rendered onto the raw video and written to its own video file.

Stage -> annotator mapping (same as the reference):
  * detection : BoxAnnotator + LabelAnnotator  (4-class palette)
  * tracking  : EllipseAnnotator (GK/player/ref) + LabelAnnotator (#id) + ball triangle
  * teams     : EllipseAnnotator colored by team_id + ball triangle
  * homography: keypoints + pitch line overlay on the broadcast frame
  * projection: top-down pitch plane with projected anchors (see scripts/)
  * ball      : triangle + trail of the ball's pitch path

Colors match the reference palettes exactly.
"""

from __future__ import annotations

import numpy as np

from src.config import CLASS_BALL, CLASS_GOALKEEPER, CLASS_PLAYER, CLASS_REFEREE
from src.analytics.adapters.video.detection.detection import Detection

# ball, goalkeeper, player, referee (reference detection palette).
DETECTION_PALETTE = ["#FF8C00", "#00BFFF", "#FF1493", "#FFD700"]
# goalkeeper, player, referee (reference post-ball palette).
TRACK_PALETTE = ["#00BFFF", "#FF1493", "#FFD700"]
BALL_COLOR = "#FFD700"          # reference ball triangle colour
UNASSIGNED_COLOR = "#9E9E9E"    # referee / no-team when rendering teams

_CLASS_TO_TRACK_IDX = {CLASS_GOALKEEPER: 0, CLASS_PLAYER: 1, CLASS_REFEREE: 2}


def _to_sv(dets: list[Detection], class_ids: list[int],
           tracker_ids: list[object] | None = None) -> "sv.Detections":
    import supervision as sv
    if not dets:
        return sv.Detections.empty()
    xyxy = np.asarray([d.xyxy for d in dets], dtype=np.float64).reshape(-1, 4)
    conf = np.asarray([d.confidence for d in dets], dtype=np.float64)
    cid = np.asarray(class_ids, dtype=int)
    data = {}
    if tracker_ids is not None:
        data["tracker_id"] = np.asarray(
            [t if t is not None else -1 for t in tracker_ids])
    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=cid, data=data)


def detection_overlay(frame: np.ndarray, dets: list[Detection]) -> np.ndarray:
    """Stage 1 — boxes + class/confidence labels on the 4-class palette."""
    import supervision as sv
    svd = _to_sv(dets, [d.class_id for d in dets])
    box = sv.BoxAnnotator(
        color=sv.ColorPalette.from_hex(DETECTION_PALETTE), thickness=2)
    label = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(DETECTION_PALETTE),
        text_color=sv.Color.from_hex("#000000"),
        text_scale=0.5)
    labels = [f"{d.class_name} {d.confidence:.2f}" for d in dets]
    anno = frame.copy()
    anno = box.annotate(anno, svd)
    anno = label.annotate(anno, svd, labels=labels)
    return anno


def tracking_overlay(frame: np.ndarray, dets: list[Detection]) -> np.ndarray:
    """Stage 2 — ellipses (GK/player/ref) + #tracker_id + ball triangle."""
    import supervision as sv
    field = [d for d in dets if d.class_name != CLASS_BALL]
    balls = [d for d in dets if d.class_name == CLASS_BALL]
    svd = _to_sv(
        field,
        [_CLASS_TO_TRACK_IDX[d.class_name] for d in field],
        tracker_ids=[d.tracker_id for d in field])
    labels = [f"#{d.tracker_id}" for d in field]
    ellipse = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(TRACK_PALETTE), thickness=2)
    label = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(TRACK_PALETTE),
        text_color=sv.Color.from_hex("#000000"),
        text_scale=0.5)
    triangle = sv.TriangleAnnotator(
        color=sv.Color.from_hex(BALL_COLOR), base=25, height=21,
        outline_thickness=1)
    anno = frame.copy()
    anno = ellipse.annotate(anno, svd)
    anno = label.annotate(anno, svd, labels=labels)
    if balls:
        anno = triangle.annotate(anno, _to_sv(balls, [0] * len(balls)))
    return anno


def teams_overlay(frame: np.ndarray, dets: list[Detection]) -> np.ndarray:
    """Stage 3 — ellipses coloured by team_id; referee/unassigned grey."""
    import supervision as sv
    field = [d for d in dets if d.class_name != CLASS_BALL]
    balls = [d for d in dets if d.class_name == CLASS_BALL]

    class_ids: list[int] = []
    for d in field:
        if d.class_name == CLASS_REFEREE or d.team_id is None:
            class_ids.append(2)          # referee slot in TRACK_PALETTE -> grey override
        else:
            class_ids.append(int(d.team_id))
    palette = sv.ColorPalette.from_hex(list(TRACK_PALETTE))
    palette.colors[2] = sv.Color.from_hex(UNASSIGNED_COLOR)  # ref stays grey
    svd = _to_sv(field, class_ids, tracker_ids=[d.tracker_id for d in field])
    labels = [f"#{d.tracker_id}" for d in field]

    ellipse = sv.EllipseAnnotator(color=palette, thickness=2)
    label = sv.LabelAnnotator(
        color=palette, text_color=sv.Color.from_hex("#000000"), text_scale=0.5)
    triangle = sv.TriangleAnnotator(
        color=sv.Color.from_hex(BALL_COLOR), base=25, height=21,
        outline_thickness=1)
    anno = frame.copy()
    anno = ellipse.annotate(anno, svd)
    anno = label.annotate(anno, svd, labels=labels)
    if balls:
        anno = triangle.annotate(anno, _to_sv(balls, [0] * len(balls)))
    return anno


def ball_overlay(frame: np.ndarray, dets: list[Detection],
                 trail: list[tuple[float, float]] | None = None) -> np.ndarray:
    """Ball focus — triangle on every ball detection plus the recent pixel trail."""
    import cv2
    import supervision as sv

    balls = [d for d in dets if d.class_name == CLASS_BALL]
    triangle = sv.TriangleAnnotator(
        color=sv.Color.from_hex(BALL_COLOR), base=25, height=21,
        outline_thickness=2)
    anno = frame.copy()
    if balls:
        anno = triangle.annotate(anno, _to_sv(balls, [0] * len(balls)))
    if trail:
        for (px, py) in trail:
            cv2.circle(anno, (int(px), int(py)), 4, (0, 200, 255), -1)
    return anno
