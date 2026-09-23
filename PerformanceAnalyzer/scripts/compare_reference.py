#!/usr/bin/env python3
"""A/B stage-by-stage comparison: PerformanceAnalyzer vs the Roboflow reference.

For each pipeline stage we run BOTH implementations on the same frames from a
source clip and report parity metrics:

  Stage 1  Detection   our Detector(hosted)  vs  reference hosted model
  Stage 2  Tracking    our ByteTrackTracker  vs  reference sv.ByteTrack
  Stage 3  Teams       our TeamClassifier    vs  reference sports.TeamClassifier
  Stage 4  Pitch kpts  our local best.pt     vs  reference hosted field model
  Stage 5  Homography  our ViewTransformer   vs  reference sports.ViewTransformer

"Parity" means: given identical input, does our clean `src/` reimplementation
produce the same output as the Roboflow reference? Where they intentionally
differ (e.g. RANSAC vs plain findHomography) we report the difference explicitly.

Usage:
    .venv/bin/python scripts/compare_reference.py \
        --video data/raw/0bfacc_0.mp4 [--frames 0 30 60] [--skip detection]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.adapters.video.reader import frame_generator  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_frames(video: str, n: int = 3, stride: int = 30) -> list[int]:
    """Sample frame indices spread across the clip."""
    total = _total_frames(video)
    idx = list(range(0, total, stride))
    if len(idx) > n:
        step = max(1, len(idx) // n)
        idx = idx[::step][:n]
    return idx


def _total_frames(video: str) -> int:
    import cv2
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(1, n)


def _grab_frames(video: str, indices: list[int]) -> dict[int, np.ndarray]:
    """Return {frame_idx: BGR frame} for the requested indices."""
    frames = {}
    gen = frame_generator(video)
    for want in sorted(indices):
        for idx, frame in enumerate(gen):
            if idx < want:
                continue
            frames[want] = frame
            break
    return frames


def _fmt(x: float, sig: int = 4) -> str:
    return f"{x:.{sig}f}"


# ---------------------------------------------------------------------------
# Stage 1 — Detection
# ---------------------------------------------------------------------------

def compare_detection(frames: dict[int, np.ndarray]) -> dict:
    """our Detector(hosted id) vs reference get_model+sv.Detections."""
    from src.config import ROBOFLOW_PLAYER_MODEL_ID
    from src.analytics.adapters.video.detection.detector import Detector

    ours = Detector(weights=ROBOFLOW_PLAYER_MODEL_ID)

    from inference import get_model
    import supervision as sv
    ref = get_model(ROBOFLOW_PLAYER_MODEL_ID,
                    api_key=os.environ.get("ROBOFLOW_API_KEY", ""))

    rows = []
    for idx, frame in frames.items():
        t0 = time.time()
        our_dets = ours.infer(frame)
        t_ours = time.time() - t0

        t0 = time.time()
        result = ref.infer(frame, confidence=0.3)[0]
        ref_dets = sv.Detections.from_inference(result)
        t_ref = time.time() - t0

        our_count = {d.class_name: sum(1 for x in our_dets if x.class_name == d.class_name)
                     for d in our_dets}
        ref_counts: dict[str, int] = {}
        for i in range(len(ref_dets)):
            name = ref_dets.data.get("class_name")
            cname = name[i] if name is not None and len(name) > i else str(ref_dets.class_id[i])
            ref_counts[str(cname)] = ref_counts.get(str(cname), 0) + 1

        # Match by IoU of boxes sharing a class name
        matched = unmatched_ours = unmatched_ref = 0
        used = set()
        for od in our_dets:
            best, best_iou = None, 0.0
            for i in range(len(ref_dets)):
                if i in used:
                    continue
                name = ref_dets.data.get("class_name")
                rname = str(name[i]) if name is not None and len(name) > i else str(ref_dets.class_id[i])
                if rname != od.class_name:
                    continue
                rb = ref_dets.xyxy[i]
                iou = _iou(od.xyxy, tuple(float(x) for x in rb))
                if iou > best_iou:
                    best_iou, best = iou, i
            if best is not None and best_iou >= 0.5:
                matched += 1
                used.add(best)
            else:
                unmatched_ours += 1
        unmatched_ref = len(ref_dets) - matched

        rows.append({
            "frame": idx,
            "ours_classes": our_count,
            "ref_classes": ref_counts,
            "matched_iou50": matched,
            "unmatched_ours": unmatched_ours,
            "unmatched_ref": unmatched_ref,
            "ours_time_ms": t_ours * 1000,
            "ref_time_ms": t_ref * 1000,
        })

    return {"stage": "detection", "rows": rows}


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


# ---------------------------------------------------------------------------
# Stage 2 — Tracking
# ---------------------------------------------------------------------------

def compare_tracking(frames: dict[int, np.ndarray]) -> dict:
    """our ByteTrackTracker vs reference sv.ByteTrack on identical detections."""
    from src.analytics.adapters.video.tracking.tracker import ByteTrackTracker

    ours_tracker = ByteTrackTracker()
    ref_tracker = _make_sv_bytetrack()

    # Reuse hosted detections so both trackers see identical inputs
    from src.config import ROBOFLOW_PLAYER_MODEL_ID
    from src.analytics.adapters.video.detection.detector import Detector
    det = Detector(weights=ROBOFLOW_PLAYER_MODEL_ID)

    rows = []
    for idx, frame in frames.items():
        dets = det.infer(frame)
        # ours
        t0 = time.time()
        our_out = ours_tracker.update(dets)
        t_ours = time.time() - t0
        # reference
        sv_dets = _to_sv(dets)
        t0 = time.time()
        ref_out = ref_tracker.update_with_detections(sv_dets)
        t_ref = time.time() - t0

        our_ids = [d.tracker_id for d in our_out]
        ref_ids = [int(x) if x is not None else None for x in ref_out.tracker_id]
        rows.append({
            "frame": idx,
            "ours_tracked": sum(1 for i in our_ids if i is not None),
            "ref_tracked": sum(1 for i in ref_ids if i is not None),
            "ours_ids": our_ids,
            "ref_ids": ref_ids,
            "ours_time_ms": t_ours * 1000,
            "ref_time_ms": t_ref * 1000,
        })
    return {"stage": "tracking", "rows": rows}


def _make_sv_bytetrack():
    import supervision as sv
    return sv.ByteTrack(
        track_activation_threshold=0.3,
        minimum_matching_threshold=0.8,
        lost_track_buffer=30,
    )


def _to_sv(dets):
    import numpy as np
    import supervision as sv
    n = len(dets)
    xyxy = np.array([d.xyxy for d in dets], dtype=np.float32) if n else np.empty((0, 4), np.float32)
    conf = np.array([d.confidence for d in dets], dtype=np.float32) if n else np.empty((0,), np.float32)
    cls = np.array([d.class_id for d in dets], dtype=int) if n else np.empty((0,), int)
    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)


# ---------------------------------------------------------------------------
# Stage 3 — Teams
# ---------------------------------------------------------------------------

def compare_teams(frames: dict[int, np.ndarray]) -> dict:
    """our TeamClassifier vs reference sports.TeamClassifier on the same crops."""
    import supervision as sv
    from src.config import ROBOFLOW_PLAYER_MODEL_ID
    from src.analytics.adapters.video.detection.detector import Detector

    det = Detector(weights=ROBOFLOW_PLAYER_MODEL_ID)
    crops: list[np.ndarray] = []
    for frame in frames.values():
        dets = det.infer(frame)
        for d in dets:
            if d.class_name in ("player", "Player", "PLAYER"):
                x1, y1, x2, y2 = d.xyxy
                crops.append(frame[int(y1):int(y2), int(x1):int(x2)])
    if len(crops) < 8:
        return {"stage": "teams", "rows": [], "note": f"only {len(crops)} player crops"}
    crops = crops[:64]

    from src.analytics.adapters.video.teams.classifier import TeamClassifier as OurTC
    ours = OurTC()
    t0 = time.time()
    ours.fit(crops)
    our_labels = ours.predict(crops)
    t_ours = time.time() - t0

    from sports.common.team import TeamClassifier as RefTC
    ref = RefTC(device="cuda")
    t0 = time.time()
    ref.fit(crops)
    ref_labels = ref.predict(crops)
    t_ref = time.time() - t0

    ari = _adjusted_rand(our_labels, ref_labels)
    return {
        "stage": "teams",
        "rows": [{
            "n_crops": len(crops),
            "ours_labels": list(our_labels),
            "ref_labels": list(ref_labels),
            "ari": ari,
            "ours_time_ms": t_ours * 1000,
            "ref_time_ms": t_ref * 1000,
        }],
        "note": "ARI=1 means identical team split (label names may swap)",
    }


def _adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


# ---------------------------------------------------------------------------
# Stage 4 — Pitch keypoints
# ---------------------------------------------------------------------------

def compare_pitch_keypoints(frames: dict[int, np.ndarray]) -> dict:
    """our local best.pt vs reference hosted field model."""
    from src.config import PITCH_KEYPOINT_MODEL_ID, ROBOFLOW_API_KEY
    from src.analytics.adapters.video.homography.keypoint_detector import PitchKeypointDetector

    ours = PitchKeypointDetector()

    from inference import get_model
    import supervision as sv
    ref = get_model(PITCH_KEYPOINT_MODEL_ID, api_key=ROBOFLOW_API_KEY)

    rows = []
    for idx, frame in frames.items():
        t0 = time.time()
        kp = ours.detect(frame)
        t_ours = time.time() - t0

        t0 = time.time()
        result = ref.infer(frame, confidence=0.3)[0]
        ref_kps = sv.KeyPoints.from_inference(result)
        t_ref = time.time() - t0

        ref_xy = np.asarray(ref_kps.xy[0], dtype=np.float64) if ref_kps.xy is not None and len(ref_kps.xy) else np.empty((0, 2))
        ref_conf = np.asarray(ref_kps.confidence[0], dtype=np.float64) if ref_kps.confidence is not None and len(ref_kps.confidence) else np.empty((0,))
        ref_mask = ref_conf > 0.5 if len(ref_conf) else np.zeros(0, dtype=bool)
        ref_xy_conf = ref_xy[ref_mask] if len(ref_mask) else np.empty((0, 2))
        ref_idx = np.nonzero(ref_mask)[0] if len(ref_mask) else np.empty((0,), int)

        our_xy = np.asarray(kp.xy, dtype=np.float64)
        our_idx = np.asarray(kp.indices)

        rows.append({
            "frame": idx,
            "ours_n": len(our_xy),
            "ref_n": len(ref_xy_conf),
            "ours_idx": sorted(int(i) for i in our_idx),
            "ref_idx": sorted(int(i) for i in ref_idx),
            "ours_time_ms": t_ours * 1000,
            "ref_time_ms": t_ref * 1000,
        })
    return {"stage": "pitch_keypoints", "rows": rows}


# ---------------------------------------------------------------------------
# Stage 5 — Homography
# ---------------------------------------------------------------------------

def compare_homography(frames: dict[int, np.ndarray]) -> dict:
    """our ViewTransformer(RANSAC) vs reference sports.ViewTransformer(plain)."""
    from src.analytics.adapters.video.homography.homography import ViewTransformer as OurVT
    from src.analytics.adapters.video.homography.pitch_model import PitchConfig
    from sports.common.view import ViewTransformer as RefVT
    from sports.configs.soccer import SoccerPitchConfiguration

    our_cfg = PitchConfig()
    ref_cfg = SoccerPitchConfiguration()
    ref_vertices = np.array(ref_cfg.vertices, dtype=np.float64)

    # Unit normalization: reference SoccerPitchConfiguration is in cm, our
    # PitchConfig is in metres. Express both canonical targets in cm so the
    # projected outputs and H matrices are directly comparable.
    our_vertices_cm = our_cfg.vertices * 100.0

    from src.analytics.adapters.video.homography.keypoint_detector import PitchKeypointDetector
    from src.analytics.adapters.video.homography.homography import StablePitchMapper
    ours_kp = PitchKeypointDetector()
    mapper = StablePitchMapper()

    rows = []
    for idx, frame in frames.items():
        kp = mapper.filter_keypoints(ours_kp.detect(frame))
        if not kp.valid:
            rows.append({"frame": idx, "note": "fewer than 4 keypoints, skipped"})
            continue
        frame_pts = np.asarray(kp.xy, dtype=np.float64)
        pitch_pts = our_vertices_cm[kp.indices]
        pitch_pts_ref = ref_vertices[kp.indices]

        t0 = time.time()
        our_vt = OurVT(frame_pts, pitch_pts)  # RANSAC
        t_ours = time.time() - t0
        t0 = time.time()
        ref_vt = RefVT(source=frame_pts.astype(np.float32),
                       target=pitch_pts_ref.astype(np.float32))  # plain LS
        t_ref = time.time() - t0

        # Compare projected canonical keypoints
        pts = np.asarray(kp.xy, dtype=np.float32)
        our_out = our_vt.transform_points(pts)
        ref_out = np.asarray(ref_vt.transform_points(pts), dtype=np.float64)
        err = np.mean(np.linalg.norm(our_out - ref_out, axis=1)) if len(our_out) else float("nan")

        # Geometric matrix difference (Frobenius on normalized H)
        m1 = our_vt.m / (our_vt.m[2, 2] or 1.0)
        m2 = ref_vt.m / (ref_vt.m[2, 2] or 1.0)
        fro = np.linalg.norm(m1 - m2)

        rows.append({
            "frame": idx,
            "n_kpts": len(frame_pts),
            "mean_proj_diff_px": err,
            "frobenius_H": fro,
            "ours_uses_ransac": True,
            "ref_uses_ransac": False,
            "ours_time_ms": t_ours * 1000,
            "ref_time_ms": t_ref * 1000,
        })
    return {"stage": "homography", "rows": rows}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(results: list[dict]):
    print("\n" + "=" * 78)
    print("PERFORMANCEANALYZER vs ROBOFLOW REFERENCE — STAGE PARITY REPORT")
    print("=" * 78)
    for r in results:
        stage = r["stage"].upper()
        rows = r["rows"]
        note = r.get("note", "")
        print(f"\n--- Stage: {stage} ({len(rows)} frame(s)) ---")
        if note:
            print(f"    note: {note}")
        if not rows:
            print("    (no rows)")
            continue
        for row in rows:
            print("   " + " | ".join(f"{k}={v}" for k, v in row.items()))
    print("\n" + "=" * 78)
    print("LEGEND")
    print("  * Detection parity  = matched boxes (IoU>=0.5) / total, per class")
    print("  * Tracking parity   = tracker ids our-vs-ref per frame (should match)")
    print("  * Teams parity      = adjusted Rand index (1.0 == identical split)")
    print("  * Pitch keypoints   = # confident kpts (conf>0.5), our local vs ref hosted")
    print("  * Homography        = mean projected-px diff + H Frobenius norm diff")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description="A/B compare PerformanceAnalyzer vs Roboflow reference")
    ap.add_argument("--video", default="data/raw/0bfacc_0.mp4")
    ap.add_argument("--frames", type=int, nargs="*", default=None,
                    help="explicit frame indices (default: 3 spread across clip)")
    ap.add_argument("--skip", type=str, nargs="*", default=[],
                    help="stages to skip, e.g. --skip teams homography")
    args = ap.parse_args()

    idx = args.frames or _pick_frames(args.video, n=3, stride=30)
    print(f"video={args.video} frames={idx}")
    frames = _grab_frames(args.video, idx)
    print(f"grabbed {len(frames)} frames")

    results = []
    stages = [
        ("detection", compare_detection),
        ("tracking", compare_tracking),
        ("teams", compare_teams),
        ("pitch_keypoints", compare_pitch_keypoints),
        ("homography", compare_homography),
    ]
    for name, fn in stages:
        if name in args.skip:
            print(f"[skip] {name}")
            continue
        print(f"[run] {name} ...")
        try:
            results.append(fn(frames))
        except Exception as e:  # keep going; report failures
            print(f"[error] {name}: {type(e).__name__}: {e}")
            results.append({"stage": name, "rows": [], "note": f"error: {e}"})

    _print_report(results)


if __name__ == "__main__":
    main()
