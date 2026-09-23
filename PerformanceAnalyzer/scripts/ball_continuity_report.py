"""Stage 4 acceptance — measure ball-trail continuity on a clip.

Runs the pipeline once (detection + homography + teams + ball Kalman) and
reports, in a single pass:

  * % frames with a raw ball detection (the reference's input signal),
  * % frames with a valid ball hypothesis (Kalman, incl. short predictions),
  * % of detection-miss frames recovered by the hypothesis,
  * possession delta: `PossessionMetric` computed on the hypothesis ball
    stream vs the raw-projected ball stream (everything else identical).

Usage:
    python scripts/ball_continuity_report.py --video data/raw/0bfacc_0.mp4 \
        [--max-frames 750] [--fps 25]
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd

from src.config import BALL_TRACKER_ID, CLASS_BALL
from src.analytics.adapters.video.metrics.possession import PossessionMetric
from src.analytics.adapters.video.pipeline import Pipeline


def _project_point(H: np.ndarray, xy: tuple[float, float]) -> np.ndarray:
    """Apply homography H (3x3) to a single image point -> pitch metres."""
    pts = np.asarray([xy], dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return out[0]


def run(source: str, max_frames: int | None, fps: float) -> dict:
    pipe = Pipeline()

    rows_h: list[dict] = []      # hypothesis ball stream + player rows
    rows_r: list[dict] = []      # raw-projected ball stream + player rows
    raw_det: dict[int, bool] = {}
    hyp_frames: set[int] = set()
    raw_ball_pos: dict[int, np.ndarray] = {}

    cols = ["frame_idx", "timestamp_sec", "track_id", "team_id",
            "pitch_x", "pitch_y", "class_name"]

    for tf in pipe.process(source, max_frames):
        ts = tf.frame_idx / fps
        raw_det[tf.frame_idx] = any(
            d.class_name == CLASS_BALL for d in tf.detections)

        # Raw projection: ball bottom-center through the current homography,
        # exactly what the reference would feed (no Kalman). Only available on
        # frames where the ball was detected and the homography is live.
        raw_pos = None
        balls = [d for d in tf.detections if d.class_name == CLASS_BALL]
        if balls and tf.homography is not None:
            best = max(balls, key=lambda d: d.confidence)
            raw_pos = _project_point(tf.homography, best.bottom_center)
            raw_ball_pos[tf.frame_idx] = raw_pos

        for d in tf.detections:
            if d.class_name == CLASS_BALL:
                continue  # ball is exported via its own stream below
            pos = (tf.pitch_positions.get(d.tracker_id)
                   if d.tracker_id is not None else None)
            player_row = {
                "frame_idx": tf.frame_idx, "timestamp_sec": round(ts, 6),
                "track_id": d.tracker_id if d.tracker_id is not None else np.nan,
                "team_id": d.team_id if d.team_id is not None else np.nan,
                "pitch_x": float(pos[0]) if pos is not None else np.nan,
                "pitch_y": float(pos[1]) if pos is not None else np.nan,
                "class_name": d.class_name,
            }
            rows_h.append(player_row)
            rows_r.append(player_row)

        if tf.ball_position is not None:
            hyp_frames.add(tf.frame_idx)
            rows_h.append({
                "frame_idx": tf.frame_idx, "timestamp_sec": round(ts, 6),
                "track_id": BALL_TRACKER_ID, "team_id": np.nan,
                "pitch_x": float(tf.ball_position[0]),
                "pitch_y": float(tf.ball_position[1]),
                "class_name": CLASS_BALL,
            })
        if raw_pos is not None:
            rows_r.append({
                "frame_idx": tf.frame_idx, "timestamp_sec": round(ts, 6),
                "track_id": BALL_TRACKER_ID, "team_id": np.nan,
                "pitch_x": float(raw_pos[0]), "pitch_y": float(raw_pos[1]),
                "class_name": CLASS_BALL,
            })

    n = len(raw_det)
    raw_frames = sum(1 for v in raw_det.values() if v)
    missed = [f for f in raw_det if not raw_det[f]]
    recovered = sum(1 for f in missed if f in hyp_frames)

    report = {
        "frames": n,
        "raw_detection_pct": round(100 * raw_frames / max(n, 1), 1),
        "hypothesis_pct": round(100 * len(hyp_frames) / max(n, 1), 1),
        "miss_frames": len(missed),
        "miss_recovered_pct": round(100 * recovered / max(len(missed), 1), 1),
    }

    # Possession delta (needs team-assigned players in both frames).
    df_h = pd.DataFrame(rows_h, columns=cols)
    df_r = pd.DataFrame(rows_r, columns=cols)
    possession = PossessionMetric()
    delta = {}
    for team in (0, 1):
        ph = possession.compute(df_h, team)
        pr = possession.compute(df_r, team)
        if ph is not None and pr is not None:
            delta[team] = {"hypothesis": ph, "raw": pr,
                           "delta": round(ph - pr, 4)}
    report["possession_delta"] = delta if delta else None
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--fps", type=float, default=25.0)
    args = p.parse_args()
    print(run(args.video, args.max_frames, args.fps))


if __name__ == "__main__":
    main()