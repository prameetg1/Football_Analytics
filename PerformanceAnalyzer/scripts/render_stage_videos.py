#!/usr/bin/env python3
"""Render one overlay video per pipeline stage, the way the Roboflow reference does.

Each stage renders its own outcome onto the raw video (or a top-down pitch for the
projection stage) with supervision annotators and writes an MP4:

    output/stages/stage_detection.mp4   boxes + class labels
    output/stages/stage_tracking.mp4    ellipses + #tracker_id + ball triangle
    output/stages/stage_teams.mp4       ellipses coloured by team_id
    output/stages/stage_ball.mp4        ball triangle + pixel trail
    output/stages/stage_pitch.mp4       top-down pitch with projected anchors

By default it uses the hosted Roboflow model (offloads the GPU — useful while the
local detector is still training); pass --weights <local.pt> after the bake-off.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/render_stage_videos.py \
        --video data/raw/0bfacc_0.mp4 --out-dir output/stages \
        --stages detection,tracking,teams,ball,pitch
"""

import argparse
from pathlib import Path

import numpy as np

from src.config import CLASS_BALL, ROBOFLOW_API_KEY, ROBOFLOW_PLAYER_MODEL_ID
from src.analytics.adapters.video.pipeline import Pipeline
from src.analytics.adapters.video.render.overlays import (
    ball_overlay,
    detection_overlay,
    teams_overlay,
    tracking_overlay,
)

STAGE_FUNCS = {
    "detection": detection_overlay,
    "tracking": tracking_overlay,
    "teams": teams_overlay,
    "ball": ball_overlay,
}

# Which pipeline stages each overlay video actually needs. Detection/tracking/ball
# only use the hosted detector + ByteTrack: we MUST NOT load the local pose
# keypoint model (homography) or SigLIP (teams) for them, or they'd steal the GPU
# from a concurrent training run. (assign_teams, use_homography)
STAGE_PIPELINE = {
    "detection": (False, False),
    "tracking": (False, False),
    "ball": (False, False),
    "teams": (True, False),
    "pitch": (True, True),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="output/stages")
    p.add_argument("--stages", type=str,
                   default="detection,tracking,teams,ball,pitch")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--weights", type=str, default=None,
                   help="local .pt, or hosted id; defaults to ROBOFLOW_PLAYER_MODEL_ID")
    p.add_argument("--scale", type=float, default=10.0,
                   help="pixels per metre for the pitch stage")
    return p.parse_args()


def _render_overlay_stage(pipe, overlay_fn, source, out_path, max_frames):
    """Write an MP4 where every frame is `overlay_fn(frame, dets)`."""
    import supervision as sv

    from src.analytics.adapters.video.reader import frame_generator

    video_info = sv.VideoInfo.from_video_path(source)
    with sv.VideoSink(out_path, video_info=video_info) as sink:
        for idx, frame in enumerate(frame_generator(source)):
            if max_frames is not None and idx >= max_frames:
                break
            dets = pipe.track_frame(frame)
            sink.write_frame(overlay_fn(frame, dets))
    print(f"  wrote {out_path}")


def _render_pitch_stage(pipe, source, out_path, max_frames, scale):
    """Top-down pitch with projected player/ball anchors (metres)."""
    import cv2
    import supervision as sv

    from src.analytics.adapters.video.homography.pitch_model import render_pitch
    from src.analytics.adapters.video.reader import frame_generator

    video_info = sv.VideoInfo.from_video_path(source)
    h, w = video_info.resolution_wh
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             video_info.fps or 25.0, (w, h))
    team_colors = {0: (255, 0, 0), 1: (0, 0, 255)}  # BGR: team0 blue, team1 red
    n = 0
    for idx, frame in enumerate(frame_generator(source)):
        if max_frames is not None and idx >= max_frames:
            break
        dets = pipe.track_frame(frame)
        positions = pipe._project(dets, frame)
        pitch = render_pitch(pipe.pitch_mapper.config, scale=scale)
        pad = pipe.pitch_mapper.config.padding
        for d in dets:
            tid = d.tracker_id
            if tid is None or tid not in positions:
                continue
            xm, ym = positions[tid]
            px = int(xm * scale) + pad
            py = int(ym * scale) + pad
            col = ((0, 255, 255) if d.class_name == CLASS_BALL
                   else team_colors.get(d.team_id, (255, 255, 255)))
            cv2.circle(pitch, (px, py), 4, col, -1)
        cv2.putText(pitch, f"frame {idx}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(cv2.resize(pitch, (w, h)))
        n += 1
    writer.release()
    print(f"  wrote {out_path} ({n} frames)")


def main():
    args = parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = args.weights or (ROBOFLOW_PLAYER_MODEL_ID if ROBOFLOW_API_KEY else None)
    print(f"detector: {weights or 'local fallback (COCO)'} "
          f"+ homography: local pose fine-tune")

    for stage in stages:
        assign_teams, use_homography = STAGE_PIPELINE.get(stage, (False, False))
        pipe = Pipeline(detector_weights=weights, track=True,
                        use_homography=use_homography,
                        assign_teams=assign_teams)
        print(f"[{stage}] "
              f"(teams={assign_teams}, homography={use_homography})")
        out = out_dir / f"stage_{stage}.mp4"
        if stage == "pitch":
            _render_pitch_stage(pipe, args.video, str(out), args.max_frames,
                                args.scale)
        elif stage in STAGE_FUNCS:
            _render_overlay_stage(pipe, STAGE_FUNCS[stage], args.video, str(out),
                                  args.max_frames)
        else:
            print(f"  unknown stage {stage!r}, skipping")
    print("done.")


if __name__ == "__main__":
    main()
