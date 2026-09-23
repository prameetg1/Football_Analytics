#!/usr/bin/env python3
"""Render an annotated video: broadcast frame + radar + C1/C2 surfaces + HUD.

Runs the full pipeline (detection -> tracking -> homography -> projection ->
teams -> ball Kalman), accumulates a live tracking table, and writes an MP4 that
is split into:
    left  : the broadcast frame (with frame counter)
    right : a top-down pitch radar showing the C1 Spearman control surface
            (blue team 0 / red team 1, refreshed every `--hud-interval`
            frames), team-coloured player dots, and the ball
    bottom: a C2 EPV value heatmap strip (dark = cold, yellow = hot)
    top   : a live metric HUD (possession %, line height, spread, block, tilt,
            formation angle, pass counts) recomputed over the frames seen so far

`--radar voronoi` swaps the control surface back to nearest-player Voronoi
ownership cells; `--no-epv` drops the EPV strip.

Stage 8 uses the local bake-off winner (DETECTION_MODEL_WEIGHTS) by default.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/render_pitch_video.py \
        --video data/raw/0bfacc_0.mp4 --max-frames 750 \
        --out output/pitch_radar_hud_v4.mp4
"""

import argparse

import numpy as np

from src.config import DETECTION_MODEL_WEIGHTS
from src.analytics.adapters.video.pipeline import Pipeline
from src.analytics.adapters.video.render.hud import (draw_control_surface, draw_epv_heatmap, draw_hud,
                            draw_voronoi_radar)
from src.analytics.adapters.video.homography.pitch_model import render_pitch


def parse_args():
    p = argparse.ArgumentParser(description="Render radar+HUD overlay video")
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--max-frames", type=int, default=150)
    p.add_argument("--out", type=str, default="output/pitch_overlay.mp4")
    p.add_argument("--scale", type=float, default=10.0)
    p.add_argument("--hud-interval", type=int, default=30,
                   help="recompute live metrics every N frames")
    p.add_argument("--weights", type=str, default=DETECTION_MODEL_WEIGHTS,
                   help="local .pt weights (defaults to the bake-off winner)")
    p.add_argument("--radar", choices=["control", "voronoi"], default="control",
                   help="radar background: C1 race control surface or Voronoi")
    p.add_argument("--no-epv", action="store_true",
                   help="do not draw the C2 EPV heatmap strip")
    p.add_argument("--epv-scale", type=float, default=2.2,
                   help="scale of the EPV strip below the radar")
    return p.parse_args()


def compute_hud_lines(df) -> list[str]:
    """Live metric lines from the accumulated tracking table.

    Stage-3 core (possession, line height, sprint) plus Stage-9 shape /
    spatial / event metrics (spread, compactness, block position, field tilt,
    formation angle, pass counts + completion).
    """
    from src.analytics.adapters.video.metrics import MetricsRunner
    from src.analytics.adapters.video.events import extract_events
    from src.analytics.models import field_tilt
    from src.analytics.adapters.video.metrics.event_aggregates import pass_completion

    runner = MetricsRunner()
    table = runner.compute(df)
    events = extract_events(df)

    def get(row, name: str) -> float:
        v = row.get(name, float("nan"))
        return float(v) if v == v else float("nan")

    lines = []
    for team in (0, 1):
        col = f"team_{team}"
        if col not in table.columns:
            lines.append(f"T{team}: no data")
            continue
        row = table[col]
        poss = row.get("PossessionMetric", float("nan"))
        poss_s = f"{poss:.0%}" if poss == poss else "n/a"
        team_events = [e for e in events if e.team_id == team]
        n_pass = len([e for e in team_events if e.kind == "Pass"])
        comp = pass_completion(team_events)
        comp_s = f"{comp:.0%}" if comp == comp else "n/a"
        lines.append(
            f"T{team}: poss {poss_s} | line {get(row,'LineHeightMetric'):.1f}m "
            f"| spread {get(row,'SpreadMetric'):.1f}m "
            f"| compact {get(row,'CompactnessMetric'):.1f}m")
        lines.append(
            f"T{team}: block {get(row,'BlockPositionMetric'):.1f}m "
            f"| tilt {field_tilt(df, team):.0f}m "
            f"| angle {get(row,'FormationAngleMetric'):.0f} "
            f"| passes {n_pass} ({comp_s})")
    return lines


def main():
    args = parse_args()
    pipe = Pipeline(detector_weights=args.weights, track=True, use_homography=True)

    import cv2

    from src.config import BALL_TRACKER_ID, CLASS_BALL
    from src.analytics.adapters.video.metrics.base import TrackingFrame
    from src.analytics.adapters.video.metrics.util import PLAYER_CLASSES
    from src.analytics.models import epv_grid
    from src.analytics.models.pitch_control import (_race_surface, _track_velocities,
                                           _velocity_map, PITCH_CONTROL_NX,
                                           PITCH_CONTROL_NY)
    from src.analytics.adapters.video.reader import frame_generator

    epv = epv_grid()          # time-invariant value grid
    team_colors = {0: (255, 0, 0), 1: (0, 0, 255)}   # BGR: team0 blue, team1 red
    writer = None
    n = 0
    batch: list[dict] = []
    hud_lines: list[str] = []
    surf = None               # C1 control surface for the latest interval
    for idx, frame in enumerate(frame_generator(args.video)):
        if args.max_frames is not None and idx >= args.max_frames:
            break
        dets = pipe.track_frame(frame)
        positions = pipe._project(dets, frame)
        ball_pos, _ = pipe._track_ball(dets, frame)
        # accumulate tracking rows for the live HUD
        for d in dets:
            tid = d.tracker_id
            pos = positions.get(tid)
            batch.append({
                "frame_idx": idx,
                "timestamp_sec": idx / 25.0,
                "track_id": tid,
                "team_id": d.team_id,
                "pitch_x": pos[0] if pos is not None else np.nan,
                "pitch_y": pos[1] if pos is not None else np.nan,
                "class_name": d.class_name,
            })
        if ball_pos is not None:
            batch.append({
                "frame_idx": idx,
                "timestamp_sec": idx / 25.0,
                "track_id": BALL_TRACKER_ID,
                "team_id": np.nan,
                "pitch_x": float(ball_pos[0]),
                "pitch_y": float(ball_pos[1]),
                "class_name": CLASS_BALL,
            })
        if idx % args.hud_interval == 0 and len(batch):
            import pandas as pd
            df_batch = pd.DataFrame(
                batch, columns=TrackingFrame.REQUIRED_COLUMNS)
            hud_lines = compute_hud_lines(df_batch)
            if args.radar == "control":
                # C1 race surface for the CURRENT frame (live snapshot)
                pl = df_batch[(df_batch["class_name"].isin(PLAYER_CLASSES))
                              & df_batch["team_id"].notna()]
                cur = pl[pl["frame_idx"] == idx]
                own = cur[cur["team_id"] == 0]
                opp = cur[cur["team_id"] == 1]
                if not own.empty and not opp.empty:
                    vel = _velocity_map(_track_velocities(df_batch))
                    surf = _race_surface(own, opp, vel,
                                         PITCH_CONTROL_NX, PITCH_CONTROL_NY)
                else:
                    surf = None

        # radar frame: team dots + control/Voronoi + ball
        pts, tids = [], []
        for d in dets:
            tid = d.tracker_id
            pos = positions.get(tid)
            if tid is None or pos is None or d.class_name == CLASS_BALL:
                continue
            pts.append(pos)
            tids.append(d.team_id if d.team_id is not None else -1)
        positions_arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        team_ids = np.asarray(tids, dtype=int)

        pitch = render_pitch(pipe.pitch_mapper.config, scale=args.scale)
        if args.radar == "control" and surf is not None:
            radar = draw_control_surface(pitch, surf, scale=args.scale)
            radar = draw_voronoi_radar(
                radar, positions_arr, team_ids, team_colors,
                scale=args.scale,
                ball_pos=ball_pos if ball_pos is not None else None,
                show_cells=False)
        else:
            radar = draw_voronoi_radar(
                pitch, positions_arr, team_ids, team_colors,
                scale=args.scale,
                ball_pos=ball_pos if ball_pos is not None else None)
        if not args.no_epv:
            epv_pitch = render_pitch(pipe.pitch_mapper.config,
                                     scale=args.epv_scale)
            strip = draw_epv_heatmap(epv_pitch, epv, scale=args.epv_scale)
            strip = cv2.resize(strip, (radar.shape[1],
                                       int(radar.shape[1] * 0.18)))
            radar = np.vstack([radar, strip])
        radar = draw_hud(radar, hud_lines or [f"frame {idx}"])

        # left side = broadcast frame resized + counter
        left = cv2.resize(frame, (radar.shape[1], radar.shape[0]))
        cv2.putText(left, f"frame {idx}", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        combo = np.hstack([left, radar])
        if writer is None:
            h, w = combo.shape[:2]
            writer = cv2.VideoWriter(
                args.out, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
        writer.write(combo)
        n += 1

    if writer is not None:
        writer.release()
    print(f"wrote {args.out} ({n} frames, hud_interval={args.hud_interval})")


if __name__ == "__main__":
    main()
