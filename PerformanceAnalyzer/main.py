#!/usr/bin/env python3
"""PerformanceAnalyzer CLI — CV-driven football analytics.

Sprint 0 provides the pipeline skeleton plus a dry-run that pulls a few frames
and prints detection/layout so the stack is proven without needing API keys.

Use it like:
    python main.py --video <path> --max-frames 20 --dry-run
    python main.py --video <path> --detect
    python main.py --video <path> --metrics            # per-team metric table
"""

import argparse
import time

from src.config import DETECTION_FALLBACK_MODEL


def parse_args():
    p = argparse.ArgumentParser(description="PerformanceAnalyzer: CV football analytics")
    p.add_argument("--video", type=str, required=True, help="Path to input video")
    p.add_argument("--detector", type=str, default=None,
                   help="YOLO weights (.pt) OR hosted Roboflow model id")
    p.add_argument("--roboflow", action="store_true",
                   help="Use the hosted fine-tuned football detector "
                        "(football-players-detection-3zvbc/11; needs ROBOFLOW_API_KEY)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Only process the first N frames")
    p.add_argument("--dry-run", action="store_true",
                   help="Load pipeline and process a few frames, print summary")
    p.add_argument("--json", type=str, default=None, help="Export detections to JSON")
    p.add_argument("--metrics", action="store_true",
                   help="Export the tracking table and print the per-team "
                        "metric report (velocity, distance, sprints, possession, line)")
    p.add_argument("--analysis-hz", type=float, default=None,
                   help="Metric export rate in Hz (default METRICS_EXPORT_HZ; "
                        "0 == full frame rate)")
    p.add_argument("--metrics-out", type=str, default=None,
                   help="Write the tracking table (csv/parquet) when --metrics")
    p.add_argument("--no-teams", action="store_true",
                   help="Disable team assignment (all players unassigned)")
    return p.parse_args()


def run_dry_run(args):
    """Prove the detection/tracking stack loads and runs on real frames."""
    from src.analytics.adapters.video.reader import VideoInfo, frame_generator

    print(f"Probing video: {args.video}")
    info = VideoInfo.from_path(args.video)
    print(f"  size={info.width}x{info.height} fps={info.fps:.1f} "
          f"frames={info.total_frames}")

    n = args.max_frames or 5
    gen = frame_generator(args.video)
    frames = [next(gen, None) for _ in range(n)]
    frames = [f for f in frames if f is not None]
    print(f"  captured {len(frames)} frames for the dry run")

    from src.analytics.adapters.video.detection.detector import Detector
    from src.config import ROBOFLOW_PLAYER_MODEL_ID

    if args.roboflow:
        weights = ROBOFLOW_PLAYER_MODEL_ID
        print(f"  using hosted football detector id={weights}")
    else:
        weights = args.detector or DETECTION_FALLBACK_MODEL
        print(f"  loading detector weights={weights} ...")
    detector = Detector(weights=weights)
    total = 0
    t0 = time.time()
    for i, frame in enumerate(frames):
        dets = detector.infer(frame)
        total += len(dets)
        if i < 3:
            classes = sorted({d.class_name for d in dets})
            print(f"    frame {i}: {len(dets)} detections -> {classes}")
    dt = time.time() - t0
    print(f"  pipeline OK: {total} detections over {len(frames)} frames "
          f"in {dt:.2f}s")
    return total


def run_metrics(args):
    """Export the tracking table and print the per-team metric report."""
    from src.analytics.adapters.video.metrics import MetricsRunner
    from src.analytics.adapters.video.pipeline import Pipeline

    pipe = Pipeline(detector_weights=args.detector, track=True,
                    use_homography=True, assign_teams=not args.no_teams)
    print(f"Computing metrics for {args.video} "
          f"(analysis_hz={args.analysis_hz or 'default'}) ...")
    t0 = time.time()
    df = pipe.to_dataframe(args.video, max_frames=args.max_frames,
                           out_path=args.metrics_out,
                           analysis_hz=args.analysis_hz)
    dt = time.time() - t0
    print(f"tracking table: {len(df)} rows in {dt:.1f}s")

    if args.metrics_out:
        print(f"  wrote {args.metrics_out}")

    runner = MetricsRunner()
    table = runner.compute(df)
    print("\nPer-team metrics:")
    print(table.to_string())
    return table


def main():
    args = parse_args()
    if args.dry_run:
        run_dry_run(args)
        return
    if args.metrics:
        run_metrics(args)
        return
    from src.analytics.adapters.video.pipeline import Pipeline

    pipe = Pipeline(detector_weights=args.detector)
    print(f"Processing {args.video} ...")
    count = 0
    for tf in pipe.process(args.video, max_frames=args.max_frames):
        count += 1
        if count % 25 == 0:
            print(f"  frame {tf.frame_idx}: {len(tf.detections)} tracked objects")
    print(f"done ({count} frames)")


if __name__ == "__main__":
    main()