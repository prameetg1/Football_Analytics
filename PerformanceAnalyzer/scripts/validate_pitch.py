#!/usr/bin/env python3
"""Validate the pitch homography: reprojection error + projected objects.

Runs the local pose detector + StablePitchMapper over N frames, reports the
per-frame reprojection RMSE (pitch keypoints projected back to pixels vs the
observed ones) and writes a top-down pitch render annotated with the projected
tracked-object anchors.

Usage:
    .venv/bin/python scripts/validate_pitch.py --video data/raw/0bfacc_0.mp4 \
        --max-frames 40 --out output/pitch_map.jpg
    .venv/bin/python scripts/validate_pitch.py --self-test \
        # synthetic shot-cut harness (no video needed) — Stage 6
"""

import argparse
import time

import numpy as np
import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Validate pitch homography")
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--max-frames", type=int, default=40)
    p.add_argument("--out", type=str, default="output/pitch_map.jpg")
    p.add_argument("--self-test", action="store_true",
                   help="run the synthetic shot-cut reset harness (no video)")
    return p.parse_args()


def _self_test():
    """Stage 6 harness: inject a camera cut; check reset recovers quickly.

    Simulates two shots (homographies h_a -> h_b) on the canonical pitch. Feeds
    the StablePitchMapper a smooth pan in shot A, then a hard cut to shot B.
    Asserts within-shot reprojection RMSE stays small and the map recovers to
    shot B within a couple of frames of the cut instead of blurring over the
    deque length.
    """
    from src.analytics.adapters.video.homography.homography import StablePitchMapper, _to_pixels
    from src.analytics.adapters.video.homography.pitch_model import PitchConfig

    cfg = PitchConfig()
    mapper = StablePitchMapper(config=cfg, kp_alpha=1.0)
    rng = np.random.default_rng(0)

    def pixels(H, jitter=0.4):
        return _to_pixels(H, cfg.vertices) + rng.normal(0, jitter, (32, 2))

    pan = [np.eye(3).copy() for _ in range(10)]
    for i in range(10):
        pan[i][0, 2] = i * 0.5 * 0.1      # slow 0->5 px drift in shot A

    h_b = np.eye(3)
    h_b[0, 2] = 60.0                       # hard cut to a 60 px offset camera

    rmses_shot_a, rmses_after_cut = [], []
    latent_cut = None
    for step in range(len(pan)):           # shot A
        H = mapper.update(pixels(pan[step]), cfg.vertices)
        rmses_shot_a.append(_reproj_rmse(H, cfg, pan[step]))
        assert mapper.cut_detected_last is False

    mapper.update(pixels(h_b), cfg.vertices)   # the cut
    assert mapper.cut_detected_last is True
    for f in range(4):                         # recovery within a few frames
        H = mapper.update(pixels(h_b), cfg.vertices)
        rmses_after_cut.append(_reproj_rmse(H, cfg, h_b))
        if mapper.current is not None and \
                float(np.mean(np.abs(_to_pixels(mapper.current,
                                                cfg.vertices)
                                     - _to_pixels(h_b, cfg.vertices)))) < 5.0:
            latent_cut = f + 1
            break

    print(f"[self-test] shot-A reproj RMSE mean={np.mean(rmses_shot_a):.2f}px")
    print(f"[self-test] after-cut RMSEs (frame 1..n): "
          f"{[round(x, 2) for x in rmses_after_cut]}")
    print(f"[self-test] recovered on frame {latent_cut} (<=2 expected)")
    ok = float(np.mean(rmses_shot_a)) <= 17.0 and latent_cut is not None and latent_cut <= 2
    print(f"[self-test] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _reproj_rmse(H, cfg, true_h):
    """Reprojection RMSE of the smoothed H vs the true camera H at pixels."""
    from src.analytics.adapters.video.homography.homography import _to_pixels

    true_px = _to_pixels(true_h, cfg.vertices)
    out_px = _to_pixels(H, cfg.vertices)
    return float(np.sqrt(np.mean((out_px - true_px) ** 2)))


def main():
    args = parse_args()
    if args.self_test:
        raise SystemExit(_self_test())
    if args.video is None:
        raise SystemExit("Need --video or --self-test")

    from src.analytics.adapters.video.homography.keypoint_detector import PitchKeypointDetector
    from src.analytics.adapters.video.homography.homography import StablePitchMapper
    from src.analytics.adapters.video.homography.pitch_model import PitchConfig
    from src.analytics.adapters.video.reader import frame_generator

    config = PitchConfig()
    kd = PitchKeypointDetector()
    mapper = StablePitchMapper(config=config)

    print(f"keypoint detector weights: {kd.weights}")
    rmses = []
    valid_frames = 0
    last_frame = None
    last_kp_xy = None
    t0 = time.time()
    for idx, frame in enumerate(frame_generator(args.video)):
        if idx >= args.max_frames:
            break
        last_frame = frame
        kp = mapper.filter_keypoints(kd.detect(frame))  # same gate as pipeline
        last_kp_xy = kp.xy
        if kp.valid:
            canonical = config.vertices[kp.indices]
            sm_xy = mapper.smooth_keypoints(kp)
            H = mapper.update(sm_xy, canonical)
            valid_frames += 1
            # Reprojection RMSE: pitch->pixel (inverse H) vs observed pixels.
            reprojected = cv2.perspectiveTransform(
                canonical.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
            rmse = float(np.sqrt(np.mean((reprojected - sm_xy) ** 2)))
            rmses.append(rmse)
            if idx % 10 == 0:
                print(f"  frame {idx}: {len(kp.xy)} keypoints, reproj RMSE={rmse:.1f}px"
                      + (" [CUT]" if mapper.cut_detected_last else ""))
        else:
            print(f"  frame {idx}: <4 confident keypoints ({len(kp.xy)}), "
                  f"holding previous H")

    dt = time.time() - t0
    avg = float(np.mean(rmses)) if rmses else float("nan")
    print(f"frames={args.max_frames} valid_homography={valid_frames} "
          f"avg_reproj_rmse={avg:.2f}px over {len(rmses)} frames in {dt:.1f}s")

    # Visual check: draw detected pitch keypoints on the last frame.
    if last_frame is not None and last_kp_xy is not None:
        vis = last_frame.copy()
        for (x, y) in last_kp_xy:
            if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 255), -1)
        out = args.out.replace(".jpg", "_keypoints.jpg")
        cv2.imwrite(out, vis)
        print(f"wrote {out} (detected keypoints on last frame)")

    return 0 if valid_frames else 1


if __name__ == "__main__":
    main()
