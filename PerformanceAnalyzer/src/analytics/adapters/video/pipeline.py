"""Orchestrates the full CV pipeline for a single video.

Mirrors the Roboflow `football_ai.py` flow as clean, composable steps:

    frames -> detections -> track -> assign teams -> homography -> pitch (x,y)

Improvements over the reference are documented per-stage:
  * detection: local OR hosted OR fallback (reference is hosted-only)
  * ball: box padded (reference behaviour) and kept out of the tracker
  * tracking: class-agnostic NMS before ByteTrack (reference behaviour)
  * teams: single-pass bootstrap-once assigner (reference is two-pass)
  * homography: local 32-kpt pose model with verified canonical ordering
    (reference uses a hosted 31-kpt model), metres not cm
  * projection: per-tracker distance outlier rejection (reference's ball-path
    outlier cleanup, generalised to all tracked objects)

Heavy imports are kept lazy so the package stays cheap to import and each stage
can be wired/dry-run independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.config import (
    BALL_BOX_PAD_PX,
    BALL_RECOVERY_CONF,
    BALL_RECOVERY_ENABLED,
    BALL_RECOVERY_IMGSZ,
    BALL_RECOVERY_MIN_CONF,
    BALL_RECOVERY_WINDOW_PX,
    BALL_TRACKER_ID,
    CLASS_BALL,
    METRICS_EXPORT_HZ,
    PITCH_MAX_STEP_M,
)
from src.analytics.adapters.video.detection.detection import Detection
from src.analytics.adapters.video.metrics.base import TrackingFrame
from src.analytics.adapters.video.reader import frame_generator


@dataclass
class TrackedFrame:
    """One processed frame."""
    frame_idx: int
    detections: list[Detection] = field(default_factory=list)
    frame: np.ndarray | None = None
    # Pitch-projected anchor points keyed by Detection.tracker_id (metres).
    pitch_positions: dict[int, np.ndarray] = field(default_factory=dict)
    homography: np.ndarray | None = None
    # Stage 4: Kalman-smoothed ball hypothesis on the pitch (metres).
    ball_position: np.ndarray | None = None
    # True while the hypothesis is backed by a recent measurement or a short
    # extrapolation; False once the track is declared lost.
    ball_valid: bool = False


def is_position_outlier(prev: np.ndarray, pos: np.ndarray,
                        max_step: float = PITCH_MAX_STEP_M) -> bool:
    """True if a projected pitch position teleports beyond `max_step` metres.

    Mirrors the reference's ball-path outlier cleanup, generalised to any
    tracked object: a jump larger than the threshold in a single frame is
    treated as a homography/mis-detection glitch and dropped.
    """
    return bool(np.linalg.norm(np.asarray(pos) - np.asarray(prev)) > max_step)


class Pipeline:
    """Default end-to-end pipeline over a video path."""

    def __init__(self, detector_weights: str | None = None,
                 track: bool = True, use_homography: bool = True,
                 assign_teams: bool = True, ball_recovery: bool = BALL_RECOVERY_ENABLED):
        from src.analytics.adapters.video.detection.detector import Detector

        self.detector = Detector(weights=detector_weights)
        self._do_track = track
        self._do_homography = use_homography
        self.tracker = None
        if track:
            from src.analytics.adapters.video.tracking.tracker import ByteTrackTracker
            self.tracker = ByteTrackTracker()
        self.keypoint_detector = None
        self.pitch_mapper = None
        if use_homography:
            from src.analytics.adapters.video.homography.keypoint_detector import PitchKeypointDetector
            from src.analytics.adapters.video.homography.homography import StablePitchMapper
            from src.analytics.adapters.video.homography.pitch_model import PitchConfig
            self.keypoint_detector = PitchKeypointDetector()
            self.pitch_mapper = StablePitchMapper(config=PitchConfig())
        self.team_assigner = None
        if assign_teams:
            from src.analytics.adapters.video.teams.assigner import TeamAssigner
            self.team_assigner = TeamAssigner()
        self._ball_recovery_enabled = ball_recovery
        # Last valid pitch position per tracker_id, for outlier rejection.
        self._last_pitch: dict[int, np.ndarray] = {}
        # Stage 4: pitch-plane ball Kalman tracker (needs homography).
        self.ball_tracker = None
        if use_homography:
            from src.analytics.adapters.video.tracking.ball_tracker import BallTracker
            self.ball_tracker = BallTracker()

    def track_frame(self, frame: np.ndarray) -> list[Detection]:
        dets = self.detector.infer(frame)
        # Stage 4b: ball recovery. Tiny balls are missed at full-frame imgsz;
        # when the best ball detection is weak/none and the Kalman hypothesis
        # gives a prior location, re-infer on an upscaled window around the
        # back-projected prediction so the small ball gets more pixels.
        if self._ball_recovery_enabled:
            dets = self._recover_ball(dets, frame)
        # Pad the ball box (reference behaviour) so the small fast ball stays a
        # stable target; the tracker bypasses ball detections entirely.
        dets = [
            d.padded(BALL_BOX_PAD_PX) if d.class_name == CLASS_BALL else d
            for d in dets
        ]
        if self.tracker is not None:
            dets = self.tracker.update(dets)
        if self.team_assigner is not None:
            dets = self.team_assigner.assign(frame, dets)
        return dets

    def _recover_ball(self, dets: list[Detection],
                      frame: np.ndarray) -> list[Detection]:
        """Targeted ball re-detection around the Kalman-predicted position."""
        if self.ball_tracker is None or self.pitch_mapper is None:
            return dets
        if self.ball_tracker.position is None:
            return dets
        ball_confs = [d.confidence for d in dets
                      if d.class_name == CLASS_BALL]
        best = max(ball_confs, default=0.0)
        if best >= BALL_RECOVERY_MIN_CONF:
            return dets
        H = self.pitch_mapper.current
        if H is None:
            return dets
        px = self.pitch_mapper.transform_to_pixels(
            np.asarray([self.ball_tracker.position], dtype=np.float64))
        if len(px) == 0:
            return dets
        cx, cy = px[0]
        recovered = self.detector.infer_region(
            frame, (cx, cy), BALL_RECOVERY_WINDOW_PX / 2,
            imgsz=BALL_RECOVERY_IMGSZ, conf=BALL_RECOVERY_CONF)
        balls = [d for d in recovered if d.class_name == CLASS_BALL]
        if balls:
            # keep the strongest recovered ball; it outranks a weak full-frame
            # ball so the Kalman gate sees the freshest evidence.
            strongest = max(balls, key=lambda d: d.confidence)
            dets = [d for d in dets if d.class_name != CLASS_BALL]
            dets.append(strongest)
        return dets

    def _project(self, dets: list[Detection],
                 frame: np.ndarray) -> dict[int, np.ndarray]:
        """Update homography from confident pitch keypoints; project anchors."""
        positions: dict[int, np.ndarray] = {}
        if self.keypoint_detector is None or self.pitch_mapper is None:
            return positions

        kp = self.keypoint_detector.detect(frame)
        kp = self.pitch_mapper.filter_keypoints(kp)
        if kp.valid:
            canonical = self.pitch_mapper.config.vertices[kp.indices]
            # Stage 6: smooth keypoint jitter before the fit (cuts reset the
            # prior inside update() via the shot-change detector).
            sm_xy = self.pitch_mapper.smooth_keypoints(kp)
            self.pitch_mapper.update(sm_xy, canonical)

        H = self.pitch_mapper.current
        if H is None:
            return positions

        anchors = np.asarray(
            [d.bottom_center for d in dets], dtype=np.float64,
        ).reshape(-1, 2)
        pitch = self.pitch_mapper.transform_points(anchors)

        for d, (x, y) in zip(dets, pitch):
            if d.tracker_id is None:
                continue
            pos = np.array([x, y])
            prev = self._last_pitch.get(d.tracker_id)
            if prev is not None and is_position_outlier(prev, pos):
                continue
            self._last_pitch[d.tracker_id] = pos
            positions[d.tracker_id] = pos
        return positions

    def _track_ball(self, dets: list[Detection], frame: np.ndarray,
                    dt: float = 1.0 / 25.0) -> tuple[np.ndarray | None, bool]:
        """Feed the projected ball into the Kalman hypothesis stream.

        Returns (ball_position, valid). The ball box is padded and kept out of
        ByteTrack (reference behaviour); here we additionally project its
        bottom-center through the homography and let the Kalman filter smooth
        it, predict through detection misses, and gate against glitches.
        """
        if self.ball_tracker is None or self.pitch_mapper is None:
            return None, False
        H = self.pitch_mapper.current
        if H is None:
            return self.ball_tracker.step(None, dt)

        balls = [d for d in dets if d.class_name == CLASS_BALL]
        if not balls:
            return self.ball_tracker.step(None, dt)
        best = max(balls, key=lambda d: d.confidence)
        pt = self.pitch_mapper.transform_points(
            np.asarray([best.bottom_center], dtype=np.float64))
        if len(pt) == 0:
            return self.ball_tracker.step(None, dt)
        return self.ball_tracker.step(pt[0], dt)

    def process(self, source_path: str, max_frames: int | None = None):
        """Yield a TrackedFrame per processed video frame."""
        from src.analytics.adapters.video.reader import VideoInfo

        fps = VideoInfo.from_path(source_path).fps or 25.0
        dt = 1.0 / fps
        for idx, frame in enumerate(frame_generator(source_path)):
            if max_frames is not None and idx >= max_frames:
                break
            dets = self.track_frame(frame)
            if self._do_homography:
                positions = self._project(dets, frame)
                ball_pos, ball_valid = self._track_ball(dets, frame, dt)
            else:
                positions = {}
                ball_pos, ball_valid = None, False
            yield TrackedFrame(
                frame_idx=idx, detections=dets, frame=frame,
                pitch_positions=positions,
                homography=self.pitch_mapper.current if self.pitch_mapper else None,
                ball_position=ball_pos, ball_valid=ball_valid,
            )

    def to_dataframe(self, source_path: str, max_frames: int | None = None,
                     out_path: str | None = None,
                     analysis_hz: float | None = None) -> "pd.DataFrame":
        """Export per-frame tracking rows to the `TrackingFrame` schema.

        One row per detected object per frame, even when the object had no
        valid pitch projection (pitch_x/y stay NaN) so track continuity and
        miss patterns survive for the metric layer. Timestamps use the source
        video fps.

        Stage 5 adaptive frame-rate: detection/tracking still run at the
        source's full FPS (track continuity), but the exported table is sampled
        down to `analysis_hz` (default METRICS_EXPORT_HZ) so the metric layer
        and table stay small. Distance/velocity metrics recompute dt from the
        exported timestamps, so the physics is preserved.

        Args:
            source_path: input video path.
            max_frames: cap on frames processed (None == whole video).
            out_path: optional CSV/parquet target (by extension).
            analysis_hz: target export rate in Hz (None == METRICS_EXPORT_HZ;
                0/None-equivalent disables downsampling via METRICS_EXPORT_HZ=0).

        Returns:
            DataFrame with columns `TrackingFrame.REQUIRED_COLUMNS`.
        """
        import pandas as pd

        from src.analytics.adapters.video.reader import VideoInfo

        fps = VideoInfo.from_path(source_path).fps or 25.0
        # sample one frame every `stride` source frames to hit analysis_hz.
        hz = METRICS_EXPORT_HZ if analysis_hz is None else analysis_hz
        stride = max(1, int(round(fps / hz))) if hz else 1

        rows: list[dict] = []
        for tf in self.process(source_path, max_frames):
            if tf.frame_idx % stride != 0:
                continue
            ts = tf.frame_idx / fps
            for d in tf.detections:
                # The ball lives in its own hypothesis stream (Stage 4): when a
                # hypothesis exists it is exported below with pitch positions,
                # so the raw (unprojectable) detection row is skipped.
                if d.class_name == CLASS_BALL and tf.ball_position is not None:
                    continue
                pos = (tf.pitch_positions.get(d.tracker_id)
                       if d.tracker_id is not None else None)
                rows.append({
                    "frame_idx": tf.frame_idx,
                    "timestamp_sec": round(ts, 6),
                    "track_id": d.tracker_id if d.tracker_id is not None else np.nan,
                    "team_id": d.team_id if d.team_id is not None else np.nan,
                    "pitch_x": float(pos[0]) if pos is not None else np.nan,
                    "pitch_y": float(pos[1]) if pos is not None else np.nan,
                    "class_name": d.class_name,
                })
            if tf.ball_position is not None:
                rows.append({
                    "frame_idx": tf.frame_idx,
                    "timestamp_sec": round(ts, 6),
                    "track_id": BALL_TRACKER_ID,
                    "team_id": np.nan,
                    "pitch_x": float(tf.ball_position[0]),
                    "pitch_y": float(tf.ball_position[1]),
                    "class_name": CLASS_BALL,
                })

        df = pd.DataFrame(rows, columns=TrackingFrame.REQUIRED_COLUMNS)
        if out_path:
            if out_path.endswith(".parquet"):
                df.to_parquet(out_path, index=False)
            else:
                df.to_csv(out_path, index=False)
        return df
