"""Stage 4 — ball-hypothesis Kalman tracker (pitch plane, metres).

The reference has NO ball tracker: it projects each raw ball detection and then
cleans the resulting path with a fixed distance threshold (that cleanup is
already mirrored as `src.analytics.adapters.video.pipeline.is_position_outlier`). This stage upgrades the
ball stream to a real track:

  * a constant-velocity Kalman filter in canonical-pitch metres smooths the raw
    projected bottom-center position;
  * it predicts forward when the ball is undetected, keeping a valid hypothesis
    through brief occlusions / detector misses;
  * detections are associated with a Mahalanobis gate so a homography glitch
    cannot teleport the trail, while a genuine re-detection (new pass / restart)
    re-anchors the track.

Gravity/drag note: the ball's 3D flight is not recoverable from its 2D
bottom-center projection, so a constant-velocity model with tuned process noise
is the standard (and honest) 2D-pitch tracking choice. The association gate
re-anchors on airborne re-detections.
"""

from __future__ import annotations

import numpy as np

from src.config import (
    BALL_ASSOCIATION_GATE_M,
    BALL_KALMAN_MEASUREMENT_NOISE,
    BALL_KALMAN_PROCESS_NOISE,
    BALL_MAX_PREDICT_FRAMES,
)


class BallTracker:
    """Kalman filter for the ball's pitch position (x, y) in metres.

    State vector: [x, y, vx, vy]. Measurements are projected bottom-center
    points. `step` returns `(position | None, valid)` every frame.
    """

    def __init__(self, process_noise: float = BALL_KALMAN_PROCESS_NOISE,
                 measurement_noise: float = BALL_KALMAN_MEASUREMENT_NOISE,
                 association_gate_m: float = BALL_ASSOCIATION_GATE_M,
                 max_predict_frames: int = BALL_MAX_PREDICT_FRAMES,
                 dt: float = 1.0 / 25.0):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.gate = association_gate_m
        self.max_predict = max_predict_frames
        self.dt = dt
        self._x: np.ndarray | None = None
        self._P: np.ndarray | None = None
        self._frames_since = 0
        self._build_matrices(dt)

    # -- model construction -------------------------------------------------
    def _build_matrices(self, dt: float):
        self.dt = float(dt)
        dt2 = self.dt * self.dt
        A = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        # Piecewise-constant white-acceleration Q block per axis.
        q2 = np.array([[dt2 * dt2 / 4, self.dt * dt2 / 2],
                       [self.dt * dt2 / 2, dt2]], dtype=np.float64)
        Q = np.zeros((4, 4))
        Q[0:2, 0:2] = q2
        Q[2:4, 2:4] = q2
        Q *= self.process_noise
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        R = np.eye(2) * self.measurement_noise
        self._A, self._Q, self._H, self._R = A, Q, H, R

    # -- public state -------------------------------------------------------
    @property
    def initialized(self) -> bool:
        return self._x is not None

    @property
    def frames_since_measurement(self) -> int:
        return self._frames_since

    @property
    def position(self) -> np.ndarray | None:
        return None if self._x is None else self._x[:2].copy()

    @property
    def velocity(self) -> np.ndarray | None:
        return None if self._x is None else self._x[2:4].copy()

    def reset(self) -> None:
        self._x = None
        self._P = None
        self._frames_since = 0

    # -- core ---------------------------------------------------------------
    def step(self, measurement: np.ndarray | None,
             dt: float | None = None) -> tuple[np.ndarray | None, bool]:
        """Advance one frame; return (hypothesis_position, valid).

        `measurement` is the projected ball point in pitch metres, or None when
        no ball was detected. The hypothesis stays valid while predicting within
        `max_predict_frames` of the last real measurement.
        """
        if dt is not None and dt != self.dt:
            self._build_matrices(dt)

        self._frames_since += 1
        self._predict()

        if measurement is None:
            return self._hypothesis()

        m = np.asarray(measurement, dtype=np.float64).ravel()[:2]
        if not self.initialized:
            self._init(m)
            return self._hypothesis()

        innov = m - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        try:
            mahalanobis_sq = float(innov.T @ np.linalg.solve(S, innov))
        except np.linalg.LinAlgError:
            mahalanobis_sq = float("inf")

        if mahalanobis_sq > self.gate * self.gate:
            # Outside the gate. A single glitch frame right after a real update
            # is ignored; a persistent far measurement (reappearing ball after
            # a miss) re-anchors the track as a new ball event.
            if self._frames_since < 2:
                return self._hypothesis()
            self._init(m)
            return self._hypothesis()

        self._update(m)
        return self._hypothesis()

    # -- internals ----------------------------------------------------------
    def _hypothesis(self) -> tuple[np.ndarray | None, bool]:
        if not self.initialized:
            return None, False
        if self._frames_since <= self.max_predict:
            return self._x[:2].copy(), True
        return None, False

    def _predict(self) -> None:
        if not self.initialized:
            return
        self._x = self._A @ self._x
        self._P = self._A @ self._P @ self._A.T + self._Q

    def _init(self, m: np.ndarray) -> None:
        self._x = np.array([m[0], m[1], 0.0, 0.0])
        self._P = np.diag([1.0, 1.0, 4.0, 4.0])
        self._frames_since = 0

    def _update(self, m: np.ndarray) -> None:
        innov = m - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = np.linalg.solve(S, (self._P @ self._H.T).T).T
        self._x = self._x + K @ innov
        I = np.eye(4)
        self._P = (I - K @ self._H) @ self._P
        self._frames_since = 0
