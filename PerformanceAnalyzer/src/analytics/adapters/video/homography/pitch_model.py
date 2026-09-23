"""Canonical metric-pitch model.

Defines the real-world pitch geometry and the ORDERED canonical keypoints a
pitch keypoint detector maps ONTO when computing the homography.

The keypoint ordering matches the `football-field-detection` Roboflow dataset
version 12 (`kpt_shape = [32, 3]`): the 32 keypoints are index-aligned with
Roboflow's `SoccerPitchConfiguration.vertices`, so
`canonical[i] == SoccerPitchConfiguration.vertices[i]` (converted to metres).
That alignment is what makes pixel->pitch homography well-posed for a model
fine-tuned on that dataset (see `References/train_pitch_keypoint_detector.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _build_canonical_vertices_m() -> np.ndarray:
    """Replicate `SoccerPitchConfiguration.vertices` (32 points, cm) in metres."""
    # Dimensions in centimetres, from Roboflow's SoccerPitchConfiguration.
    W, L = 7000, 12000
    PW, PL = 4100, 2015   # penalty box width / length
    GW, GL = 1832, 550    # goal box width / length
    CR = 915              # centre circle radius
    PS = 1100             # penalty spot distance
    hw, hl = W / 2, L / 2
    v = [
        (0, 0),                                  # 01
        (0, (W - PW) / 2),                       # 02
        (0, (W - GW) / 2),                       # 03
        (0, (W + GW) / 2),                       # 04
        (0, (W + PW) / 2),                       # 05
        (0, W),                                  # 06
        (GL, (W - GW) / 2),                      # 07
        (GL, (W + GW) / 2),                      # 08
        (PS, hw),                                # 09
        (PL, (W - PW) / 2),                      # 10
        (PL, (W - GW) / 2),                      # 11
        (PL, (W + GW) / 2),                      # 12
        (PL, (W + PW) / 2),                      # 13
        (hl, 0),                                 # 14
        (hl, hw - CR),                           # 15
        (hl, hw + CR),                           # 16
        (hl, W),                                 # 17
        (L - PL, (W - PW) / 2),                  # 18
        (L - PL, (W - GW) / 2),                  # 19
        (L - PL, (W + GW) / 2),                  # 20
        (L - PL, (W + PW) / 2),                  # 21
        (L - PS, hw),                            # 22
        (L - GL, (W - GW) / 2),                  # 23
        (L - GL, (W + GW) / 2),                  # 24
        (L, 0),                                  # 25
        (L, (W - PW) / 2),                       # 26
        (L, (W - GW) / 2),                       # 27
        (L, (W + GW) / 2),                       # 28
        (L, (W + PW) / 2),                       # 29
        (L, W),                                  # 30
        (hl - CR, hw),                           # 31
        (hl + CR, hw),                           # 32
    ]
    return np.asarray(v, dtype=np.float64) / 100.0  # cm -> m


# Labels as emitted by the `football-field-detection` keypoint model ordering
# (index-aligned with `vertices`).
KEYPOINT_LABELS = [
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
    "11", "12", "13", "15", "16", "17", "18", "20", "21", "22",
    "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
    "14", "19",
]


@dataclass
class PitchConfig:
    """Real-world pitch geometry (metres) + canonical keypoints.

    `vertices` is the (32, 2) canonical keypoint array in metres, in the exact
    order the fine-tuned keypoint detector emits them.
    """
    length: float = 120.0      # x axis (m)
    width: float = 70.0        # y axis (m)
    padding: int = 50          # px padding when rendering the pitch image

    keypoint_names: list[str] = field(default_factory=lambda: list(KEYPOINT_LABELS))

    @property
    def vertices(self) -> np.ndarray:
        """(32, 2) canonical keypoints in metres: [[x, y], ...]."""
        return _build_canonical_vertices_m()


def render_pitch(config: PitchConfig, scale: float = 0.1) -> np.ndarray:
    """Render a simple top-down pitch image (BGR) for viz.

    Image shape is (height, width) where height == width_scaled and
    width == length_scaled (broadcast camera looks down the length, so we lay
    the length along the horizontal axis).
    """
    import cv2

    H = int(config.width * scale) + 2 * config.padding
    W = int(config.length * scale) + 2 * config.padding
    img = np.full((H, W, 3), 46, dtype=np.uint8)

    x0, y0 = config.padding, config.padding
    x1 = int(config.length * scale) + config.padding
    y1 = int(config.width * scale) + config.padding
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 2)
    cv2.line(img, (x0 + (x1 - x0) // 2, y0), (x0 + (x1 - x0) // 2, y1),
             (255, 255, 255), 2)
    cxx = x0 + (x1 - x0) // 2
    cyy = y0 + (y1 - y0) // 2
    cv2.circle(img, (cxx, cyy), max(1, int(9.15 * scale)), (255, 255, 255), 2)
    return img
