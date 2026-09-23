"""Detection data structures.

`Detection` is the canonical cross-layer object produced by the detector and
consumed by tracking/teams/homography. Kept free of any single library so the
pipeline stages stay decoupled.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    """A single scene object locked in a frame."""
    tracker_id: int | None = None
    class_id: int = -1
    class_name: str = ""
    confidence: float = 0.0
    xyxy: tuple[float, float, float, float] = (0, 0, 0, 0)
    # bottom-centre anchor (x, y) in the source image (filled by homography)
    anchor_xy: tuple[float, float] | None = None
    # team id assigned by the TeamAssigner (0..n_teams-1) or None.
    team_id: int | None = None

    @property
    def x1(self) -> float:
        return self.xyxy[0]

    @property
    def y1(self) -> float:
        return self.xyxy[1]

    @property
    def x2(self) -> float:
        return self.xyxy[2]

    @property
    def y2(self) -> float:
        return self.xyxy[3]

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Contact point under the feet — used for homography mapping."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def padded(self, px: float) -> "Detection":
        """Return a copy with the box expanded by `px` on every side.

        The reference pads the ball box (10 px) so the small, fast-moving ball
        stays a stable detection/target across frames.
        """
        x1, y1, x2, y2 = self.xyxy
        return Detection(
            tracker_id=self.tracker_id,
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.confidence,
            xyxy=(max(0.0, x1 - px), max(0.0, y1 - px),
                  x2 + px, y2 + px),
            anchor_xy=self.anchor_xy,
            team_id=self.team_id,
        )