"""SkillCorner -> canonical pitch coordinate mapping.

SkillCorner tracks positions in meters, zero at the pitch centre, on a pitch of
`pitch_length` x `pitch_width` metres. Our canonical frame is x in [0,120],
y in [0,70], defending goal at x=0 for both sides. We rescale linearly and
clamp, and mirror when the player's team attacks against the canonical frame
(away team; home team plays left->right).

SkillCorner also marks teams by `home_team_side`: a two-element list that tells
per period which direction each team attacks ("left_to_right" / "right_to_left").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PitchSpec:
    """Physical dimensions + canonicalization metadata for a SkillCorner match."""
    length: float = 104.0
    width: float = 68.0
    canonical_length: float = 120.0
    canonical_width: float = 70.0

    def to_canonical(self, x: float, y: float) -> tuple[float, float]:
        """Meter coords (origin centre) -> canonical [0,120]x[0,70]."""
        cx = (x + self.length / 2.0) / self.length * self.canonical_length
        cy = (y + self.width / 2.0) / self.width * self.canonical_width
        return round(min(max(cx, 0.0), self.canonical_length), 4), \
            round(min(max(cy, 0.0), self.canonical_width), 4)