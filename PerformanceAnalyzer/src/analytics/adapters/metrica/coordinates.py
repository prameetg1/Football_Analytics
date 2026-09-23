"""Metrica Sports -> canonical pitch coordinate mapping.

Metrica sample data publishes tracking/event positions normalized to [0,1] x
[0,1] with the origin at the TOP-LEFT, pitch drawn so the home team attacks
left->right and the kick-off point at (0.5, 0.5). It is a fixed broadcast
frame (not per-possession), like the IDSSE/DFL export, so we keep the same
orientation for both teams.

Our canonical frame is x in [0,120] (length), y in [0,70] (width), with y = 0
at the BOTTOM (consistent with the StatsBomb/IDSSE adapters). Because Metrica
uses a top-left origin, we flip y (1 - y) before rescaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricaPitch:
    """Metrica [0,1]x[0,1] (origin top-left) -> canonical [0,120]x[0,70]."""
    length: float = 120.0
    width: float = 70.0

    def _finite(self, v: float) -> bool:
        return math.isfinite(v)

    def to_finite(self, x: float, y: float) -> tuple[float, float] | None:
        """Return canonical metres, or None if any input is non-finite/NaN."""
        if not (self._finite(x) and self._finite(y)):
            return None
        cx = x * self.length
        cy = (1.0 - y) * self.width   # flip y: Metrica y=0 is top
        return round(min(max(cx, 0.0), self.length), 4), \
            round(min(max(cy, 0.0), self.width), 4)

    def to_canonical(self, x: float, y: float) -> tuple[float, float]:
        """Normalized [0,1] coords (origin top-left) -> canonical metres.

        Non-finite (NaN/empty) inputs clamp to 0; callers that must drop such
        players should use `to_finite` instead.
        """
        cx = x * self.length
        cy = (1.0 - y) * self.width   # flip y: Metrica y=0 is top
        return round(min(max(cx, 0.0), self.length), 4), \
            round(min(max(cy, 0.0), self.width), 4)
