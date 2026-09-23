"""Wyscout open-data adapter (events + lineups).

Wyscout positions (Pappalardo et al. 2019) are in [0,100] x [0,100] percent of
the whole pitch (origin top-left), *always* in the home team's attacking
direction. We normalize to the canonical frame: mirror x for the away team so
both teams attack toward x = 120, rescale [0,100] -> [0,120] x [0,70].

The JSON format here is the "regular Wyscout form" served by the kloppy mirror
(`processed-v2/files/<match_id>.json`): a dict with `events`, `teams`, and
`players` lists. Event types come from `eventName`; pass accuracy comes from
the `tags` array (1801 = accurate).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WyscoutPitch:
    """Wyscout [0,100]x[0,100] coordinate system -> canonical [0,120]x[0,70].

    Wyscout puts the origin at the top-left and always shows the home team
    attacking left->right. For the away team we mirror x so both teams defend
    toward x=0 (our canonical frame).
    """
    length: float = 120.0
    width: float = 70.0

    def to_canonical(self, x: float, y: float, home_attack_right: bool) -> tuple[float, float]:
        cx = x / 100.0 * self.length
        cy = (100.0 - y) / 100.0 * self.width   # flip y: Wyscout origin top-left
        if not home_attack_right:
            cx = self.length - cx
        return round(cx, 4), round(cy, 4)