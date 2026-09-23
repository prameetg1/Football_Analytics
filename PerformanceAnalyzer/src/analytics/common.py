"""Shared foundational types for the analytics layer.

These are provider-agnostic contracts that BOTH the StatsBomb source adapter
and the (dormant) Video/CV adapter rely on, so neither may hide them in their
own dependency tree. Owning them here guarantees the data-source side never
imports from the video pipeline (and vice-versa).

* `TrackingFrame` — the per-frame tracking-table schema contract. It is what
  every CV metric reads AND what the StatsBomb 360 freeze-frame alignment
  produces, so both sides agree on column names.
* `PLAYER_CLASSES` — the entity classes a metric treats as a team player.
"""

from __future__ import annotations

from src.config import CLASS_GOALKEEPER, CLASS_PLAYER

# A metric-relevant entity is a team-assigned outfield player or goalkeeper.
PLAYER_CLASSES = (CLASS_PLAYER, CLASS_GOALKEEPER)


class TrackingFrame:
    """Minimal schema contract for the per-frame tracking table.

    The tracking stage emits rows with these columns; every metric reads from
    them. We document it here so metrics stay decoupled from where the frame
    came from (CV tracking vs. StatsBomb events vs. synthetic tests).
    """

    REQUIRED_COLUMNS = [
        "frame_idx", "timestamp_sec", "track_id", "team_id",
        "pitch_x", "pitch_y", "class_name",
    ]
