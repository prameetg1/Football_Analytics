"""Event model for the Stage-9 event-extraction layer.

An `Event` is a discrete on-pitch happening derived from the per-frame tracking
table (as opposed to a scalar metric). Events are what feed the event-aggregate
metrics (Stage B), the pitch-control/EPV layer (Stage C), and the StatsBomb
gate (recall/precision per event class).

`kind` uses the StatsBomb vocabulary where one exists so gate matching stays
direct: "Pass", "Shot", "Ball Recovery"/"Dispossessed" (turnovers), "Clearance",
"Pressure". Fields that aren't universally present live in `qualifiers` so the
matching layer can stay a uniform (timestamp, x, y) protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Event:
    """A discrete on-pitch event at a canonical pitch-metre location."""

    kind: str                       # StatsBomb-style event type name
    timestamp_sec: float            # time since kick-off (s)
    x: float                        # canonical pitch_x (m)
    y: float                        # canonical pitch_y (m)
    team_id: int | None = None      # owning/attacking team (0/1), NaN-aware
    qualifiers: dict[str, object] = field(default_factory=dict)

    def as_tuple(self) -> tuple[float, float, float]:
        """(timestamp_sec, x, y) — the uniform protocol used by gate matching."""
        return (self.timestamp_sec, self.x, self.y)
