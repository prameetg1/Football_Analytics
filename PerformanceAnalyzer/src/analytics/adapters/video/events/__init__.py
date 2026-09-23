"""Stage 9 event-extraction layer.

Discrete events (pass, recovery, shot, clearance, set-piece, pressure, carry,
through ball, duel) are derived from the per-frame tracking table and validated
against StatsBomb ground truth via the Stage-9 gate
(`src.analytics.adapters.statsbomb.validate.event_match`).
"""

from src.analytics.adapters.video.events.extractor import (
    ball_ownership_chain,
    events_by_kind,
    extract_events,
)
from src.analytics.adapters.video.events.models import Event

__all__ = [
    "Event",
    "ball_ownership_chain",
    "extract_events",
    "events_by_kind",
]
