"""Runtime seam for the dormant Video adapter.

`run_video_to_events` is the one place the CV pipeline would be wired into the
normalized `analytics.schema`. It is intentionally exercised only when the
adapter is enabled; while dormant it raises, so no half-built mapping is ever
quietly produced. The mapping itself is the parked-Video work this project
defers to when Video/CV is re-enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.analytics.schema import Event


@dataclass
class VideoRunResult:
    """Normalized events produced by running the CV pipeline over a video."""
    events: list[Event] = field(default_factory=list)


def run_video_to_events(*, source_path: str) -> VideoRunResult:
    """Run the CV pipeline and map its output to normalized events.

    Not implemented while the adapter is dormant. When enabled end-to-end, this
    would:
      1. `Pipeline().to_dataframe(source_path)` -> tracking frames,
      2. `extract_events(...)` -> CV event vocabulary,
      3. project CV events/positions onto `analytics.schema.Event`.
    """
    raise NotImplementedError(
        "run_video_to_events is not implemented while the Video adapter is "
        "dormant; it is the parked seam for re-enabling Video/CV input.")
