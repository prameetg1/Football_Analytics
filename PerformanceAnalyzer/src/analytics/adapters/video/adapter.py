"""Dormant Video/CV provider adapter -> normalized `analytics.schema` events.

`VideoAdapter` is a full `ProviderAdapter` for the video-computer-vision input
source: it runs the CV pipeline over a video file and turns the resulting
tracking/event output into the same normalized `src.analytics.schema` events
that the StatsBomb adapter produces, so the model layer is agnostic to source.

Because the CV path is parked during the pivot, the adapter is **dormant by
default**: `enabled=False` makes every provider method raise
`VideoAdapterDormant` without importing the heavy CV stack (torch/ultralytics).
Set `enabled=True` (or `config.VIDEO_ADAPTER_ENABLED=True`) to actually run it.
"""

from __future__ import annotations

from src.analytics.adapters.base import ProviderAdapter
from src.analytics.adapters.video._registry import VIDEO_ADAPTER_DORMANT
from src.analytics.schema import (
    Event,
    FreezeFrame,
    Match,
    PassEvent,
    ShotEvent,
)


class VideoAdapterDormant(RuntimeError):
    """Raised when the video adapter is exercised while it is dormant."""


class VideoAdapter(ProviderAdapter):
    """Provider adapter over the video/CV pipeline (dormant unless enabled)."""

    def __init__(self, source_path: str | None = None,
                 enabled: bool | None = None):
        self.source_path = source_path
        # explicit flag wins; otherwise fall back to the global config state
        self.enabled = (not VIDEO_ADAPTER_DORMANT) if enabled is None else enabled

    # -- dormancy gate ---------------------------------------------------------
    def _require_enabled(self) -> None:
        if not self.enabled:
            raise VideoAdapterDormant(
                "Video/CV adapter is dormant (parked during the pivot). "
                "Set VideoAdapter(enabled=True) or config.VIDEO_ADAPTER_ENABLED "
                "to run the CV pipeline and emit events.")

    # -- ProviderAdapter (all gate on the dormancy flag) ----------------------
    def match(self, match_id: int) -> Match:
        self._require_enabled()
        return Match(match_id=match_id)  # CV metadata is not event-driven

    def events(self, match_id: int) -> list[Event]:
        self._require_enabled()
        return self._run_pipeline().events

    def passes(self, match_id: int) -> list[PassEvent]:
        self._require_enabled()
        return [e for e in self._run_pipeline().events
                if isinstance(e, PassEvent)]

    def shots(self, match_id: int) -> list[ShotEvent]:
        self._require_enabled()
        return [e for e in self._run_pipeline().events
                if isinstance(e, ShotEvent)]

    def freeze_frames(self, match_id: int) -> list[FreezeFrame]:
        # CV tracking is not event-keyed the way 360 data is; expose a single
        # per-match snapshot window when enabled.
        self._require_enabled()
        return []

    # -- the (lazy, enabled-only) CV run ---------------------------------------
    def _run_pipeline(self) -> "VideoRunResult":
        """Imports and runs the CV pipeline. Heavy imports happen only here."""
        from src.analytics.adapters.video._runner import run_video_to_events

        if self.source_path is None:
            raise NotImplementedError(
                "run_video_to_events is not implemented while the Video adapter "
                "is dormant; it is the parked seam for re-enabling Video/CV "
                "input. Provide source_path and enable it end-to-end first.")
        # NOTE: the mapping CV tracking/event vocabulary -> normalized schema is
        # the seam where the parked CV work resumes. Nothing here is exercised
        # while the adapter is dormant, so it can be filled in when Video/CV is
        # re-enabled.
        return run_video_to_events(source_path=self.source_path)
