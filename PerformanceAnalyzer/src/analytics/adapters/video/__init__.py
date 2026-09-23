"""Video/CV provider adapter.

This is a full `ProviderAdapter` for the video-computer-vision input source:
it turns a broadcast/tactical video into normalized `src.analytics.schema`
events, taking coordinates from the tracking/homography pipeline and events
from the CV event extractor.

**Dormant by design.** The CV path was parked during the pivot (arbitrary-camera
broadcast video is unreliable; statistical models are now fed by structured
third-party data). The whole CV stack lives here so it is a first-class data
source again — but it refuses to run unless explicitly enabled via
`VideoAdapter(enabled=True)` / `config.VIDEO_ADAPTER_ENABLED`. This keeps the
model layer provider-agnostic and prevents accidental heavyweight inference
(importing this module imports torch/ultralytics lazily, never eagerly).
"""

from __future__ import annotations

from src.analytics.adapters.video.adapter import (
    VideoAdapter,
    VideoAdapterDormant,
)
from src.config import VIDEO_ADAPTER_ENABLED
from src.analytics.adapters.video._registry import VIDEO_ADAPTER_DORMANT

__all__ = [
    "VideoAdapter",
    "VideoAdapterDormant",
    "VIDEO_ADAPTER_DORMANT",
    "VIDEO_ADAPTER_ENABLED",
]
