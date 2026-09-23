"""Dormant-adapter registry for the video/CV provider.

Kept separate from `adapter.py` so the dormancy flag can be inspected cheaply
(no torch/ultralytics import) from anywhere, including the analytics facade and
tests. A dormant adapter is registered but refuses to produce events.
"""

from __future__ import annotations

from src.config import VIDEO_ADAPTER_ENABLED

VIDEO_ADAPTER_DORMANT = not VIDEO_ADAPTER_ENABLED
