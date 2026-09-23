"""Timestamp-level event <-> tracking alignment across providers.

Events and dense tracking may come from *different* adapters (e.g. StatsBomb
events resolved alongside IDSSE/SkillCorner tracking). Both are normalized to
`timestamp_sec` = seconds since kick-off on the same canonical pitch, so they
already share a compatible time axis. What's missing is a thin glue that pairs
a discrete event with the tracking `Frame` describing the pitch at that moment
— the state (ball position, each player, formations) a model needs at event
time without re-implementing bisection in every model.

This module provides that glue. It assumes `frames` are sorted ascending by
`timestamp_sec` (as the `DenseTracking.tracking()` contract guarantees).
"""

from __future__ import annotations

from bisect import bisect_left

from src.analytics.schema import Frame


def _frame_times(frames: list[Frame]) -> list[float]:
    return [f.timestamp_sec for f in frames]


def nearest_frame_index(frames: list[Frame], ts: float,
                        mode: str = "after") -> int:
    """Index of the frame nearest `ts` along a chosen direction.

    `frames` must be sorted ascending by `timestamp_sec`.

    * `after`  -> the first frame with timestamp >= ts (the state on arrival).
    * `before` -> the last frame with timestamp <= ts (the state on departure).
    * `nearest`-> whichever of those two is temporally closer to `ts`.

    A out-of-range `ts` clamps to the nearest end of the series.
    """
    if not frames:
        raise ValueError("cannot align to an empty frame series")
    times = _frame_times(frames)
    pos = bisect_left(times, ts)  # first index with times[pos] >= ts
    if mode == "after":
        return pos if pos < len(frames) else len(frames) - 1
    if mode == "before":
        return max(pos - 1, 0)
    # nearest
    if pos >= len(frames):
        return len(frames) - 1
    if pos == 0:
        return 0
    i_before, i_after = pos - 1, pos
    return (i_after if (times[i_after] - ts) < (ts - times[i_before])
            else i_before)


def frame_at(frames: list[Frame], ts: float, mode: str = "after") -> Frame:
    """The aligned tracking frame for a timestamp (`nearest_frame_index`)."""
    return frames[nearest_frame_index(frames, ts, mode)]


def align_events_to_frames(events, frames: list[Frame],
                           mode: str = "after"):
    """Pair each event with the tracking frame describing the pitch at that time.

    `events` is any iterable of schema `Event`-like objects carrying
    `timestamp_sec`. Yields `(event, frame)` pairs in the events' original
    order. The same frame may back several close events (it is the *nearest*
    sample, not an interpolation).
    """
    for ev in events:
        yield ev, frame_at(frames, ev.timestamp_sec, mode)


def possession_at(frames: list[Frame], ts: float,
                  mode: str = "before") -> int | None:
    """The possession side (`frame.possession_side`) at `ts`, if recorded."""
    return frame_at(frames, ts, mode).possession_side