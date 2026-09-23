"""Offline tests for the timestamp-level event <-> tracking alignment layer."""

from src.analytics.alignment import (
    align_events_to_frames,
    frame_at,
    nearest_frame_index,
)
from src.analytics.schema import Event, Frame


def _frames(ts_list):
    return [Frame(timestamp_sec=t, players=[], ball=None) for t in ts_list]


def test_nearest_after():
    frames = _frames([0.0, 1.0, 2.0, 3.0])
    assert nearest_frame_index(frames, 2.4, "after") == 3
    assert nearest_frame_index(frames, 2.0, "after") == 2
    assert nearest_frame_index(frames, 5.0, "after") == 3  # clamp high


def test_nearest_before():
    frames = _frames([0.0, 1.0, 2.0, 3.0])
    assert nearest_frame_index(frames, 2.4, "before") == 2
    assert nearest_frame_index(frames, -1.0, "before") == 0  # clamp low
    assert nearest_frame_index(frames, 0.0, "before") == 0


def test_nearest_exact_tie_picks_before():
    frames = _frames([0.0, 2.0])
    assert nearest_frame_index(frames, 1.0, "nearest") == 0  # tie -> before
    frames = _frames([0.0, 2.0, 4.0])
    assert nearest_frame_index(frames, 1.1, "nearest") == 1  # 2.0 closest
    assert nearest_frame_index(frames, 2.9, "nearest") == 1  # 2.0 closest
    assert nearest_frame_index(frames, 3.1, "nearest") == 2  # 4.0 closest


def test_frame_at_and_align():
    frames = _frames([0.0, 1.0, 2.0, 3.0])
    assert frame_at(frames, 1.2, "before").timestamp_sec == 1.0
    assert frame_at(frames, 1.2, "after").timestamp_sec == 2.0
    events = [Event(type="Pass", timestamp_sec=1.5, x=0.0, y=0.0),
              Event(type="Shot", timestamp_sec=2.9, x=0.0, y=0.0)]
    pairs = list(align_events_to_frames(events, frames))
    assert [p[0].type for p in pairs] == ["Pass", "Shot"]
    assert [p[1].timestamp_sec for p in pairs] == [2.0, 3.0]


def test_align_respects_original_order():
    frames = _frames([0.0, 5.0, 10.0])
    events = [Event(type="Shot", timestamp_sec=9.0, x=0.0, y=0.0),
              Event(type="Pass", timestamp_sec=1.0, x=0.0, y=0.0)]
    pairs = list(align_events_to_frames(events, frames))
    assert [p[0].type for p in pairs] == ["Shot", "Pass"]
