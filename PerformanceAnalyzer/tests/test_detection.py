"""Tests for detection data structures."""

from src.analytics.adapters.video.detection.detection import Detection


def test_bottom_center():
    d = Detection(class_name="player", xyxy=(10.0, 20.0, 30.0, 60.0))
    bc = d.bottom_center
    assert bc == (20.0, 60.0)


def test_defaults():
    d = Detection()
    assert d.tracker_id is None
    assert d.anchor_xy is None