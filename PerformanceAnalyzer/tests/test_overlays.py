"""Smoke tests for the per-stage supervision overlays + Stage 8 radar/HUD."""

from __future__ import annotations

import numpy as np
import pytest

from src.analytics.adapters.video.detection.detection import Detection
from src.analytics.adapters.video.homography.pitch_model import PitchConfig
from src.analytics.adapters.video.render.hud import (
    draw_control_surface,
    draw_epv_heatmap,
    draw_hud,
    draw_voronoi_radar,
    finite_voronoi_polygons,
    radar_with_hud,
)
from src.analytics.adapters.video.render.overlays import (
    ball_overlay,
    detection_overlay,
    teams_overlay,
    tracking_overlay,
)


@pytest.fixture
def frame() -> np.ndarray:
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def make_dets() -> list[Detection]:
    return [
        Detection(class_id=2, class_name="player", confidence=0.9,
                  xyxy=(100, 100, 160, 240), tracker_id=1, team_id=0),
        Detection(class_id=2, class_name="player", confidence=0.8,
                  xyxy=(300, 120, 360, 250), tracker_id=2, team_id=1),
        Detection(class_id=1, class_name="goalkeeper", confidence=0.7,
                  xyxy=(600, 100, 640, 230), tracker_id=3, team_id=0),
        Detection(class_id=3, class_name="referee", confidence=0.6,
                  xyxy=(800, 110, 850, 240), tracker_id=4),
        Detection(class_id=0, class_name="ball", confidence=0.5,
                  xyxy=(900, 300, 910, 311)),
    ]


@pytest.mark.parametrize("overlay_fn", [
    detection_overlay, tracking_overlay, teams_overlay,
])
def test_overlays_run(frame, overlay_fn):
    out = overlay_fn(frame, make_dets())
    assert out.shape == frame.shape
    assert out.dtype == np.uint8


def test_overlays_empty(frame):
    out = detection_overlay(frame, [])
    assert out.shape == frame.shape
    assert np.array_equal(out, frame)


def test_ball_overlay_trail(frame):
    out = ball_overlay(frame, make_dets(), trail=[(100.0, 100.0), (105.0, 100.0)])
    assert out.shape == frame.shape


class TestStage8RadarHUD:
    @staticmethod
    def _pts():
        rng = np.random.default_rng(0)
        return rng.uniform([20, 10], [100, 60], size=(12, 2))

    def test_voronoi_one_polygon_per_point(self):
        pts = self._pts()
        polys = finite_voronoi_polygons(pts)
        assert len(polys) == len(pts)
        assert all(p.shape[0] >= 3 for p in polys)   # all closed cells

    def test_voronoi_degenerates_with_fewer_than_two_points(self):
        for pts in (np.empty((0, 2)), np.array([[5.0, 5.0]])):
            polys = finite_voronoi_polygons(pts)
            assert len(polys) == len(pts)

    def test_draw_voronoi_radar_shape(self):
        pitch = np.zeros((800, 1300, 3), dtype=np.uint8)
        pts = self._pts()
        out = draw_voronoi_radar(
            pitch, pts, np.array([0, 1] * 6), {0: (255, 0, 0), 1: (0, 0, 255)},
            scale=10.0, ball_pos=np.array([60.0, 35.0]))
        assert out.shape == pitch.shape
        assert out.dtype == np.uint8
        # drawing actually changed something
        assert not np.array_equal(out, pitch)

    def test_radar_with_hud_composes(self):
        pts = self._pts()
        out = radar_with_hud(
            PitchConfig(), pts, np.array([0, 1] * 6),
            {0: (255, 0, 0), 1: (0, 0, 255)},
            ["team_0 pos 52%", "team_1 pos 48%", "sprint 320m"])
        assert out.shape == (800, 1300, 3)

    def test_hud_renders_text_panel(self):
        base = np.zeros((300, 400, 3), dtype=np.uint8)
        out = draw_hud(base, ["team_0 pos 52%", "sprint 320m"])
        assert out.shape == base.shape
        assert not np.array_equal(out, base)

    def test_control_surface_blue_team0_red_team1(self):
        pitch = np.zeros((800, 1300, 3), dtype=np.uint8)
        surface = np.zeros((16, 24))
        surface[:, :12] = 0.9          # left: team 0 wins
        surface[:, 12:] = 0.1          # right: team 1 wins
        out = draw_control_surface(pitch, surface, scale=10.0)
        assert out.shape == pitch.shape
        assert out.dtype == np.uint8
        left = out[60:700, 60:660].mean(axis=(0, 1))
        right = out[60:700, 670:1240].mean(axis=(0, 1))
        assert left[0] > left[2] + 40   # blue channel dominates team-0 half
        assert right[2] > right[0] + 40 # red channel dominates team-1 half

    def test_epv_heatmap_highlights_goal_mouth(self):
        pitch = np.zeros((800, 1300, 3), dtype=np.uint8)
        grid = np.zeros((16, 24))
        grid[:, -2:] = 0.8             # value concentrated at x=120
        out = draw_epv_heatmap(pitch, grid, scale=10.0)
        assert out.shape == pitch.shape
        assert out.dtype == np.uint8
        mouth = out[60:700, 1200:1240].mean(axis=(0, 1))
        center = out[60:700, 600:640].mean(axis=(0, 1))
        assert mouth.sum() > center.sum()

    def test_epv_heatmap_zero_grid_is_noop(self):
        pitch = np.zeros((800, 1300, 3), dtype=np.uint8)
        out = draw_epv_heatmap(pitch, np.zeros((16, 24)), scale=10.0)
        assert np.array_equal(out, pitch)
