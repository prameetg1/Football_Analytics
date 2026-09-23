"""Tests for the homography stage (pixel <-> pitch mapping)."""

import numpy as np
import pytest

from src.analytics.adapters.video.homography.homography import StablePitchMapper, ViewTransformer, _to_pixels
from src.analytics.adapters.video.homography.keypoint_detector import PitchKeypoints
from src.analytics.adapters.video.homography.pitch_model import PitchConfig


def _config():
    return PitchConfig()


def test_reprojection_is_close_to_identity():
    cfg = _config()
    verts = cfg.vertices
    src = verts + 10.0  # fake "pixel" observation on a still camera
    transformer = ViewTransformer(src, verts)
    back = transformer.transform_points(src)
    err = float(np.mean(np.abs(back - verts)))
    assert err < 5.0


def test_needs_four_correspondences():
    cfg = _config()
    with pytest.raises(ValueError):
        ViewTransformer(cfg.vertices[:3], cfg.vertices[:3])


def test_temporal_smoothing_updates_current_matrix():
    cfg = _config()
    verts = cfg.vertices
    mapper = StablePitchMapper(config=cfg, maxlen=5)
    mapper.update(verts + 10.0, verts)
    assert mapper.current is not None and mapper.current.shape == (3, 3)
    mapper.update(verts + 12.0, verts)
    # After two updates the mean-of-two should be "between" 10 and 12 offsets.
    assert mapper.current.shape == (3, 3)


def test_transform_empty_points_ok():
    cfg = _config()
    verts = cfg.vertices
    mapper = StablePitchMapper(config=cfg)
    mapper.update(verts + 10.0, verts)
    out = mapper.transform_points(np.empty((0, 2)))
    assert out.shape == (0, 2)


def test_pitch_config_shape():
    assert PitchConfig().vertices.ndim == 2
    assert PitchConfig().vertices.shape[1] == 2


def test_pitch_config_has_32_canonical_keypoints_in_metres():
    verts = PitchConfig().vertices
    assert verts.shape == (32, 2)
    # Full pitch bounds: length 120 m, width 70 m.
    assert float(verts[:, 0].max()) == 120.0
    assert float(verts[:, 1].max()) == 70.0
    # Corners are index-aligned with SoccerPitchConfiguration ordering.
    assert np.allclose(verts[0], [0, 0])
    assert np.allclose(verts[5], [0, 70])       # top-left corner
    assert np.allclose(verts[24], [120, 0])     # bottom-right corner
    assert np.allclose(verts[29], [120, 70])    # top-right corner
    assert np.allclose(verts[13], [60, 0])      # halfway line bottom
    assert np.allclose(verts[30], [60 - 9.15, 35])  # centre circle left
    assert np.allclose(verts[31], [60 + 9.15, 35])  # centre circle right


def test_keypoint_labels_match_vertex_count():
    cfg = PitchConfig()
    assert len(cfg.keypoint_names) == len(cfg.vertices)


# ---------------------------------------------------------------------------
# Stage 6 — stateful homography + shot-cut reset
# ---------------------------------------------------------------------------

def _pixels_for(cfg, H, scale=10.0, jitter=0.0, seed=0):
    """Canonical vertices -> noisy pixel observations through inverse H."""
    rng = np.random.default_rng(seed)
    px = _to_pixels(H, cfg.vertices)
    if jitter:
        px = px + rng.normal(0, jitter, size=px.shape)
    return px


def _pan_h(offset_px, x_scale=1.0):
    """A pure translation homography pixel->pitch (invertible)."""
    H = np.eye(3)
    H[0, 0], H[1, 1] = 1.0 / x_scale, 1.0 / x_scale
    H[0, 2] = offset_px
    return H


def _kps(px):
    """Wrap pixel coords into a PitchKeypoints with canonical indices."""
    return PitchKeypoints(xy=np.asarray(px, dtype=np.float64),
                          confidence=np.ones(len(px)),
                          indices=np.arange(len(px)))


def test_shot_cut_detected_and_prior_reset():
    cfg = PitchConfig()
    mapper = StablePitchMapper(config=cfg, maxlen=5, kp_alpha=1.0)
    h_a = _pan_h(0.0)
    # Warm up shot A.
    for _ in range(6):
        px = _pixels_for(cfg, h_a, jitter=0.5, seed=1)
        mapper.update(px, cfg.vertices)
    assert mapper.cut_detected_last is False

    # Cut: a very different camera pose (60 px translation) — mean displacement
    # of canonical points through the two homographies exceeds the 40 px gate.
    h_b = _pan_h(60.0)
    px_b = _pixels_for(cfg, h_b, jitter=0.5, seed=2)
    mapper.update(px_b, cfg.vertices)
    assert mapper.cut_detected_last is True
    # After the cut the smoothed state equals the NEW homography (no old-frame
    # blur): projecting canonical points back to pixels matches h_b.
    back = _to_pixels(mapper.current, cfg.vertices)
    assert float(np.mean(np.abs(back - _to_pixels(h_b, cfg.vertices)))) < 5.0


def test_no_cut_within_slow_pan():
    cfg = PitchConfig()
    mapper = StablePitchMapper(config=cfg, maxlen=5, kp_alpha=1.0)
    # A slow pan: 0 -> 8 px translation over frames. Mean displacement per step
    # stays below the 40 px cut threshold.
    for i in range(10):
        h = _pan_h(float(i) * 0.8)
        px = _pixels_for(cfg, h, jitter=0.3, seed=3)
        mapper.update(px, cfg.vertices)
        assert mapper.cut_detected_last is False, f"false cut at frame {i}"


def test_keypoint_smoothing_blends_by_index():
    cfg = PitchConfig()
    mapper = StablePitchMapper(config=cfg, kp_alpha=0.5)
    px = cfg.vertices + 10.0
    sm = mapper.smooth_keypoints(_kps(px))
    assert np.allclose(sm, px)                      # first sight: passthrough
    px2 = cfg.vertices + 20.0
    sm2 = mapper.smooth_keypoints(_kps(px2))
    assert np.allclose(sm2, (px2 + px) / 2)


def test_reset_clears_state_and_cut_flag():
    cfg = PitchConfig()
    mapper = StablePitchMapper(config=cfg, kp_alpha=1.0)
    mapper.update(_pixels_for(cfg, _pan_h(0.0)), cfg.vertices)
    assert mapper.current is not None
    mapper.reset()
    assert mapper.current is None
    assert mapper.cut_detected_last is False
    assert mapper.smooth_keypoints(_kps(cfg.vertices)).shape == (32, 2)