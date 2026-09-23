"""Stage 5: adaptive team re-fit — drift detection + bounded, label-preserving
re-fit; and the assigner's rolling-window cadence / identity stability.
"""

import numpy as np
import pytest

from src.analytics.adapters.video.detection.detection import Detection
from src.analytics.adapters.video.teams.assigner import TeamAssigner
from src.analytics.adapters.video.teams.classifier import TeamClassifier, match_cluster_centers


def _det(cls: int, name: str, xyxy, conf: float = 0.9) -> Detection:
    return Detection(class_id=cls, class_name=name, confidence=conf, xyxy=xyxy)


class _IdentityReducer:
    """Stub in for UMAP: no dimensionality reduction (crops already 2D)."""

    def fit_transform(self, x):
        return x

    def transform(self, x):
        return x


@pytest.fixture(autouse=True)
def _no_siglip(monkeypatch):
    """Route embed_crops through an identity so tests skip SigLIP/UMAP load."""
    monkeypatch.setattr(
        "src.analytics.adapters.video.teams.classifier.embed_crops",
        lambda crops, batch_size=512: np.asarray(crops, dtype=np.float32),
    )


def _blobs(n: int, centre: tuple[float, float],
           scale: float, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(centre, scale, size=(n, 2))


class TestMatchClusterCenters:
    def test_id_when_centres_aligned(self):
        old = np.array([[0.0, 0.0], [10.0, 0.0]])
        new = np.array([[0.1, 0.1], [9.9, -0.1]])
        perm = match_cluster_centers(old, new)
        assert (perm == [0, 1]).all()

    def test_remaps_swapped_back_to_old_identity(self):
        old = np.array([[0.0, 0.0], [10.0, 0.0]])
        new = np.array([[10.0, 0.0], [0.0, 0.0]])  # KMeans labelled 1 first
        perm = match_cluster_centers(old, new)
        assert (perm == [1, 0]).all()
        assert np.allclose(new[perm], old, atol=1e-6)


class TestClassifierRefit:
    def test_refit_preserves_label_identity(self):
        tc = TeamClassifier(n_teams=2, umap_components=2)
        tc._reducer = _IdentityReducer()
        fit_crops = np.vstack([_blobs(30, (0.0, 0.0), 0.0),
                               _blobs(30, (10.0, 0.0), 0.0)])
        tc.fit(fit_crops.tolist())

        probe_a = np.array([[0.0, 0.0]])
        probe_b = np.array([[10.0, 0.0]])
        a0, b0 = int(tc.predict(probe_a)[0]), int(tc.predict(probe_b)[0])
        assert a0 != b0

        # drift: re-cluster on shifted blobs (KMeans ordering arbitrary)
        drift = np.vstack([_blobs(30, (5.0, 0.0), 0.0),
                           _blobs(30, (15.0, 0.0), 0.0)])
        tc.refit(drift.tolist())

        a1, b1 = int(tc.predict(probe_a)[0]), int(tc.predict(probe_b)[0])
        assert a1 == a0 and b1 == b0 and a1 != b1

    def test_refit_updates_baseline_tightness(self):
        tc = TeamClassifier(n_teams=2, umap_components=2)
        tc._reducer = _IdentityReducer()
        tc.fit(np.vstack([_blobs(20, (0.0, 0.0), 0.0),
                          _blobs(20, (10.0, 0.0), 0.0)]).tolist())
        base0 = tc.baseline_tightness
        tc.refit(np.vstack([_blobs(20, (0.0, 0.0), 1.0),
                            _blobs(20, (10.0, 0.0), 1.0)]).tolist())
        assert tc.baseline_tightness is not None
        assert tc.baseline_tightness > base0


class _DriftClassifier:
    """Deterministic fake: configurable tightness, counts refits."""

    def __init__(self, tightness=1.0):
        self.n_teams = 2
        self._baseline = 1.1
        self.tightness_val = tightness
        self.refits = 0
        self._fitted = True

    @property
    def fitted(self):
        return self._fitted

    def fit(self, data):
        self._crops = list(data)
        return self

    def predict(self, crops):
        return np.array([i % 2 for i in range(len(crops))])  # alternate teams

    @property
    def baseline_tightness(self):
        return self._baseline

    def tightness(self, crops):
        return self.tightness_val

    def refit(self, crops):
        self.refits += 1
        return self


def _crop_dets(n: int = 6):
    return [_det(2, "player", (10 + i, 10, 60 + i, 120)) for i in range(n)]


class TestAssignerDrift:
    def _make(self, **kw):
        kw.setdefault("bootstrap_crops", 4)
        a = TeamAssigner(**kw)
        fake = _DriftClassifier()
        a._classifier = fake
        a._frame_count = a.bootstrap_crops  # pretend bootstrap ran on frame 0
        return a

    def test_no_refit_without_degradation(self):
        a = self._make(check_interval_frames=2, window_crops=8,
                       tightness_ratio=2.0)
        a._classifier.tightness_val = 1.0          # <= 2.0, no drift
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        for _ in range(6):
            a.assign(frame, _crop_dets(6))
        assert a.refit_count == 0

    def test_refit_triggered_on_drift(self):
        a = self._make(check_interval_frames=2, window_crops=6,
                       tightness_ratio=1.5)
        a._classifier.tightness_val = 5.0          # >> baseline*ratio
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        for _ in range(5):
            a.assign(frame, _crop_dets(6))
        assert a.refit_count >= 1

    def test_window_grows_bounded_and_keeps_identity(self):
        a = self._make(check_interval_frames=1000, window_crops=10,
                       tightness_ratio=1.0)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        for _ in range(3):
            a.assign(frame, _crop_dets(6))
        assert len(a._window) == 10                 # bounded at window_crops
        teams_a = [d.team_id for d in a.assign(frame, _crop_dets(4))]
        teams_b = [d.team_id for d in a.assign(frame, _crop_dets(4))]
        assert teams_a == teams_b