"""Team assignment — SigLIP -> UMAP -> K-Means appearance clustering.

Replicates the Roboflow `TeamClassifier` approach:

  1. embed each player crop with a SigLIP vision transformer,
  2. reduce to a low dim with UMAP,
  3. cluster the two outfield teams with K-Means.

The model is *fit once* on a bootstrap sample and then only *predicts* per
frame, which prevents mid-match label flips (naive per-frame fitting would
swap teams). Goalkeepers are assigned by centroid proximity to each team's
outfield centroid (Roboflow heuristic).
"""

from __future__ import annotations

import numpy as np

from src.config import (
    EMBEDDING_BATCH,
    N_TEAMS,
    SIGLIP_FP16,
    SIGLIP_MODEL,
    UMAP_N_COMPONENTS,
)

# SigLIP model + processor are cached once (module-level) so they are loaded a
# single time and reused across fit()/predict() calls and across instances —
# matching the reference TeamClassifier which loads the model once in __init__.
# Loading ~1GB of weights per call was the dominant cost (13x slowdown vs ref).
_SIGLIP = {"model": None, "processor": None, "device": None, "fp16": None}


def _get_siglip():
    """Return the cached (model, processor, device) tuple, loading once."""
    if _SIGLIP["model"] is None:
        import torch
        from transformers import AutoProcessor, SiglipVisionModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL)
        fp16 = SIGLIP_FP16 and device == "cuda"
        if fp16:
            model = model.half()
        _SIGLIP["model"] = model.to(device)
        _SIGLIP["processor"] = AutoProcessor.from_pretrained(SIGLIP_MODEL)
        _SIGLIP["device"] = device
        _SIGLIP["fp16"] = fp16
    return _SIGLIP["model"], _SIGLIP["processor"], _SIGLIP["device"]


def embed_crops(crops: list[np.ndarray], batch_size: int = EMBEDDING_BATCH) -> np.ndarray:
    """SigLIP embeddings for a list of BGR crops. Returns (N, 768) float32."""
    import torch
    from PIL import Image

    model, processor, device = _get_siglip()

    # BGR -> RGB and make a list of PIL images.
    images = [Image.fromarray(crop[:, :, ::-1]) for crop in crops]

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = {}
            for key, value in processor(images=batch, return_tensors="pt").items():
                value = value.to(device)
                if value.is_floating_point():
                    value = value.to(model.dtype)
                inputs[key] = value
            outputs = model(**inputs)
            # Always return float32: UMAP is fp32 and fp16 would lose precision.
            emb = torch.mean(outputs.last_hidden_state, dim=1).float().cpu().numpy()
            embeddings.append(emb)
    return np.concatenate(embeddings, axis=0)


def match_cluster_centers(old_centers: np.ndarray,
                          new_centers: np.ndarray) -> np.ndarray:
    """Permutation aligning `new_centers` to `old_centers`.

    Returns `perm` such that `new_centers[perm[i]]` is the new cluster closest
    to old cluster `i`. KMeans label ids are arbitrary across fits, so a refit
    could silently swap the two team ids; this remap pins the new clusters back
    to their pre-refit identity (no mid-match label flips).
    """
    from scipy.optimize import linear_sum_assignment

    cost = np.linalg.norm(
        old_centers[:, None, :] - new_centers[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.empty(len(old_centers), dtype=int)
    perm[row_ind] = col_ind
    return perm


class TeamClassifier:
    """Learns team affiliation from player appearance."""

    def __init__(self, n_teams: int = N_TEAMS,
                 umap_components: int = UMAP_N_COMPONENTS):
        import umap
        from sklearn.cluster import KMeans

        self.n_teams = n_teams
        self._reducer = umap.UMAP(n_components=umap_components, random_state=0)
        self._kmeans = KMeans(n_clusters=n_teams, random_state=0)
        self._fitted = False
        # Within-cluster tightness (mean dist to own centroid, projection space)
        # measured on the bootstrap fit. Drift detection compares fresh-window
        # tightness against this baseline.
        self._baseline_tightness: float | None = None

    @staticmethod
    def _within_cluster_mean(proj: np.ndarray, labels: np.ndarray) -> float:
        centers = np.asarray([proj[labels == i].mean(axis=0)
                              for i in np.unique(labels)])
        dists = np.linalg.norm(proj - centers[labels], axis=1)
        return float(dists.mean())

    def fit(self, crops: list[np.ndarray]) -> "TeamClassifier":
        """Learn the two team clusters from the bootstrap crops."""
        data = embed_crops(crops)
        proj = self._reducer.fit_transform(data)
        self._kmeans.fit(proj)
        self._fitted = True
        self._baseline_tightness = self._within_cluster_mean(proj, self._kmeans.labels_)
        return self

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def baseline_tightness(self) -> float | None:
        return self._baseline_tightness

    def tightness(self, crops: list[np.ndarray]) -> float:
        """Bounded re-fit is the only change — the projection stays fixed."""
        if not self._fitted:
            raise RuntimeError("Call fit() before tightness()")
        data = embed_crops(crops)
        proj = self._reducer.transform(data)
        labels = self._kmeans.predict(proj)
        return self._within_cluster_mean(proj, labels)

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        """Return an array of team ids (0..n_teams-1) for each crop."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        data = embed_crops(crops)
        proj = self._reducer.transform(data)
        return self._kmeans.predict(proj).astype(int)

    def refit(self, crops: list[np.ndarray]) -> "TeamClassifier":
        """Re-cluster on new crops while *preserving* team label identity.

        Bounded: only the KMeans is refit — the UMAP projection (and the
        embeddings) stay as-is, so the label-preserving centre remap in
        `match_cluster_centers` is meaningful and cheap. The baseline tightness
        is reset to the refit window so we don't immediately re-drift.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before refit()")
        data = embed_crops(crops)
        proj = self._reducer.transform(data)
        from sklearn.cluster import KMeans

        new_km = KMeans(n_clusters=self.n_teams, random_state=0).fit(proj)
        perm = match_cluster_centers(self._kmeans.cluster_centers_,
                                     new_km.cluster_centers_)
        new_km.cluster_centers_ = new_km.cluster_centers_[perm]
        self._kmeans = new_km
        self._baseline_tightness = self._within_cluster_mean(
            proj, new_km.predict(proj))
        return self


def resolve_goalkeeper_team_ids(
    player_anchors: np.ndarray,      # (P, 2) bottom-centre of outfield players
    player_teams: np.ndarray,        # (P,) team labels for those players
    goalkeeper_anchors: np.ndarray,  # (G, 2) bottom-centre of goalkeepers
) -> np.ndarray:
    """Assign each goalkeeper to the team whose player centroid is closer."""
    labels = np.unique(player_teams)
    centroids = np.array([
        player_anchors[player_teams == t].mean(axis=0) for t in labels
    ])
    out = []
    for gk in goalkeeper_anchors:
        dists = np.linalg.norm(centroids - gk, axis=1)
        out.append(int(labels[np.argmin(dists)]))
    return np.asarray(out, dtype=int)