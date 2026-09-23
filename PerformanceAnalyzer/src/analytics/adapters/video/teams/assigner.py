"""Team assignment orchestration — bootstrap-once, then adaptive per-frame.

The reference does a *two-pass*: it first collects player crops across the
whole video (stride 30), fits SigLIP->UMAP->KMeans, then re-processes to assign
teams. We improve that to a single streaming pass:

  * collect player crops until `bootstrap_crops` are seen,
  * fit the classifier exactly once (prevents mid-match label flips),
  * assign team_id to every subsequent player (and goalkeeper via the
    nearest-team-centroid heuristic).

Stage 5 adds adaptive robustness: after bootstrap we keep a *rolling window* of
player crops; every `check_interval_frames` we measure the window's cluster
tightness against the bootstrap baseline and, if it degrades by more than
`tightness_ratio` (kit/lighting drift scattering the embeddings), trigger a
*bounded re-fit*. The refit re-clusters on the recent crops but preserves team
label identity (`TeamClassifier.refit`), so it never flips teams mid-match.

Players with no team assignment (referee, or pre-bootstrap) keep team_id=None.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.config import (
    CLASS_GOALKEEPER,
    CLASS_PLAYER,
    TEAM_BOOTSTRAP_CROPS,
    TEAM_REFIT_CHECK_INTERVAL_FRAMES,
    TEAM_REFIT_TIGHTNESS_RATIO,
    TEAM_REFIT_WINDOW_CROPS,
)
from src.analytics.adapters.video.detection.detection import Detection
from src.analytics.adapters.video.teams.classifier import TeamClassifier, resolve_goalkeeper_team_ids


class TeamAssigner:
    """Assigns team ids to player/goalkeeper detections."""

    def __init__(self, bootstrap_crops: int = TEAM_BOOTSTRAP_CROPS,
                 stride: int = 1,
                 window_crops: int = TEAM_REFIT_WINDOW_CROPS,
                 check_interval_frames: int = TEAM_REFIT_CHECK_INTERVAL_FRAMES,
                 tightness_ratio: float = TEAM_REFIT_TIGHTNESS_RATIO):
        self.bootstrap_crops = bootstrap_crops
        self.stride = stride            # sample every Nth detection (matches ref)
        self._classifier: TeamClassifier | None = None
        self._crops: list[np.ndarray] = []
        self._frame_count = 0
        self._enabled = True
        # Stage 5: rolling window for drift detection + bounded re-fit.
        self.window_crops = window_crops
        self.check_interval_frames = check_interval_frames
        self.tightness_ratio = tightness_ratio
        self._window: deque[np.ndarray] = deque(maxlen=window_crops)
        self._frames_since_check = 0
        self.refit_count = 0
        self.refit_frames: list[int] = []

    @property
    def fitted(self) -> bool:
        return self._classifier is not None and self._classifier.fitted

    def disable(self) -> None:
        self._enabled = False

    def _collect(self, frame: np.ndarray, dets: list[Detection]) -> None:
        if self._frame_count % self.stride != 0:
            return
        for d in dets:
            if d.class_name != CLASS_PLAYER:
                continue
            crop = frame[
                int(max(0, d.y1)): int(max(0, d.y2)),
                int(max(0, d.x1)): int(max(0, d.x2)),
            ]
            if crop.size == 0:
                continue
            self._crops.append(crop)
            if len(self._crops) >= self.bootstrap_crops:
                return

    def _fit(self) -> None:
        try:
            self._classifier = TeamClassifier()
            self._classifier.fit(self._crops)
            self._crops = []
        except Exception as exc:  # pragma: no cover - SigLIP/network failure
            # Teams are a soft feature: never crash the pipeline on HF/network
            # issues — degrade to no team assignment.
            print(f"[teams] disabled: {exc}")
            self._classifier = None
            self._enabled = False
            self._crops = []

    def _player_crops(self, frame: np.ndarray,
                      players: list[Detection]) -> tuple[list[np.ndarray],
                                                         list[int]]:
        crops, idx = [], []
        for i, d in enumerate(players):
            c = frame[int(max(0, d.y1)): int(max(0, d.y2)),
                      int(max(0, d.x1)): int(max(0, d.x2))]
            if c.size > 0:
                crops.append(c)
                idx.append(i)
        return crops, idx

    def _maybe_refit(self, frame: np.ndarray) -> None:
        """Periodic drift check; bounded re-fit when tightness degrades."""
        if (self._classifier is None or not self._classifier.fitted
                or len(self._window) < 2 * self._classifier.n_teams):
            return
        baseline = self._classifier.baseline_tightness
        if baseline is None or baseline <= 0:
            return
        try:
            current = self._classifier.tightness(list(self._window))
        except Exception as exc:  # pragma: no cover - embed failure
            print(f"[teams] drift check failed, skipped: {exc}")
            return
        if current > self.tightness_ratio * baseline:
            self._classifier.refit(list(self._window))
            self.refit_count += 1
            self.refit_frames.append(self._frame_count)
            print(f"[teams] refit #{self.refit_count} at frame "
                  f"{self._frame_count}: tightness {current:.3f} vs "
                  f"baseline {baseline:.3f}")
            self._window.clear()

    def assign(self, frame: np.ndarray, dets: list[Detection]) -> list[Detection]:
        """Set `d.team_id` on player/goalkeeper detections in-place."""
        if not self._enabled:
            return dets
        self._frame_count += 1

        players = [d for d in dets if d.class_name == CLASS_PLAYER]
        keepers = [d for d in dets if d.class_name == CLASS_GOALKEEPER]

        if not self.fitted:
            self._collect(frame, dets)
            if len(self._crops) >= self.bootstrap_crops:
                self._fit()
            return dets  # nothing assigned until bootstrap completes

        if players:
            crops, idx = self._player_crops(frame, players)
            if crops:
                teams = self._classifier.predict(crops)
                for i, tid in zip(idx, teams):
                    players[i].team_id = int(tid)
                # Stage 5: feed the fresh crops into the rolling window.
                self._window.extend(crops)

        if keepers and players:
            p_anchors = np.asarray([d.bottom_center for d in players])
            p_teams = np.asarray(
                [d.team_id if d.team_id is not None else -1 for d in players],
                dtype=int)
            valid = p_teams >= 0
            if valid.any():
                gk_anchors = np.asarray([d.bottom_center for d in keepers])
                ids = resolve_goalkeeper_team_ids(
                    p_anchors[valid], p_teams[valid], gk_anchors)
                for d, tid in zip(keepers, ids):
                    d.team_id = int(tid)

        # Stage 5: run the drift check on the configured cadence.
        self._frames_since_check += 1
        if self._frames_since_check >= self.check_interval_frames:
            self._frames_since_check = 0
            self._maybe_refit(frame)

        return dets
