"""Stage 9, Stage C: shot model fitted on real StatsBomb shot data.

The EPV grid's scoring probability (`_score_probability` in
`src/spatial/pitch_control.py`) is a hand-set exponential decay. This module
replaces it with a logistic model fit on StatsBomb shots: P(goal | location)
from (distance to goal mouth, lateral deviation), fitted with L2
regularization so the small-sample curve stays smooth and monotone.

Coordinate convention: the EPV grid places the attacking goal at x = 120,
y = 35 on the canonical 120 x 70 frame. StatsBomb raw shot locations are x in
[0,120] (already toward the attacked goal) and y in [0,80]; y is rescaled by
70/80. `attacking` mirroring (alignment.py) is a no-op for x after conversion
because the mirror lands back on the original StatsBomb x.

The fitted coefficients are cached as JSON under `data/statsbomb/shot_model.json`
so `epv_grid` uses real data when present and falls back to the hand-set
exponential otherwise (offline / CI).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    EPV_SHOT_MODEL_PATH,
    STATSBOMB_PITCH_WIDTH,
)

PITCH_LENGTH = 120.0
PITCH_WIDTH = 70.0

# Max shot distance considered when fitting (m); further shots add noise.
MAX_SHOT_DIST_M = 45.0


def extract_shots(events: list[dict]) -> pd.DataFrame:
    """StatsBomb events -> (x, y, goal) shots on the canonical EPV frame.

    x is the raw StatsBomb x (attacked goal at 120 after mirroring), y scaled
    80 -> 70. Outcome "Goal" labels the row.
    """
    rows = []
    for e in events:
        if e["type"]["name"] != "Shot":
            continue
        loc = e.get("location") or []
        if len(loc) < 2:
            continue
        y = loc[1] * (70.0 / STATSBOMB_PITCH_WIDTH)
        goal = (e.get("shot") or {}).get("outcome", {}).get("name") == "Goal"
        rows.append({"x": float(loc[0]), "y": float(y), "goal": bool(goal)})
    return pd.DataFrame(rows, columns=["x", "y", "goal"])


def _features(shots: pd.DataFrame) -> np.ndarray:
    """(n, 2): [dist_to_goal, |lateral deviation from centre|].

    Deliberately no quadratic distance term: the empirical conversion rate is
    monotone decreasing in distance, and a d^2 term lets the fitted curve peak
    away from the goal (small-sample artifact). Keeping the logit linear in d
    guarantees EPV grows toward the goal mouth.
    """
    d = np.hypot(shots["x"] - PITCH_LENGTH, shots["y"] - PITCH_WIDTH / 2)
    return np.column_stack([d, np.abs(shots["y"] - PITCH_WIDTH / 2)])


@dataclass
class ShotModel:
    """Fitted logistic P(goal | x, y). `coef_` covers the 2 features."""
    intercept: float
    coef_: list[float]

    @classmethod
    def fit(cls, shots: pd.DataFrame,
            c: float = 1.0) -> "ShotModel":
        """L2-regularised logistic regression on real shot locations."""
        from sklearn.linear_model import LogisticRegression

        sub = shots[np.hypot(shots["x"] - PITCH_LENGTH,
                             shots["y"] - PITCH_WIDTH / 2) <= MAX_SHOT_DIST_M]
        if len(sub) < 20:
            raise ValueError(
                f"need >= 20 shots to fit the shot model, got {len(sub)}")
        X = _features(sub)
        clf = LogisticRegression(C=c, solver="lbfgs", max_iter=2000)
        clf.fit(X, sub["goal"])
        return cls(intercept=float(clf.intercept_[0]),
                   coef_=[float(w) for w in clf.coef_[0]])

    def predict(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        """P(goal) at grid centres (arrays broadcast to a common shape)."""
        gx, gy = np.broadcast_arrays(gx, gy)
        d = np.hypot(gx - PITCH_LENGTH, gy - PITCH_WIDTH / 2)
        X = np.column_stack([d.ravel(),
                             np.abs(gy.ravel() - PITCH_WIDTH / 2)])
        z = self.intercept + X @ np.asarray(self.coef_)
        return 1.0 / (1.0 + np.exp(-z)).reshape(gx.shape)

    # -- persistence ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {"intercept": self.intercept, "coef_": self.coef_}

    @classmethod
    def from_dict(cls, d: dict) -> "ShotModel":
        return cls(intercept=float(d["intercept"]),
                   coef_=[float(w) for w in d["coef_"]])


def save_shot_model(model: ShotModel, path: Path | str) -> None:
    Path(path).write_text(json.dumps(model.to_dict()))


@lru_cache(maxsize=1)
def load_shot_model(path: Path | str | None = None) -> ShotModel | None:
    """Cached shot model from disk, or None when it is not fitted yet."""
    p = Path(path or EPV_SHOT_MODEL_PATH)
    if not p.exists():
        return None
    try:
        return ShotModel.from_dict(json.loads(p.read_text()))
    except (ValueError, KeyError, TypeError):
        return None


def score_probability(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Scoring probability on a grid: fitted shot model or exponential fallback."""
    model = load_shot_model()
    if model is not None:
        return model.predict(gx, gy)
    from src.analytics.models.pitch_control import _score_probability_fallback
    return _score_probability_fallback(gx, gy)


def fit_and_cache(match_ids: list[int], path: Path | str | None = None,
                  loader=None) -> ShotModel:
    """Download shots for `match_ids`, fit the model and cache it."""
    from src.analytics.adapters.statsbomb.loader import StatsBombLoader

    loader = loader or StatsBombLoader()
    frames = []
    for mid in match_ids:
        try:
            shots = extract_shots(loader.events(mid))
        except Exception as exc:  # noqa: BLE001 — a missing match shouldn't abort
            print(f"  ! match {mid}: {exc}")
            continue
        frames.append(shots)
    all_shots = pd.concat(frames, ignore_index=True)
    model = ShotModel.fit(all_shots)
    save_shot_model(model, path or EPV_SHOT_MODEL_PATH)
    return model


def _grid_check() -> None:
    """Sanity: model output shape matches the pitch-control grid."""
    nx, ny = 24, 16
    xs = (np.arange(nx) + 0.5) * PITCH_LENGTH / nx
    ys = (np.arange(ny) + 0.5) * PITCH_WIDTH / ny
    gx, gy = np.meshgrid(xs, ys)
    p = score_probability(gx, gy)
    assert p.shape == (ny, nx)


if __name__ == "__main__":
    print("module OK; _grid_check:", _grid_check())
