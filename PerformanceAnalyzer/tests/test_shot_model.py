"""Stage 9, Stage C: fitted shot model (C2 scoring probability)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.adapters.statsbomb.shot_model import (
    MAX_SHOT_DIST_M,
    ShotModel,
    _features,
    extract_shots,
    load_shot_model,
    score_probability,
)
from src.analytics.models.pitch_control import PITCH_CONTROL_NX, PITCH_CONTROL_NY


def _shot_event(x: float, y: float, goal: bool) -> dict:
    return {"type": {"name": "Shot"},
            "location": [x, y],
            "shot": {"outcome": {"name": "Goal" if goal else "Saved"}}}


class TestExtractShots:
    def test_parses_goal_and_location(self):
        df = extract_shots([_shot_event(115.0, 40.0, True),
                            _shot_event(80.0, 20.0, False)])
        assert len(df) == 2
        assert df.iloc[0].to_dict() == {"x": 115.0, "y": 35.0, "goal": True}
        assert df.iloc[1]["y"] == pytest.approx(17.5)   # 80 -> 70 scaling
        assert bool(df.iloc[1]["goal"]) is False

    def test_ignores_non_shots(self):
        ev = [{"type": {"name": "Pass"}, "location": [50.0, 40.0]}]
        assert extract_shots(ev).empty


class TestFit:
    def _realistic_shots(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        rows = []
        for _ in range(400):
            d = rng.uniform(3, 45)
            ang = rng.uniform(-20, 20)
            x = 120 - d
            y = 35 + ang
            p_goal = 0.5 / (1 + d / 8)      # closer -> more likely to score
            rows.append({"x": x, "y": y, "goal": rng.random() < p_goal})
        return pd.DataFrame(rows)

    def test_fit_is_monotone_decreasing_in_distance(self):
        model = ShotModel.fit(self._realistic_shots())
        # a farther shot (same angle) must never be more likely to score
        xs = np.array([110.0, 105.0])
        ys = np.full_like(xs, 35.0)
        p = model.predict(xs, ys)
        assert p[0] > p[1]

    def test_fit_requires_enough_shots(self):
        with pytest.raises(ValueError):
            ShotModel.fit(pd.DataFrame(
                [{"x": 100.0, "y": 35.0, "goal": True}]))

    def test_features_shape(self):
        shots = pd.DataFrame({"x": [100.0], "y": [20.0], "goal": [True]})
        assert _features(shots).shape == (1, 2)


class TestScoreProbability:
    def test_grid_shape(self):
        p = score_probability(
            np.zeros((PITCH_CONTROL_NY, PITCH_CONTROL_NX)),
            np.zeros((PITCH_CONTROL_NY, PITCH_CONTROL_NX)))
        assert p.shape == (PITCH_CONTROL_NY, PITCH_CONTROL_NX)

    def test_lower_far_from_goal(self):
        xs = (np.arange(24) + 0.5) * 120 / 24
        ys = np.full(24, 35.0)
        p = score_probability(xs, ys)
        assert p[-1] > p[0]      # nearest the goal mouth scores highest

    def test_fallback_when_no_model(self):
        assert load_shot_model("/nonexistent/shot_model.json") is None

    def test_max_distance_gate(self):
        assert MAX_SHOT_DIST_M > 0
