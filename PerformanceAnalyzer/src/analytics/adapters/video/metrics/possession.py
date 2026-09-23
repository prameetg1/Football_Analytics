"""Ball possession metric.

Assigns each ball frame to the team whose nearest outfield player/keeper is
the ball; the metric is that team's fraction of all ball frames (0..1).
Ownership is a state, not a per-frame coin-flip: a team switch only commits
after the new team has been nearest for `hysteresis_frames` consecutive
frames, which removes rapid back-and-forth flicker near contested balls.
A generous `loose_m` acts only as a safety net so a ball far from everyone
(a goal kick, a long clearance) is not force-assigned; if a ball stays loose
for `loose_frames` consecutive frames the owner resets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CLASS_BALL
from src.analytics.adapters.video.metrics.base import MetricBase
from src.analytics.adapters.video.metrics.util import PLAYER_CLASSES


class PossessionMetric(MetricBase):
    """Stateful fraction of ball frames a team's player is nearest the ball."""

    def __init__(
        self,
        loose_m: float = 8.0,
        hysteresis_frames: int = 12,
        loose_frames: int | None = None,
    ):
        self.loose_m = loose_m
        self.hysteresis_frames = hysteresis_frames
        self.loose_frames = loose_frames or hysteresis_frames

    def _raw_ownership(self, df: pd.DataFrame) -> pd.Series:
        """ball-frame -> nearest team_id (NaN == loose / no players)."""
        ball = df[(df["class_name"] == CLASS_BALL)].dropna(
            subset=["pitch_x", "pitch_y"])
        players = df[df["class_name"].isin(PLAYER_CLASSES)].dropna(
            subset=["pitch_x", "pitch_y", "team_id"])
        owners: dict[int, float] = {}
        for frame_idx, b in ball.groupby("frame_idx").first().iterrows():
            pl = players[players["frame_idx"] == frame_idx]
            if pl.empty:
                continue
            d = np.hypot(pl["pitch_x"] - b["pitch_x"],
                         pl["pitch_y"] - b["pitch_y"]).to_numpy()
            i = int(d.argmin())
            owners[frame_idx] = pl.iloc[i]["team_id"] if d[i] <= self.loose_m else np.nan
        return pd.Series(owners, dtype=float)

    def _ownership(self, df: pd.DataFrame) -> pd.Series:
        """Smoothed ball-frame -> owning team_id (NaN == loose)."""
        raw = self._raw_ownership(df)
        if raw.empty:
            return raw
        out: dict[int, float] = {}
        current = np.nan
        pending = np.nan
        pending_count = 0
        loose_streak = 0
        for fidx, v in raw.items():
            if pd.isna(v):
                loose_streak += 1
                if loose_streak >= self.loose_frames:
                    current = np.nan
                    pending = np.nan
                    pending_count = 0
                out[fidx] = current
                continue
            loose_streak = 0
            if pd.isna(current):
                # cold start: first nearest team owns immediately
                current = v
                pending = np.nan
                pending_count = 0
            elif v == current:
                pending = np.nan
                pending_count = 0
            elif v == pending:
                pending_count += 1
            else:
                pending = v
                pending_count = 1
            if v != current and pending_count >= self.hysteresis_frames:
                current = v
                pending = np.nan
                pending_count = 0
            out[fidx] = current
        return pd.Series(out, dtype=float)

    def compute(self, df: pd.DataFrame, team_id: int) -> float | None:
        ownership = self._ownership(df)
        if ownership.empty:
            return None
        owned = float((ownership == team_id).mean())
        return round(owned, 4)

    def description(self) -> str:
        return (
            f"Ball possession fraction (nearest player, hysteresis "
            f"{self.hysteresis_frames} f, loose>{self.loose_m:g} m)"
        )
