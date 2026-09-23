"""Team-shape analytics (Stage 9, Stage D).

Shape metrics describe a team's spatial configuration on the pitch from player
positions alone — no ball/event dependency. Each metric is computed per frame
and averaged over frames (matching `LineHeightMetric`'s convention) so a single
scalar per team per metric falls out of the tracking table.

All metrics respect the canonical pitch frame: pitch_x in [0,120] along the
length, pitch_y in [0,70] across the width.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    CLASS_PLAYER,
    SHAPE_DEFENSIVE_THIRD_M,
    SHAPE_FINAL_THIRD_M,
)
from src.analytics.adapters.video.metrics.base import MetricBase
from src.analytics.adapters.video.metrics.util import PLAYER_CLASSES, team_players

PITCH_LENGTH_M = 120.0
PITCH_WIDTH_M = 70.0


def per_frame(positions: pd.DataFrame, fn) -> list[float]:
    """Apply `fn(frame_positions_array)` to each frame's player positions."""
    return [float(v)
            for _, grp in positions.groupby("frame_idx", sort=False)
            if (v := fn(grp)) is not None]


class SpreadMetric(MetricBase):
    """D1 — mean distance of players from the team's frame centroid (m)."""

    def compute(self, df: pd.DataFrame, team_id: int) -> float | None:
        sub = team_players(df, team_id)
        if sub.empty:
            return None

        def _spread(grp):
            pts = grp[["pitch_x", "pitch_y"]].to_numpy()
            if len(pts) < 2:
                return np.nan
            c = pts.mean(axis=0)
            return float(np.linalg.norm(pts - c, axis=1).mean())

        vals = [v for v in per_frame(sub, _spread) if np.isfinite(v)]
        return round(float(np.mean(vals)), 2) if vals else None

    def description(self) -> str:
        return "Spread (m): mean distance of players from team centroid"


class StretchIndexMetric(MetricBase):
    """D3 — pitch length (x-extent) x width (y-extent), a proxy for how much
    of the pitch the team occupies / how stretched it is."""

    def compute(self, df: pd.DataFrame, team_id: int) -> float | None:
        sub = team_players(df, team_id)
        if sub.empty:
            return None

        def _stretch(grp):
            xs = grp["pitch_x"].to_numpy()
            ys = grp["pitch_y"].to_numpy()
            if len(xs) < 2:
                return np.nan
            return float((xs.max() - xs.min()) * (ys.max() - ys.min()))

        vals = [v for v in per_frame(sub, _stretch) if np.isfinite(v)]
        return round(float(np.mean(vals)), 2) if vals else None

    def description(self) -> str:
        return "Stretch index (m^2): pitch length-extent x width-extent"


class CompactnessMetric(MetricBase):
    """D4 — mean pairwise distance between a team's players (m).

    High compactness == players tightly clustered; typically desired out of
    possession (defensive block).
    """

    def compute(self, df: pd.DataFrame, team_id: int) -> float | None:
        sub = team_players(df, team_id)
        if sub.empty:
            return None

        def _compact(grp):
            pts = grp[["pitch_x", "pitch_y"]].to_numpy()
            if len(pts) < 2:
                return np.nan
            d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
            iu = np.triu_indices(len(pts), k=1)
            return float(d[iu].mean())

        vals = [v for v in per_frame(sub, _compact) if np.isfinite(v)]
        return round(float(np.mean(vals)), 2) if vals else None

    def description(self) -> str:
        return "Compactness (m): mean pairwise player distance"


class SpatialEntropyMetric(MetricBase):
    """D5 — positional dispersion as Shannon entropy over a coarse grid.

    Players are binned into an (nx, ny) grid over the pitch; entropy is
    -sum(p*log p) over occupied cells, normalized to [0, 1] by log(#cells).
    Higher == more spread / less predictable structure.
    """

    def __init__(self, grid: tuple[int, int] = (6, 4)):
        self.nx, self.ny = grid

    def compute(self, df: pd.DataFrame, team_id: int) -> float | None:
        sub = team_players(df, team_id)
        if sub.empty:
            return None

        def _entropy(grp):
            pts = grp[["pitch_x", "pitch_y"]].to_numpy()
            if len(pts) < 2:
                return np.nan
            ix = np.clip((pts[:, 0] / PITCH_LENGTH_M * self.nx).astype(int),
                         0, self.nx - 1)
            iy = np.clip((pts[:, 1] / PITCH_WIDTH_M * self.ny).astype(int),
                         0, self.ny - 1)
            cells = ix * self.ny + iy
            _, counts = np.unique(cells, return_counts=True)
            p = counts / counts.sum()
            h = float(-(p * np.log(p)).sum())
            return h / np.log(self.nx * self.ny)

        vals = [v for v in per_frame(sub, _entropy) if np.isfinite(v)]
        return round(float(np.mean(vals)), 4) if vals else None

    def description(self) -> str:
        return (f"Spatial entropy ({self.nx}x{self.ny} grid): normalized "
                f"Shannon dispersion of player positions")


class TeamCentroidMetric(MetricBase):
    """D8 — mean of the team's player positions (pitch centre of the block)."""

    def compute(self, df: pd.DataFrame, team_id: int) -> float | None:
        sub = team_players(df, team_id)
        if sub.empty:
            return None

        def _cent(grp):
            return float(grp["pitch_x"].mean())

        vals = [v for v in per_frame(sub, _cent) if np.isfinite(v)]
        return round(float(np.mean(vals)), 2) if vals else None

    def description(self) -> str:
        return "Team centroid (m): mean pitch_x of the team's players"


class TeamWidthMetric(MetricBase):
    """D2 — mean pitch_y extent (width) of the team's block (m)."""

    def compute(self, df: pd.DataFrame, team_id: int) -> float | None:
        sub = team_players(df, team_id)
        if sub.empty:
            return None

        def _width(grp):
            ys = grp["pitch_y"].to_numpy()
            return float(ys.max() - ys.min()) if len(ys) > 1 else np.nan

        vals = [v for v in per_frame(sub, _width) if np.isfinite(v)]
        return round(float(np.mean(vals)), 2) if vals else None

    def description(self) -> str:
        return "Team width (m): mean pitch_y extent of the team's block"


class TeamLengthMetric(MetricBase):
    """D2 — mean pitch_x extent (length) of the team's block (m)."""

    def compute(self, df: pd.DataFrame, team_id: int) -> float | None:
        sub = team_players(df, team_id)
        if sub.empty:
            return None

        def _length(grp):
            xs = grp["pitch_x"].to_numpy()
            return float(xs.max() - xs.min()) if len(xs) > 1 else np.nan

        vals = [v for v in per_frame(sub, _length) if np.isfinite(v)]
        return round(float(np.mean(vals)), 2) if vals else None

    def description(self) -> str:
        return "Team length (m): mean pitch_x extent of the team's block"


class DefensiveLineShiftMetric(MetricBase):
    """D6 — mean pitch_y of the defensive line (lateral shift across width).

    Complements `LineHeightMetric` (which gives the line's depth in x).
    """

    def __init__(self, n_defenders: int = 3):
        self.n_defenders = n_defenders

    def compute(self, df: pd.DataFrame, team_id: int) -> float | None:
        sub = team_players(df, team_id)
        if sub.empty:
            return None
        shifts: list[float] = []
        for _, grp in sub.groupby("frame_idx"):
            if len(grp) < self.n_defenders:
                continue
            # deepest 3 outfield players = defensive line
            depths = grp.nsmallest(self.n_defenders, "pitch_x")
            shifts.append(float(depths["pitch_y"].mean()))
        if not shifts:
            return None
        return round(float(np.mean(shifts)), 2)

    def description(self) -> str:
        return f"Defensive line lateral shift (m), {self.n_defenders} deepest"


def per_phase(df: pd.DataFrame, team_id: int, phase: str,
              ownership: pd.Series | None = None) -> float | None:
    """Conditional shape value: spread for frames in a possession phase.

    `phase` in {"in_possession", "out_of_possession"}. `ownership` is a
    frame_idx -> owning team_id series (from the possession metric); when None
    it is derived by nearest player. Returns the mean `SpreadMetric` value.
    """
    if ownership is None:
        from src.analytics.adapters.video.metrics.possession import PossessionMetric
        ownership = PossessionMetric()._ownership(df)
    keep = (ownership == team_id) if phase == "in_possession" \
        else (ownership != team_id)
    sub = df[(df["team_id"] == team_id)
             & (df["class_name"].isin(PLAYER_CLASSES))
             & (df["frame_idx"].isin(ownership[keep].index))].dropna(
                 subset=["pitch_x", "pitch_y"])
    if sub.empty:
        return None
    return SpreadMetric().compute(sub, team_id)


class ShapeVsPossessionMetric(MetricBase):
    """D10 — ratio of out-of-possession compactness to in-possession spread.

    A defensive block should be more compact when defending (ratio < 1).
    Uses `SpreadMetric` on possession-conditioned frames.
    """

    def compute(self, df: pd.DataFrame, team_id: int) -> float | None:
        inn = per_phase(df, team_id, "in_possession")
        out = per_phase(df, team_id, "out_of_possession")
        if inn is None or out is None or inn == 0:
            return None
        return round(out / inn, 3)

    def description(self) -> str:
        return ("Shape vs possession: out-of-possession spread / "
                "in-possession spread (D10)")


class ShapeByBallZoneMetric(MetricBase):
    """D13 — team spread when the ball is in each of three zones.

    Returns a tuple (defensive, middle, final) of the team's mean spread in
    each ball zone; encoded as a single float via tuple for the runner is not
    possible, so this returns a dict-like mapping in `compute_map`.
    """

    def compute(self, df: pd.DataFrame, team_id: int) -> float | None:
        return self.compute_map(df, team_id)  # type: ignore[return-value]

    def compute_map(self, df: pd.DataFrame,
                    team_id: int) -> dict[str, float | None]:
        ball = df[df["class_name"] == "ball"].dropna(
            subset=["pitch_x", "pitch_y"])
        if ball.empty:
            return {"defensive": None, "middle": None, "final": None}
        zone: dict[str, float | None] = {}
        for z, lo, hi in (("defensive", 0.0, SHAPE_DEFENSIVE_THIRD_M),
                          ("middle", SHAPE_DEFENSIVE_THIRD_M, SHAPE_FINAL_THIRD_M),
                          ("final", SHAPE_FINAL_THIRD_M, PITCH_LENGTH_M)):
            fids = set(ball[(ball["pitch_x"] >= lo)
                            & (ball["pitch_x"] < hi)]["frame_idx"])
            sub = df[(df["team_id"] == team_id)
                     & (df["frame_idx"].isin(fids))
                     & (df["class_name"].isin(PLAYER_CLASSES))].dropna(
                         subset=["pitch_x", "pitch_y"])
            zone[z] = SpreadMetric().compute(sub, team_id) if not sub.empty else None
        return zone

    def description(self) -> str:
        return "Shape by ball zone: spread when ball in def/mid/final third (D13)"


class BlockPositionMetric(MetricBase):
    """D7 — midfield block depth (mean pitch_x of outfielders between the
    back line and the front line).

    With the back line taken as the team's `n_back` deepest players and the
    front line as the `n_front` highest, the block position is the mean x of
    everyone in between. Paired with `LineHeightMetric` (back line) it gives
    the pressing line height and the block's vertical footprint.
    """

    def __init__(self, n_back: int = 3, n_front: int = 3):
        self.n_back = n_back
        self.n_front = n_front

    def compute(self, df: pd.DataFrame, team_id: int) -> float | None:
        sub = team_players(df, team_id)
        if sub.empty:
            return None
        depths: list[float] = []
        for _, grp in sub.groupby("frame_idx"):
            if len(grp) < self.n_back + self.n_front + 1:
                continue
            xs = grp["pitch_x"]
            back = xs.nsmallest(self.n_back).index
            front = xs.nlargest(self.n_front).index
            middle = xs.drop(back).drop(front)
            if middle.empty:
                continue
            depths.append(float(middle.mean()))
        if not depths:
            return None
        return round(float(np.mean(depths)), 2)

    def description(self) -> str:
        return (f"Block position (m): mean pitch_x of the midfield block "
                f"({self.n_back} back / {self.n_front} front excluded) (D7)")


class MomentOfInertiaMetric(MetricBase):
    """D9 — radius of gyration (m): sqrt of the mean squared distance of the
    team's players from their centroid.

    Outlier-weighted dispersion: the same quantity as spread but quadratic, so
    a single isolated player pulls it up harder — a sensitive compactness
    measure for defensive shape.
    """

    def compute(self, df: pd.DataFrame, team_id: int) -> float | None:
        sub = team_players(df, team_id)
        if sub.empty:
            return None

        def _gyration(grp):
            pts = grp[["pitch_x", "pitch_y"]].to_numpy()
            if len(pts) < 2:
                return np.nan
            c = pts.mean(axis=0)
            r2 = np.linalg.norm(pts - c, axis=1) ** 2
            return float(np.sqrt(r2.mean()))

        vals = [v for v in per_frame(sub, _gyration) if np.isfinite(v)]
        return round(float(np.mean(vals)), 2) if vals else None

    def description(self) -> str:
        return "Moment of inertia (m): sqrt mean squared dist from centroid (D9)"


class FormationAngleMetric(MetricBase):
    """D9 — orientation of the team's principal axis, degrees from the pitch
    length axis.

    The eigenvector of the position covariance with the largest eigenvalue
    shows which way the formation is stretched; the angle captures formation
    "rotation" (a narrow block shifted sideways reads 90 degrees).
    """

    def compute(self, df: pd.DataFrame, team_id: int) -> float | None:
        sub = team_players(df, team_id)
        if sub.empty:
            return None

        def _angle(grp):
            pts = grp[["pitch_x", "pitch_y"]].to_numpy()
            if len(pts) < 2:
                return np.nan
            cov = np.cov(pts, rowvar=False)
            w, v = np.linalg.eigh(cov)
            main = v[:, int(np.argmax(w))]
            ang = np.degrees(np.arctan2(main[1], main[0])) % 180.0
            return float(ang)

        vals = [v for v in per_frame(sub, _angle) if np.isfinite(v)]
        return round(float(np.mean(vals)), 2) if vals else None

    def description(self) -> str:
        return ("Formation angle (deg): principal-axis orientation vs pitch "
                "length, 0..180 (D9)")


class LineHeightMetric(MetricBase):
    """Mean defensive-line height (pitch metres from the defending goal).

    Uses only outfield players (`class_name == player`); keepers never count
    towards the back line. Set `include_goalkeeper=True` to opt in.
    """

    def __init__(self, defend_x: float = 0.0, n_defenders: int = 3,
                 include_goalkeeper: bool = False):
        self.defend_x = defend_x
        self.n_defenders = n_defenders
        self._classes = PLAYER_CLASSES if include_goalkeeper else (CLASS_PLAYER,)

    def compute(self, df: pd.DataFrame, team_id: int) -> float | None:
        sub = team_players(df, team_id, classes=self._classes)
        if sub.empty:
            return None
        heights = []
        for _, grp in sub.groupby("frame_idx"):
            if len(grp) < self.n_defenders:
                continue
            depths = (grp["pitch_x"] - self.defend_x).abs()
            line = grp.loc[depths.nsmallest(self.n_defenders).index, "pitch_x"]
            heights.append(float(line.mean()))
        if not heights:
            return None
        return round(float(np.mean(heights)), 2)

    def description(self) -> str:
        return (f"Defensive line height (m), {self.n_defenders} deepest, "
                f"defending x={self.defend_x:g}")
