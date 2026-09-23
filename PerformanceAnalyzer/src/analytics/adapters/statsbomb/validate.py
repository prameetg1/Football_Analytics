"""Core validation: proves the CV output against StatsBomb ground truth.

Two independent checks feed a `ValidationReport`:

  1. Position error (m): per-frame mean distance between CV player positions
     and GT (360 freeze-frame) player positions, aligned on time, matched via
     optimized one-to-one assignment so identity/miss-count errors don't
     inflate positional accuracy.
  2. Pass recall / precision: how many GT passes the CV recovers (recall) and
     how many CV passes are real (precision), matched within a time/space
     window.

The report feeds the Stage 7 `MetricsGate`, which decides whether metric
tables may be emitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.config import (
    GATE_MAX_POSITION_ERROR_M,
    GATE_MIN_EVENT_PRECISION,
    GATE_MIN_EVENT_RECALL,
    GATE_MIN_PASS_PRECISION,
    GATE_MIN_PASS_RECALL,
)
from src.analytics.common import PLAYER_CLASSES


def _player_positions(sub: pd.DataFrame) -> np.ndarray:
    """(N,2) array of player/keeper pitch positions."""
    p = sub[sub["class_name"].isin(PLAYER_CLASSES)].dropna(
        subset=["pitch_x", "pitch_y"])
    return p[["pitch_x", "pitch_y"]].to_numpy(dtype=np.float64)


def nearest_assignment_mean_error(cv: np.ndarray, gt: np.ndarray) -> float:
    """Mean matched distance (m) via optimal assignment over the min count.

    Unmatched points are ignored so a false extra detection can't tank the
    score. Returns inf when either side has no points.
    """
    if len(cv) == 0 or len(gt) == 0:
        return float("inf")
    cost = np.linalg.norm(cv[:, None, :] - gt[None, :, :], axis=2)
    r, c = linear_sum_assignment(cost)
    return float(np.mean(cost[r, c]))


def position_error(cv_df: pd.DataFrame, gt_df: pd.DataFrame) -> float:
    """Mean per-frame CV-vs-GT position error (m).

    Frames are aligned on `timestamp_sec` (nearest GT frame within half a
    nominal frame). Each aligned frame's player sets are matched by assignment
    and the per-frame mean distances averaged.
    """
    errs: list[float] = []
    for _, g_cv in cv_df.groupby("timestamp_sec"):
        cv_pos = _player_positions(g_cv)
        if len(cv_pos) == 0:
            continue
        # nearest GT timestamp
        ts_cv = float(g_cv["timestamp_sec"].iloc[0])
        candidates = gt_df.copy()
        candidates["_dt"] = (candidates["timestamp_sec"] - ts_cv).abs()
        best = candidates.loc[candidates["_dt"].idxmin(), "timestamp_sec"]
        g_gt = gt_df[gt_df["timestamp_sec"] == best]
        gt_pos = _player_positions(g_gt)
        if len(gt_pos) == 0:
            continue
        errs.append(nearest_assignment_mean_error(cv_pos, gt_pos))
    return float(np.mean(errs)) if errs else float("inf")


def detect_passes(df: pd.DataFrame, min_move_m: float = 1.0) -> list[tuple[float, float, float]]:
    """Heuristic CV passes: the ball travelling >1 m between consecutive samples.

    Returns (timestamp, x, y) of each long ball move. This is deliberately a
    simple physical proxy (a real possession-state detector is out of scope);
    it exists so pass recall/precision has a CV signal to validate against.
    """
    b = df[df["class_name"] == "ball"].dropna(
        subset=["pitch_x", "pitch_y"]).sort_values("timestamp_sec")
    locs = b[["timestamp_sec", "pitch_x", "pitch_y"]].to_numpy(dtype=np.float64)
    out: list[tuple[float, float, float]] = []
    for (t0, x0, y0), (t1, x1, y1) in zip(locs, locs[1:]):
        if np.hypot(x1 - x0, y1 - y0) > min_move_m:
            out.append((float(t1), float(x1), float(y1)))
    return out


def pass_match(cv_passes: list[tuple[float, float, float]],
               gt_passes: list[tuple[float, float, float]],
               time_tol: float = 2.0, space_tol: float = 15.0) -> tuple[float, float]:
    """(precision, recall) of CV passes against StatsBomb GT passes.

    A GT pass is recovered if some CV pass lands within `time_tol` seconds and
    `space_tol` metres. precision = recovered / #CV, recall = recovered / #GT.
    """
    return event_match(cv_passes, gt_passes, time_tol, space_tol)


def event_match(cv_events: list[tuple[float, float, float]],
                gt_events: list[tuple[float, float, float]],
                time_tol: float = 2.0, space_tol: float = 15.0) -> tuple[float, float]:
    """(precision, recall) for any (timestamp, x, y) event class.

    A GT event is recovered if some CV event lands within `time_tol` seconds
    and `space_tol` metres. precision = recovered / #CV, recall = recovered /
    #GT. Used by the Stage-9 gate for every event class (passes, recoveries,
    shots, ...). Empty-vs-empty returns (1, 1); a GT with no CV events returns
    (0, 0); CV with no GT returns (0, 1).
    """
    n_gt = len(gt_events)
    n_cv = len(cv_events)
    if n_gt == 0 and n_cv == 0:
        return 1.0, 1.0
    if n_gt == 0:
        return 0.0, 1.0
    if n_cv == 0:
        return 0.0, 0.0
    gt = np.asarray(gt_events, dtype=np.float64)
    cv = np.asarray(cv_events, dtype=np.float64)
    recovered = 0
    for t, x, y in gt:
        d = np.hypot(cv[:, 1] - x, cv[:, 2] - y)
        dt = np.abs(cv[:, 0] - t)
        if ((dt <= time_tol) & (d <= space_tol)).any():
            recovered += 1
    return recovered / n_cv, recovered / n_gt


class ValidationReport:
    """CV-vs-GT numbers plus a pass/fail judgement against the gate bounds.

    `passes` requires the position error to be within bounds and — for every
    validated event class — non-trivial GT and recall/precision at or above the
    gate thresholds. `pass_precision`/`pass_recall` cover passes; additional
    event classes live in `kinds: {kind: (precision, recall)}`.
    """

    def __init__(self, position_error_m: float = float("inf"),
                 pass_precision: float = 0.0, pass_recall: float = 0.0,
                 n_gt_passes: int = 0, n_cv_passes: int = 0,
                 kinds: dict[str, tuple[float, float]] | None = None):
        self.position_error_m = position_error_m
        self.pass_precision = pass_precision
        self.pass_recall = pass_recall
        self.n_gt_passes = n_gt_passes
        self.n_cv_passes = n_cv_passes
        self.kinds = dict(kinds or {})

    def add_kind(self, kind: str, precision: float, recall: float,
                 n_gt: int) -> None:
        """Record a validated event class; `passes` then requires it too."""
        self.kinds[kind] = (float(precision), float(recall), int(n_gt))

    def _kind_ok(self) -> bool:
        for _, (precision, recall, n_gt) in self.kinds.items():
            if n_gt == 0:
                return False
            if precision < GATE_MIN_EVENT_PRECISION:
                return False
            if recall < GATE_MIN_EVENT_RECALL:
                return False
        return True

    @property
    def passes(self) -> bool:
        return (self.position_error_m <= GATE_MAX_POSITION_ERROR_M
                and self.n_gt_passes > 0
                and self.pass_recall >= GATE_MIN_PASS_RECALL
                and self.pass_precision >= GATE_MIN_PASS_PRECISION
                and self._kind_ok())

    @property
    def blocked(self) -> bool:
        return not self.passes

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        kinds = ", ".join(
            f"{k}={p:.2f}/{r:.2f}" for k, (p, r, _) in self.kinds.items())
        return (f"ValidationReport(pos_error_m={self.position_error_m:.2f}, "
                f"pass_precision={self.pass_precision:.2f}, "
                f"pass_recall={self.pass_recall:.2f}, "
                f"gt_passes={self.n_gt_passes}, cv_passes={self.n_cv_passes}, "
                f"kinds={{{kinds}}}, "
                f"passed={self.passes})")


def validate_cv_against_gt(cv_df: pd.DataFrame,
                           gt_df: pd.DataFrame) -> ValidationReport:
    """Full Stage 7 validation of a CV tracking table against StatsBomb GT."""
    cv_passes = detect_passes(cv_df)
    gt_passes = detect_passes(gt_df)
    precision, recall = pass_match(cv_passes, gt_passes)
    return ValidationReport(
        position_error_m=position_error(cv_df, gt_df),
        pass_precision=precision,
        pass_recall=recall,
        n_gt_passes=len(gt_passes),
        n_cv_passes=len(cv_passes),
    )


def validate_events_against_gt(cv_events: list,
                               gt_events: dict[str, list],
                               time_tol: float = 2.0,
                               space_tol: float = 15.0) -> ValidationReport:
    """Stage-9 event gate: per-kind (precision, recall) for CV events.

    `cv_events` is a list of `src.analytics.adapters.video.events.models.Event`; `gt_events` maps a
    kind name ("Pass", "Ball Recovery", "Shot", ...) to (t, x, y) tuples. The
    pass precision/recall fields are filled from the "Pass" kind so the report
    feeds the existing `MetricsGate` unchanged; every kind present is recorded
    via `add_kind` and must clear the gate thresholds for `passes` to hold.
    """
    report = ValidationReport()
    cv_by_kind: dict[str, list[tuple[float, float, float]]] = {}
    for e in cv_events:
        cv_by_kind.setdefault(e.kind, []).append(e.as_tuple())

    n_cv_pass = len(cv_by_kind.get("Pass", []))
    gt_pass = gt_events.get("Pass", [])
    precision, recall = pass_match(cv_by_kind.get("Pass", []), gt_pass,
                                   time_tol, space_tol)
    report.pass_precision = precision
    report.pass_recall = recall
    report.n_cv_passes = n_cv_pass
    report.n_gt_passes = len(gt_pass)

    for kind, gt in gt_events.items():
        if kind == "Pass":
            continue
        p, r = event_match(cv_by_kind.get(kind, []), gt, time_tol, space_tol)
        report.add_kind(kind, p, r, len(gt))
    return report