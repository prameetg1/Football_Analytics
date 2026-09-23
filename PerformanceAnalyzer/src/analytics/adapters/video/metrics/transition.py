"""Stage 9, Stage D: transition-time shape analytics (D11, D12).

These need the *ordered* frame stream plus the event stream (turnovers), so
they are functions over (df, events) rather than per-frame `MetricBase` scalars:

  - transition_stretch(events, spread_series, window)   D11 — mean spread in
    the N frames after each ball recovery, vs the pre-transition baseline.
  - formation_recovery_frames(...)                      D12 — frames until the
    team's spread returns to within a tolerance of its pre-transition mean.

`spread_series` is frame_idx -> team's spread; provided by the caller so the
same geometry powers several metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TURNOVER_KIND = "Ball Recovery"


def spread_series(df: pd.DataFrame, team_id: int) -> pd.Series:
    """frame_idx -> mean distance of the team's players from their centroid."""
    from src.analytics.adapters.video.metrics.shape import SpreadMetric
    sub = df[(df["team_id"] == team_id)].dropna(subset=["pitch_x", "pitch_y"])
    rows: dict[int, float] = {}
    for fidx, grp in sub.groupby("frame_idx"):
        pts = grp[["pitch_x", "pitch_y"]].to_numpy()
        if len(pts) < 2:
            continue
        c = pts.mean(axis=0)
        rows[int(fidx)] = float(np.linalg.norm(pts - c, axis=1).mean())
    return pd.Series(rows, dtype=float).sort_index()


def transition_stretch(df: pd.DataFrame, events: list,
                       team_id: int, window: int = 25,
                       tol_m: float = 1.0) -> dict:
    """D11 — stretch spike after losing the ball.

    For every Ball Recovery *against* `team_id` (losing_team == team_id),
    compare mean spread in the `window` frames after the turnover to the mean
    spread in the `window` frames before it. Returns the mean delta and the
    mean post-turnover spread.
    """
    spread = spread_series(df, team_id)
    if spread.empty:
        return {"delta_m": 0.0, "post_m": 0.0, "n": 0}
    frames = np.sort(spread.index.to_numpy())
    deltas: list[float] = []
    posts: list[float] = []
    for e in events:
        if e.kind != TURNOVER_KIND:
            continue
        if e.qualifiers.get("losing_team") != team_id:
            continue
        t = e.timestamp_sec
        # nearest frame at/before the event
        post = frames[frames <= t]
        if len(post) == 0:
            continue
        fi = int(post[-1])
        window_after = spread[fi:fi + window]
        window_before = spread[fi - window:fi]
        if len(window_after) == 0 or len(window_before) == 0:
            continue
        posts.append(float(window_after.mean()))
        deltas.append(float(window_after.mean() - window_before.mean()))
    if not deltas:
        return {"delta_m": 0.0, "post_m": 0.0, "n": 0}
    return {"delta_m": round(float(np.mean(deltas)), 2),
            "post_m": round(float(np.mean(posts)), 2),
            "n": len(deltas)}


def formation_recovery_frames(df: pd.DataFrame, events: list,
                              team_id: int, window: int = 25,
                              tol_m: float = 1.0) -> dict:
    """D12 — frames for the team to return to a compact shape after ball loss.

    After each turnover against the team, count frames until the spread drops
    back to within `tol_m` of the pre-transition mean (or the window elapses).
    Returns the mean recovery time (frames) and how often it recovered.
    """
    spread = spread_series(df, team_id)
    if spread.empty:
        return {"mean_frames": 0.0, "recovered_share": 0.0, "n": 0}
    frames = np.sort(spread.index.to_numpy())
    times: list[int] = []
    recovered = 0
    n = 0
    for e in events:
        if e.kind != TURNOVER_KIND:
            continue
        if e.qualifiers.get("losing_team") != team_id:
            continue
        t = e.timestamp_sec
        post = frames[frames <= t]
        if len(post) == 0:
            continue
        fi = int(post[-1])
        before = spread[fi - window:fi]
        after = spread[fi:fi + window]
        if len(before) == 0 or len(after) == 0:
            continue
        baseline = float(before.mean())
        n += 1
        k = 0
        for k, v in enumerate(after):
            if abs(float(v) - baseline) <= tol_m:
                recovered += 1
                break
        times.append(k + 1)
    if n == 0:
        return {"mean_frames": 0.0, "recovered_share": 0.0, "n": 0}
    return {"mean_frames": round(float(np.mean(times)), 2),
            "recovered_share": round(recovered / n, 3),
            "n": n}
