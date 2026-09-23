"""Stage 9, Stage C: pitch control and possession value.

  - `pitch_control_surface` (C1)   Spearman-style race: each point on the pitch
                                   is "won" by the team whose player reaches it
                                   first, with constant-acceleration kinematics
                                   from the player's current position + velocity
  - `epv_grid` (C2)                Expected Possession Value via value iteration
  - `expected_threat_grid` (C3)    xT = value moved per transition (per m)
  - `field_tilt` (C4)              time-weighted territory of the ball
  - `space_creation_runs` (C5)     runs into uncovered space

All operate on the canonical pitch frame (x in [0,120], y in [0,70]) and return
plain numpy grids so callers (metrics, render, gate) stay decoupled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    EPV_DISCOUNT,
    EPV_HORIZON_STEPS,
    EPV_KEEP_PROB,
    EPV_STEP_M,
    PITCH_CONTROL_A_MAX_MS2,
    PITCH_CONTROL_NX,
    PITCH_CONTROL_NY,
    PITCH_CONTROL_TIME_SIGMA,
    SPACE_CREATION_FRAME_GAP,
    SPACE_CREATION_MIN_RUN_M,
    SPACE_CREATION_OPP_CONTROL_MAX,
)
from src.analytics.common import PLAYER_CLASSES

PITCH_LENGTH = 120.0
PITCH_WIDTH = 70.0


def _grid_centers(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    xs = (np.arange(nx) + 0.5) * PITCH_LENGTH / nx
    ys = (np.arange(ny) + 0.5) * PITCH_WIDTH / ny
    gx, gy = np.meshgrid(xs, ys)
    return gx, gy


def _track_velocities(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """track_id -> (timestamp_sec, vx, vy) by central differences.

    Used by the C1 race model so a player's current momentum (not just
    position) decides who reaches a location first.
    """
    players = df[df["class_name"].isin(PLAYER_CLASSES)].dropna(
        subset=["pitch_x", "pitch_y", "timestamp_sec", "track_id"])
    out: dict[str, pd.DataFrame] = {}
    for tid, grp in players.groupby("track_id"):
        grp = grp.sort_values("timestamp_sec")
        t = grp["timestamp_sec"].to_numpy()
        if len(t) < 2:
            continue
        x = grp["pitch_x"].to_numpy().astype(float)
        y = grp["pitch_y"].to_numpy().astype(float)
        out[str(tid)] = pd.DataFrame({
            "timestamp_sec": t,
            "vx": np.gradient(x, t),
            "vy": np.gradient(y, t),
        })
    return out


def _time_to_reach(points: np.ndarray, pos: np.ndarray, vel: np.ndarray,
                   a_max: float) -> np.ndarray:
    """Time for a player at `pos` with velocity `vel` to reach `points`.

    Constant-acceleration kinematics (Spearman 2018): if the player is already
    moving toward a point they keep accelerating; otherwise they must first
    stop, then accelerate back. `points` is (N, 2), `pos`/`vel` are (2,).
    """
    d = points - pos[None, :]
    dist = np.linalg.norm(d, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = d / dist[:, None]
    v = u @ vel            # speed component toward the point (signed), (N,)

    t = np.zeros_like(dist)
    moving = v >= 0
    away = ~moving
    t[moving] = (-v[moving]
                 + np.sqrt(v[moving] ** 2 + 2 * a_max * dist[moving])) / a_max
    t[away] = (np.abs(v[away]) / a_max
               + np.sqrt(2 * dist[away] / a_max + (v[away] / a_max) ** 2))
    t[dist == 0] = 0.0
    return t


def _velocity_map(velocities: dict[str, pd.DataFrame]
                  ) -> dict[tuple[str, float], tuple[float, float]]:
    """(track_id, timestamp) -> (vx, vy) flat lookup for the C1 race."""
    vel: dict[tuple[str, float], tuple[float, float]] = {}
    for tid, vdf in velocities.items():
        for ts, vx, vy in zip(vdf["timestamp_sec"], vdf["vx"], vdf["vy"]):
            vel[(str(tid), round(float(ts), 4))] = (float(vx), float(vy))
    return vel


def _spearman_surface(own: pd.DataFrame, opp: pd.DataFrame,
                      velocities: dict[str, pd.DataFrame],
                      nx: int, ny: int,
                      a_max: float = PITCH_CONTROL_A_MAX_MS2,
                      time_sigma: float = PITCH_CONTROL_TIME_SIGMA) -> np.ndarray:
    """Per-frame control surface: min-time race between the two teams."""
    vel = _velocity_map(velocities)
    return _race_surface(own, opp, vel, nx, ny, a_max, time_sigma)


def _race_surface(own: pd.DataFrame, opp: pd.DataFrame,
                  vel: dict[tuple[str, float], tuple[float, float]],
                  nx: int, ny: int,
                  a_max: float = PITCH_CONTROL_A_MAX_MS2,
                  time_sigma: float = PITCH_CONTROL_TIME_SIGMA) -> np.ndarray:
    """Per-frame race surface with a pre-built velocity map (fast path)."""
    gx, gy = _grid_centers(nx, ny)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    own_t = np.full(pts.shape[0], np.inf)
    opp_t = np.full(pts.shape[0], np.inf)

    def race_min(group: pd.DataFrame, out: np.ndarray) -> None:
        for _, row in group.iterrows():
            px, py = row["pitch_x"], row["pitch_y"]
            if pd.isna(px) or pd.isna(py):     # occlusion artifact
                continue
            v = vel.get((str(row["track_id"]),
                         round(float(row["timestamp_sec"]), 4)), (0.0, 0.0))
            pos = np.array([float(px), float(py)])
            t = _time_to_reach(pts, pos, np.asarray(v, dtype=float), a_max)
            out[:] = np.minimum(out, t)

    race_min(own, own_t)
    race_min(opp, opp_t)
    with np.errstate(over="ignore"):
        dt = own_t - opp_t          # positive -> opponent reaches first
        p = 1.0 / (1.0 + np.exp(dt / time_sigma))
    return p.reshape(ny, nx)


def pitch_control_surface(df: pd.DataFrame, team_id: int,
                          nx: int = PITCH_CONTROL_NX,
                          ny: int = PITCH_CONTROL_NY) -> np.ndarray:
    """C1 — probability `team_id` reaches each grid cell first (Spearman race).

    Every cell is won by whichever team's player gets there first under
    constant-acceleration kinematics from their current position and velocity
    (momentum matters: a player sprinting toward a point reaches it sooner).
    Multiple frames are evaluated snapshot-by-snapshot and averaged.

    Returns an (ny, nx) probability grid (fraction of local control).
    """
    players = df[df["class_name"].isin(PLAYER_CLASSES)].dropna(
        subset=["pitch_x", "pitch_y", "team_id"])
    if players[players["team_id"] == team_id].empty:
        return np.full((ny, nx), np.nan)
    velocities = _track_velocities(df)
    surfaces: list[np.ndarray] = []
    for fidx, frame in players.groupby("frame_idx"):
        own = frame[frame["team_id"] == team_id]
        if own.empty:
            continue
        opp = frame[frame["team_id"] != team_id]
        if opp.empty:
            surfaces.append(np.full((ny, nx), 1.0))
            continue
        surfaces.append(_spearman_surface(own, opp, velocities, nx, ny))
    if not surfaces:
        return np.full((ny, nx), np.nan)
    return np.mean(np.stack(surfaces), axis=0)


def _score_probability_fallback(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Hand-set exponential decay used before the fitted shot model exists."""
    dist_to_goal = np.sqrt((gx - PITCH_LENGTH) ** 2
                           + (gy - PITCH_WIDTH / 2) ** 2)
    prob = np.exp(-dist_to_goal / 18.0)
    prob *= (gx > PITCH_LENGTH - 20)   # only attacking third carries value
    return prob


def _score_probability(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Scoring probability per grid cell from the fitted StatsBomb shot model
    (falls back to a hand-set exponential when the model is not fitted)."""
    from src.analytics.adapters.statsbomb.shot_model import score_probability
    return score_probability(gx, gy)


def _transition_index(idx: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Possible next cells from a flat (ny*nx,) index: move ±EPV_STEP_M."""
    i, j = np.unravel_index(idx, (ny, nx))
    di = int(round(EPV_STEP_M / (PITCH_LENGTH / nx))) or 1
    dj = int(round(EPV_STEP_M / (PITCH_WIDTH / ny))) or 1
    i = np.clip(i + 1, 0, ny - 1)
    j = np.clip(j, 0, nx - 1)
    return i * nx + j


def epv_grid(nx: int = PITCH_CONTROL_NX,
             ny: int = PITCH_CONTROL_NY) -> np.ndarray:
    """Expected Possession Value via value iteration (C2).

    A Markov chain over the grid: from each cell the ball moves forward one
    step with probability `keep`, else the possession dies (terminal). The
    value of a cell is its immediate scoring probability plus the discounted
    expected value of the next cell. Solved by `EPV_HORIZON_STEPS` iterations
    of Bellman backup. Returns an (ny, nx) value grid (0..1, goal-weighted).
    """
    gx, gy = _grid_centers(nx, ny)
    score = _score_probability(gx, gy).ravel()
    n_cells = nx * ny
    next_cell = _transition_index(np.arange(n_cells), nx, ny)
    v = np.zeros(n_cells)
    for _ in range(EPV_HORIZON_STEPS):
        # scoring absorbs the possession: value now + discounted future value
        # weighted by (keep possession AND not scoring now)
        v = score + EPV_DISCOUNT * EPV_KEEP_PROB * (1 - score) * v[next_cell]
    return v.reshape(ny, nx)


def expected_threat_grid(nx: int = PITCH_CONTROL_NX,
                         ny: int = PITCH_CONTROL_NY) -> np.ndarray:
    """Expected threat (xT) — value *gained per step* moving the ball (C3)."""
    v = epv_grid(nx, ny)
    gx, gy = _grid_centers(nx, ny)
    # xT at a cell is the marginal value of advancing the ball from there.
    dv = np.zeros_like(v)
    dv[:, 1:] = v[:, 1:] - v[:, :-1]       # x-gradient (advancing)
    dv[:, 0] = v[:, 0]
    return np.clip(dv, 0, None)


def field_tilt(df: pd.DataFrame, team_id: int) -> float:
    """C4 — mean ball x while `team_id` owns it (territory proxy)."""
    ball = df[df["class_name"] == "ball"].dropna(
        subset=["pitch_x", "pitch_y", "timestamp_sec"])
    players = df[df["class_name"].isin(PLAYER_CLASSES)].dropna(
        subset=["pitch_x", "pitch_y", "team_id", "timestamp_sec"])
    xs: list[float] = []
    for _, b in ball.iterrows():
        ts = round(float(b["timestamp_sec"]), 4)
        pl = players[players["timestamp_sec"].round(4) == ts]
        if pl.empty:
            continue
        d = np.hypot(pl["pitch_x"] - b["pitch_x"], pl["pitch_y"] - b["pitch_y"])
        if d.min() > 8.0:
            continue
        owner = pl.iloc[int(d.argmin())]["team_id"]
        if owner == team_id:
            xs.append(float(b["pitch_x"]))
    return float(np.mean(xs)) if xs else 0.0


def _opponent_team_id(df: pd.DataFrame, team_id: int) -> int | None:
    teams = df[df["class_name"].isin(PLAYER_CLASSES)]["team_id"].dropna().unique()
    opp = [t for t in teams if t != team_id]
    return int(opp[0]) if opp else None


def space_creation_runs(df: pd.DataFrame, team_id: int,
                        opp_control_max: float = SPACE_CREATION_OPP_CONTROL_MAX,
                        min_run_m: float = SPACE_CREATION_MIN_RUN_M,
                        frame_gap: int = SPACE_CREATION_FRAME_GAP,
                        nx: int = PITCH_CONTROL_NX,
                        ny: int = PITCH_CONTROL_NY) -> list[dict]:
    """C5 — runs by `team_id` players into uncovered space.

    For every frame each `team_id` player is tagged with the threat at their
    exact position: how soon the nearest opponent can reach it under
    constant-acceleration kinematics (momentum included), mapped through a
    logistic with `PITCH_CONTROL_TIME_SIGMA` (opponent already there -> 0.5,
    more than ~0.5 s away -> <0.15). A space run is a maximal sequence of
    consecutive frames (gaps up to `frame_gap` tolerated) where the player
    stays under `opp_control_max` threat and advances at least `min_run_m`.
    Runs are reported with their endpoints, distance, worst (max) threat and a
    `threatening` flag (ending in the opponent's final third).

    Returns a list of dicts:
      {track_id, start (x, y), end (x, y), distance_m, max_opp_control,
       frames, threatening}.
    """
    opp = _opponent_team_id(df, team_id)
    if opp is None:
        return []
    players = df[(df["class_name"].isin(PLAYER_CLASSES))
                 & (df["team_id"] == team_id)].dropna(
                     subset=["pitch_x", "pitch_y", "frame_idx", "track_id"])
    if players.empty:
        return []
    # one velocity table for the whole clip; reuse across every frame
    velocities = _track_velocities(df)
    vel = _velocity_map(velocities)
    opp_rows = df[(df["class_name"].isin(PLAYER_CLASSES))
                  & (df["team_id"] == opp)].dropna(
                      subset=["pitch_x", "pitch_y"])

    runs: list[dict] = []
    for tid, grp in players.groupby("track_id"):
        grp = grp.sort_values("frame_idx")
        pts: list[tuple[int, float, float, float]] = []   # (frame, x, y, ctrl)
        for fidx, fg in grp.groupby("frame_idx"):
            if fg.empty:
                continue
            oppf = opp_rows[opp_rows["frame_idx"] == fidx]
            if oppf.empty:
                continue
            # opponent threat at the runner's exact position: how soon can the
            # nearest opponent get there, given their momentum?
            px, py = float(fg["pitch_x"].iloc[0]), float(fg["pitch_y"].iloc[0])
            pos = np.array([px, py])
            opp_t = np.inf
            for _, orow in oppf.iterrows():
                v = vel.get((str(orow["track_id"]),
                             round(float(orow["timestamp_sec"]), 4)),
                            (0.0, 0.0))
                t = _time_to_reach(pos[None, :],
                                   np.array([float(orow["pitch_x"]),
                                             float(orow["pitch_y"])]),
                                   np.asarray(v, dtype=float),
                                   PITCH_CONTROL_A_MAX_MS2)[0]
                if np.isfinite(t) and t < opp_t:
                    opp_t = t
            if not np.isfinite(opp_t):
                continue
            # opponent already on the spot (t=0) -> tie at 0.5; far -> ~0
            control = 1.0 / (1.0 + np.exp(opp_t / PITCH_CONTROL_TIME_SIGMA))
            pts.append((int(fidx), px, py, float(control)))
        # segment consecutive frames into runs below the control ceiling
        i = 0
        while i < len(pts):
            j = i
            while (j + 1 < len(pts)
                   and pts[j + 1][0] - pts[j][0] <= frame_gap
                   and pts[j + 1][3] <= opp_control_max):
                j += 1
            seg = pts[i:j + 1]
            if seg and all(p[3] <= opp_control_max for p in seg):
                x0, y0 = seg[0][1], seg[0][2]
                x1, y1 = seg[-1][1], seg[-1][2]
                dist = float(np.hypot(x1 - x0, y1 - y0))
                if dist >= min_run_m:
                    runs.append({
                        "track_id": tid,
                        "start": (round(x0, 2), round(y0, 2)),
                        "end": (round(x1, 2), round(y1, 2)),
                        "distance_m": round(dist, 2),
                        "max_opp_control": round(max(p[3] for p in seg), 3),
                        "frames": len(seg),
                        "threatening": x1 >= PITCH_LENGTH - 20.0,
                    })
            i = j + 1
    return runs
