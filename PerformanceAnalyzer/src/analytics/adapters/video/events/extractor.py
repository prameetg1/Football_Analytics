"""Event extraction from the per-frame tracking table.

Stage 9, Stage A. Turns the (player positions, ball, ownership) tracking frame
into discrete events:

  - Pass            — ball-owner changes to a different player, same team
  - Ball Recovery   — ball-owner changes to the opposition (a turnover)
  - Shot            — the ball enters the goal mouth at speed
  - Clearance       — the ball leaves the defensive third quickly
  - Throw-in        — the ball crosses the touchline
  - Goal Kick       — the ball crosses the goal line away from the corner zones
  - Corner          — the ball crosses the goal line near a corner
  - Pressure        — opposition within N m of the ball while it is owned
  - Carry           — one owner keeps the ball while it travels >= N m
  - Through Ball    — a pass whose receiver is beyond the opposition's line
  - Duel            — both teams have a player near a *loose* ball

Owner == the nearest player to the ball within `loose_m` (same notion the
possession metric uses). Every event carries a canonical (timestamp, x, y),
the uniform protocol the StatsBomb gate matches on.

`extract_events` runs every detector and merges results sorted by time; a
`kinds` filter selects a subset (used by the gate and tests).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    EVENT_CARRY_MIN_M,
    EVENT_CLEARANCE_DEFENSIVE_THIRD_M,
    EVENT_CLEARANCE_MAX_SEC,
    EVENT_CLEARANCE_MIDFIELD_M,
    EVENT_DUEL_RADIUS_M,
    EVENT_DUEL_MIN_FRAMES,
    EVENT_MIN_PASS_MOVE_M,
    EVENT_MIN_TURNOVER_MOVE_M,
    EVENT_PRESSURE_MIN_SAMPLES,
    EVENT_PRESSURE_RADIUS_M,
    EVENT_SHOT_GOAL_LINE_M,
    EVENT_SHOT_MIN_SPEED_MS,
    EVENT_THROUGH_BALL_DEFENDERS,
    EVENT_THROUGH_BALL_LINE_M,
)
from src.analytics.adapters.video.events.models import Event
from src.analytics.adapters.video.metrics.base import TrackingFrame
from src.analytics.adapters.video.metrics.util import PLAYER_CLASSES

PITCH_LENGTH = 120.0
PITCH_WIDTH = 70.0
GOAL_WIDTH = 7.32
GOAL_Y = PITCH_WIDTH / 2


def ball_ownership_chain(df: pd.DataFrame,
                         loose_m: float = 8.0) -> list[dict]:
    """Per-ball-frame nearest player: [{timestamp_sec, x, y, track_id, team_id}].

    Frames without a ball row (or with no player within `loose_m`) are skipped,
    so a flying ball simply has no owner sample until a player is nearest again.
    """
    ball = df[df["class_name"] == "ball"].dropna(
        subset=["pitch_x", "pitch_y"]).sort_values("timestamp_sec")
    players = df[df["class_name"].isin(PLAYER_CLASSES)].dropna(
        subset=["pitch_x", "pitch_y", "team_id"])
    chain: list[dict] = []
    for _, b in ball.iterrows():
        pl = players[players["timestamp_sec"] == b["timestamp_sec"]]
        if pl.empty:
            continue
        d = np.hypot(pl["pitch_x"] - b["pitch_x"],
                     pl["pitch_y"] - b["pitch_y"]).to_numpy()
        i = int(d.argmin())
        if d[i] > loose_m:
            continue
        chain.append({
            "timestamp_sec": float(b["timestamp_sec"]),
            "x": float(b["pitch_x"]),
            "y": float(b["pitch_y"]),
            "track_id": pl.iloc[i]["track_id"],
            "team_id": int(pl.iloc[i]["team_id"]),
        })
    return chain


def _passes_and_recoveries(chain: list[dict],
                           min_pass_m: float,
                           min_turnover_m: float) -> list[Event]:
    """Pass / Ball Recovery from ownership transitions in the chain."""
    events: list[Event] = []
    last_owner: dict | None = None
    for cur in chain:
        if last_owner is None:
            last_owner = cur
            continue
        moved = float(np.hypot(cur["x"] - last_owner["x"],
                               cur["y"] - last_owner["y"]))
        same_team = (cur["team_id"] == last_owner["team_id"])
        if same_team and cur["track_id"] != last_owner["track_id"]:
            if moved >= min_pass_m:
                events.append(Event(
                    kind="Pass",
                    timestamp_sec=cur["timestamp_sec"],
                    x=cur["x"],
                    y=cur["y"],
                    team_id=cur["team_id"],
                    qualifiers={
                        "passer": last_owner["track_id"],
                        "receiver": cur["track_id"],
                        "distance_m": round(moved, 2),
                    },
                ))
        elif not same_team and moved >= min_turnover_m:
            events.append(Event(
                kind="Ball Recovery",
                timestamp_sec=cur["timestamp_sec"],
                x=cur["x"],
                y=cur["y"],
                team_id=cur["team_id"],
                qualifiers={
                    "losing_team": last_owner["team_id"],
                    "distance_m": round(moved, 2),
                },
            ))
        last_owner = cur
    return events


def _shots(chain: list[dict]) -> list[Event]:
    """Shot — ball entering the goal mouth at speed (at most one per visit)."""
    events: list[Event] = []
    last_ball: dict | None = None
    in_mouth = False
    last_owner: dict | None = None
    for cur in chain:
        x, y, t = cur["x"], cur["y"], cur["timestamp_sec"]
        mouth = (abs(y - GOAL_Y) <= GOAL_WIDTH / 2
                 and (x <= EVENT_SHOT_GOAL_LINE_M
                      or x >= PITCH_LENGTH - EVENT_SHOT_GOAL_LINE_M))
        if last_ball is not None and mouth and not in_mouth:
            dt = t - last_ball["timestamp_sec"]
            dist = float(np.hypot(x - last_ball["x"], y - last_ball["y"]))
            speed = dist / dt if dt > 0 else 0.0
            if speed >= EVENT_SHOT_MIN_SPEED_MS:
                shooter = last_owner["team_id"] if last_owner else None
                events.append(Event(
                    kind="Shot",
                    timestamp_sec=t,
                    x=x,
                    y=y,
                    team_id=shooter,
                    qualifiers={"speed_ms": round(speed, 2)},
                ))
        in_mouth = mouth
        last_ball = cur
        last_owner = cur
    return events


def _clearances(df: pd.DataFrame) -> list[Event]:
    """Clearance — ball leaving the defensive third to past midfield quickly."""
    ball = df[df["class_name"] == "ball"].dropna(
        subset=["pitch_x", "pitch_y"]).sort_values("timestamp_sec")
    locs = ball[["timestamp_sec", "pitch_x", "pitch_y"]].to_numpy(dtype=np.float64)
    events: list[Event] = []
    for (t0, x0, y0), (t1, x1, y1) in zip(locs, locs[1:]):
        dt = t1 - t0
        if x0 > EVENT_CLEARANCE_DEFENSIVE_THIRD_M:
            continue
        if x1 < EVENT_CLEARANCE_MIDFIELD_M:
            continue
        if dt <= 0 or dt > EVENT_CLEARANCE_MAX_SEC:
            continue
        events.append(Event(
            kind="Clearance",
            timestamp_sec=float(t1),
            x=float(x1),
            y=float(y1),
            team_id=None,
            qualifiers={"from_defensive_third_m": float(x0),
                        "seconds": round(float(dt), 2)},
        ))
    return events


def _set_pieces(df: pd.DataFrame) -> list[Event]:
    """Throw-in / Goal Kick / Corner — the ball crossing the pitch boundary.

    Touchline crossing (y out of [0,70]) is a Throw-in; goal-line crossing
    (x out of [0,120]) near a corner zone is a Corner, otherwise a Goal Kick.
    """
    ball = df[df["class_name"] == "ball"].dropna(
        subset=["pitch_x", "pitch_y"]).sort_values("timestamp_sec")
    events: list[Event] = []
    for _, b in ball.iterrows():
        x, y = float(b["pitch_x"]), float(b["pitch_y"])
        t = float(b["timestamp_sec"])
        if y < 0 or y > PITCH_WIDTH:
            events.append(Event(kind="Throw-in", timestamp_sec=t, x=x, y=y))
        elif x < 0 or x > PITCH_LENGTH:
            corner_zone = (y <= EVENT_SHOT_GOAL_LINE_M * 2
                           or y >= PITCH_WIDTH - EVENT_SHOT_GOAL_LINE_M * 2)
            kind = "Corner" if corner_zone else "Goal Kick"
            events.append(Event(kind=kind, timestamp_sec=t, x=x, y=y))
    return events


def _pressures(df: pd.DataFrame, chain: list[dict]) -> list[Event]:
    """Pressure — opposition within N m of a ball owned by the other team."""
    owned = {round(c["timestamp_sec"], 4): c for c in chain}
    if not owned:
        return []
    players = df[df["class_name"].isin(PLAYER_CLASSES)].dropna(
        subset=["pitch_x", "pitch_y", "team_id", "timestamp_sec"])
    events: list[Event] = []
    for ts, c in owned.items():
        pl = players[players["timestamp_sec"].round(4) == ts]
        opp = pl[pl["team_id"] != c["team_id"]]
        if opp.empty:
            continue
        d = np.hypot(opp["pitch_x"] - c["x"], opp["pitch_y"] - c["y"])
        if (d <= EVENT_PRESSURE_RADIUS_M).any():
            events.append(Event(
                kind="Pressure",
                timestamp_sec=c["timestamp_sec"],
                x=c["x"],
                y=c["y"],
                team_id=c["team_id"],
                qualifiers={"pressers": int((d <= EVENT_PRESSURE_RADIUS_M).sum())},
            ))
    return events


def _carries(chain: list[dict]) -> list[Event]:
    """Carry — one owner keeps the ball while it travels >= N m."""
    events: list[Event] = []
    run: list[dict] = []
    for cur in chain:
        if run and cur["track_id"] != run[-1]["track_id"]:
            _emit_carry(events, run)
            run = []
        run.append(cur)
    _emit_carry(events, run)
    return events


def _emit_carry(events: list[Event], run: list[dict]) -> None:
    if len(run) < 2:
        return
    dist = float(np.hypot(run[-1]["x"] - run[0]["x"],
                          run[-1]["y"] - run[0]["y"]))
    if dist >= EVENT_CARRY_MIN_M:
        events.append(Event(
            kind="Carry",
            timestamp_sec=run[-1]["timestamp_sec"],
            x=run[-1]["x"],
            y=run[-1]["y"],
            team_id=run[0]["team_id"],
            qualifiers={"player": run[0]["track_id"],
                        "distance_m": round(dist, 2)},
        ))


def _through_balls(df: pd.DataFrame, passes: list[Event]) -> list[Event]:
    """Through Ball — a pass whose receiver is beyond the opposition's line.

    The opposition's defensive line is the mean x of their `n` deepest
    outfield players at the pass instant. A pass is a through ball when its
    receiver's x is beyond that line (towards the goal the passer attacks).
    """
    if not passes:
        return []
    players = df[df["class_name"] == "player"].dropna(
        subset=["pitch_x", "pitch_y", "team_id", "timestamp_sec"])
    events: list[Event] = []
    for p in passes:
        ts = round(p.timestamp_sec, 4)
        opp = players[(players["team_id"] != p.team_id)
                      & (players["timestamp_sec"].round(4) == ts)]
        if len(opp) < EVENT_THROUGH_BALL_DEFENDERS:
            continue
        # deepest defenders = smallest pitch_x (defending goal at x=0)
        depths = opp.nsmallest(EVENT_THROUGH_BALL_DEFENDERS, "pitch_x")
        line = float(depths["pitch_x"].mean())
        if p.x > line + EVENT_THROUGH_BALL_LINE_M:
            events.append(Event(
                kind="Through Ball",
                timestamp_sec=p.timestamp_sec,
                x=p.x,
                y=p.y,
                team_id=p.team_id,
                qualifiers={"defensive_line_m": round(line, 2)},
            ))
    return events


def _duels(df: pd.DataFrame) -> list[Event]:
    """Duel — a loose ball with both teams within N m of it simultaneously.

    Temporal clustering: consecutive frames where the condition holds are one
    duel, reported at its first frame, and runs shorter than
    `EVENT_DUEL_MIN_FRAMES` are discarded as noise.
    """
    ball = df[df["class_name"] == "ball"].dropna(
        subset=["pitch_x", "pitch_y"]).sort_values("timestamp_sec")
    players = df[df["class_name"].isin(PLAYER_CLASSES)].dropna(
        subset=["pitch_x", "pitch_y", "team_id", "timestamp_sec"])
    frames: list[tuple[float, float, float, int]] = []   # (t, x, y, near)
    for _, b in ball.iterrows():
        ts = round(float(b["timestamp_sec"]), 4)
        pl = players[players["timestamp_sec"].round(4) == ts]
        if pl.empty:
            continue
        d = np.hypot(pl["pitch_x"] - b["pitch_x"],
                     pl["pitch_y"] - b["pitch_y"])
        near = pl[d <= EVENT_DUEL_RADIUS_M]
        if len(near["team_id"].unique()) >= 2:
            frames.append((float(b["timestamp_sec"]),
                           float(b["pitch_x"]), float(b["pitch_y"]),
                           int(len(near))))
    events: list[Event] = []
    run: list[tuple[float, float, float, int]] = []
    for f in frames:
        if run and abs(f[0] - run[-1][0]) > 1.0:   # time gap ends the run
            if len(run) >= EVENT_DUEL_MIN_FRAMES:
                t, x, y = run[0][0], run[0][1], run[0][2]
                events.append(Event(
                    kind="Duel", timestamp_sec=t, x=x, y=y, team_id=None,
                    qualifiers={"players_near": max(n[3] for n in run),
                                "frames": len(run)},
                ))
            run = []
        run.append(f)
    if len(run) >= EVENT_DUEL_MIN_FRAMES:
        events.append(Event(
            kind="Duel", timestamp_sec=run[0][0], x=run[0][1], y=run[0][2],
            team_id=None,
            qualifiers={"players_near": max(n[3] for n in run),
                        "frames": len(run)},
        ))
    return events


# kind -> detector(df) ; chain-based detectors get the chain too.
def _run_all(df: pd.DataFrame, chain: list[dict]) -> list[Event]:
    return (
        _passes_and_recoveries(chain, EVENT_MIN_PASS_MOVE_M,
                               EVENT_MIN_TURNOVER_MOVE_M)
        + _shots(chain)
        + _clearances(df)
        + _set_pieces(df)
        + _pressures(df, chain)
        + _carries(chain)
        + _duels(df)
    )


def extract_events(df: pd.DataFrame,
                   kinds: tuple[str, ...] | None = None) -> list[Event]:
    """All Stage-A events from a tracking frame, sorted by time.

    `kinds` restricts the output to those event types (None == all). Through
    Balls are derived from the extracted passes, so they are only produced
    when "Pass" is in the kind set (or kinds is None).
    """
    chain = ball_ownership_chain(df)
    events = _run_all(df, chain)
    if kinds is not None and "Pass" in kinds:
        events += _through_balls(df, [e for e in events if e.kind == "Pass"])
    if kinds is not None:
        events = [e for e in events if e.kind in kinds]
    return sorted(events, key=lambda e: e.timestamp_sec)


def events_by_kind(events: list[Event]) -> dict[str, list[Event]]:
    """Group events by `kind` (only kinds actually present)."""
    out: dict[str, list[Event]] = {}
    for e in events:
        out.setdefault(e.kind, []).append(e)
    return out
