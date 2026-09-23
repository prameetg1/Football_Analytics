"""Alignment: StatsBomb raw data -> our canonical tracking frame + events.

StatsBomb positions use x in [0,120] (length) and y in [0,80] (width), in the
*attacking* direction of the possession team. Our pipeline exports pitch metres
on the canonical `PitchConfig` frame (length 120, width 70). This module:

  * rescales y: 80 -> 70 so GT and CV positions are comparable,
  * builds a `TrackingFrame`-schema DataFrame from 360 freeze-frame player
    positions (the ground truth),
  * normalizes team ids to 0/1 (lineup order) and 120-x to a fixed frame so
    both teams share one coordinate system,
  * extracts pass events (timestamp, team, x/y) for recall/precision checks.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from src.config import (
    CLASS_BALL,
    CLASS_GOALKEEPER,
    CLASS_PLAYER,
    STATSBOMB_NOMINAL_FPS,
    STATSBOMB_PITCH_LENGTH,
    STATSBOMB_PITCH_WIDTH,
)
from src.analytics.common import TrackingFrame


def seconds_from_timestamp(ts: str) -> float:
    """'mm:ss.mmm' -> seconds since kick-off."""
    mm, ss, ms = ts.split(":")
    return int(mm) * 60 + int(ss) + float(ms)


def convert_pitch_xy(x: float, y: float, attacking: bool) -> tuple[float, float]:
    """StatsBomb (x, y) on [0,120]x[0,80] -> canonical pitch metres [0,120]x[0,70].

    `attacking` tells us the team attacks left->right in StatsBomb's frame. To
    pin both teams to one coordinate system (attack always right->left for
    team 0), we mirror x when attacking (so a shot toward the same goal reads
    the same x for both teams). y is rescaled to the 70 m width.
    """
    mx = (STATSBOMB_PITCH_LENGTH - x) if attacking else x
    my = y * (70.0 / STATSBOMB_PITCH_WIDTH)
    return float(mx), float(my)


def team_map(lineups: list[dict]) -> dict[int, int]:
    """StatsBomb team_id -> our 0/1 (lineup order)."""
    return {int(t["team_id"]): i for i, t in enumerate(lineups)}


def lineup_player_names(lineups: list[dict]) -> dict[int, set[str]]:
    """our team id -> set of player names on the team."""
    return {i: {p["player_name"] for p in t["lineup"]}
            for i, t in enumerate(lineups)}


def build_gt_frame(loader, match_id: int, with_360: bool = True) -> pd.DataFrame:
    """Ground-truth `TrackingFrame` from a match's 360 freeze-frame data.

    Each 360 event becomes one timestamped frame: a row per player with a
    pitch position, plus a ball row. Player rows use the player name (when the
    freeze frame identifies it) as `track_id` so position-error alignment can
    link CV tracks to GT players by name; fall back to a positional anchor when
    the name is absent.

    Returns a DataFrame with `TrackingFrame.REQUIRED_COLUMNS`.
    """
    match = loader.load_match(match_id, with_360=with_360)
    tmap = team_map(match.lineups)
    # event uuid -> event (to resolve timestamp + possession team)
    ev_by_id = {e["id"]: e for e in match.events}

    rows: list[dict] = []
    for i, ff in enumerate(match.three_sixty):
        event = ev_by_id.get(ff["event_uuid"])
        if event is None:
            continue
        ts_sec = seconds_from_timestamp(event["timestamp"])
        attacking = (event["possession_team"]["id"] in tmap)
        for p in ff["freeze_frame"]:
            name = (p.get("player") or {}).get("name", "")
            loc = p.get("location") or []
            if len(loc) < 2:
                continue
            x, y = convert_pitch_xy(loc[0], loc[1], attacking)
            team_our = tmap[event["possession_team"]["id"]] if p["teammate"] \
                else 1 - tmap[event["possession_team"]["id"]]
            rows.append({
                "frame_idx": i,
                "timestamp_sec": round(ts_sec, 4),
                "track_id": name or f"anchor_{i}_{len(rows)}",
                "team_id": team_our,
                "pitch_x": x,
                "pitch_y": y,
                "class_name": CLASS_GOALKEEPER if p.get("keeper") else CLASS_PLAYER,
            })
        ball = event.get("location")
        if ball and len(ball) >= 2:
            bx, by = convert_pitch_xy(ball[0], ball[1], attacking)
            rows.append({
                "frame_idx": i,
                "timestamp_sec": round(ts_sec, 4),
                "track_id": "ball",
                "team_id": np.nan,
                "pitch_x": bx,
                "pitch_y": by,
                "class_name": CLASS_BALL,
            })

    df = pd.DataFrame(rows, columns=TrackingFrame.REQUIRED_COLUMNS)
    if df.empty:
        raise ValueError(
            f"no 360 freeze-frame data for match {match_id} — cannot build GT")
    return df


def extract_pass_events(loader, match_id: int) -> pd.DataFrame:
    """Pass events: timestamp_sec, team_id (0/1), x/y (canonical metres), outcome.

    `outcome` is "Complete" or "Incomplete". Used by the CV-vs-event pass
    recall/precision validation. CV passes are detected at the receiver's
    contact point, so `end_x`/`end_y` (the StatsBomb pass end location) are
    the matching coordinates; `x`/`y` are the pass origin.
    """
    events = loader.events(match_id)
    tmap = team_map(loader.lineups(match_id))
    rows: list[dict] = []
    for e in events:
        if e["type"]["name"] != "Pass":
            continue
        loc = e.get("location") or []
        if len(loc) < 2:
            continue
        attacking = e["team"]["id"] in tmap
        x, y = convert_pitch_xy(loc[0], loc[1], attacking)
        end = (e.get("pass") or {}).get("end_location") or []
        end_x, end_y = (convert_pitch_xy(end[0], end[1], attacking)
                        if len(end) >= 2 else (np.nan, np.nan))
        outcome = (e.get("pass") or {}).get("outcome", {}).get("name", "Complete")
        rows.append({
            "timestamp_sec": round(seconds_from_timestamp(e["timestamp"]), 4),
            "team_id": tmap[e["team"]["id"]],
            "pitch_x": x,
            "pitch_y": y,
            "end_x": end_x,
            "end_y": end_y,
            "outcome": outcome,
        })
    return pd.DataFrame(rows)


def extract_event_locations(loader, match_id: int, kinds: tuple[str, ...]) -> pd.DataFrame:
    """Generic (timestamp, team, canonical x/y) extraction for StatsBomb event
    types — used as GT for the Stage-9 event gate.

    Shot events use `end_location` (the goal-mouth contact point) as the
    matching coordinates because CV shots are detected where the ball enters
    the goal mouth, not where it was struck.
    """
    events = loader.events(match_id)
    tmap = team_map(loader.lineups(match_id))
    rows: list[dict] = []
    for e in events:
        if e["type"]["name"] not in kinds:
            continue
        loc = e.get("location") or []
        if len(loc) < 2:
            continue
        attacking = e["team"]["id"] in tmap
        use = loc
        if e["type"]["name"] == "Shot":
            end = (e.get("shot") or {}).get("end_location") or []
            use = end if len(end) >= 2 else loc
        x, y = convert_pitch_xy(use[0], use[1], attacking)
        rows.append({
            "timestamp_sec": round(seconds_from_timestamp(e["timestamp"]), 4),
            "team_id": tmap[e["team"]["id"]],
            "pitch_x": x,
            "pitch_y": y,
        })
    return pd.DataFrame(rows)


def iter_gt_frames(df: pd.DataFrame) -> Iterator[tuple[float, pd.DataFrame]]:
    """(timestamp_sec, per-frame player/ball rows) sorted by time."""
    for _, grp in df.sort_values("timestamp_sec").groupby("timestamp_sec"):
        yield float(grp["timestamp_sec"].iloc[0]), grp
