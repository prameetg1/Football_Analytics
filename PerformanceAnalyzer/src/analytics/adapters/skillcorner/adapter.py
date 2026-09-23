"""SkillCorner dense-tracking adapter.

Reads the files `SkillCornerRetriever` caches under
`data/retrieval/skillcorner/<match_id>/`:

* `<id>_match.json`              — teams, players, periods, pitch size.
* `<id>_tracking_extrapolated.jsonl` — per-frame JSONL: ball + all players at
  10 fps, positions in metres (origin centre), player ids resolved.
* `<id>_dynamic_events.csv`      — event stream with rich qualifiers (xT,
  xPass, defensive-shape state, ...) we do **not** recompute.

Implements `DenseTracking` (real per-player continuous positions), plus
`EventsProvider` and `LineupsProvider` from the same files. Coordinates are
normalized to the shared canonical pitch (x in [0,120], y in [0,70]) and the
goal frames align with the StatsBomb convention. This is the reference kind of
true tracking data that StatsBomb's 360 can never provide: a constructor for
the team-only models tested against synthetic dense frames.

Skills / copyright note: SkillCorner Open Data is MIT-licensed broadcast-derived
tracking (10 fps, extrapolated). It is fine to reuse for this project.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.analytics.adapters.base import (
    DenseTracking,
    EventsProvider,
    LineupsProvider,
)
from src.analytics.schema import Event, Frame, Match, PassEvent, Player, ShotEvent, Team
from src.analytics.adapters.skillcorner.coordinates import PitchSpec

_GOALKEEPER_POSITION_HINTS = {"goalkeeper", "gk", "keeper", "goalkeeper_reginald"}


class SkillCornerAdapter(DenseTracking, EventsProvider, LineupsProvider):
    """Normalizes one cached SkillCorner match dir into schema types."""

    source = "skillcorner"

    def __init__(self, match_dir: str | Path):
        self.match_dir = Path(match_dir)
        self._match: Match | None = None
        self._frames: list[Frame] | None = None
        self._events: list[Event] | None = None

    # -- payload loading ----------------------------------------------------
    def _load_match_json(self) -> dict:
        candidates = list(self.match_dir.glob("*_match.json"))
        if not candidates:
            raise FileNotFoundError(
                f"no *_match.json found in {self.match_dir}")
        with open(candidates[0]) as fh:
            return json.load(fh)

    def _tracking_path(self) -> Path:
        candidates = list(self.match_dir.glob("*_tracking*.jsonl"))
        if not candidates:
            raise FileNotFoundError(
                f"no tracking jsonl found in {self.match_dir}")
        return candidates[0]

    # -- LineupsProvider ----------------------------------------------------
    def match(self, match_id: int) -> Match:
        if self._match is not None:
            return self._match
        raw = self._load_match_json()
        pid_to_team = {}
        players = []
        for p in raw.get("players", []):
            pid_to_team[p["id"]] = p.get("team_id")
            players.append(Player(
                player_id=int(p["id"]),
                name=f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
                position=(p.get("player_role") or {}).get("name", ""),
                shirt_number=p.get("number"),
            ))
        home_id = (raw.get("home_team") or {}).get("id")
        tids = []
        for t in (raw.get("home_team"), raw.get("away_team")):
            if not t:
                continue
            tids.append(t["id"])
        # SkillCorner home = index 0
        teams = [Team(team_id=int(t["id"]),
                      name=(t.get("short_name") or t.get("name")),
                      side=i)
                 for i, t in enumerate(
                     (raw.get("home_team"), raw.get("away_team"))) if t]
        self._match = Match(
            match_id=int(match_id),
            teams=teams,
            players=players,
            home_team=raw.get("home_team", {}).get("short_name", ""),
            away_team=raw.get("away_team", {}).get("short_name", ""),
        )
        self._players_by_pid = pid_to_team
        return self._match

    # -- DenseTracking ------------------------------------------------------
    def tracking(self, match_id: int) -> list[Frame]:
        if self._frames is not None:
            return self._frames
        raw = self._load_match_json()
        spec = PitchSpec(float(raw.get("pitch_length", 104.0)),
                         float(raw.get("pitch_width", 68.0)))
        pid_to_team = {p["id"]: p.get("team_id")
                       for p in raw.get("players", [])}
        home_id = (raw.get("home_team") or {}).get("id")
        self._side_of_team = {}
        for pid, team in pid_to_team.items():
            self._side_of_team[team] = 0 if team == home_id else 1
        frames_sort: list[tuple[float, Frame]] = []
        with open(self._tracking_path()) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ts = self._timestamp_seconds(row)
                players = []
                for p in row.get("player_data", []):
                    pid = p.get("player_id")
                    team = pid_to_team.get(pid)
                    if team is None:
                        continue
                    side = self._side_of_team.get(team, 0)
                    cx, cy = spec.to_canonical(float(p["x"]), float(p["y"]))
                    players.append((pid, side, cx, cy,
                                    self._is_keeper(pid)))
                ball = None
                b = row.get("ball_data") or {}
                if b.get("x") is not None:
                    bx, byy = spec.to_canonical(float(b["x"]), float(b["y"]))
                    ball = (bx, byy)
                frames_sort.append((ts, Frame(timestamp_sec=ts,
                                              players=players, ball=ball)))
        frames_sort.sort(key=lambda t: t[0])
        self._frames = [f for _, f in frames_sort]
        return self._frames

    def _is_keeper(self, pid: int) -> bool:
        m = self.match(0).players if self._match is None else self._match.players
        for p in m:
            if p.player_id == pid and \
                    p.position.lower().strip() in _GOALKEEPER_POSITION_HINTS:
                return True
        return False

    def _timestamp_seconds(self, row: dict) -> float:
        ts = row.get("timestamp") or row.get("frame_time")
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, str):
            try:
                return self._seconds_from_ts_str(ts)
            except (ValueError, AttributeError):
                pass
        # 10 fps nominal feed: fall back to frame index.
        return float(row.get("frame", 0)) / 10.0

    def _seconds_from_ts_str(self, ts: str) -> float:
        mm, ss, ms = ts.split(":")
        return int(mm) * 60 + int(ss) + float(ms)

    # -- EventsProvider -----------------------------------------------------
    def events(self, match_id: int) -> list[Event]:
        if self._events is not None:
            return self._events
        csv_path = self._dynamic_events_path()
        out: list[Event] = []
        if csv_path is not None:
            out = self._parse_dynamic_events(csv_path)
        out.sort(key=lambda e: e.timestamp_sec)
        self._events = out
        return out

    def passes(self, match_id: int) -> list[PassEvent]:
        return [e for e in self.events(match_id) if isinstance(e, PassEvent)]

    def shots(self, match_id: int) -> list[ShotEvent]:
        return [e for e in self.events(match_id) if isinstance(e, ShotEvent)]

    def _dynamic_events_path(self) -> Path | None:
        candidates = list(self.match_dir.glob("*_dynamic_events.csv"))
        return candidates[0] if candidates else None

    def _parse_dynamic_events(self, csv_path: Path) -> list[Event]:
        import csv
        out: list[Event] = []
        rows = list(csv.DictReader(open(csv_path, newline="")))
        if not rows:
            return out
        spec = PitchSpec()
        side_cache: dict[str, int] = {}
        for r in rows:
            if not (r.get("x_start") and r.get("y_start")):
                continue
            try:
                x, y = spec.to_canonical(float(r["x_start"]), float(r["y_start"]))
            except (TypeError, ValueError):
                continue
            side = self._side_for_team_id(r.get("team_id"), side_cache)
            ts = float(r.get("minute_start", 0)) * 60 + float(r.get("second_start", 0))
            period = int(r.get("period") or 1)
            etype = (r.get("event_type") or "").lower()
            sub = (r.get("event_subtype") or "").lower()
            end_type = (r.get("end_type") or "").lower()
            start_type = (r.get("start_type") or "").lower()
            player_id = r.get("player_id")
            pid = int(player_id) if player_id and player_id.isdigit() else None
            # Map the SkillCorner possession-action vocabulary onto our event
            # types: a pass ends when end_type == "pass", a shot ends with
            # "shot"; the remaining actions stay generic possession/run events.
            kind = "Shot" if end_type == "shot" else (
                "Pass" if end_type == "pass" else "Event")
            common = dict(
                type=kind,
                timestamp_sec=round(ts, 4),
                x=x, y=y,
                team_side=side,
                player_id=pid,
                period=period,
            )
            if kind == "Pass":
                ex = ey = None
                if r.get("x_end") and r.get("y_end"):
                    try:
                        ex, ey = spec.to_canonical(float(r["x_end"]),
                                                   float(r["y_end"]))
                    except (TypeError, ValueError):
                        pass
                outcome = "complete" if r.get("pass_outcome") == "successful" \
                    else "incomplete"
                out.append(PassEvent(end_x=ex, end_y=ey, outcome=outcome,
                                     receiver_id=None, pass_type="",
                                     body_part="", height="",
                                     length=float(r["pass_distance"])
                                     if r.get("pass_distance") else None,
                                     angle=None, is_switch=False, is_cross=False,
                                     through_ball=False, shot_assist=False,
                                     goal_assist=False, **common))
            elif kind == "Shot":
                result = "goal" if r.get("goal", "").lower() == "true" else \
                    ("saved" if r.get("xshot_player_possession_end") else "missed")
                out.append(ShotEvent(result=result, xg=None, body_part="",
                                     shot_type=sub, technique="",
                                     first_time=False, **common))
            else:
                out.append(Event(**common))
        return out

    def _side_for_team_id(self, team_id: str | None,
                          cache: dict[str, int]) -> int | None:
        if not team_id:
            return None
        if team_id in cache:
            return cache[team_id]
        m = self.match(0)
        side = None
        for i, t in enumerate(m.teams):
            if str(t.team_id) == team_id:
                side = i
                break
        cache[team_id] = side
        return side