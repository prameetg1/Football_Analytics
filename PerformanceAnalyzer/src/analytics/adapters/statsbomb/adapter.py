"""StatsBomb provider adapter: raw open-data JSON -> normalized schema.

Converts the nested StatsBomb payloads (via `StatsBombLoader`) into the
provider-agnostic `src.analytics.schema` types. Reuses the coordinate
conventions already established in `src.analytics.adapters.statsbomb.alignment` so Video/CV and
StatsBomb-derived metrics remain directly comparable.

Pitch frame: x in [0,120], y in [0,70], both teams' defending goal at x=0
(team 0 attacks left->right). StatsBomb stores x already toward the attacking
goal and y in [0,80]; we mirror x when `attacking` so a fixed frame is shared
and rescale y by 70/80.
"""

from __future__ import annotations

from functools import lru_cache

from src.analytics.adapters.base import (
    EventsProvider,
    LineupsProvider,
    SparseTracking,
)
from src.analytics.schema import (
    CarryEvent,
    Event,
    Frame,
    FreezeFrame,
    Match,
    PassEvent,
    Player,
    ShotEvent,
    Team,
)
from src.analytics.adapters.statsbomb.alignment import (
    convert_pitch_xy,
    seconds_from_timestamp,
    team_map,
)
from src.analytics.adapters.statsbomb.loader import StatsBombLoader

# StatsBomb "Ball Receipt*", "Miscontrol" etc. carry no location meaningfully
# for our models; we keep only event types models consume, and forward the raw
# type string for everything else so adapters stay lossless.
_COORDINATE_TYPES = {
    "Pass", "Shot", "Carry", "Ball Recovery", "Clearance", "Pressure",
    "Duel", "Throw-in", "Goal Kick", "Corner", "Foul Committed", "Foul Won",
    "Dispossessed", "Interception", "Block", "Dribble", "Miscontrol",
    "Ball Receipt*", "Offside", "Shield", "50/50", "Goal Keeper",
}

# outcome -> normalized PassOutcome
_PASS_OUTCOME = {"Complete": "complete", "Incomplete": "incomplete"}
# shot outcome -> normalized ShotResult
_SHOT_OUTCOME = {
    "Goal": "goal", "Saved": "saved", "Off Target": "off_target",
    "Blocked": "blocked", "Wayward": "off_target", "Post": "post",
    "Saved to Post": "post",
}


class StatsBombAdapter(EventsProvider, SparseTracking, LineupsProvider):
    """Provider adapter over StatsBomb open data.

    Capabilities: Events + SparseTracking (360) + Lineups. StatsBomb 360 is
    event-keyed snapshots with no persistent per-player identity, so this does
    NOT implement `DenseTracking`; `tracking()` is present only for interface
    completeness and always returns `[]`.
    """

    # capability labels this source genuinely provides (see resolver.CAPABILITIES)
    CAPABILITIES = {"events", "sparse_tracking", "lineups"}

    def __init__(self, loader: StatsBombLoader | None = None,
                 with_360: bool = True):
        self.loader = loader or StatsBombLoader()
        self._with_360 = with_360
        # caches (match_id -> result) for the call-heavy stages
        self._match_cache: dict[int, Match] = {}
        self._event_cache: dict[int, list[Event]] = {}
        self._freeze_cache: dict[int, list[FreezeFrame]] = {}

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _side_xy(x: float, y: float, attacking: bool) -> tuple[float, float]:
        return convert_pitch_xy(x, y, attacking)

    def _team(self, raw_team: dict, side: int) -> Team:
        return Team(team_id=int(raw_team["id"]),
                    name=raw_team.get("name", ""), side=side)

    # -- ProviderAdapter -----------------------------------------------------
    def match(self, match_id: int) -> Match:
        if match_id in self._match_cache:
            return self._match_cache[match_id]
        lineups = self.loader.lineups(match_id)
        tmap = team_map(lineups)
        teams: list[Team] = []
        players: list[Player] = []
        home = away = ""
        for t in lineups:
            side = tmap[int(t["team_id"])]
            teams.append(self._team({"id": t["team_id"], "name": t["team_name"]},
                                    side))
            for p in t["lineup"]:
                players.append(Player(
                    player_id=int(p["player_id"]),
                    name=p.get("player_name", ""),
                    position=(p.get("position") or {}).get("name", ""),
                    shirt_number=p.get("jersey_number"),
                ))
        m = Match(match_id=match_id, teams=teams, players=players,
                  home_team=teams[0].name if teams else "",
                  away_team=teams[1].name if len(teams) > 1 else "")
        self._match_cache[match_id] = m
        return m

    def events(self, match_id: int) -> list[Event]:
        if match_id in self._event_cache:
            return self._event_cache[match_id]
        raw = self.loader.events(match_id)
        tmap = team_map(self.loader.lineups(match_id))
        out: list[Event] = []
        for e in raw:
            tname = e["type"]["name"]
            loc = e.get("location") or []
            if tname not in _COORDINATE_TYPES or len(loc) < 2:
                continue
            attacking = e.get("possession_team", {}).get("id") in tmap
            side = tmap[e["team"]["id"]] if e.get("team") else None
            x, y = self._side_xy(loc[0], loc[1], attacking)
            ev = self._to_event(e, tname, x, y, side, tmap)
            if ev is not None:
                out.append(ev)
        out.sort(key=lambda ev: ev.timestamp_sec)
        self._event_cache[match_id] = out
        return out

    def _to_event(self, e: dict, tname: str, x: float, y: float,
                  side: int | None, tmap: dict) -> Event | None:
        ts = seconds_from_timestamp(e["timestamp"])
        pid = (e.get("player") or {}).get("id")
        common = dict(
            type=tname, timestamp_sec=ts, x=x, y=y,
            team_side=side, player_id=pid,
            period=int(e.get("period", 0)),
            minute=int(e.get("minute", 0)),
            second=int(e.get("second", 0)),
            duration=float(e.get("duration") or 0.0),
            under_pressure=bool(e.get("under_pressure", False)),
            counterpress=bool(e.get("counterpress", False)),
            play_pattern=(e.get("play_pattern") or {}).get("name", ""),
            position=(e.get("position") or {}).get("name", ""),
        )
        if tname == "Pass":
            p = e.get("pass") or {}
            end = p.get("end_location") or []
            attacking = e.get("possession_team", {}).get("id") in tmap
            ex = ey = None
            if len(end) >= 2:
                ex, ey = self._side_xy(end[0], end[1], attacking)
            return PassEvent(
                end_x=ex, end_y=ey,
                outcome=_PASS_OUTCOME.get((p.get("outcome") or {}).get("name"),
                                          "complete"),
                receiver_id=(p.get("recipient") or {}).get("id"),
                pass_type=(p.get("type") or {}).get("name", ""),
                body_part=(p.get("body_part") or {}).get("name", ""),
                height=(p.get("height") or {}).get("name", ""),
                length=float(p["length"]) if p.get("length") is not None else None,
                angle=float(p["angle"]) if p.get("angle") is not None else None,
                is_switch=bool(p.get("switch", False)),
                is_cross=bool(p.get("cross", False)),
                through_ball=bool(p.get("through_ball", False)),
                shot_assist=bool(p.get("shot_assist", False)),
                goal_assist=bool(p.get("goal_assist", False)),
                **common,
            )
        if tname == "Carry":
            c = e.get("carry") or {}
            end = c.get("end_location") or []
            attacking = e.get("possession_team", {}).get("id") in tmap
            ex = ey = None
            if len(end) >= 2:
                ex, ey = self._side_xy(end[0], end[1], attacking)
            return CarryEvent(end_x=ex, end_y=ey, **common)
        if tname == "Shot":
            s = e.get("shot") or {}
            out = _SHOT_OUTCOME.get((s.get("outcome") or {}).get("name"),
                                    "saved")
            return ShotEvent(
                result=out,
                xg=(s.get("statsbomb_xg") if "statsbomb_xg" in s else None),
                body_part=(s.get("body_part") or {}).get("name", ""),
                shot_type=(s.get("type") or {}).get("name", ""),
                technique=(s.get("technique") or {}).get("name", ""),
                first_time=bool(s.get("first_time", False)),
                **common,
            )
        # generic event: forward qualifiers that could matter
        return Event(**common)

    def passes(self, match_id: int) -> list[PassEvent]:
        return [e for e in self.events(match_id) if isinstance(e, PassEvent)]

    def shots(self, match_id: int) -> list[ShotEvent]:
        return [e for e in self.events(match_id) if isinstance(e, ShotEvent)]

    def freeze_frames(self, match_id: int) -> list[FreezeFrame]:
        if not self._with_360:
            return []
        if match_id in self._freeze_cache:
            return self._freeze_cache[match_id]
        try:
            raw360 = self.loader.three_sixty(match_id)
        except Exception:
            return []
        if not raw360:
            return []
        events = self.loader.events(match_id)
        ev_by_id = {e["id"]: e for e in events}
        tmap = team_map(self.loader.lineups(match_id))
        out: list[FreezeFrame] = []
        for ff in raw360:
            event = ev_by_id.get(ff.get("event_uuid"))
            if event is None:
                continue
            ts = seconds_from_timestamp(event["timestamp"])
            attacking = event.get("possession_team", {}).get("id") in tmap
            poss_side = (tmap[event["possession_team"]["id"]]
                         if attacking else None)
            players: list[tuple] = []
            ball = None
            for p in ff.get("freeze_frame", []):
                loc = p.get("location") or []
                if len(loc) < 2:
                    continue
                x, y = self._side_xy(loc[0], loc[1], attacking)
                own_side = tmap[event["possession_team"]["id"]] \
                    if p.get("teammate") else 1 - tmap[event["possession_team"]["id"]]
                players.append(((p.get("player") or {}).get("id"),
                                own_side, x, y, bool(p.get("keeper"))))
            ballloc = event.get("location") or []
            if len(ballloc) >= 2:
                bx, by = self._side_xy(ballloc[0], ballloc[1], attacking)
                ball = (bx, by)
            out.append(FreezeFrame(timestamp_sec=ts,
                                   event_uuid=ff.get("event_uuid"),
                                   possession_side=poss_side,
                                   players=players, ball=ball,
                                   event_type=(event.get("type") or {}).get(
                                       "name", ""),
                                   period=int(event.get("period", 0)),
                                   visible_area=list(ff.get("visible_area") or [])))
        out.sort(key=lambda ff: ff.timestamp_sec)
        self._freeze_cache[match_id] = out
        return out

    def tracking(self, match_id: int) -> list[Frame]:
        """No dense tracking available: StatsBomb 360 is event-keyed only."""
        return []
