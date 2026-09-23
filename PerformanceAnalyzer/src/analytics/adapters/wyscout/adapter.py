"""Wyscout open-data adapter (events + lineups).

Reads one cached Wyscout match file (from `WyscoutRetriever`) and normalizes it
to `src.analytics.schema`:

* `Event` for every Wyscout action (mapping `eventName` onto our vocabulary),
* `PassEvent` for passes (accuracy from tag 1801),
* `ShotEvent` for shots,
* `Match` metadata + lineups from `teams`/`players`.

Implementations are capability mixins `EventsProvider` + `LineupsProvider`
(dense tracking is not part of this open dataset).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.analytics.adapters.base import EventsProvider, LineupsProvider
from src.analytics.schema import Event, Match, PassEvent, Player, ShotEvent, Team
from src.analytics.adapters.wyscout.coordinates import WyscoutPitch

_TAG_ACCURATE_PASS = 1801
_WYSCOUT_TO_KIND = {
    "Pass": "Pass",
    "Shot": "Shot",
    "Smart pass": "Pass",
    "Cross": "Pass",
    "Hand pass": "Pass",
    "High pass": "Pass",
    "Carry": "Carry",
    "Duel": "Event",
    "Free Kick": "Event",
    "Corner": "Event",
    "Foul": "Event",
    "Goal": "Goal",
    "Offside": "Event",
    "Clearance": "Event",
    "Interruption": "Event",
}


class WyscoutAdapter(EventsProvider, LineupsProvider):
    """Normalizes a cached Wyscout match JSON into schema types."""

    source = "wyscout"

    def __init__(self, match_path: str | Path):
        self.match_path = Path(match_path)
        self._match: Match | None = None
        self._events: list[Event] | None = None

    def _load(self) -> dict:
        with open(self.match_path) as fh:
            return json.load(fh)

    # -- LineupsProvider ----------------------------------------------------
    def match(self, match_id: int) -> Match:
        if self._match is not None:
            return self._match
        payload = self._load()
        raw_teams = payload.get("teams") or {}
        if isinstance(raw_teams, dict) and not any(
                isinstance(v, dict) and "wyId" in v for v in raw_teams.values()):
            raw_teams = [v.get("team", v) for v in raw_teams.values()
                         if isinstance(v, dict)]
        if isinstance(raw_teams, dict):
            raw_teams = [raw_teams[k] for k in sorted(raw_teams)]
        teams: list[Team] = []
        for i, t in enumerate(raw_teams):
            t = t.get("team", t) if isinstance(t, dict) else t
            if not isinstance(t, dict):
                continue
            teams.append(Team(team_id=int(t["wyId"]),
                              name=t.get("name", "") or t.get("officialName", ""),
                              side=i))
        raw_players = payload.get("players") or []
        if isinstance(raw_players, dict):
            # keyed by team id -> list of player dicts per team
            raw_players = [p for v in raw_players.values()
                           for p in (v if isinstance(v, list) else [v])]
        players = []
        for p in raw_players:
            p = p.get("player", p) if isinstance(p, dict) else p
            if not isinstance(p, dict) or "wyId" not in p:
                continue
            players.append(Player(
                player_id=int(p["wyId"]),
                name=f"{p.get('firstName', '')} {p.get('lastName', '')}".strip(),
                position=(p.get("role") or {}).get("shortName", "") or
                (p.get("role") or {}).get("name", ""),
                shirt_number=(p.get("shirtNumber")) or None))
        self._match = Match(
            match_id=int(match_id),
            home_team=teams[0].name if teams else "",
            away_team=teams[1].name if len(teams) > 1 else "",
            teams=teams,
            players=players,
        )
        return self._match

    # -- EventsProvider -----------------------------------------------------
    def events(self, match_id: int) -> list[Event]:
        if self._events is not None:
            return self._events
        payload = self._load()
        self.match(match_id)  # ensure teams populated
        spec = WyscoutPitch()
        sides: dict[int, int] = {t.team_id: t.side for t in self._match.teams}
        home_id = sides and self._match.teams[0].team_id
        out: list[Event] = []
        for e in payload.get("events", []):
            name = e.get("eventName", "")
            kind = _WYSCOUT_TO_KIND.get(name, "Event")
            pos = (e.get("positions") or [])
            if len(pos) < 1:
                continue
            team_id = e.get("teamId")
            side = sides.get(team_id) if team_id is not None else None
            home_attack_right = (team_id == home_id)
            x, y = spec.to_canonical(float(pos[0]["x"]), float(pos[0]["y"]),
                                     home_attack_right)
            ts = e.get("eventSec", 0.0)
            period = 1 if e.get("matchPeriod") == "1H" else 2
            pid = e.get("playerId")
            tags = {t.get("id") for t in e.get("tags", [])}
            common = dict(type=kind, timestamp_sec=round(float(ts), 4),
                          x=x, y=y, team_side=side, player_id=pid,
                          period=period)
            if kind == "Pass":
                end = pos[1] if len(pos) > 1 else None
                ex = ey = None
                if end:
                    ex, ey = spec.to_canonical(float(end["x"]), float(end["y"]),
                                               home_attack_right)
                out.append(PassEvent(
                    end_x=ex, end_y=ey,
                    outcome="complete" if _TAG_ACCURATE_PASS in tags else "incomplete",
                    receiver_id=e.get("receiverId"),
                    pass_type=e.get("subEventName", ""), **common))
            elif kind == "Shot":
                result = ("goal" if name == "Goal" else
                          ("off_target" if _TAG_ACCURATE_PASS not in tags
                           else "saved"))
                out.append(ShotEvent(result=result,
                                     xg=e.get("xg"),
                                     shot_type=e.get("subEventName", ""),
                                     **common))
            else:
                out.append(Event(**common))
        out.sort(key=lambda ev: ev.timestamp_sec)
        self._events = out
        return out

    def passes(self, match_id: int) -> list[PassEvent]:
        return [e for e in self.events(match_id) if isinstance(e, PassEvent)]

    def shots(self, match_id: int) -> list[ShotEvent]:
        return [e for e in self.events(match_id) if isinstance(e, ShotEvent)]