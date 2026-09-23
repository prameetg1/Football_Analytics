"""Provider-agnostic normalized analytics schema.

This package decouples statistical modelling from any single data provider
(StatsBomb, Opta, StatsPerform, Wyscout, ...). Raw provider payloads are
converted by `src.analytics.adapters` into these normalized types, and every
model in `src.analytics.models` consumes only these types — never a provider's
raw JSON.

Coordinate convention (kept identical to the rest of the repo): canonical
pitch x in [0, 120] (length), y in [0, 70] (width), goal mouth at y = 35 and
attacking direction normalized so the *defending* goal of both teams is at
x = 0 (i.e. a fixed frame, not per-possession). `timestamp_sec` is seconds
since kick-off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Position = tuple[float, float]  # (x, y) canonical metres

PassOutcome = Literal["complete", "incomplete"]
ShotResult = Literal["goal", "saved", "off_target", "blocked", "post"]

# StatsBomb-style event type vocabulary (ported from src/events/models.py kind
# naming so CV-derived and provider-derived events stay directly comparable).
EVENT_TYPES = Literal[
    "Pass", "Shot", "Carry", "Ball Recovery", "Dispossessed", "Clearance",
    "Pressure", "Duel", "Throw-in", "Goal Kick", "Corner", "Foul Committed",
    "Foul Won", "Goal", "Own Goal Against", "Substitution",
    "Tactical Shift", "Half Start", "Half End", "Bad Behaviour",
]


@dataclass(frozen=True)
class Player:
    """A player within one team's lineup for a match."""
    player_id: int
    name: str
    position: str = ""           # normalized position label (e.g. "Goalkeeper")
    shirt_number: int | None = None


@dataclass(frozen=True)
class Team:
    """One match side."""
    team_id: int                 # provider-facing id
    name: str
    # our normalized 0/1 side index (lineup order) — mirrors alignment.team_map
    side: int


@dataclass(frozen=True)
class Match:
    """Normalized metadata + lineups for one match."""
    match_id: int
    competition: str = ""
    season: str = ""
    home_team: str = ""
    away_team: str = ""
    home_side: int = 0
    teams: list[Team] = field(default_factory=list)
    players: list[Player] = field(default_factory=list)


@dataclass(frozen=True)
class Event:
    """A discrete on-pitch event on the normalized canonical frame."""
    type: str                                    # EVENT_TYPES
    timestamp_sec: float
    x: float
    y: float
    team_side: int | None = None                 # normalized 0/1 side
    player_id: int | None = None
    period: int = 0
    minute: int = 0
    second: int = 0
    duration: float = 0.0                        # how long the action took (s)
    under_pressure: bool = False                 # opponent pressure during action
    counterpress: bool = False                   # immediate counter-pressure
    play_pattern: str = ""                       # regular / corner / throw-in ...
    position: str = ""                           # player's lineup position label
    qualifiers: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PassEvent(Event):
    """A pass with origin, destination and outcome."""
    end_x: float | None = None
    end_y: float | None = None
    outcome: PassOutcome = "complete"
    receiver_id: int | None = None
    pass_type: str = ""                          # "corner", "throw_in", ...
    body_part: str = ""
    height: str = ""                             # "ground", "low", "high"
    length: float | None = None                  # pass length in metres
    angle: float | None = None                   # pass angle
    is_switch: bool = False                      # switch of play
    is_cross: bool = False
    through_ball: bool = False
    shot_assist: bool = False
    goal_assist: bool = False
    # In-progress passes (not yet resolved) are excluded by the adapters.


@dataclass(frozen=True)
class CarryEvent(Event):
    """A ball carry: location moved ball from start to end.""" 
    end_x: float | None = None
    end_y: float | None = None


@dataclass(frozen=True)
class ShotEvent(Event):
    """A shot with location, result and optional xG."""
    result: ShotResult = "saved"
    xg: float | None = None
    body_part: str = ""
    shot_type: str = ""                          # "open_play", "set_piece", ...
    technique: str = ""
    first_time: bool = False


@dataclass(frozen=True)
class FreezeFrame:
    """One timestamped snapshot of every visible player (provider 360 data).

    `players` is a list of (player_id, team_side, x, y, is_goalkeeper) tuples.
    Event context (`event_type`, `period`) plus the broadcast-camera `visible_area`
    are kept so pitch-control / pressure / flow models at event time know the
    possession, phase and what the cameraman could see.
    """
    timestamp_sec: float
    event_uuid: str = ""
    possession_side: int | None = None
    players: list[tuple[int | None, int, float, float, bool]] = field(
        default_factory=list)
    ball: Position | None = None
    event_type: str = ""                         # the triggering event's type
    period: int = 0
    visible_area: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class Frame:
    """A time-aligned sample of every tracked/visible player at `timestamp_sec`.

    Like `FreezeFrame` but a pure position sample (not tied to one event): the
    360 freeze-frames are *event-keyed*, so the model layer re-times them into
    `Frame` objects when building velocities, flow fields and formations.
    `players` is a list of (player_id, team_side, x, y, is_goalkeeper) tuples.
    """
    timestamp_sec: float
    players: list[tuple[int | None, int, float, float, bool]] = field(
        default_factory=list)
    ball: Position | None = None
    possession_side: int | None = None
