"""VAEP on-ball valuation (Valuing Actions by Estimating Probabilities).

Every on-ball action changes the match's expected-scoring and
expected-conceding probabilities. VAEP prices an action as the *change* it
causes:

    value(action) = P(score | after) - P(score | before)
                  - (P(concede | after) - P(concede | before))

Shots get their scoring-after term from the provider xG (`ShotEvent.xg`);
passes/carries move possession from the start cell to the destination cell,
and a lost ball lets the opponent inherit part of the destination hazard.

Coordinate contract matches the repository event stream: adapter events sit
on the canonical frame where both sides attack toward `attacking_goal_x` (0
in the alignment-mirrored events, 120 in raw StatsBomb/Europass frames) —
the surface decays from that goal mouth. Pass `attacking_goal_x=120.0`
when feeding attacking-frame coordinates directly.

This is the Numbers Game valuation thread made concrete: rate players by
the change in chance-quality they *create* minus the chance-quality their
mistakes *concede*, not by goals/assists headlines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.analytics.schema import Event, PassEvent, ShotEvent

PITCH_LENGTH = 120.0
PITCH_WIDTH = 70.0

_GRID_NX = 12
_GRID_NY = 8

# Ball being lost hands the opposition a possession worth ~40% of the
# destination cell (a recovered possession starts further from goal on average).
_LOST_BALL_HAZARD_SHARE = 0.4


def hazard_surface(attacking_goal_x: float = 0.0,
                   nx: int = _GRID_NX, ny: int = _GRID_NY) -> np.ndarray:
    """Per-cell scoring probability for a possession starting in that cell.

    Exponential decay from the attacking goal mouth (default x=0, matching
    the alignment-mirrored canonical events; pass `attacking_goal_x=120.0`
    for attacking-frame coordinates). Cells outside the pitch are 0.
    """
    surf = np.zeros((ny, nx))
    for j in range(ny):
        cy = (j + 0.5) * PITCH_WIDTH / ny
        for i in range(nx):
            cx = (i + 0.5) * PITCH_LENGTH / nx
            d = float(np.hypot(cx - attacking_goal_x, cy - 35.0))
            surf[j, i] = 0.25 * np.exp(-d / 25.0)
    return surf


def _cell_index(x: float, y: float, nx: int, ny: int) -> tuple[int, int]:
    j = int(np.clip(y // (PITCH_WIDTH / ny), 0, ny - 1))
    i = int(np.clip(x // (PITCH_LENGTH / nx), 0, nx - 1))
    return i, j


def _hazard_at(surface: np.ndarray, x: float, y: float) -> float:
    i, j = _cell_index(x, y, surface.shape[1], surface.shape[0])
    return float(surface[j, i])


@dataclass
class VaepAction:
    """Valuation read-out for one on-ball action."""
    event_type: str
    team_side: int | None
    player_id: int | None = None
    minute: int = 0
    x: float = 0.0
    y: float = 0.0
    end_x: float | None = None
    end_y: float | None = None
    outcome: str = ""
    xg: float | None = None
    value: float = 0.0
    scoring_add: float = 0.0
    conceding_add: float = 0.0


@dataclass
class VaepSummary:
    """Per-player (and per-team) VAEP aggregates for a match."""
    by_player: dict[int, dict] = field(default_factory=dict)
    by_team: dict[int, float] = field(default_factory=dict)
    valued: int = 0

    def player_value(self, player_id: int) -> float:
        return self.by_player.get(player_id, {}).get("value", 0.0)

    def top_players(self, k: int = 5) -> list[tuple[int, float]]:
        return sorted(((p, v["value"]) for p, v in self.by_player.items()),
                      key=lambda kv: -kv[1])[:k]


def value_action(
    action: Event,
    surface: np.ndarray | None = None,
    attacking_goal_x: float = 0.0,
) -> VaepAction:
    """VAEP value of a single on-ball action under a hazard surface.

    For shots the scoring-after term is the provider xG (1.0 if it went in);
    otherwise it is the destination-cell hazard. The conceding term is the
    destination hazard the opponent inherits when the ball is lost.
    """
    if surface is None:
        surface = hazard_surface(attacking_goal_x)
    start_h = _hazard_at(surface, action.x, action.y)
    lost = 0.0

    if isinstance(action, ShotEvent):
        scor_before = 0.0
        scor_after = (float(action.xg) if action.xg is not None else start_h)
        end_h = scor_after
    else:
        scor_before = start_h
        ex = getattr(action, "end_x", None)
        ey = getattr(action, "end_y", None)
        if ex is not None and ey is not None:
            end_h = _hazard_at(surface, ex, ey)
        else:
            end_h = start_h
        scor_after = end_h
        outcome = getattr(action, "outcome", "")
        if outcome != "complete":
            lost = 1.0

    conceding_add = float(lost * _LOST_BALL_HAZARD_SHARE * max(end_h, 0.0))
    value = float(scor_after - scor_before) - conceding_add
    return VaepAction(
        event_type=action.type,
        team_side=action.team_side,
        player_id=action.player_id,
        minute=action.minute,
        x=action.x, y=action.y,
        end_x=getattr(action, "end_x", None),
        end_y=getattr(action, "end_y", None),
        outcome=getattr(action, "outcome", ""),
        xg=getattr(action, "xg", None),
        value=value,
        scoring_add=float(scor_after - scor_before),
        conceding_add=conceding_add,
    )


def vaep(
    actions: list[Event],
    shots: list[ShotEvent],
    attacking_goal_x: float = 0.0,
) -> tuple[list[VaepAction], VaepSummary]:
    """Value an on-ball event stream and aggregate per player/team.

    `actions` carry the passes/carries (and any shots already present);
    `shots` adds the highest-value actions with provider xG. The summary keeps
    per-player (value/actions/scoring/conceding) and per-team totals.
    """
    surface = hazard_surface(attacking_goal_x)
    valued: list[VaepAction] = []
    for ev in actions:
        if ev.team_side is None:
            continue
        valued.append(value_action(ev, surface))
    for sh in shots:
        valued.append(value_action(sh, surface))

    summary = VaepSummary(valued=len(valued))
    for v in valued:
        if v.team_side is not None:
            summary.by_team[v.team_side] = summary.by_team.get(v.team_side, 0.0) + v.value
        if v.player_id is not None:
            d = summary.by_player.setdefault(v.player_id, {
                "value": 0.0, "actions": 0, "scoring": 0.0, "conceding": 0.0})
            d["value"] += v.value
            d["actions"] += 1
            d["scoring"] += v.scoring_add
            d["conceding"] += v.conceding_add
    return valued, summary