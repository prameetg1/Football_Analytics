"""Stage 9, Stage B: event aggregates derived from the extracted event stream.

These are not per-frame scalar metrics (they need the *ordered event list*), so
they live outside the `MetricBase` family and take `list[Event]`:

  - pass_completion(events)          fraction of passes followed by a same-team
                                     Pass or non-recovery contact (receiver
                                     keeps it) before a turnover
  - pass_count / forward_pass_count  total and forward (x-advancing) passes
  - possession_sequences(events)     [(n_passes, n_carries, turnover?)]
  - ppda(events, team_id)            opponent passes / team's defensive actions
  - shots_per_possession(events)
  - turnover_rate(events, team_id)   forced vs unforced regains
  - pressing_success(events)         share of pressures followed by a regain
  - pressures_per_min(events)

The StatsBomb gate validates the underlying events; these aggregates are the
"product" numbers we surface once the events pass.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from src.analytics.adapters.video.events.models import Event

TURNOVER_KIND = "Ball Recovery"
DEFENSIVE_KINDS = ("Ball Recovery", "Interception", "Block", "Duel")


def pass_completion(events: list[Event]) -> float:
    """Fraction of Pass events not immediately followed by a turnover.

    A pass is deemed complete when the *next* event from the receiver's team is
    another Pass (chain continues) or the receiver keeps it (no turnover
    before the team's next ball contact). Cheap proxy: a pass is incomplete if
    the very next event is a turnover that the *losing* team is the passer's.
    """
    if not events:
        return 0.0
    n_pass = 0
    n_complete = 0
    for i, e in enumerate(events):
        if e.kind != "Pass":
            continue
        n_pass += 1
        # find the next event involving this team's possession
        nxt = None
        for j in range(i + 1, len(events)):
            ev = events[j]
            if ev.kind == "Pass" and ev.team_id == e.team_id:
                nxt = ev
                break
            if ev.kind == TURNOVER_KIND and ev.qualifiers.get("losing_team") == e.team_id:
                nxt = ev
                break
            if ev.kind == TURNOVER_KIND and ev.team_id == e.team_id:
                nxt = ev   # own regain is not the previous pass's failure
                break
            if ev.kind in ("Shot",):
                nxt = ev
                break
        complete = True
        if nxt is not None:
            if nxt.kind == TURNOVER_KIND and nxt.qualifiers.get("losing_team") == e.team_id:
                complete = False
            elif nxt.kind == TURNOVER_KIND and nxt.qualifiers.get("losing_team") != e.team_id:
                complete = True   # a regain while still owning is fine
        if complete:
            n_complete += 1
    return round(n_complete / n_pass, 4) if n_pass else 0.0


def forward_passes(events: list[Event]) -> int:
    """Passes whose destination x exceeds their origin x (attacking direction)."""
    n = 0
    prev_ball: dict[int, float] = {}
    for e in events:
        if e.kind != "Pass":
            continue
        # forward = destination is deeper in the attacking half than the origin.
        # Without a frame history here, use receiver x vs the team's mean:
        # handled by the caller-provided origin via qualifiers when available.
        origin = e.qualifiers.get("origin_x")
        if origin is not None and e.x > float(origin):
            n += 1
    return n


def possession_sequences(events: list[Event]) -> list[dict]:
    """Break the event stream into possessions.

    A sequence runs while the same `team_id` performs events; it ends when a
    Ball Recovery hands the ball to the opposition (or the stream ends).
    Returns [{"team_id", "n_passes", "n_carries", "n_shots", "ended_in_turnover"}].
    """
    sequences: list[dict] = []
    current: dict | None = None

    def flush():
        if current is not None:
            sequences.append(current)

    for e in events:
        if e.kind == TURNOVER_KIND:
            if current is not None:
                current["ended_in_turnover"] = True
                flush()
            # the regaining team starts a new possession with this event
            current = {"team_id": e.team_id, "n_passes": 0, "n_carries": 0,
                       "n_shots": 0, "ended_in_turnover": False}
            continue
        if current is None or e.team_id != current["team_id"]:
            flush()
            current = {"team_id": e.team_id, "n_passes": 0, "n_carries": 0,
                       "n_shots": 0, "ended_in_turnover": False}
        if e.kind == "Pass":
            current["n_passes"] += 1
        elif e.kind == "Carry":
            current["n_carries"] += 1
        elif e.kind == "Shot":
            current["n_shots"] += 1
    flush()
    return sequences


def shots_per_possession(events: list[Event]) -> float:
    seqs = [s for s in possession_sequences(events)
            if s["n_passes"] > 0 or s["n_carries"] > 0 or s["n_shots"] > 0]
    if not seqs:
        return 0.0
    shots = sum(s["n_shots"] for s in seqs)
    return round(shots / len(seqs), 4)


def turnovers_by_team(events: list[Event]) -> dict[int, int]:
    c: Counter = Counter()
    for e in events:
        if e.kind == TURNOVER_KIND and e.qualifiers.get("losing_team") is not None:
            c[int(e.qualifiers["losing_team"])] += 1
    return dict(c)


def ppda(events: list[Event], team_id: int) -> float:
    """PPDA for `team_id`: opponent passes allowed / team's defensive actions.

    Defensive actions are regains, interceptions, blocks and duels attributed
    to `team_id` (as the gaining/interfering team).
    """
    opp_passes = sum(1 for e in events if e.kind == "Pass" and e.team_id != team_id)
    def_actions = sum(1 for e in events
                      if e.kind in DEFENSIVE_KINDS
                      and e.team_id == team_id)
    if def_actions == 0:
        return float("inf")
    return round(opp_passes / def_actions, 2)


def turnover_rate(events: list[Event], team_id: int) -> float:
    """Fraction of the team's possessions ending in a turnover."""
    seqs = [s for s in possession_sequences(events)
            if s["team_id"] == team_id]
    if not seqs:
        return 0.0
    lost = sum(1 for s in seqs if s["ended_in_turnover"])
    return round(lost / len(seqs), 4)


def pressing_success(events: list[Event]) -> float:
    """Share of Pressure events followed by a regain within 2 events."""
    if not events:
        return 0.0
    n_press = 0
    n_success = 0
    for i, e in enumerate(events):
        if e.kind != "Pressure":
            continue
        n_press += 1
        for ev in events[i + 1:i + 3]:
            if ev.kind == TURNOVER_KIND and ev.qualifiers.get("losing_team") != e.team_id:
                n_success += 1
                break
    return round(n_success / n_press, 4) if n_press else 0.0


def pressures_per_min(events: list[Event], minutes: float) -> float:
    n = sum(1 for e in events if e.kind == "Pressure")
    if minutes <= 0:
        return 0.0
    return round(n / minutes, 2)
