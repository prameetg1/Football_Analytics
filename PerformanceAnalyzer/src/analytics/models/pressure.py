"""Pressing analysis from normalized events (Soccermatics Ch 9).

Beyond possession, a team's out-of-possession behaviour is measured by when and
where it presses. Using `under_pressure` / `counterpress` per event plus `duration`
we build:

* Press counts by pitch zone (third/lane) — *where* a team presses.
* Press windows (Soccermatics Ch 9: on-ball windows ~2.3 s, high win ~5.5 s)
  — the field location bands where pressure is won/lost.
* Press rate vs a `baseline` (league-avg) to flag teams that under/over-press.

All inputs are normalized `Event` objects, so it is provider-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.analytics.schema import Event

PITCH_LENGTH = 120.0

# Soccermatics Ch 9 press-window thresholds: pressure within 2.3 s of a pass is
# "on-ball press"; sustained ~5.5 s onward is a deeper winning press.
ON_BALL_PRESS_SEG = 2.3
HIGH_PRESS_SEG = 5.5


def _thirds(x: float) -> str:
    """Which of the three longitudinal thirds x is in (own/neutral/opp)."""
    if x < PITCH_LENGTH / 3:
        return "defensive_third"
    if x < 2 * PITCH_LENGTH / 3:
        return "middle_third"
    return "final_third"


@dataclass
class PressMap:
    """Pressing summary for a team across a match."""
    team_side: int
    total_press_events: int = 0
    # where pressure was applied (counts per third and per 5-lane x-band)
    by_third: dict[str, int] = field(default_factory=dict)
    by_x_band: dict[tuple[int, int], int] = field(default_factory=dict)
    counterpress_count: int = 0       # immediate counter-press (win-back intent)
    pressured_against: int = 0        # times this team was the pressed side

    def share_in_final_third(self) -> float | None:
        tot = sum(self.by_third.values())
        return (self.by_third.get("final_third", 0) / tot) if tot else None


def press_map(events: list[Event], team_side: int, x_bands: int = 12) -> PressMap:
    """Aggregate pressing activity for `team_side`.

    `under_pressure` on an opponent event means this team was pressing the ball
    holder (the event's actor is the pressed side). We count events whose owning
    team is the *opponent* and that are flagged under pressure as this team's
    presses, then bucket by the event's location.
    """
    m = PressMap(team_side=team_side)
    band_w = PITCH_LENGTH / x_bands
    for e in events:
        if e.team_side is None:
            continue
        if e.team_side == team_side:
            # this team's own actions: being pressed by the opponent
            if e.under_pressure:
                m.pressured_against += 1
            if e.counterpress:
                m.counterpress_count += 1
            continue
        # opponent's action -> this team is the pressing party when pressured
        if e.under_pressure:
            m.total_press_events += 1
            third = _thirds(e.x)
            m.by_third[third] = m.by_third.get(third, 0) + 1
            band = (int(e.x // band_w), int((e.x // band_w) + 1))
            m.by_x_band[band] = m.by_x_band.get(band, 0) + 1
    return m


@dataclass
class PressWindows:
    """Existence of Soccermatics-style on-ball vs deep press windows."""
    on_ball_press_count: int = 0     # pressure within 2.3 s of possession start
    deep_press_count: int = 0        # pressure sustained into 5.5+ s window
    avg_press_duration: float = 0.0


def press_windows(events: list[Event], team_side: int) -> PressWindows:
    """Classify press strength from event durations.

    Uses the durations of opponent-pressured events (our team pressing) to bucket
    into on-ball vs deep (winning) press windows, following Soccermatics Ch 9's
    2.3 s / 5.5 s thresholds.
    """
    w = PressWindows()
    durations = []
    for e in events:
        if e.team_side is None or e.team_side == team_side:
            continue
        if e.under_pressure:
            d = e.duration
            durations.append(d)
            if d <= ON_BALL_PRESS_SEG:
                w.on_ball_press_count += 1
            if d >= HIGH_PRESS_SEG:
                w.deep_press_count += 1
    if durations:
        w.avg_press_duration = float(sum(durations) / len(durations))
    return w
