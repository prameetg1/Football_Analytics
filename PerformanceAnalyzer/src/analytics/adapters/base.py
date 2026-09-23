"""Provider adapter interface — capability mixins.

A `ProviderAdapter` turns a provider's raw payloads into the normalized
`src.analytics.schema` types so model code never imports provider-specific JSON.

Instead of one monolithic interface, we model *capabilities*: any concrete
adapter implements the mixins for the payload kinds its source can actually
produce. This lets a model declare what primitives it needs and lets an
orchestrator resolve them from however many sources it takes — a single
provider if it can, or a cross-source merge when required (e.g. StatsBomb
events + IDSSE dense tracking).

Capabilities
============
* `EventsProvider`    — discrete event stream (events/passes/shots).
* `SparseTracking`    — event-keyed position snapshots (StatsBomb 360):
                      `freeze_frames`, NO continuous per-player series.
* `DenseTracking`     — continuous per-player position series with player
                      identity: `tracking`. (IDSSE/SkillCorner/PFF/Video.)
* `LineupsProvider`   — match metadata + team/player lineups.

A source may be several of these at once — StatsBomb implements Events,
SparseTracking and Lineups (its 360 is "sparse tracking": sampled snapshots,
no persistent player identity). Models always consume schema types, never the
provider, so swapping the backing source only changes resolution, not code.

Coordinate/ownership rules are shared: each adapter normalizes the canonical
pitch itself (x in [0,120], y in [0,70], defending goal at x=0 for both sides)
so downstream models don't think about per-possession mirroring.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.analytics.schema import (
    Event,
    Frame,
    FreezeFrame,
    Match,
    PassEvent,
    ShotEvent,
)


@runtime_checkable
class EventsProvider(Protocol):
    """Provides discrete on-ball events, normalized to schema types."""

    def events(self, match_id: int) -> list[Event]:
        """All events for the match, normalized, sorted by timestamp."""
        ...

    def passes(self, match_id: int) -> list[PassEvent]:
        """Only pass events."""
        ...

    def shots(self, match_id: int) -> list[ShotEvent]:
        """Only shot events."""
        ...


@runtime_checkable
class SparseTracking(Protocol):
    """Provides event-keyed position snapshots (StatsBomb-style 360).

    These are snapshots tied to specific events (a handful per match), with
    *no* persistent per-player identity, so per-player analyses are impossible
    — only team-level aggregates.
    """

    def freeze_frames(self, match_id: int) -> list[FreezeFrame]:
        """Event-keyed position snapshots (empty if provider lacks them)."""
        ...


@runtime_checkable
class DenseTracking(Protocol):
    """Provides continuous per-player position series (true tracking).

    Every frame carries the full visible set with real `player_id`s, so
    per-player formations, flow, and high-fidelity pitch control are possible.
    """

    def tracking(self, match_id: int) -> list[Frame]:
        """Continuous per-player position samples, sorted by time."""
        ...


@runtime_checkable
class LineupsProvider(Protocol):
    """Provides match metadata and team/player lineups."""

    def match(self, match_id: int) -> Match:
        """Normalized match metadata + lineups."""
        ...


# Backward-compatible union alias: the old monolithic `ProviderAdapter` name
# matched everything; keep it for imports that used capability-agnostic code.
class ProviderAdapter(EventsProvider, SparseTracking, DenseTracking,
                      LineupsProvider, Protocol):
    """Union of every capability.

    Used by code that wants "give me whatever this source has" without
    specifying primitives. Model/orchestrator code should prefer the specific
    capability mixins instead.
    """
