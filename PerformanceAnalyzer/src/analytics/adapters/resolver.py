"""Capability resolver: map a model's primitive needs onto concrete adapters.

Models declare *what* they need (events, tracking, lineups, ...) without
naming a provider. `Resolver` turns that into a concrete `ResolvedProvider`
that joins the answer together — ideally from a single source, or as a
cross-source merge (e.g. StatsBomb events + a dense-tracking provider) when no
one source satisfies everything.

The merge is intentionally thin: it picks, per capability, the adapter that
provides it (in registration or explicit order) and delegates. Real
timestamp-level cross-source alignment (Kloppy-style) is a separate concern on
top of the resolver; here we assume capability-level substitution is enough for
most analyses (each capability is self-consistent within its own source).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.analytics.adapters.base import (
    DenseTracking,
    EventsProvider,
    LineupsProvider,
    SparseTracking,
)
from src.analytics.schema import Event, Frame, FreezeFrame, Match, PassEvent, ShotEvent

# capability -> the (single) protocol method used to detect support
_CAPABILITY_METHOD: dict[str, str] = {
    "events": "events",
    "sparse_tracking": "freeze_frames",
    "dense_tracking": "tracking",
    "lineups": "match",
}

CAPABILITIES = tuple(_CAPABILITY_METHOD)


@dataclass
class ResolvedProvider:
    """A merged view over whichever adapters satisfy a requirement set."""

    events_provider: EventsProvider | None = None
    sparse_provider: SparseTracking | None = None
    dense_provider: DenseTracking | None = None
    lineups_provider: LineupsProvider | None = None
    # capability label -> source adapter label actually used (for reporting)
    sources: dict[str, str] = field(default_factory=dict)

    # -- individual capabilities -------------------------------------------
    def match(self, match_id: int) -> Match | None:
        if self.lineups_provider is None:
            return None
        return self.lineups_provider.match(match_id)

    def events(self, match_id: int) -> list[Event]:
        if self.events_provider is None:
            return []
        return self.events_provider.events(match_id)

    def passes(self, match_id: int) -> list[PassEvent]:
        if self.events_provider is None:
            return []
        return self.events_provider.passes(match_id)

    def shots(self, match_id: int) -> list[ShotEvent]:
        if self.events_provider is None:
            return []
        return self.events_provider.shots(match_id)

    def freeze_frames(self, match_id: int) -> list[FreezeFrame]:
        if self.sparse_provider is None:
            return []
        return self.sparse_provider.freeze_frames(match_id)

    def tracking(self, match_id: int) -> list[Frame]:
        if self.dense_provider is None:
            return []
        return self.dense_provider.tracking(match_id)

    # -- query --------------------------------------------------------------
    def has(self, capability: str) -> bool:
        return self.sources.get(capability) is not None

    def missing(self, needs: set[str]) -> set[str]:
        return {c for c in needs if c not in self.sources}


def _implements(adapter: object, capability: str) -> bool:
    # Prefer an explicit capability declaration; fall back to method presence
    # (which risks treating a `[]`-returning stub as a real capability).
    declared = getattr(adapter, "CAPABILITIES", None)
    if declared is not None:
        return capability in declared
    method = _CAPABILITY_METHOD[capability]
    return callable(getattr(adapter, method, None))


def _label(adapter: object) -> str:
    return type(adapter).__name__


class Resolver:
    """Selects adapters by required capabilities.

    Args:
        adapters: ordered list of concrete adapter instances. Earlier entries
            win ties for the same capability.
    """

    def __init__(self, adapters: list[object]):
        self.adapters = list(adapters)

    def resolve(self, match_id: int, needs: set[str] | None = None) -> ResolvedProvider:
        """Resolve the union of `needs` (defaults to all capabilities)."""
        needs = set(needs) if needs is not None else set(CAPABILITIES)
        rp = ResolvedProvider()
        for cap in CAPABILITIES:  # stable order
            if cap not in needs:
                continue
            for ad in self.adapters:
                if _implements(ad, cap):
                    cls = type(ad)
                    if cap == "events" and rp.events_provider is None:
                        rp.events_provider = ad
                    elif cap == "sparse_tracking" and rp.sparse_provider is None:
                        rp.sparse_provider = ad
                    elif cap == "dense_tracking" and rp.dense_provider is None:
                        rp.dense_provider = ad
                    elif cap == "lineups" and rp.lineups_provider is None:
                        rp.lineups_provider = ad
                    rp.sources[cap] = _label(ad)
                    break
        return rp

    def register(self, adapter: object) -> "Resolver":
        """Append an adapter (e.g. discovered at runtime)."""
        self.adapters.append(adapter)
        return self


def build_resolver(*extra_adapters: object) -> Resolver:
    """Default resolver seeded with the active built-in source adapters.

    Pass `SkillCornerAdapter(match_dir)` instances (or any other adapter) as
    `extra_adapters` to add dense-tracking capability at runtime.
    """
    from src.analytics.adapters import StatsBombAdapter
    return Resolver([StatsBombAdapter(), *extra_adapters])
