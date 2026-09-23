"""StatsBomb retriever — reuses the existing StatsBombLoader for acquisition.

The loader already handles fetch + cache of events/lineups/360 under
`data/statsbomb/`. This retriever simply exposes it through the unified
`Retriever` interface so the resolver/registry can treat StatsBomb like any
other source.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.analytics.adapters.statsbomb.loader import StatsBombLoader
from src.analytics.retrieval.base import RetrievalError

# Match payload kinds the loader can fetch.
_KINDS = ("events", "lineups", "360")


class StatsBombRetriever:
    """Unified Retriever facade over StatsBombLoader."""

    source = "statsbomb"

    def __init__(self, loader: StatsBombLoader | None = None):
        self._loader = loader or StatsBombLoader()

    def list_resources(self) -> list[str]:
        # Match ids we already have cached (from any kind of payload).
        ids = set()
        data_dir = self._loader.data_dir
        for p in data_dir.glob("match_*_events.json"):
            ids.add(p.name.split("_")[1])
        return sorted(ids)

    def fetch(self, resource_id: str) -> Path:
        match_id = int(resource_id)
        loader = self._loader
        # Force-fetch each kind through the loader's own cache.
        try:
            loader.events(match_id)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"statsbomb events {match_id}: {exc}") from exc
        loader.lineups(match_id)
        loader.three_sixty(match_id)
        # Return the events cache path as the canonical resource location.
        ev_path = loader._cache_path(match_id, "events")  # noqa: SLF001
        if not ev_path.exists():
            raise RetrievalError(f"statsbomb events {match_id} not cached")
        return ev_path

    @property
    def loader(self) -> StatsBombLoader:
        return self._loader
