"""Retriever layer: uniform way to get + cache raw data from any source.

A `Retriever` is the acquisition half of an adapter: it knows *how to obtain*
a source's raw payloads (HTTP download, scraping a website, calling a library)
and *caches* them to `data/retrieval/<source>/...` so the source's adapter can
convert them offline. Adapters stay responsible for *interpreting* payloads
into `src.analytics.schema`.

This is deliberately distinct from the adapter layer: a source can be scraped
(Retriever) and then normalized (Adapter) without the model layer ever knowing
how the bytes were obtained.

Politeness & caching
--------------------
Retrievers should:
* write every fetched payload under `data/retrieval/<source>/` and return the
  cached copy on repeat calls (no re-hitting the server),
* back off / rotate user agents when scraping (see `scrape_http`),
* be lazy about heavy third-party libraries (import `soccerdata` only when a
  SoccerData retriever is actually instantiated).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from src.analytics.schema import Frame, Match

# Every source's raw data lands under data/retrieval/<source>/
RETRIEVAL_ROOT = Path(__file__).resolve().parents[3] / "data" / "retrieval"


@runtime_checkable
class Retriever(Protocol):
    """Minimal interface every retriever implements."""

    source: str

    def list_resources(self) -> list[str]:
        """Identifiers (match ids / dataset names / files) this retriever can
        fetch. Used for discovery and by registry tooling."""
        ...

    def fetch(self, resource_id: str) -> Path:
        """Obtain and cache the raw payload for `resource_id`, returning the
        local cache path. Raises `RetrievalError` on failure."""
        ...


class RetrievalError(RuntimeError):
    """Raised when a retriever cannot obtain a payload."""


def cache_dir(source: str) -> Path:
    """Per-source cache directory under data/retrieval/."""
    d = RETRIEVAL_ROOT / source
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_empty_provider(match_or_none: Match | None) -> bool:
    """True when a source returned no lineups (e.g. a failed fetch)."""
    return match_or_none is None


__all__ = [
    "Retriever",
    "RetrievalError",
    "cache_dir",
    "RETRIEVAL_ROOT",
    "Frame",
    "Match",
]
