"""Retrieval layer: fetch + cache raw data from many sources.

`Retriever` (base.py) is the acquisition half of an adapter: it knows how to
pull a source's raw payloads and cache them under `data/retrieval/<source>/`.
Adapters then reinterpret those payloads into `src.analytics.schema`.

Sources live in `registry.py`; built-ins include the StatsBomb loader, a
generic cached HTTP downloader (for dense-tracking open datasets), and the
soccerdata-backed scrapers (FBref / Understat / WhoScored / SofaScore / ESPN).
"""

from __future__ import annotations

from src.analytics.retrieval.base import (
    Retriever,
    RetrievalError,
    RETRIEVAL_ROOT,
    cache_dir,
)
from src.analytics.retrieval.registry import (
    get,
    available,
    register,
    register_builtins,
)

__all__ = [
    "Retriever",
    "RetrievalError",
    "RETRIEVAL_ROOT",
    "cache_dir",
    "get",
    "available",
    "register",
    "register_builtins",
]
