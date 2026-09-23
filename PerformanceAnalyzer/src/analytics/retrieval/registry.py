"""Registry of retrievers by source name.

`get(source)` returns a (possibly lazily-instantiated) retriever for a source,
or `None` if no retriever is registered. Kept dependency-light: registered
retrievers import their heavy third-party libraries lazily (inside the factory
or the retriever's own import), so importing this module never pulls in
`soccerdata`, `requests`, etc.
"""

from __future__ import annotations

from typing import Callable

# source name -> factory returning a Retriever instance.
_FACTORIES: dict[str, Callable[[], object]] = {}


def register(source: str, factory: Callable[[], object]) -> None:
    _FACTORIES[source] = factory


def get(source: str) -> object | None:
    factory = _FACTORIES.get(source)
    return factory() if factory is not None else None


def available() -> list[str]:
    return sorted(_FACTORIES)


def register_builtins() -> None:
    """Register all built-in retrievers (idempotent)."""
    from src.analytics.retrieval.http import HTTPFileRetriever
    from src.analytics.retrieval.statsbomb import StatsBombRetriever

    _FACTORIES.setdefault("http_file", lambda: HTTPFileRetriever())
    _FACTORIES.setdefault("statsbomb", lambda: StatsBombRetriever())
    register_tracking_builtins()
    register_wyscout_builtins()
    register_openfootball_builtin()
    # soccerdata-backed scrapers register lazily (they import the lib on use).
    register_soccerdata_builtins()


def register_wyscout_builtins() -> None:
    """Register the Wyscout open-data retriever."""
    from src.analytics.retrieval.wyscout import WyscoutRetriever

    _FACTORIES.setdefault("wyscout", lambda: WyscoutRetriever())


def register_tracking_builtins() -> None:
    """Register dense-tracking dataset retrievers (SkillCorner/IDSSE/Metrica)."""
    from src.analytics.retrieval.tracking import (
        IDSSERetriever,
        MetricaRetriever,
        SkillCornerRetriever,
    )

    _FACTORIES.setdefault("skillcorner", lambda: SkillCornerRetriever())
    _FACTORIES.setdefault("idsse", lambda: IDSSERetriever())
    _FACTORIES.setdefault("metrica", lambda: MetricaRetriever())


def register_soccerdata_builtins() -> None:
    """Register the soccerdata-backed scrapers (FBref/Understat/WhoScored/...)."""
    from src.analytics.retrieval.soccerdata import SOCCERDATA_SOURCES

    for name in SOCCERDATA_SOURCES:
        if name not in _FACTORIES:
            _FACTORIES[name] = _make_soccerdata_factory(name)


def register_openfootball_builtin() -> None:
    """Register the CC0 OpenFootball (football.json) retriever."""
    from src.analytics.retrieval.openfootball import OpenFootballRetriever

    _FACTORIES.setdefault("openfootball", lambda: OpenFootballRetriever())


def _make_soccerdata_factory(source: str) -> Callable[[], object]:
    def _factory() -> object:
        from src.analytics.retrieval.soccerdata import SoccerDataRetriever
        return SoccerDataRetriever(source)
    return _factory


register_builtins()

__all__ = [
    "register",
    "get",
    "available",
    "register_builtins",
    "register_soccerdata_builtins",
]
