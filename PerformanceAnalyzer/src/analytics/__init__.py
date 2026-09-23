"""Provider-agnostic analytics layer.

Turn normalized match data (from any provider adapter) into statistical models
that feed analytics products (e.g. DefenseAnalytics). Decouples modelling from
data sourcing: the schema/adapter layer swallows provider differences, and the
model layer consumes only normalized `src.analytics.schema` types.

Layout
------
* `schema.py` / `common.py`  normalized types + shared foundational contracts
                            (TrackingFrame, PLAYER_CLASSES, pitch frame)
* `adapters/`               input sources -> normalized schema:
                              statsbomb/  (active structured data source)
                              video/      (CV pipeline, DORMANT until enabled)
* `models/`                 statistical models (events + pitch surfaces)
* `analyze.py`              high-level bundle: adapter -> MatchReport

Quick start (StatsBomb):
    from src.analytics.adapters import StatsBombAdapter
    from src.analytics.analyze import analyze_match

    adapter = StatsBombAdapter()
    report = analyze_match(adapter, 3869685)
"""

from src.analytics import models
from src.analytics.analyze import analyze_match, MatchReport
from src.analytics.adapters import (
    ProviderAdapter,
    StatsBombAdapter,
    VideoAdapter,
    VideoAdapterDormant,
)

__all__ = [
    "models", "analyze_match", "MatchReport",
    "ProviderAdapter", "StatsBombAdapter", "VideoAdapter", "VideoAdapterDormant",
]
