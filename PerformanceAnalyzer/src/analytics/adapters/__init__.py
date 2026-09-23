"""Provider adapters.

Each adapter normalizes one input source into `src.analytics.schema` so the
model layer is provider-agnostic. Capabilities are exposed as mixins
(`EventsProvider`, `SparseTracking`, `DenseTracking`, `LineupsProvider`) so a
source can implement the primitive(s) it genuinely supports.

Current source adapters:
* `StatsBombAdapter` — Events + SparseTracking + Lineups (active).
* `VideoAdapter`     — DenseTracking + Events via CV pipeline (dormant until
                       explicitly enabled).
"""

from src.analytics.adapters.base import (
    ProviderAdapter,
    EventsProvider,
    SparseTracking,
    DenseTracking,
    LineupsProvider,
)
from src.analytics.adapters.idsse import IDSSEAdapter
from src.analytics.adapters.metrica import MetricaAdapter
from src.analytics.adapters.openfootball import OpenFootballAdapter
from src.analytics.adapters.skillcorner import SkillCornerAdapter
from src.analytics.adapters.soccerdata import SoccerDataAdapter
from src.analytics.adapters.statsbomb import StatsBombAdapter
from src.analytics.adapters.video import (
    VideoAdapter,
    VideoAdapterDormant,
    VIDEO_ADAPTER_DORMANT,
)
from src.analytics.adapters.wyscout import WyscoutAdapter

__all__ = [
    "ProviderAdapter",
    "EventsProvider",
    "SparseTracking",
    "DenseTracking",
    "LineupsProvider",
    "StatsBombAdapter",
    "SkillCornerAdapter",
    "WyscoutAdapter",
    "IDSSEAdapter",
    "MetricaAdapter",
    "OpenFootballAdapter",
    "SoccerDataAdapter",
    "VideoAdapter",
    "VideoAdapterDormant",
    "VIDEO_ADAPTER_DORMANT",
]
