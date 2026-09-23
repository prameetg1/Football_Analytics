"""soccerdata-backed scrapers (FBref / Understat / WhoScored / SofaScore / ESPN).

The `soccerdata` library implements polite HTML/API scraping with local caching
and user-agent / proxy / rate-limit controls for these sites. We wrap it behind
the unified `Retriever` interface and cache what we pull under
`data/retrieval/<source>/` for offline reuse.

`soccerdata` is imported lazily (inside the constructor) so this module is safe
to import without it installed — factories that need it will surface a clear
`RetrievalError` at construction time instead of at import time.

NOTE on licensing: scraping FBref (Opta-sourced) / WhoScored / SofaScore does
NOT transfer a redistribution licence. Use the data for personal analysis; do
not republish raw tables.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.analytics.retrieval.base import RetrievalError, cache_dir

# league slug -> soccerdata league key (a reasonable default catalogue).
# Users can pass custom leagues/seasons via constructor kwargs.
# NOTE: `football-data` is NOT a soccerdata class (it is a separate site/API),
# so it is intentionally excluded here; `whoscored`/`espn`/`sofascore` are
# match-level-only sources (schedule/lineups), they expose no player_season.
_SOURCE_LEAGUES: dict[str, str] = {
    "fbref": "ENG-Premier League",
    "understat": "ENG-Premier League",
    "whoscored": "ENG-Premier League",
    "sofascore": "ENG-Premier League",
    "espn": "ENG-Premier League",
}

SOCCERDATA_SOURCES = tuple(_SOURCE_LEAGUES)


class SoccerDataRetriever:
    """Scrape + cache one site's data via the `soccerdata` library."""

    def __init__(self, source: str, season: str = "2021",
                 leagues: str | list[str] | None = None):
        if source not in _SOURCE_LEAGUES:
            raise RetrievalError(f"unsupported soccerdata source '{source}'")
        self.source = source
        self.season = season
        self.leagues = leagues or _SOURCE_LEAGUES[source]
        # Season-scoped cache dir (e.g. data/retrieval/fbref/2021/) so fetching
        # another season never clobbers the default season's tables.
        season_dir = str(season) if season is not None else "default"
        self._dir = cache_dir(source) / season_dir
        # Lazy import: constructing the retriever is what pulls in `soccerdata`.
        self._sd = self._load_soccerdata()

    def _load_soccerdata(self) -> object:
        # Keep soccerdata's HTTP cache on the spacious WorkDirectory mount
        # instead of ~/soccerdata on the root filesystem. Must be set before
        # soccerdata is imported (its _config caches the dir at import time).
        # Resolve `data/` through its symlink to reach the DataStore root.
        data_store = cache_dir("soccerdata").resolve().parent.parent.parent
        os.environ.setdefault(
            "SOCCERDATA_DIR",
            str(data_store / "soccerdata"),
        )
        try:
            import soccerdata as sd  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RetrievalError(
                "the 'soccerdata' library is required for this source; "
                "install it (pip install soccerdata)") from exc
        return sd

    # -- helpers ------------------------------------------------------------
    def _client(self):
        if self.source == "understat":
            return self._sd.Understat(self.leagues, self.season)
        if self.source == "sofascore":
            # class is `Sofascore` (lowercase s) in soccerdata 1.9.x
            return self._sd.Sofascore(self.leagues, self.season)
        if self.source == "espn":
            return self._sd.ESPN(self.leagues, self.season)
        if self.source == "whoscored":
            return self._sd.WhoScored(self.leagues, self.season)
        return self._sd.FBref(self.leagues, self.season)

    def _cache_path(self, name: str) -> Path:
        return self._dir / f"{name}.json"

    def _write_df(self, name: str, df) -> Path:
        path = self._cache_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(df.to_json(orient="records", date_format="iso"))
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"could not serialise {name}: {exc}") from exc
        return path

    # -- Retriever interface -------------------------------------------------
    def list_resources(self) -> list[str]:
        return [p.stem for p in self._dir.glob("*.json")]

    def fetch(self, resource_id: str) -> Path:
        """resource_id is a category key: schedule | player_season | team_season
        | shots | events | elo. Sources that lack a category raise RetrievalError;
        call `available_resources()` to see what a given source supports."""
        client = self._client()
        if resource_id == "schedule":
            if not hasattr(client, "read_schedule"):
                raise RetrievalError(f"{self.source} has no schedule data")
            return self._write_df("schedule", client.read_schedule())
        if resource_id == "player_season":
            if not hasattr(client, "read_player_season_stats"):
                raise RetrievalError(
                    f"{self.source} has no player-season data")
            return self._write_df("player_season",
                                  client.read_player_season_stats())
        if resource_id == "team_season":
            if not hasattr(client, "read_team_season_stats"):
                raise RetrievalError(f"{self.source} has no team-season data")
            return self._write_df("team_season",
                                  client.read_team_season_stats())
        if resource_id == "shots":
            if not hasattr(client, "read_shot_events"):
                raise RetrievalError(
                    f"{self.source} has no shot-level data")
            return self._write_df("shots", client.read_shot_events())
        if resource_id == "events":
            if not hasattr(client, "read_events"):
                raise RetrievalError(f"{self.source} has no event stream")
            return self._write_df("events", client.read_events())
        if resource_id == "elo" and self.source == "clubelo":
            return self._write_df("elo", client.read_elo())
        raise RetrievalError(f"unknown resource '{resource_id}' for "
                             f"{self.source}")

    # category -> method name, for callers that list a source's capabilities
    _RESOURCE_METHOD = {
        "schedule": "read_schedule",
        "player_season": "read_player_season_stats",
        "team_season": "read_team_season_stats",
        "shots": "read_shot_events",
        "events": "read_events",
    }

    def available_resources(self) -> list[str]:
        """Category keys this source can actually fetch (by method presence)."""
        client = self._client()
        return [res for res, meth in self._RESOURCE_METHOD.items()
                if hasattr(client, meth)]
