"""OpenFootball (openfootball/football.json) retriever — CC0 schedule/doc data.

OpenFootball / football.db publishes free, open, public-domain football data
(CC0) as per-league JSON season files plus the underlying Football.TXT sources.
Retrieval is a plain GitHub raw download — no API key, no scraping wall, with
explicit redistribution rights. It is a *match metadata* source (fixtures,
results, dates, rounds, goal scorers); it contains no pass/shot coordinates,
so the matching adapter exposes `LineupsProvider` (match metadata) and a
Goal-only `EventsProvider`.

Files land under `data/retrieval/openfootball/<season>/<league>.json` and
`.../<league>.txt` (the league's Football.TXT source with the goal detail):

* `<season>/<league>.json`  — full season fixture list (from football.json).
* `<season>/<league>.txt`     — Football.TXT source (used for goal events).
"""

from __future__ import annotations

from pathlib import Path

from src.analytics.retrieval.base import RetrievalError
from src.analytics.retrieval.http import HTTPFileRetriever

_FOOTBALL_JSON = "https://raw.githubusercontent.com/openfootball/football.json/master"
_ENGLAND = "https://raw.githubusercontent.com/openfootball/england/master"

# league key -> (football.json path, england txt path)
# `en.1` is the English Premier League season file used across football.json.
_LEAGUES = {
    "en.1": ("{season}/en.1.json", "{season}/1-premierleague.txt"),
}


class OpenFootballRetriever:
    """Fetch + cache one league-season of OpenFootball data (JSON + txt)."""

    source = "openfootball"

    def __init__(self, http: HTTPFileRetriever | None = None):
        self._http = http or HTTPFileRetriever()

    def list_resources(self) -> list[str]:
        return sorted(_LEAGUES)

    def _season_dir(self, season: str, league: str) -> Path:
        from src.analytics.retrieval.base import cache_dir
        d = cache_dir(self.source) / season / league
        d.mkdir(parents=True, exist_ok=True)
        return d

    def fetch(self, resource_id: str, season: str = "2024-25") -> Path:
        """Download + cache one league-season. Returns the JSON cache path."""
        if resource_id not in _LEAGUES:
            raise RetrievalError(
                f"unknown openfootball league '{resource_id}'; "
                f"known: {sorted(_LEAGUES)}")
        json_rel, txt_rel = _LEAGUES[resource_id]
        season_dir = self._season_dir(season, resource_id)
        json_dest = season_dir / "league.json"
        if not json_dest.exists():
            url = f"{_FOOTBALL_JSON}/{json_rel.format(season=season)}"
            try:
                cached = self._http.fetch(url)
            except RetrievalError as exc:  # noqa: BLE001
                raise RetrievalError(
                    f"could not fetch openfootball {resource_id} {season}: "
                    f"{exc}") from exc
            cached.rename(json_dest)
        txt_dest = season_dir / "league.txt"
        if not txt_dest.exists():
            url = f"{_ENGLAND}/{txt_rel.format(season=season)}"
            try:
                cached = self._http.fetch(url)
            except RetrievalError:  # noqa: BLE001
                pass  # txt optional (used for goal events only)
            else:
                cached.rename(txt_dest)
        return json_dest

    def file_path(self, season: str, league: str, kind: str) -> Path:
        return self._season_dir(season, league) / kind