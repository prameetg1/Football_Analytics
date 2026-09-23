"""Dense-tracking dataset retrievers (IDSSE/SkillCorner/PFF).

These aren't live APIs — they're published open datasets fetched as files.

* SkillCorner Open Data  — GitHub raw JSON/JSONL (MIT), broadcast-derived at
  10 fps with player identities; several matches across top-5 leagues + a
  2024/25 A-League batch.
* IDSSE / Sportec DFL    — official 25 Hz TRACAB optical tracking + events for
  7 Bundesliga matches (figshare, CC-BY 4.0); the highest-fidelity public feed.
* PFF FC                 — broadcast-derived 35-match WC2022 set (request-gated;
  loadable via `kloppy.pff` once you have the files).

Retrievers expose per-match `fetch(match_id)` that returns the local directory
of raw tracking+metadata payloads ready for a `DenseTrackingAdapter`.
"""

from __future__ import annotations

from pathlib import Path

from src.analytics.retrieval.base import RetrievalError
from src.analytics.retrieval.http import HTTPFileRetriever

# ---------------------------------------------------------------------------
# SkillCorner open data match catalogue — A-League 2024/25 batch (10 matches)
# as currently published on the opendata repo master branch. Files per match:
#   {id}_match.json, {id}_tracking_extrapolated.jsonl,
#   {id}_dynamic_events.csv, {id}_phases_of_play.csv
# ---------------------------------------------------------------------------
SKILLCORNER_ALEAGUE: dict[str, tuple[str, str]] = {
    "1886347": ("A-League", "A-League"),
    "1899585": ("A-League", "A-League"),
    "1925299": ("A-League", "A-League"),
    "1953632": ("A-League", "A-League"),
    "1996435": ("A-League", "A-League"),
    "2006229": ("A-League", "A-League"),
    "2011166": ("A-League", "A-League"),
    "2013725": ("A-League", "A-League"),
    "2015213": ("A-League", "A-League"),
    "2017461": ("A-League", "A-League"),
}


def _skillcorner_base() -> str:
    return ("https://raw.githubusercontent.com/SkillCorner/opendata/"
            "master/data/matches")


class SkillCornerRetriever:
    """Fetch + cache SkillCorner broadcast tracking for known match ids."""

    source = "skillcorner"

    def __init__(self, http: HTTPFileRetriever | None = None):
        self._http = http or HTTPFileRetriever()

    def list_resources(self) -> list[str]:
        return list(SKILLCORNER_ALEAGUE)

    def _match_dir(self, match_id: str) -> Path:
        from src.analytics.retrieval.base import cache_dir
        return cache_dir(self.source) / match_id

    def fetch(self, match_id: str) -> Path:
        if match_id not in SKILLCORNER_ALEAGUE:
            raise RetrievalError(f"unknown skillcorner match '{match_id}'")
        base = _skillcorner_base()
        match_dir = self._match_dir(match_id)
        match_dir.mkdir(parents=True, exist_ok=True)
        urls = {
            f"{match_id}_match.json":
                f"{base}/{match_id}/{match_id}_match.json",
            f"{match_id}_tracking_extrapolated.jsonl":
                f"{base}/{match_id}/{match_id}_tracking_extrapolated.jsonl",
            f"{match_id}_dynamic_events.csv":
                f"{base}/{match_id}/{match_id}_dynamic_events.csv",
            f"{match_id}_phases_of_play.csv":
                f"{base}/{match_id}/{match_id}_phases_of_play.csv",
        }
        for name, url in urls.items():
            path = match_dir / name
            if not path.exists():
                cached = self._http.fetch(url)
                cached.rename(path)
        return match_dir


# ---------------------------------------------------------------------------
# IDSSE / Sportec DFL Bundesliga (Bassek et al. 2025, CC-BY 4.0)
#
# Originally figshare item 28196177 (now broken -> returns empty). The PySport
# community mirrored the raw DFL XML files to Hugging Face
# (`pysport/idsse-data`), which is the stable route used by kloppy since v6.x.
# Per match there are three XML files (matchinformation / events_raw /
# positions_raw_observed), 7 matches at 25 Hz TRACAB.
# ---------------------------------------------------------------------------
IDSSE_HF_DATASET = "pysport/idsse-data"
IDSSE_MATCH_IDS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY",
                   "J03WPY", "J03WQQ", "J03WR9")
_IDSSE_KIND_SUFFIX = {
    "matchinformation": "DFL_02_01_matchinformation_{comp}_DFL-MAT-{mid}.xml",
    "events": "DFL_03_02_events_raw_{comp}_DFL-MAT-{mid}.xml",
    "tracking": "DFL_04_03_positions_raw_observed_{comp}_DFL-MAT-{mid}.xml",
}
# 7 files bear "DFL-COM-000001", the other 14 "DFL-COM-000002"; same prefix as
# matchinformation for each mid (verified against the HF tree).
_IDSSE_COMP = {
    "J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001",
    "J03WOH": "DFL-COM-000002", "J03WOY": "DFL-COM-000002",
    "J03WPY": "DFL-COM-000002", "J03WQQ": "DFL-COM-000002",
    "J03WR9": "DFL-COM-000002",
}


def _idsse_filename(match_id: str, kind: str) -> str:
    comp = _IDSSE_COMP[match_id]
    return _IDSSE_KIND_SUFFIX[kind].format(comp=comp, mid=match_id)


class IDSSERetriever:
    """Fetch + cache the IDSSE/Sportec DFL open dataset (HF mirror)."""

    source = "idsse"

    def __init__(self, http: HTTPFileRetriever | None = None):
        self._http = http or HTTPFileRetriever()

    def list_resources(self) -> list[str]:
        return sorted(IDSSE_MATCH_IDS)

    def _match_dir(self, match_id: str) -> Path:
        from src.analytics.retrieval.base import cache_dir
        return cache_dir(self.source) / match_id

    def fetch(self, resource_id: str, kinds: tuple[str, ...] | None = None) -> Path:
        """Download + cache XML files for a match (default: all three kinds)."""
        if resource_id not in IDSSE_MATCH_IDS:
            raise RetrievalError(
                f"unknown idsse match '{resource_id}'; known: {IDSSE_MATCH_IDS}")
        kinds = kinds or ("matchinformation", "events", "tracking")
        match_dir = self._match_dir(resource_id)
        match_dir.mkdir(parents=True, exist_ok=True)
        base = f"https://huggingface.co/datasets/{IDSSE_HF_DATASET}/resolve/main"
        for kind in kinds:
            fname = _idsse_filename(resource_id, kind)
            dest = match_dir / fname
            if dest.exists():
                continue
            url = f"{base}/{fname}"
            try:
                cached = self._http.fetch(url)
            except RetrievalError:
                continue
            cached.rename(dest)
        return match_dir

    def file_path(self, match_id: str, kind: str) -> Path:
        fname = _idsse_filename(match_id, kind)
        p = self._match_dir(match_id) / fname
        return p


# ---------------------------------------------------------------------------
# Metrica Sports sample tracking + events (github metrica-sports/sample-data)
#
# Three anonymized sample matches published in the repo's `data/` tree.
# Sample Games 1 & 2 use the standard CSV layout (tracking at 25 Hz in
# normalized [0,1]x[0,1] coords, origin top-left, kick-off at (0.5, 0.5);
# a multi-row header) plus a per-match events CSV. Sample Game 3 uses the
# EPTS/FIFA + JSON format (recommended via kloppy) and is not consumed here.
# Redistribution-friendly open sample data (metricca permits acknowledged use).
# Each CSV lands under data/retrieval/metrica/<tag>/.
# ---------------------------------------------------------------------------
_METRICA_BASE = ("https://raw.githubusercontent.com/metrica-sports/"
                 "sample-data/master/data")

# tag -> the per-game files we pull (tracking home/away + events).
_METRICA_GAMES: dict[str, str] = {
    "sample_game_1": "Sample_Game_1",
    "sample_game_2": "Sample_Game_2",
}


class MetricaRetriever:
    """Download + cache the Metrica sample tracking/events CSVs."""

    source = "metrica"

    def __init__(self, http: HTTPFileRetriever | None = None):
        self._http = http or HTTPFileRetriever()

    def list_resources(self) -> list[str]:
        return sorted(_METRICA_GAMES)

    def _match_dir(self, match_id: str) -> Path:
        from src.analytics.retrieval.base import cache_dir
        return cache_dir(self.source) / match_id

    def fetch(self, match_id: str) -> Path:
        folder = _METRICA_GAMES.get(match_id)
        if folder is None:
            raise RetrievalError(
                f"unknown metrica match '{match_id}'; "
                f"known: {sorted(_METRICA_GAMES)}")
        match_dir = self._match_dir(match_id)
        match_dir.mkdir(parents=True, exist_ok=True)
        home = f"{folder}_RawTrackingData_Home_Team.csv"
        away = f"{folder}_RawTrackingData_Away_Team.csv"
        events = f"{folder}_RawEventsData.csv"
        files = {"tracking_home.csv": home,
                 "tracking_away.csv": away,
                 "events.csv": events}
        for local, remote in files.items():
            dest = match_dir / local
            if dest.exists():
                continue
            url = f"{_METRICA_BASE}/{folder}/{remote}"
            try:
                cached = self._http.fetch(url)
            except RetrievalError as exc:  # noqa: BLE001
                raise RetrievalError(
                    f"could not fetch metrica {remote}: {exc}") from exc
            cached.rename(dest)
        return match_dir

    def file_path(self, match_id: str, kind: str) -> Path:
        return self._match_dir(match_id) / f"{kind}"
