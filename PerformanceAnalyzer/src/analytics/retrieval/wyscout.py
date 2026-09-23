"""Wyscout open event data retriever (Pappalardo et al. 2019).

The original open dataset (CC BY 4.0) — 2017/18 top-5 European leagues + the
2018 World Cup and Euro 2016 — is published as a FigShare *collection*. Its
files have been mirrored in a stable, per-match form on GitHub
(`koenvo/wyscout-soccer-match-event-dataset`, `processed-v2/files/<match_id>.json`),
which is what kloppy's `load_wyscout_open_data` uses directly.

This retriever pulls individual match event files from that mirror into
`data/retrieval/wyscout/matches/<match_id>.json`, where a `WyscoutAdapter`
reads them with zero network. It also records the teams/players/match metadata
files from the FigShare collection when available.

Copyright note: Wyscout open data is CC BY 4.0 (Pappalardo et al., Sci Data 6,
2019). Local analysis use only; do not redistribute commercially.
"""

from __future__ import annotations

from pathlib import Path

from src.analytics.retrieval.base import RetrievalError
from src.analytics.retrieval.http import HTTPFileRetriever
from src.config import WYSCOUT_DATA_DIR

# Per-match JSON on the koenvo mirror (kloppy uses this exact base + /<id>.json)
_MIRROR = ("https://raw.githubusercontent.com/"
           "koenvo/wyscout-soccer-match-event-dataset/main/processed-v2/files")


class WyscoutRetriever:
    """Fetch + cache Wyscout open event files per match (from GitHub mirror)."""

    source = "wyscout"

    def __init__(self, http: HTTPFileRetriever | None = None,
                 data_dir: Path | str | None = None):
        self._http = http or HTTPFileRetriever()
        self.data_dir = Path(data_dir or WYSCOUT_DATA_DIR)
        self.matches_dir = self.data_dir / "matches"
        self.matches_dir.mkdir(parents=True, exist_ok=True)

    def list_resources(self) -> list[str]:
        return sorted(
            p.stem for p in self.matches_dir.glob("*.json") if p.is_file())

    def fetch(self, match_id: str) -> Path:
        """Download + cache one match's events from the koenvo mirror."""
        dest = self.matches_dir / f"{match_id}.json"
        if dest.exists():
            return dest
        url = f"{_MIRROR}/{match_id}.json"
        cached = self._http.fetch(url)
        cached.rename(dest)
        return dest