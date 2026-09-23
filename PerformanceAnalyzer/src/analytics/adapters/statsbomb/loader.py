"""StatsBomb open-data loader.

Fetches and caches match events, lineups and 360 freeze-frame data from the
StatsBomb open-data GitHub mirror. Data is cached under `STATSBOMB_DATA_DIR`
so a match is downloaded once and reused offline.

The raw payloads are returned verbatim (nested dicts) — interpretation and
conversion to our tracking schema live in `alignment.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import requests

from src.config import STATSBOMB_DATA_DIR, STATSBOMB_DATA_URL


@dataclass
class StatsBombMatch:
    """One match's raw open-data payloads."""
    match_id: int
    events: list[dict] = field(default_factory=list)
    lineups: list[dict] = field(default_factory=list)
    three_sixty: list[dict] = field(default_factory=list)


class StatsBombLoader:
    """Fetch + cache StatsBomb open data for a match id."""

    def __init__(self, data_dir: Path | str | None = None,
                 base_url: str = STATSBOMB_DATA_URL):
        self.data_dir = Path(data_dir or STATSBOMB_DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "performance-analyzer/1.0"

    # -- cache helpers -------------------------------------------------------
    def _cache_path(self, match_id: int, kind: str) -> Path:
        return self.data_dir / f"match_{match_id}_{kind}.json"

    def _load_cached(self, match_id: int, kind: str) -> list[dict] | None:
        path = self._cache_path(match_id, kind)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _save(self, match_id: int, kind: str, payload: list[dict]) -> None:
        self._cache_path(match_id, kind).write_text(json.dumps(payload))

    def _fetch(self, url: str, match_id: int, kind: str) -> list[dict]:
        cached = self._load_cached(match_id, kind)
        if cached is not None:
            return cached
        resp = self._session.get(url, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        self._save(match_id, kind, payload)
        return payload

    # -- public API ----------------------------------------------------------
    def events(self, match_id: int) -> list[dict]:
        return self._fetch(f"{self.base_url}/events/{match_id}.json",
                           match_id, "events")

    def matches(self, competition_id: int, season_id: int) -> list[dict]:
        """Match list for a competition/season (StatsBomb `matches/` index)."""
        return self._fetch(
            f"{self.base_url}/matches/{competition_id}/{season_id}.json",
            competition_id, f"season_{season_id}")

    def lineups(self, match_id: int) -> list[dict]:
        return self._fetch(f"{self.base_url}/lineups/{match_id}.json",
                           match_id, "lineups")

    def three_sixty(self, match_id: int) -> list[dict]:
        return self._fetch(f"{self.base_url}/three-sixty/{match_id}.json",
                           match_id, "360")

    def load_match(self, match_id: int,
                   with_360: bool = False) -> StatsBombMatch:
        """Fetch (or load cached) the full match payload set."""
        ev = self.events(match_id)
        lineups = self.lineups(match_id)
        ts = self.three_sixty(match_id) if with_360 else []
        return StatsBombMatch(match_id=match_id, events=ev,
                              lineups=lineups, three_sixty=ts)
