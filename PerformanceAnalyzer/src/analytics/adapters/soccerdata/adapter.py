"""soccerdata schedule adapter (lineups + match-level events from cached tables).

Reads the JSON tables the `soccerdata` scrapers cache under
`data/retrieval/<source>/schedule.json`:

* `fbref`      — 380-match EPL schedule (home/away/score/game_id).
* `understat`  — the same with home_xg/away_xg and per-team codes.
* `espn`       — EPL schedule (game_id, home/away).
* `sofascore`  — EPL schedule (rounds, home/away scores).

These are *match metadata* sources — no player lineups, no tracking. The
adapter exposes `LineupsProvider` (`match()` with the two teams, no player
roster) and an `EventsProvider`. For `understat` a cached `shots.json` is
consumed too, giving a real shot-event stream (9k shots/session, normalized
coordinates, xG, outcome). Column layout mirrors what soccerdata (v1.9.x)
emits. Scraped tables are personal-use only (no redistribution licence).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.analytics.adapters.base import EventsProvider, LineupsProvider
from src.analytics.schema import Event, Match, Player, ShotEvent, Team

_SUPPORTED = ("fbref", "understat", "espn", "sofascore")

# understat shot `result` -> our ShotResult vocabulary.
_SHOT_RESULT = {
    "Goal": "goal", "Saved Shot": "saved", "Missed Shot": "off_target",
    "Blocked Shot": "blocked", "Shot On Post": "post",
}
_OWN_GOAL = "Own Goal"


def _int_team_id(name: str) -> int:
    """Stable small int id from a team name (no provider numeric id)."""
    return (abs(hash(name)) % 1_000_000_000) if name else 0


class SoccerDataAdapter(LineupsProvider, EventsProvider):
    """Consumes one cached soccerdata `schedule.json` table as schema types."""

    source = "soccerdata"

    def __init__(self, source: str, schedule_path: str | Path,
                 shots_path: str | Path | None = None,
                 season: str = "2020-21"):
        if source not in _SUPPORTED:
            raise ValueError(
                f"unsupported soccerdata schedule source '{source}'; "
                f"expected one of {_SUPPORTED}")
        self.source = source
        self.season = season
        self.schedule_path = Path(schedule_path)
        if shots_path is None and source == "understat":
            shots = Path(schedule_path).parent / "shots.json"
            shots_path = shots if shots.exists() else None
        self.shots_path = Path(shots_path) if shots_path else None
        self._rows: list[dict] | None = None
        self._game_index: dict[int, dict] | None = None
        self._shots: list[dict] | None = None

    # -- loading ------------------------------------------------------------
    def _load(self) -> list[dict]:
        if self._rows is None:
            with open(self.schedule_path) as fh:
                self._rows = json.load(fh)
        return self._rows

    def _by_game(self) -> dict[int, dict]:
        if self._game_index is None:
            rows = [r for r in self._load() if r.get("game_id") is not None]
            idx: dict[int, dict] = {}
            self._game_raw_id: dict[int, object] = {}
            for r in rows:
                kid = self._key(r["game_id"])
                idx[kid] = r
                self._game_raw_id[kid] = r["game_id"]
            self._game_index = idx
        return self._game_index

    @staticmethod
    def _key(raw) -> int:
        """Stable int key for a game_id (int, hex string, or hash fallback)."""
        if isinstance(raw, int):
            return raw
        try:
            return int(str(raw), 16)   # fbref uses hex ids like 'db261cb0'
        except ValueError:
            return abs(hash(str(raw))) % 2**32

    def list_matches(self) -> list[int]:
        return sorted(self._by_game())

    # -- LineupsProvider ----------------------------------------------------
    def match(self, match_id: int) -> Match:
        row = self._by_game().get(int(match_id))
        if row is None:
            raise KeyError(f"no schedule row for game_id {match_id} "
                           f"in {self.schedule_path}")
        home = row.get("home_team") or ""
        away = row.get("away_team") or ""
        # prefer numeric provider ids when the source has them (understat).
        home_id = row.get("home_team_id")
        away_id = row.get("away_team_id")
        return Match(
            match_id=int(match_id),
            competition="ENG-Premier League",
            season=self.season,
            home_team=home,
            away_team=away,
            teams=[
                Team(team_id=int(home_id) if home_id is not None
                     else _int_team_id(home), name=home, side=0),
                Team(team_id=int(away_id) if away_id is not None
                     else _int_team_id(away), name=away, side=1),
            ],
            players=[],
        )

    # -- EventsProvider -----------------------------------------------------
    def events(self, match_id: int) -> list[Event]:
        """Shots (if this is understat with `shots.json`) else scoreline goals."""
        own = self.shots(match_id)
        if own:
            return own
        # Fall back to synthesized scoreline goals (fbref/sofascore).
        row = self._by_game().get(int(match_id))
        if row is None:
            raise KeyError(f"no schedule row for game_id {match_id} "
                           f"in {self.schedule_path}")
        home_goals, away_goals = self._scoreline(row)
        out: list[Event] = []
        for _ in range(home_goals):
            out.append(Event(type="Goal", timestamp_sec=0.0,
                             x=0.0, y=0.0, team_side=0))
        for _ in range(away_goals):
            out.append(Event(type="Goal", timestamp_sec=0.0,
                             x=0.0, y=0.0, team_side=1))
        return out

    def passes(self, match_id: int):
        return []

    def shots(self, match_id: int) -> list[ShotEvent]:
        """Understat shot stream for a match (empty if no `shots.json`)."""
        if self.shots_path is None:
            return []
        if self._shots is None:
            with open(self.shots_path) as fh:
                self._shots = json.load(fh)
        kid = self._key(match_id)
        out: list[ShotEvent] = []
        for s in self._shots:
            if self._key(s.get("game_id")) != kid:
                continue
            result = _SHOT_RESULT.get(s.get("result"))
            side = self._team_side_of(kid, s.get("team_id"))
            if side is None:
                continue
            out.append(ShotEvent(
                type="Goal" if s.get("result") == _OWN_GOAL else "Shot",
                timestamp_sec=float(s.get("minute", 0)) * 60.0,
                x=self._norm_x(float(s.get("location_x", 0.5))),
                y=self._norm_y(float(s.get("location_y", 0.5))),
                team_side=side,
                player_id=self._intable(s.get("player_id")),
                period=1, minute=int(s.get("minute", 0)),
                result=result if result else "missed",
                xg=float(s.get("xg")) if s.get("xg") is not None else None,
                body_part=s.get("body_part") or "",
                shot_type=s.get("situation") or "",
                technique="", first_time=False,
            ))
        out.sort(key=lambda e: e.timestamp_sec)
        return out

    @staticmethod
    def _norm_x(x: float) -> float:
        """understat normalized [0,1] x -> canonical [0,120]."""
        return round(min(max(x, 0.0), 1.0) * 120.0, 2)

    @staticmethod
    def _norm_y(y: float) -> float:
        """understat normalized [0,1] y -> canonical [0,70]."""
        return round(min(max(y, 0.0), 1.0) * 70.0, 2)

    @staticmethod
    def _intable(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    def _team_side_of(self, match_id: int, provider_team_id) -> int | None:
        """Map a team's provider id (understat) to our 0/1 side index."""
        row = self._by_game().get(self._key(match_id))
        if row is None:
            return None
        pid = int(provider_team_id)
        if row.get("home_team_id") is not None and \
                int(row["home_team_id"]) == pid:
            return 0
        if row.get("away_team_id") is not None and \
                int(row["away_team_id"]) == pid:
            return 1
        return None

    @staticmethod
    def _scoreline(row: dict) -> tuple[int, int]:
        home = row.get("home_goals")
        away = row.get("away_goals")
        if home is None:
            home = row.get("home_score")   # sofascore
        if away is None:
            away = row.get("away_score")
        if home is not None and away is not None:
            return int(home), int(away)
        score = row.get("score") or ""   # fbref: "1–0"
        parts = score.replace("–", "-") if score else ""
        if "-" in str(parts):
            h, _, a = str(parts).partition("-")
            try:
                return int(h.strip()), int(a.strip())
            except ValueError:
                pass
        return 0, 0