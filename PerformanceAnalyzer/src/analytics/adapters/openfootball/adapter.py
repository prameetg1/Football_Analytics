"""OpenFootball (football.json) adapter — match metadata + Goal events.

Reads the CC0 OpenFootball cache under
`data/retrieval/openfootball/<season>/<league>/` produced by
`OpenFootballRetriever`:

* `league.json` — full-season fixture list (date, round, home/away, score).
* `league.txt`  — the Football.TXT source annotating each match with goal
                  events (scorer, minute, `+` stoppage, `(p)` penalty and
                  `(og)` own-goal markers).

The dataset is *score- and scorer-level only*: team names and scorelines are
real, but there are no pass/shot coordinates and no player lineups. Accordingly
this adapter offers:

* `LineupsProvider.match()`  — teams + competition/season metadata (no roster).
* `EventsProvider`          — `Goal` events only (from the txt), with minute-
                              derived `timestamp_sec` and the canonical kick-off
                              point as a document position (the source has no
                              event coordinates).

This gives the resolver a genuinely open, non-scraped Event/Lineup source
beyond the scraping wall. Free, public-domain (CC0) data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.analytics.adapters.base import EventsProvider, LineupsProvider
from src.analytics.schema import Event, Match, PassEvent, ShotEvent, Team

# Default position for goal events (no coordinates in the dataset): canonical
# kick-off point, consistent with the "defend at x=0" fixed frame.
_DEFAULT_X, _DEFAULT_Y = 60.0, 35.0

# One comma-separated goal token inside the trailing (...):
#   `Name 37'`, `Name 90+4'`, `55'(p)`, `58'(og)`, bare `46'`
_GOAL_TOKEN = re.compile(
    r"(?P<name>[^0-9',(]+?)?"
    r"(?P<min>\d{1,3})(?:\+(?P<stop>\d{1,2}))?'"
    r"\s*(?:\((?P<mark>p|og)\))?",
    re.IGNORECASE)


class OpenFootballAdapter(LineupsProvider, EventsProvider):
    """Consumes one cached OpenFootball league-season as schema types."""

    source = "openfootball"
    CAPABILITIES = {"events", "lineups"}

    def __init__(self, season_dir: str | Path):
        self.season_dir = Path(season_dir)
        self._match: Match | None = None
        self._events: list[Event] | None = None
        self._rows: list[dict] | None = None

    # -- loading ------------------------------------------------------------
    def _json_path(self) -> Path:
        p = self.season_dir / "league.json"
        if not p.exists():
            raise FileNotFoundError(f"no league.json in {self.season_dir}")
        return p

    def _txt_path(self) -> Path:
        return self.season_dir / "league.txt"

    def _load(self) -> list[dict]:
        if self._rows is None:
            with open(self._json_path()) as fh:
                self._rows = json.load(fh).get("matches", [])
        return self._rows

    # -- LineupsProvider ----------------------------------------------------
    def match(self, match_id: int) -> Match:
        rows = self._load()
        idx = int(match_id)
        if idx < 0 or idx >= len(rows):
            raise KeyError(f"no row {idx} in {self._json_path()} "
                           f"({len(rows)} matches)")
        row = rows[idx]
        teams = [
            Team(team_id=_team_id(row.get("team1", "")),
                 name=row.get("team1", ""), side=0),
            Team(team_id=_team_id(row.get("team2", "")),
                 name=row.get("team2", ""), side=1),
        ]
        self._match = Match(
            match_id=idx,
            competition=self._competition(),
            season=self._season(),
            home_team=row.get("team1", ""),
            away_team=row.get("team2", ""),
            teams=teams,
            players=[],
        )
        return self._match

    def _competition(self) -> str:
        return "ENG-Premier League" if "en.1" in str(self.season_dir.name) \
            else str(self.season_dir.name)

    def _season(self) -> str:
        return self.season_dir.parent.name

    # -- EventsProvider (Goal-only) ----------------------------------------
    def events(self, match_id: int) -> list[Event]:
        rows = self._load()
        idx = int(match_id)
        if idx < 0 or idx >= len(rows):
            raise KeyError(f"no row {idx} in {self._json_path()}")
        goals = self._goals_for(idx, rows[idx])
        out: list[Event] = []
        for typ, scorer, minute, side in goals:
            ts = float(minute * 60)
            out.append(Event(
                type="Goal", timestamp_sec=ts,
                x=_DEFAULT_X, y=_DEFAULT_Y,
                team_side=side, period=1 if minute < 45 else 2,
                minute=minute, second=0,
                qualifiers={
                    "scorer": scorer,
                    "event_type": typ,          # goal / penalty / own_goal
                    "source": "openfootball",
                },
            ))
        out.sort(key=lambda e: e.timestamp_sec)
        self._events = out
        return out

    def _goals_for(self, idx: int, row: dict) -> list[tuple[str, str, int, int]]:
        txt = self._txt_path()
        goals: list[tuple[str, str, int, int]] = []
        if txt.exists():
            block = self._find_match_block(txt.read_text(), row)
            if block:
                goals = self._goals_from_block(block)
        # Fall back to the JSON scoreline when the txt has no scorer detail
        # (older Football.TXT format), giving a stable non-empty Goal stream.
        if not goals:
            goals = self._goals_from_score(row)
        return goals

    @staticmethod
    def _goals_from_score(row: dict) -> list[tuple[str, str, int, int]]:
        """Synthesize one Goal per FTP goal (scorer unknown, minute default)."""
        score = row.get("score")
        if isinstance(score, dict):
            ft = score.get("ft") or [0, 0]
        elif isinstance(score, list) and len(score) >= 2:
            ft = score
        else:
            ft = [0, 0]
        goals: list[tuple[str, str, int, int]] = []
        for side, n in enumerate(ft[:2]):
            for _ in range(int(n or 0)):
                goals.append(("goal", "", 45, side))
        return goals

    def _find_match_block(self, text: str, row: dict) -> str | None:
        """Isolate the team line + its immediate scoring-annotation lines.

        The txt layout is:
           19:00   Liverpool  4-2 (1-0)  Bournemouth
                   (Hugo EKITIKE 37', Cody GAKPO 49'; Antoine SEMENYO 64')
        We grab the team/score line (contains both names) and any directly
        following indented parentheses-only annotations from the *same* match
        (stop at the next team/score line).
        """
        team1 = self._alias(row.get("team1", ""))
        team2 = self._alias(row.get("team2", ""))
        lines = text.splitlines()
        for i, line in enumerate(lines):
            hay = line.lower()
            if team1 in hay and team2 in hay and _looks_like_match(line):
                block = [line]
                for j in range(i + 1, min(i + 4, len(lines))):
                    ln = lines[j]
                    if not ln.strip() or _looks_like_match(ln):
                        break
                    block.append(ln)
                return "\n".join(block)
        return None

    @staticmethod
    def _aliases(text: str) -> str:
        """Normalize raw txt: 'Liverpool FC', 'AFC Bournemouth', ..."""
        t = text.lower().replace("-", " ").replace("  ", " ").strip()
        for suffix in (" fc", " afc", " cf", " sc", " united fc",
                       " united", " town fc", " hotspur fc"):
            t = t.replace(suffix, "")
        t = t.replace("  ", " ").strip()
        return t

    @staticmethod
    def _alias(name: str) -> str:
        """Normalize a JSON long name to the txt's short form."""
        t = name.lower().replace("-", " ").replace("  ", " ").strip()
        for prefix in ("afc ", "fc ", "as ",):
            t = t.replace(prefix, "") if t.startswith(prefix) else t
        for suffix in (" fc", " afc", " cf", " sc",
                       " united fc", " town fc", " hotspur fc"):
            t = t.replace(suffix, "")
        t = t.replace("  ", " ").strip()
        return t

    @staticmethod
    def _norm(name: str) -> str:
        return name.lower().replace("-", " ").replace("  ", " ").strip()

    @staticmethod
    def _goals_from_block(block: str) -> list[tuple[str, str, int, int]]:
        """Parse scoring annotations like
        `(Hugo EKITIKE 37', Cody GAKPO 49'; Antoine SEMENYO 64', 76')`.

        Home goals before `;`, away after. Each token is `Name 37'` or a bare
        repeat-minute `76'` (belongs to the previous scorer). `(p)` and `(og)`
        mark penalties / own-goals. The block's first line (the team/score
        line with its `(1-0)` half-time score) is skipped: only the trailing
        annotation lines are parsed.
        """
        goals: list[tuple[str, str, int, int]] = []
        # The block's first line is the team/score line; everything after it
        # is the scoring annotation (which may span multiple lines).
        annotation = "\n".join(block.splitlines()[1:])
        if not annotation.strip():
            return goals
        m = re.search(r"\((.*?)\)", annotation, re.DOTALL)
        if not m:
            return goals
        halves = [h.strip() for h in m.group(1).split(";")]
        for side, half in enumerate(halves):
                if not half:
                    continue
                current: str | None = None
                for tok in half.split(","):
                    tok = tok.strip()
                    if not tok:
                        continue
                    tok = tok + "'" if tok[-1].isdigit() else tok
                    gm = _GOAL_TOKEN.search(tok)
                    if not gm:
                        continue
                    if gm.group("name"):
                        current = gm.group("name").strip()
                    if current is None:
                        continue  # drop bare minute with no prior scorer
                    minute = int(gm.group("min")) + int(gm.group("stop") or 0)
                    mark = (gm.group("mark") or "").lower()
                    typ = {"p": "penalty", "og": "own_goal"}.get(mark, "goal")
                    goals.append((typ, current, minute, side))
        return goals

    @staticmethod
    def _norm(name: str) -> str:
        return name.lower().replace("-", " ").replace("  ", " ").strip()

    def passes(self, match_id: int) -> list[PassEvent]:
        return []

    def shots(self, match_id: int) -> list[ShotEvent]:
        return []


def _looks_like_match(line: str) -> bool:
    """True when a txt line carries a scoreline (`4-2`, `0-0`, ...)."""
    return bool(re.search(r"\s\d+\s*-\s*\d+\b", line))


def _team_id(name: str) -> int:
    return (abs(hash(name)) % 1_000_000_000) if name else 0