"""Metrica Sports sample-data adapter (dense tracking + events + lineups).

Reads the CSV set that `MetricaRetriever` caches under
`data/retrieval/metrica/<game>/`:

* `tracking_home.csv` / `tracking_away.csv` — optical tracking at 25 Hz for
  every outfield player + the ball, positions normalized to [0,1]x[0,1] with
  the origin at the top-left and kick-off at (0.5, 0.5).
* `events.csv` — synchronized event stream (passes, shots, duels, recoveries,
  restarts, cards, ...).

Both tracking files begin with 3 header rows (team-label row, player-number
row, then `Period,Frame,Time [s],PlayerN,,...,Ball,`); after those the frame
counter and `Time [s]` run continuously from the kick-off across both halves,
so row `i` in the home file aligns with row `i` in the away file. We merge the
two files on that alignment and emit one canonical `Frame` per sample.

Metrica uses a top-left origin, so we flip y (`1 - y`) before rescaling and
clamp to the canonical frame x in [0,120], y in [0,70]. Because the dataset is
anonymized, team names are synthetic and `player_id`s are the shirt numbers
with the side derived from which tracking file the player came from.

Data: Metrica Sports sample data (github.com/metrica-sports/sample-data),
three anonymized sample EPL matches, redistribution-friendly for research.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.analytics.adapters.base import (
    DenseTracking,
    EventsProvider,
    LineupsProvider,
)
from src.analytics.adapters.metrica.coordinates import MetricaPitch
from src.analytics.schema import (
    Event,
    Frame,
    Match,
    PassEvent,
    Player,
    ShotEvent,
    Team,
)

_CANONICAL_LENGTH = 120.0
_CANONICAL_WIDTH = 70.0
_HZ = 25.0            # Metrica nominal sample rate

# Metrica event Type -> our EVENT_TYPES / schema kind.
_SET_PIECE_KIND = {
    "KICK OFF": "Half Start",
    "THROW IN": "Throw-in",
    "CORNER KICK": "Corner",
    "FREE KICK": "Goal Kick",          # defensive restart; kept generic under it
}
_TYPE_TO_KIND = {
    "RECOVERY": "Ball Recovery",
    "BALL LOST": "Dispossessed",
    "CHALLENGE": "Duel",
    "FAULT RECEIVED": "Foul Won",
    "CARD": "Foul Committed",
}

# Metrica SHOT <Subtype> -> our ShotResult vocabulary.
_SHOT_RESULT = {
    "goal": "goal",
    "ON TARGET-GOAL": "goal",
    "HEAD-ON TARGET-GOAL": "goal",
    "saved": "saved",
    "ON TARGET-SAVED": "saved",
    "HEAD-ON TARGET-SAVED": "saved",
    "off_target": "off_target",
    "OFF TARGET": "off_target",
    "OFF TARGET-OUT": "off_target",
    "HEAD-OFF TARGET-OUT": "off_target",
    "post": "post",
    "HEAD-WOODWORK-OUT": "post",
    "blocked": "blocked",
    "BLOCKED": "blocked",
}


class MetricaAdapter(EventsProvider, DenseTracking, LineupsProvider):
    """Normalizes one cached Metrica sample-match directory into schema types."""

    source = "metrica"
    CAPABILITIES = {"events", "dense_tracking", "lineups"}

    def __init__(self, match_dir: str | Path, sample_every: int = 5):
        self.match_dir = Path(match_dir)
        self.sample_every = max(int(sample_every), 1)
        self._match: Match | None = None
        self._events: list[Event] | None = None
        self._frames: list[Frame] | None = None
        self._pitch = MetricaPitch()

    # -- file discovery -----------------------------------------------------
    def _path(self, name: str) -> Path:
        p = self.match_dir / name
        if not p.exists():
            raise FileNotFoundError(f"no '{name}' in {self.match_dir}")
        return p

    # -- LineupsProvider ----------------------------------------------------
    def match(self, match_id: int) -> Match:
        if self._match is not None:
            return self._match
        teams: list[Team] = []
        players: list[Player] = []
        for side, label in ((0, "Home"), (1, "Away")):
            teams.append(Team(team_id=label, name=label, side=side))
            # Anonymized: reconstruct shirt numbers from the tracking header.
            fname = "tracking_home.csv" if side == 0 else "tracking_away.csv"
            for idx, num in enumerate(self._player_numbers(self._path(fname))):
                players.append(Player(
                    player_id=int(num),
                    name=f"{label} #{num}",
                    position=self._keeper_position(num, side),
                    shirt_number=int(num)))
        self._match = Match(
            match_id=int(match_id),
            competition="EPL (Metrica sample)",
            season="sample",
            home_team="Home", away_team="Away",
            teams=teams, players=players,
        )
        self._player_num_by_id = {p.player_id: p.shirt_number
                                  for p in players}
        return self._match

    def _keeper_position(self, number: int, side: int) -> str:
        # Home goalkeepers wear 1 in the sample; away wears 21.
        return "Goalkeeper" if (side == 0 and number == 1) or (
            side == 1 and number == 21) else ""

    def _player_numbers(self, path: Path) -> list[int]:
        """Shirt numbers of outfielders from the 2nd header row (skip Ball)."""
        nums: list[int] = []
        with open(path) as fh:
            reader = csv.reader(fh)
            next(reader)  # team-label row
            number_row = next(reader)
            header_rows = list(next(reader))  # player-name row
        for i, name in enumerate(header_rows[3:]):
            if name.startswith("Player") and i % 2 == 0:
                num = number_row[3 + i]
                try:
                    nums.append(int(num))
                except (TypeError, ValueError):
                    pass
        return nums

    # -- EventsProvider -----------------------------------------------------
    def events(self, match_id: int) -> list[Event]:
        if self._events is not None:
            return self._events
        path = self._path("events.csv")
        out: list[Event] = []
        with open(path) as fh:
            for row in csv.DictReader(fh):
                etype = (row.get("Type") or "").strip().upper()
                subtype = (row.get("Subtype") or "").strip().upper()
                period = int(row.get("Period", 0))
                start_s = float(row.get("Start Time [s]") or 0.0)
                pos = self._pitch.to_finite(
                    self._as_float(row.get("Start X")),
                    self._as_float(row.get("Start Y")))
                cx = cy = 0.0
                if pos is not None:
                    cx, cy = pos
                side = 0 if (row.get("Team") or "").strip() == "Home" else 1
                pid = self._player_id_of(side, row.get("From"))
                minute = int(start_s // 60)
                second = int(start_s % 60)

                if etype == "PASS":
                    end = self._pitch.to_finite(
                        self._as_float(row.get("End X")),
                        self._as_float(row.get("End Y")))
                    ex, ey = (end if end is not None else (cx, cy))
                    outcome = "complete" if self._pass_accurate(subtype) \
                        else "incomplete"
                    out.append(PassEvent(
                        type="Pass", timestamp_sec=start_s,
                        x=cx, y=cy, team_side=side, player_id=pid,
                        period=period, minute=minute, second=second,
                        end_x=ex, end_y=ey, outcome=outcome,
                        receiver_id=self._player_id_of(side, row.get("To")),
                    ))
                elif etype == "SHOT":
                    result = _SHOT_RESULT.get(
                        subtype, _SHOT_RESULT.get("saved", "saved"))
                    out.append(ShotEvent(
                        type="Shot", timestamp_sec=start_s,
                        x=cx, y=cy, team_side=side, player_id=pid,
                        period=period, minute=minute, second=second,
                        result=result,
                    ))
                elif etype == "SET PIECE":
                    kind = _SET_PIECE_KIND.get(subtype, "Goal Kick")
                    out.append(Event(
                        type=kind, timestamp_sec=start_s, x=cx, y=cy,
                        team_side=side, player_id=pid, period=period,
                        minute=minute, second=second,
                        play_pattern=subtype.lower(),
                    ))
                else:
                    kind = _TYPE_TO_KIND.get(etype)
                    if kind is None:
                        continue
                    out.append(Event(
                        type=kind, timestamp_sec=start_s, x=cx, y=cy,
                        team_side=side, player_id=pid, period=period,
                        minute=minute, second=second,
                        play_pattern=subtype.lower() or etype.lower(),
                    ))
        out.sort(key=lambda e: (e.timestamp_sec, str(type(e).__name__)))
        self._events = out
        return out

    @staticmethod
    def _as_float(value) -> float:
        """Parse a CSV cell to float; empty/'NaN'/'None' -> NaN so callers can
        drop it via `to_finite`."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    @staticmethod
    def _pass_accurate(subtype: str) -> bool:
        return not subtype or "UNSUCCESSFUL" not in subtype.upper()

    def _player_id_of(self, side: int, token: str | None) -> int | None:
        """Events reference players as 'Player<N>'; map to our shirt-number id."""
        if not token or not token.startswith("Player"):
            return None
        t = token[len("Player"):]
        try:
            return int(t)
        except ValueError:
            return None

    # -- EventsProvider helpers --------------------------------------------
    def passes(self, match_id: int) -> list[PassEvent]:
        return [e for e in self.events(match_id) if isinstance(e, PassEvent)]

    def shots(self, match_id: int) -> list[ShotEvent]:
        return [e for e in self.events(match_id) if isinstance(e, ShotEvent)]

    # -- DenseTracking ------------------------------------------------------
    def tracking(self, match_id: int,
                 sample_every: int | None = None) -> list[Frame]:
        """Merge home + away tracking CSV rows into per-sample `Frame`s.

        Both tracking files advance in lockstep from the kick-off (25 Hz,
        continuous across halves), so row `i` of the home file aligns with
        row `i` of the away file. We sample every `sample_every`-th row
        (default 5 -> 5 Hz) to bound memory, mirroring IDSSE.
        """
        if self._frames is not None:
            return self._frames
        step = sample_every or self.sample_every
        self.match(match_id)
        home_path = self._path("tracking_home.csv")
        away_path = self._path("tracking_away.csv")
        home_players = self._tracking_spec(home_path, side=0)
        away_players = self._tracking_spec(away_path, side=1)

        frames: list[Frame] = []
        with open(home_path) as hh, open(away_path) as ah:
            hreader = csv.reader(hh)
            areader = csv.reader(ah)
            for _ in range(3):
                next(hreader)
                next(areader)
            for row_i, (hrow, arow) in enumerate(zip(hreader, areader)):
                if row_i % step != 0:
                    continue
                ts = round(float(hrow[2]), 4)
                players = self._collect_players(hrow, home_players, 0) + \
                    self._collect_players(arow, away_players, 1)
                ball = self._ball(hrow)
                if ball is None:
                    ball = self._ball(arow)
                frames.append(Frame(timestamp_sec=ts,
                                    players=players, ball=ball))
        self._frames = frames
        return frames

    @staticmethod
    def _tracking_spec(path: Path, side: int) -> list[tuple[int, int]]:
        """Return (csv_x_idx, player_id) pairs for the shirt-numbered players.

        After the 3 header rows each player occupies two adjacent columns
        (x then y); the header names the x-column `PlayerN` and the y-column
        (and the two ball columns) are blank, so we detect players by the
        `PlayerN` name in the header.
        """
        with open(path) as fh:
            reader = csv.reader(fh)
            next(reader)  # team label
            number_row = next(reader)  # shirt numbers
            name_row = next(reader)     # 'PlayerN' names
        spec: list[tuple[int, int]] = []
        for i, name in enumerate(name_row[3:]):
            if name.startswith("Player"):
                num = number_row[3 + i]
                try:
                    pid = int(num)
                except (TypeError, ValueError):
                    pid = int(name[len("Player"):])
                spec.append((3 + i, pid))
        # drop the trailing (Ball, ...) pair if included by name
        return [(i, pid) for i, pid in spec]

    def _collect_players(self, row: list[str],
                         spec: list[tuple[int, int]], side: int) -> list:
        out = []
        for x_idx, pid in spec:
            try:
                x = float(row[x_idx])
                y = float(row[x_idx + 1])
            except (TypeError, ValueError, IndexError):
                continue
            pos = self._pitch.to_finite(x, y)
            if pos is None:
                continue  # out-of-camera (NaN) player — not visible this frame
            cx, cy = pos
            out.append((pid, side, cx, cy, self._is_keeper(pid, side)))
        return out

    def _ball(self, row: list[str]) -> tuple[float, float] | None:
        # Ball is the last two columns of the track row.
        try:
            x = float(row[-2])
            y = float(row[-1])
        except (TypeError, ValueError, IndexError):
            return None
        return self._pitch.to_finite(x, y)

    def _is_keeper(self, pid: int, side: int) -> bool:
        return self._keeper_position(pid, side) == "Goalkeeper"
