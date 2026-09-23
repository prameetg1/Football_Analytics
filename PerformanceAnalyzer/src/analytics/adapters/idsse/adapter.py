"""IDSSE / Sportec DFL adapter (dense 25 Hz tracking + events + lineups).

Reads the raw DFL XML files that `IDSSERetriever` caches under
`data/retrieval/idsse/<match_id>/`:

* `<...>_matchinformation_....xml`   — teams, players, kickoff, pitch size.
* `<...>_events_raw_....xml`         — synchronized event stream (passes, shots,
                                       duels, restarts, ...).
* `<...>_positions_raw_observed_....xml` — TRACAB 25 Hz optical tracking for every
                                      player + ball + referees.

Each `<FrameSet>` block holds one person (player / referee / BALL) for one half
and contains `<Frame N T X Y D S A M/>` samples. `N` resets per half
(first half 10000.., second half 100000..), `T` is the absolute ISO timestamp.

Positions are metres on a DFL pitch (origin at the corner of the home
defending half, x in [0,105], y in [0,68]). We rescale to the shared canonical
frame x in [0,120], y in [0,70] keeping the same orientation for both teams
(the DFL export is already a fixed-frame broadcast system), so both teams
attack towards their own canonical goal — matching the resolver convention.

`tracking()` returns `Frame` objects with full per-player identity; because a
full match is ~2M position samples the default keeps every 5th frame (5 Hz) via
`sample_every`. Pass `sample_every=1` for the raw 25 Hz stream.

Data: Bassek et al. (2025), Sci Data 12, 195, CC-BY 4.0, distributed with
DFL/Sportec authorization via the pysport Hugging Face mirror.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from src.analytics.adapters.base import (
    DenseTracking,
    EventsProvider,
    LineupsProvider,
)
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
_HZ = 25.0  # TRACAB nominal sample rate

# DFL event child tags -> our EVENT_TYPES vocabulary.
_TAG_TO_KIND = {
    "Pass": "Pass",
    "ShotAtGoal": "Shot",
    "TacklingGame": "Duel",
    "Foul": "Foul Committed",
    "FreeKick": "Foul Won",
    "ThrowIn": "Throw-in",
    "GoalKick": "Goal Kick",
    "Corner": "Corner",
    "Substitution": "Substitution",
    "KickOff": "Half Start",
    "OtherBallAction": "Ball Recovery",
}

_GOALKEEPER_DPL = {"TW"}


class IDSSEAdapter(DenseTracking, EventsProvider, LineupsProvider):
    """Normalizes one cached IDSSE/DFL match directory into schema types."""

    source = "idsse"
    CAPABILITIES = {"events", "lineups", "dense_tracking"}

    def __init__(self, match_dir: str | Path, sample_every: int = 5):
        self.match_dir = Path(match_dir)
        self.sample_every = max(int(sample_every), 1)
        self._match: Match | None = None
        self._events: list[Event] | None = None
        self._frames: list[Frame] | None = None
        self._kickoff: datetime | None = None

    # -- file discovery -----------------------------------------------------
    def _find(self, substring: str) -> Path:
        matches = list(self.match_dir.glob(f"*{substring}*.xml"))
        if not matches:
            raise FileNotFoundError(f"no '{substring}' xml in {self.match_dir}")
        return matches[0]

    def _matchinfo_path(self) -> Path:
        return self._find("_matchinformation_")

    def _events_path(self) -> Path:
        return self._find("_events_raw_")

    def _tracking_path(self) -> Path:
        return self._find("_positions_raw_observed_")

    # -- time helpers -------------------------------------------------------
    @staticmethod
    def _parse_iso(ts: str) -> datetime:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _kickoff_time(self) -> datetime:
        if self._kickoff is None:
            root = ET.parse(self._matchinfo_path()).getroot()
            gen = root.find(".//General")
            kick = gen.get("KickoffTime") if gen is not None else None
            self._kickoff = self._parse_iso(kick) if kick else None
        return self._kickoff

    def _seconds(self, ts: str) -> float:
        kick = self._kickoff_time()
        return round(max((self._parse_iso(ts) - kick).total_seconds(), 0.0), 4)

    # -- coordinate mapping -------------------------------------------------
    @staticmethod
    def _to_canonical(x: float, y: float) -> tuple[float, float]:
        cx = min(max(x / 105.0 * _CANONICAL_LENGTH, 0.0), _CANONICAL_LENGTH)
        cy = min(max(y / 68.0 * _CANONICAL_WIDTH, 0.0), _CANONICAL_WIDTH)
        return round(cx, 4), round(cy, 4)

    # -- LineupsProvider ----------------------------------------------------
    def match(self, match_id: int) -> Match:
        if self._match is not None:
            return self._match
        root = ET.parse(self._matchinfo_path()).getroot()
        gen = root.find(".//General")
        env = root.find(".//Environment")
        kick = gen.get("KickoffTime") if gen is not None else None
        self._kickoff = self._parse_iso(kick) if kick else None
        home_id = gen.get("HomeTeamId") if gen is not None else None
        guest_id = gen.get("GuestTeamId") if gen is not None else None
        home_name = gen.get("HomeTeamName") if gen is not None else ""
        guest_name = gen.get("GuestTeamName") if gen is not None else ""
        competition = gen.get("CompetitionName", "") if gen is not None else ""
        season = gen.get("Season", "") if gen is not None else ""

        teams: list[Team] = []
        players: list[Player] = []
        for team_el in root.findall(".//Team"):
            tid = team_el.get("TeamId")
            if tid not in (home_id, guest_id):
                continue
            side = 0 if tid == home_id else 1
            teams.append(Team(team_id=tid,
                              name=team_el.get("TeamName", ""), side=side))
            for p in team_el.findall(".//Player"):
                players.append(Player(
                    player_id=p.get("PersonId"),
                    name=f"{p.get('FirstName', '')} {p.get('LastName', '')}".strip(),
                    position=p.get("PlayingPosition", ""),
                    shirt_number=int(p.get("ShirtNumber"))
                    if p.get("ShirtNumber") else None,
                ))
        self._match = Match(
            match_id=int(match_id),
            competition=competition,
            season=season,
            home_team=home_name, away_team=guest_name,
            teams=teams, players=players,
        )
        self._home_id = home_id
        return self._match

    # -- EventsProvider -----------------------------------------------------
    def events(self, match_id: int) -> list[Event]:
        if self._events is not None:
            return self._events
        self.match(match_id)
        root = ET.parse(self._events_path()).getroot()

        # First-level child element names of an Event (in DFL export order).
        _SHOT_RESULT = {"SuccessfulShot": "goal", "SavedShot": "saved",
                        "BlockedShot": "blocked", "ShotWide": "off_target",
                        "ShotWoodWork": "post"}
        out: list[Event] = []
        for ev in root.iter("Event"):
            ts = self._seconds(ev.get("EventTime", ""))
            cx, cy = self._to_canonical(
                float(ev.get("X-Position", 0.0)), float(ev.get("Y-Position", 0.0)))
            period = self._period(ts)
            children = list(ev)

            play = next((c for c in children if c.tag == "Play"), None)
            shot = next(ev.iter("ShotAtGoal"), None)
            shot_result = None
            if shot is not None:
                for c in list(shot):
                    if c.tag in _SHOT_RESULT:
                        shot_result = c.tag
                        break

            if play is not None and any(c.tag in ("Pass", "Cross")
                                        for c in list(play)):
                kind = "Pass"
                pid = play.get("Player")
                side = self._side_for(play.get("Team"))
                outcome = ("complete" if play.get("Evaluation")
                           == "successfullyCompleted" else "incomplete")
                out.append(PassEvent(
                    type=kind, timestamp_sec=ts, x=cx, y=cy,
                    team_side=side, player_id=pid, period=period,
                    end_x=None, end_y=None, outcome=outcome,
                    receiver_id=play.get("Recipient"),
                    pass_type="", body_part="",
                    height=play.get("Height", ""), length=None,
                    angle=self._float_or_none(play.get("PlayAngle")),
                    is_cross=any(c.tag == "Cross" for c in list(play))))
            elif shot is not None:
                kind = "Shot"
                pid = shot.get("Player")
                side = self._side_for(shot.get("Team"))
                out.append(ShotEvent(
                    type=kind, timestamp_sec=ts, x=cx, y=cy,
                    team_side=side, player_id=pid, period=period,
                    result=_SHOT_RESULT.get(shot_result, "saved"),
                    xg=float(shot.get("xG")) if shot.get("xG") else None,
                    body_part="", shot_type=shot.get("TypeOfShot", ""),
                    technique="", first_time=False))
            else:
                for child in children:
                    tag = child.tag
                    if tag not in _TAG_TO_KIND:
                        continue
                    kind = _TAG_TO_KIND[tag]
                    pid = child.get("Player")
                    side = self._side_for(child.get("Team"))
                    out.append(Event(type=kind, timestamp_sec=ts, x=cx, y=cy,
                                     team_side=side, player_id=pid,
                                     period=period))
                    break
        out.sort(key=lambda e: (e.timestamp_sec, str(type(e).__name__)))
        self._events = out
        return out

    def passes(self, match_id: int) -> list[PassEvent]:
        return [e for e in self.events(match_id) if isinstance(e, PassEvent)]

    def shots(self, match_id: int) -> list[ShotEvent]:
        return [e for e in self.events(match_id) if isinstance(e, ShotEvent)]

    def _side_for(self, team_id: str) -> int | None:
        for t in self.match(0).teams:
            if t.team_id == team_id:
                return t.side
        return None

    @staticmethod
    def _float_or_none(value):
        try:
            return float(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _period(ts: float) -> int:
        return 1 if ts < 2700.0 else 2

    # -- DenseTracking ------------------------------------------------------
    def tracking(self, match_id: int,
                 sample_every: int | None = None) -> list[Frame]:
        """Stream the raw tracking XML into `Frame` position samples.

        N resets per half (first half 10000.., second half 100000..), and every
        person's `FrameSet` advances in lockstep (same kickoff, +0.04 s per N),
        so we bucket persons by (half, N) merged into one `Frame` each.
        """
        if self._frames is not None:
            return self._frames
        self.match(match_id)
        step = sample_every or self.sample_every
        side_of_team = {t.team_id: t.side for t in self._match.teams}

        # (half_base_label) -> (N_base, datetime) discovered from first frame
        base: dict[str, tuple[int, datetime]] = {}
        # key (half, N//step) -> (timestamp_sec, {person: tuple}, ball)
        buckets: dict[tuple[str, int], tuple[float, dict, tuple | None]] = {}
        stack: list[dict] = []

        for event, elem in ET.iterparse(self._tracking_path(),
                                        events=("start", "end")):
            tag = elem.tag.split("}")[-1]
            if event == "start":
                if tag == "FrameSet":
                    stack.append(dict(elem.attrib))
                elif tag == "Frame" and stack:
                    fs = stack[-1]
                    team_id = fs.get("TeamId")
                    half = fs.get("GameSection") or "firstHalf"
                    bh = half  # bucket half label
                    n = int(elem.get("N"))
                    key = n // step
                    bucket = buckets.get((bh, key))
                    if bucket is None:
                        n_base, t_base = base.get(bh, (n, self._parse_iso(
                            elem.get("T", ""))))
                        base.setdefault(bh, (n_base, t_base))
                        ts = t_base.timestamp() + (n - n_base) / _HZ
                        ts -= self._kickoff_time().timestamp()
                        ts = round(max(ts, 0.0), 4)
                        bucket = (ts, {}, None)
                        buckets[(bh, key)] = bucket
                    x = float(elem.get("X", 0.0))
                    y = float(elem.get("Y", 0.0))
                    cx, cy = self._to_canonical(x, y)
                    if team_id == "BALL":
                        buckets[(bh, key)] = (bucket[0], bucket[1], (cx, cy))
                    else:
                        person = fs.get("PersonId")
                        side = side_of_team.get(team_id)
                        if side is None or person is None:
                            continue
                        bucket[1][person] = (person, side, cx, cy,
                                             self._is_keeper(person))
            else:  # end
                elem.clear()
                if tag == "FrameSet" and stack:
                    stack.pop()

        frames = [Frame(timestamp_sec=ts, players=list(persons.values()),
                        ball=ball)
                  for (_, key), (ts, persons, ball) in
                  sorted(buckets.items())]
        self._frames = frames
        return frames

    def _is_keeper(self, person_id: str) -> bool:
        for p in self._match.players:
            if p.player_id == person_id and p.position in _GOALKEEPER_DPL:
                return True
        return False