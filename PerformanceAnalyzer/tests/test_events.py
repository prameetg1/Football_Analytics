"""Stage 9, Stage A: event extraction — passes, recoveries, shots.

Synthetic tracking frames with a known ball path + player ownership prove the
extractor emits the right events at the right (timestamp, x, y), and that the
Stage-9 gate matches CV events against StatsBomb-shaped GT.
"""

import numpy as np
import pandas as pd
import pytest

from src.analytics.adapters.video.events import Event, extract_events
from src.analytics.adapters.video.metrics.base import TrackingFrame
from src.analytics.adapters.statsbomb.validate import (
    ValidationReport,
    event_match,
    validate_events_against_gt,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=TrackingFrame.REQUIRED_COLUMNS)


def _ball_chain(times: list[float], xs: list[float],
                ys: list[float] | None = None) -> list[dict]:
    """Ball-only samples (no owner) — drives shot detection."""
    ys = ys or [35.0] * len(times)
    return [{"timestamp_sec": t, "x": x, "y": y}
            for t, x, y in zip(times, xs, ys)]


def _chain_df(chain: list[dict]) -> pd.DataFrame:
    """Tracking frame from a ball chain with a player following the ball.

    The dummy player stays within `loose_m` of the ball so the ownership chain
    (which shot detection reads) stays populated even while the ball is flying.
    """
    rows = []
    for i, s in enumerate(chain):
        rows.append({"frame_idx": i, "timestamp_sec": s["timestamp_sec"],
                     "track_id": "ball", "team_id": np.nan,
                     "pitch_x": s["x"], "pitch_y": s["y"],
                     "class_name": "ball"})
        rows.append({"frame_idx": i, "timestamp_sec": s["timestamp_sec"],
                     "track_id": "p0", "team_id": 0,
                     "pitch_x": s["x"] - 2.0, "pitch_y": s["y"],
                     "class_name": "player"})
    return _frame(rows)


class TestOwnershipChain:
    def test_single_frame_no_events(self):
        rows = [
            {"frame_idx": 0, "timestamp_sec": 0.0, "track_id": "ball",
             "team_id": np.nan, "pitch_x": 10.0, "pitch_y": 35.0,
             "class_name": "ball"},
            {"frame_idx": 0, "timestamp_sec": 0.0, "track_id": "p1",
             "team_id": 0, "pitch_x": 10.5, "pitch_y": 35.0,
             "class_name": "player"},
            {"frame_idx": 0, "timestamp_sec": 0.0, "track_id": "p2",
             "team_id": 1, "pitch_x": 40.0, "pitch_y": 35.0,
             "class_name": "player"},
        ]
        assert extract_events(_frame(rows)) == []

    def test_loose_ball_skipped(self):
        rows = [
            {"frame_idx": 0, "timestamp_sec": 0.0, "track_id": "ball",
             "team_id": np.nan, "pitch_x": 100.0, "pitch_y": 35.0,
             "class_name": "ball"},
            {"frame_idx": 0, "timestamp_sec": 0.0, "track_id": "p1",
             "team_id": 0, "pitch_x": 10.0, "pitch_y": 35.0,
             "class_name": "player"},
        ]
        assert extract_events(_frame(rows)) == []


def _pass_frame():
    """Team-0 pass from p1 to p2 at t=1.0, ball moving 20m in between."""
    rows = []
    for i, (t, ballx, p1x, p2x) in enumerate([
        (0.0, 10.0, 10.0, 40.0),    # p1 has the ball
        (0.5, 20.0, 10.5, 40.0),    # ball in flight, nearest owner stays p1
        (1.0, 30.0, 10.8, 30.0),    # p2 receives at x=30
        (1.5, 31.0, 11.0, 30.0),    # p2 retains
    ]):
        rows.append({"frame_idx": i, "timestamp_sec": t, "track_id": "ball",
                     "team_id": np.nan, "pitch_x": ballx, "pitch_y": 35.0,
                     "class_name": "ball"})
        rows.append({"frame_idx": i, "timestamp_sec": t, "track_id": "p1",
                     "team_id": 0, "pitch_x": p1x, "pitch_y": 35.0,
                     "class_name": "player"})
        rows.append({"frame_idx": i, "timestamp_sec": t, "track_id": "p2",
                     "team_id": 0, "pitch_x": p2x, "pitch_y": 35.0,
                     "class_name": "player"})
    return _frame(rows)


class TestPassDetection:
    def test_same_team_owner_change_is_pass(self):
        events = extract_events(_pass_frame())
        kinds = [e.kind for e in events]
        assert kinds == ["Pass"]
        p = events[0]
        assert p.timestamp_sec == pytest.approx(1.0)
        assert p.x == pytest.approx(30.0)
        assert p.team_id == 0

    def test_same_team_but_same_player_no_pass(self):
        rows = []
        for i, (t, ballx) in enumerate([(0.0, 10.0), (1.0, 10.2)]):
            rows.append({"frame_idx": i, "timestamp_sec": t,
                         "track_id": "ball", "team_id": np.nan,
                         "pitch_x": ballx, "pitch_y": 35.0,
                         "class_name": "ball"})
            rows.append({"frame_idx": i, "timestamp_sec": t,
                         "track_id": "p1", "team_id": 0,
                         "pitch_x": ballx, "pitch_y": 35.0,
                         "class_name": "player"})
        assert extract_events(_frame(rows)) == []

    def test_cross_team_change_is_turnover(self):
        rows = []
        for i, (t, ballx, owner, team) in enumerate([
            (0.0, 10.0, "p1", 0), (0.5, 15.0, "p1", 0),
            (1.0, 20.0, "q1", 1), (1.5, 21.0, "q1", 1),
        ]):
            rows.append({"frame_idx": i, "timestamp_sec": t,
                         "track_id": "ball", "team_id": np.nan,
                         "pitch_x": ballx, "pitch_y": 35.0,
                         "class_name": "ball"})
            rows.append({"frame_idx": i, "timestamp_sec": t,
                         "track_id": owner, "team_id": team,
                         "pitch_x": ballx, "pitch_y": 35.0,
                         "class_name": "player"})
        events = extract_events(_frame(rows), kinds=("Ball Recovery",))
        assert [e.kind for e in events] == ["Ball Recovery"]
        e = events[0]
        assert e.team_id == 1
        assert e.qualifiers["losing_team"] == 0


class TestShotDetection:
    def test_ball_into_mouth_at_speed_is_shot(self):
        chain = _ball_chain(
            times=[0.0, 0.1, 0.2, 0.3],
            xs=[90.0, 100.0, 110.0, 118.0],   # straight at the x=120 goal
        )
        events = extract_events(_chain_df(chain), kinds=("Shot",))
        assert [e.kind for e in events] == ["Shot"]
        s = events[0]
        assert s.timestamp_sec == pytest.approx(0.3)
        assert s.x == pytest.approx(118.0)
        assert abs(s.y - 35.0) <= 3.66

    def test_slow_ball_not_a_shot(self):
        chain = _ball_chain(
            times=[0.0, 2.0, 4.0, 6.0],
            xs=[90.0, 100.0, 110.0, 118.0],   # 5 m/s, under the speed gate
        )
        assert extract_events(_chain_df(chain), kinds=("Shot",)) == []

    def test_one_shot_not_two_while_in_mouth(self):
        chain = _ball_chain(
            times=[0.0, 0.1, 0.2, 0.3, 0.4],
            xs=[90.0, 100.0, 110.0, 118.0, 119.0],
        )
        events = extract_events(_chain_df(chain), kinds=("Shot",))
        assert [e.kind for e in events] == ["Shot"]

    def test_shot_reams_after_leaving_mouth(self):
        chain = _ball_chain(
            times=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            xs=[90.0, 100.0, 118.0, 100.0, 110.0, 118.0],
        )
        shots = extract_events(_chain_df(chain), kinds=("Shot",))
        assert len(shots) == 2

    def test_shot_outside_goal_width_not_counted(self):
        chain = _ball_chain(
            times=[0.0, 0.1, 0.2, 0.3],
            xs=[90.0, 100.0, 110.0, 118.0],
            ys=[35.0, 35.0, 35.0, 20.0],        # wide, y=20 far from goal
        )
        assert extract_events(_chain_df(chain), kinds=("Shot",)) == []


class TestGateMatching:
    def test_event_match_identical(self):
        ev = [(1.0, 20.0, 30.0), (5.0, 40.0, 30.0)]
        p, r = event_match(ev, ev)
        assert p == pytest.approx(1.0) and r == pytest.approx(1.0)

    def test_event_match_empty_gt(self):
        p, r = event_match([(1.0, 20.0, 30.0)], [])
        assert p == pytest.approx(0.0) and r == pytest.approx(1.0)

    def test_event_match_empty_both(self):
        assert event_match([], []) == (1.0, 1.0)

    def test_pass_match_is_event_match(self):
        from src.analytics.adapters.statsbomb.validate import pass_match
        ev = [(1.0, 20.0, 30.0), (5.0, 40.0, 30.0)]
        assert pass_match(ev, ev) == event_match(ev, ev)

    def test_kind_reported_only_when_gt_present(self):
        cv = [Event("Pass", 1.0, 20.0, 30.0, 0),
              Event("Shot", 2.0, 118.0, 35.0, 0)]
        report = validate_events_against_gt(
            cv, {"Pass": [(1.0, 20.0, 30.0)]})
        assert report.kinds == {}
        assert report.pass_recall == pytest.approx(1.0)

    def test_kind_fail_blocks_report(self):
        cv = [Event("Pass", 1.0, 20.0, 30.0, 0),
              Event("Shot", 100.0, 118.0, 35.0, 0)]   # shot at wrong time
        report = validate_events_against_gt(
            cv, {"Pass": [(1.0, 20.0, 30.0)],
                 "Shot": [(2.0, 118.0, 35.0)]})
        assert "Shot" in report.kinds
        assert report.kinds["Shot"][1] == 0.0        # recall 0 -> gate blocks
        assert report.passes is False

    def test_report_passes_with_good_kinds(self):
        cv = [Event("Pass", 1.0, 20.0, 30.0, 0),
              Event("Shot", 2.0, 118.0, 35.0, 0)]
        report = validate_events_against_gt(
            cv, {"Pass": [(1.0, 20.0, 30.0)],
                 "Shot": [(2.0, 118.0, 35.0)]})
        report.position_error_m = 1.0
        assert report.passes is True

    def test_empty_gt_kind_blocks(self):
        report = ValidationReport(kinds={"Shot": (1.0, 1.0, 0)})
        assert report.passes is False


def _rows_with_players(ball: list[tuple[float, float, float]],
                       players: list[tuple[str, int, float, float]]) -> pd.DataFrame:
    """Ball samples + static players. ball == [(t, x, y), ...].

    `players` == [(track_id, team_id, x, y), ...]; a track id may repeat to
    move the same player across frames.
    """
    rows = []
    for i, (t, x, y) in enumerate(ball):
        rows.append({"frame_idx": i, "timestamp_sec": t, "track_id": "ball",
                     "team_id": np.nan, "pitch_x": x, "pitch_y": y,
                     "class_name": "ball"})
        for j, (tid, team, px, py) in enumerate(players):
            rows.append({"frame_idx": i, "timestamp_sec": t,
                         "track_id": tid, "team_id": team,
                         "pitch_x": px, "pitch_y": py,
                         "class_name": "player"})
    return _frame(rows)


class TestClearance:
    def test_ball_leaving_defensive_third(self):
        # ball from x=30 to x=70 in 1 s -> a clearance
        df = _rows_with_players(
            ball=[(0.0, 30.0, 35.0), (1.0, 70.0, 35.0)],
            players=[("a", 0, 30.0, 35.0)])
        events = extract_events(df, kinds=("Clearance",))
        assert [e.kind for e in events] == ["Clearance"]
        assert events[0].timestamp_sec == pytest.approx(1.0)

    def test_slow_exit_not_clearance(self):
        df = _rows_with_players(
            ball=[(0.0, 30.0, 35.0), (4.0, 70.0, 35.0)],  # 4 s > max 3 s
            players=[("a", 0, 30.0, 35.0)])
        assert extract_events(df, kinds=("Clearance",)) == []

    def test_ball_never_in_midfield_not_clearance(self):
        df = _rows_with_players(
            ball=[(0.0, 30.0, 35.0), (1.0, 50.0, 35.0)],  # x=50 < 60
            players=[("a", 0, 30.0, 35.0)])
        assert extract_events(df, kinds=("Clearance",)) == []


class TestSetPieces:
    def test_touchline_crossing_is_throw_in(self):
        df = _rows_with_players(
            ball=[(0.0, 50.0, 69.0), (1.0, 50.0, 71.0)],  # y > 70
            players=[("a", 0, 50.0, 35.0)])
        events = extract_events(df, kinds=("Throw-in",))
        assert [e.kind for e in events] == ["Throw-in"]

    def test_goal_line_corner_zone(self):
        df = _rows_with_players(
            ball=[(0.0, 119.0, 1.0), (1.0, 121.0, 1.0)],  # x>120, y near corner
            players=[("a", 0, 60.0, 35.0)])
        events = extract_events(df, kinds=("Corner", "Goal Kick"))
        assert [e.kind for e in events] == ["Corner"]

    def test_goal_line_middle_is_goal_kick(self):
        df = _rows_with_players(
            ball=[(0.0, 119.0, 35.0), (1.0, 121.0, 35.0)],  # x>120, y middle
            players=[("a", 0, 60.0, 35.0)])
        events = extract_events(df, kinds=("Corner", "Goal Kick"))
        assert [e.kind for e in events] == ["Goal Kick"]


class TestPressure:
    def test_opposition_near_ball_is_pressure(self):
        df = _rows_with_players(
            ball=[(0.0, 50.0, 35.0), (1.0, 51.0, 35.0)],
            players=[("a", 0, 50.0, 35.0),        # owner
                     ("b", 1, 51.5, 35.0)])       # presser within 3 m
        events = extract_events(df, kinds=("Pressure",))
        assert events and events[0].team_id == 0
        assert events[0].qualifiers["pressers"] >= 1

    def test_far_opposition_not_pressure(self):
        df = _rows_with_players(
            ball=[(0.0, 50.0, 35.0), (1.0, 51.0, 35.0)],
            players=[("a", 0, 50.0, 35.0),
                     ("b", 1, 60.0, 35.0)])        # 10 m away
        assert extract_events(df, kinds=("Pressure",)) == []


class TestCarry:
    def test_same_owner_moves_ball(self):
        df = _rows_with_players(
            ball=[(0.0, 40.0, 35.0), (0.5, 42.0, 35.0), (1.0, 44.0, 35.0)],
            players=[("a", 0, 40.0, 35.0), ("a", 0, 44.0, 35.0)])
        events = extract_events(df, kinds=("Carry",))
        assert [e.kind for e in events] == ["Carry"]
        assert events[0].qualifiers["distance_m"] == pytest.approx(4.0)

    def test_short_movement_not_carry(self):
        df = _rows_with_players(
            ball=[(0.0, 40.0, 35.0), (1.0, 41.0, 35.0)],   # 1 m < 3 m
            players=[("a", 0, 40.0, 35.0), ("a", 0, 41.0, 35.0)])
        assert extract_events(df, kinds=("Carry",)) == []


class TestDuel:
    def test_both_teams_near_loose_ball(self):
        # ball far from every player (loose) but both teams within 5 m for
        # two consecutive frames -> one duel reported at the first frame
        df = _rows_with_players(
            ball=[(0.0, 60.0, 35.0), (1.0, 60.5, 35.0)],
            players=[("a", 0, 60.0, 36.0), ("b", 1, 61.0, 35.5)])
        events = extract_events(df, kinds=("Duel",))
        assert len(events) == 1
        assert events[0].kind == "Duel"
        assert events[0].timestamp_sec == pytest.approx(0.0)
        assert events[0].qualifiers["frames"] == 2

    def test_single_frame_not_duel(self):
        df = _rows_with_players(
            ball=[(0.0, 60.0, 35.0)],
            players=[("a", 0, 60.0, 36.0), ("b", 1, 61.0, 35.5)])
        assert extract_events(df, kinds=("Duel",)) == []

    def test_one_team_near_not_duel(self):
        df = _rows_with_players(
            ball=[(0.0, 60.0, 35.0), (1.0, 60.5, 35.0)],
            players=[("a", 0, 60.0, 36.0), ("b", 1, 80.0, 80.0)])
        assert extract_events(df, kinds=("Duel",)) == []


class TestThroughBall:
    def test_pass_beyond_defensive_line(self):
        df = _rows_with_players(
            ball=[(0.0, 30.0, 35.0), (1.0, 80.0, 35.0)],
            players=[("a", 0, 30.0, 35.0),   # passer/owner
                     ("b", 0, 80.0, 35.0),   # receiver
                     ("d1", 1, 50.0, 20.0),  # defender line ~ x=50
                     ("d2", 1, 55.0, 40.0),
                     ("d3", 1, 48.0, 55.0)])
        # receiver at x=80, line at mean(50,55,48)=51 -> 80 > 51+15 -> through ball
        events = extract_events(df, kinds=("Pass", "Through Ball"))
        kinds = [e.kind for e in events]
        assert "Pass" in kinds and "Through Ball" in kinds

    def test_pass_within_defensive_line_not_through(self):
        df = _rows_with_players(
            ball=[(0.0, 30.0, 35.0), (1.0, 60.0, 35.0)],
            players=[("a", 0, 30.0, 35.0),
                     ("b", 0, 60.0, 35.0),
                     ("d1", 1, 50.0, 20.0),
                     ("d2", 1, 55.0, 40.0),
                     ("d3", 1, 48.0, 55.0)])
        events = extract_events(df, kinds=("Pass", "Through Ball"))
        assert [e.kind for e in events] == ["Pass"]
