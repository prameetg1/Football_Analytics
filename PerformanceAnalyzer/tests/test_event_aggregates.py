"""Stage 9, Stage B: event aggregates over the extracted event stream."""

from __future__ import annotations

import pytest

from src.analytics.adapters.video.events import Event
from src.analytics.adapters.video.metrics.event_aggregates import (
    forward_passes,
    pass_completion,
    possession_sequences,
    ppda,
    pressures_per_min,
    pressing_success,
    shots_per_possession,
    turnover_rate,
    turnovers_by_team,
)


def _ev(kind, t, x=10.0, y=35.0, team=0, **q) -> Event:
    return Event(kind=kind, timestamp_sec=t, x=x, y=y, team_id=team,
                 qualifiers=q)


PASS = "Pass"
REC = "Ball Recovery"
SHOT = "Shot"
CARRY = "Carry"
PRESS = "Pressure"


def _stream() -> list[Event]:
    """Team 0: pass, pass, turnover to team 1; team 1 shot; team 0 carry."""
    return [
        _ev(PASS, 1.0, team=0),
        _ev(PASS, 2.0, team=0, origin_x=5.0),          # forward pass
        _ev(REC, 3.0, team=1, losing_team=0),
        _ev(SHOT, 4.0, team=1),
        _ev(CARRY, 5.0, team=0, distance_m=6.0),
    ]


class TestPassCompletion:
    def test_complete_chain(self):
        events = [_ev(PASS, 1.0, team=0), _ev(PASS, 2.0, team=0)]
        assert pass_completion(events) == pytest.approx(1.0)

    def test_pass_followed_by_turnover_incomplete(self):
        events = [_ev(PASS, 1.0, team=0),
                  _ev(REC, 2.0, team=1, losing_team=0)]
        assert pass_completion(events) == pytest.approx(0.0)

    def test_pass_followed_by_shot_complete(self):
        events = [_ev(PASS, 1.0, team=0), _ev(SHOT, 2.0, team=0)]
        assert pass_completion(events) == pytest.approx(1.0)

    def test_empty(self):
        assert pass_completion([]) == pytest.approx(0.0)


class TestForwardPasses:
    def test_counts_origin_x_advancing(self):
        events = [_ev(PASS, 1.0, x=10.0, team=0, origin_x=5.0),
                  _ev(PASS, 2.0, x=10.0, team=0, origin_x=15.0)]
        assert forward_passes(events) == 1


class TestPossessionSequences:
    def test_sequences_split_on_turnover(self):
        seqs = possession_sequences(_stream())
        assert len(seqs) == 3
        assert seqs[0]["n_passes"] == 2 and seqs[0]["team_id"] == 0
        assert seqs[0]["ended_in_turnover"] is True
        assert seqs[1]["team_id"] == 1 and seqs[1]["n_shots"] == 1
        assert seqs[2]["n_carries"] == 1

    def test_empty(self):
        assert possession_sequences([]) == []


class TestShotsPerPossession:
    def test_value(self):
        # 3 progressing possessions, 1 shot among them
        assert shots_per_possession(_stream()) == pytest.approx(1 / 3, abs=1e-4)


class TestTurnovers:
    def test_turnovers_by_team(self):
        assert turnovers_by_team(_stream()) == {0: 1}

    def test_turnover_rate(self):
        # team 0 has 2 possessions, 1 lost
        assert turnover_rate(_stream(), 0) == pytest.approx(0.5)


class TestPPDA:
    def test_ppda_counts_opp_passes_over_actions(self):
        events = [
            _ev(PASS, 1.0, team=0),   # opponent pass for team 1's PPDA
            _ev(REC, 2.0, team=1, losing_team=0),
        ]
        # team 1 defends against team 0's 1 pass, with 1 recovery action
        assert ppda(events, 1) == pytest.approx(1.0)

    def test_ppda_no_actions_inf(self):
        assert ppda([_ev(PASS, 1.0, team=0)], 1) == float("inf")


class TestPressing:
    def test_pressing_success_when_regain_follows(self):
        events = [_ev(PRESS, 1.0, team=0),
                  _ev(REC, 2.0, team=0, losing_team=1)]
        assert pressing_success(events) == pytest.approx(1.0)

    def test_pressing_success_zero_without_regain(self):
        events = [_ev(PRESS, 1.0, team=0), _ev(PASS, 2.0, team=1)]
        assert pressing_success(events) == pytest.approx(0.0)

    def test_pressures_per_min(self):
        events = [_ev(PRESS, 1.0, team=0)] * 6
        assert pressures_per_min(events, 2.0) == pytest.approx(3.0)
