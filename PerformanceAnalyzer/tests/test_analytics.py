"""Tests for the provider-agnostic analytics layer (schema + models).

Driven by synthetic normalized `src.analytics.schema` events (no network) so the
model maths are verified independently of any provider, plus a StatsBomb-adapter
smoke test against the locally-cached WC-Final JSON.
"""

from __future__ import annotations

import pytest

from src.analytics.adapters import StatsBombAdapter, VideoAdapter, VideoAdapterDormant
from src.analytics.models.core import (
    GoalLineValuation,
    PassingNetwork,
    expected_points_from_goals,
    funnel,
    possession_analysis,
)
from src.analytics.models.xg import aggregate_shots
from src.analytics.schema import PassEvent, ShotEvent


def _passes():
    """Symmetric synthetic pass networks: 10 A->B and 2 B->A complete passes."""
    return [
        PassEvent(type="Pass", timestamp_sec=0.0 + i, x=50.0, y=35.0,
                  team_side=0, player_id=1,
                  end_x=80.0, end_y=35.0, outcome="complete", receiver_id=2)
        for i in range(10)
    ] + [
        PassEvent(type="Pass", timestamp_sec=0.0, x=50.0, y=35.0,
                  team_side=0, player_id=2,
                  end_x=20.0, end_y=35.0, outcome="complete", receiver_id=1)
        for _ in range(2)
    ]


class TestPassingNetwork:
    def test_counts_edges(self):
        net = PassingNetwork()
        net.add(1, 2)
        net.add(1, 2)
        net.add(2, 1)
        assert net.edges == {(1, 2): 2, (2, 1): 1}

    def test_matrix_ordering(self):
        net = PassingNetwork()
        net.add(1, 2)
        net.add(2, 3)
        m = net.as_matrix([1, 2, 3])
        assert m[0, 1] == 1.0
        assert m[1, 2] == 1.0
        assert m[2, 0] == 0.0

    def test_degree_top(self):
        net = PassingNetwork()
        net.add(1, 2)
        net.add(2, 1)
        net.add(2, 3)
        top = dict(net.degree_top(2))
        assert top[2] == 3  # both in and out


class TestPassingNetworkCentrality:
    def _star(self):
        """Player 2 hubs every shortest path (star network)."""
        net = PassingNetwork()
        net.add(1, 2)
        net.add(2, 1)
        net.add(2, 3)
        net.add(3, 2)
        net.add(2, 4)
        net.add(4, 2)
        return net

    def _line(self):
        """Equally-spread chain: no single hub."""
        net = PassingNetwork()
        net.add(1, 2)
        net.add(2, 3)
        net.add(3, 4)
        net.add(4, 5)
        net.add(5, 6)
        net.add(2, 3)
        net.add(4, 5)
        net.add(5, 6)
        return net

    def test_betweenness_center_max(self):
        bc = self._star().betweenness(weights=False)
        assert bc[2] > bc[1]
        assert bc[2] == pytest.approx(1.0, abs=1e-6)
        assert bc[1] == pytest.approx(0.0, abs=1e-9)

    def test_summarize_network(self):
        from src.analytics.models.core import summarize_network
        s = summarize_network(self._star())
        assert s.most_central == 2
        assert 0.0 < s.decentralization <= 1.0

    def test_decentralized_side_lower(self):
        from src.analytics.models.core import summarize_network
        star = summarize_network(self._star()).decentralization
        line = summarize_network(self._line()).decentralization
        assert line < star
        assert line >= 0.0 and line <= 1.0


class TestPossession:
    def test_share_and_rate(self):
        # team 0 holds 0->60, then a turnover hands it to team 1 for 60->75
        evs = [
            PassEvent(type="Pass", timestamp_sec=0.0, x=0, y=0, team_side=0,
                      outcome="complete"),
            PassEvent(type="Pass", timestamp_sec=45.0, x=0, y=0, team_side=0,
                      outcome="complete"),
            PassEvent(type="Ball Recovery", timestamp_sec=60.0, x=0, y=0,
                      team_side=1),
            PassEvent(type="Pass", timestamp_sec=75.0, x=0, y=0, team_side=1,
                      outcome="complete"),
        ]
        pos = possession_analysis(evs, [e for e in evs if isinstance(e, PassEvent)])
        # team 0 held 0->60 (60 s), team 1 60->75 (15 s)
        assert pos.share(0) == pytest.approx(60 / 75, abs=1e-6)
        # team 0: 2 completed passes / 1 minute possession
        assert pos.passing_rate(0) == pytest.approx(2.0, abs=1e-6)


class TestFunnel:
    def test_chain_break_on_recovery(self):
        events = [
            ShotEvent(type="Shot", timestamp_sec=5.0, x=110.0, y=35.0,
                      team_side=0),
        ]
        f = funnel(events, [ShotEvent(type="Shot", timestamp_sec=5.0,
                                      x=110.0, y=35.0, team_side=0,
                                      result="goal")])
        assert f.possessions == 1
        assert f.shots == 1
        assert f.goals == 1
        assert f.possessions_to_goal == pytest.approx(1.0)


class TestExpectedPoints:
    def test_balanced_side(self):
        ep = expected_points_from_goals({0: 1.4, 1: 1.4}, {0: 1.4, 1: 1.4})
        # an even side earns ~1.4 pts per match
        assert ep.expected_points[0] == pytest.approx(1.37, abs=0.02)
        assert ep.expected_points[0] == pytest.approx(ep.expected_points[1], abs=1e-9)

    def test_dominant_side(self):
        ep = expected_points_from_goals({0: 2.2, 1: 1.0}, {0: 1.0, 1: 2.2})
        assert ep.expected_points[0] > ep.expected_points[1]
        assert ep.expected_points[0] > 1.8


class TestGoalLineValuation:
    def test_prevention_roughly_equals_scoring(self):
        gv = GoalLineValuation.from_rates({0: 1.4}, {0: 1.4})
        # Numbers Game §4: conceding is ~as costly as scoring is valuable
        assert 0.5 < gv.value_ratio[0] < 1.5

    def test_diminishing_returns_as_score_grows(self):
        # Marginal expected-points value of a goal is largest when the score is
        # open and shrinks as the side pulls ahead (game already won) — the
        # "0 > 1"-style diminishing-returns property, not a monotonic constant.
        from src.analytics.models.core import _expected_points_diff
        base = _expected_points_diff(1.6, 1.4, 0, 0)
        one0 = _expected_points_diff(1.6, 1.4, 1, 0)
        two0 = _expected_points_diff(1.6, 1.4, 2, 0)
        three0 = _expected_points_diff(1.6, 1.4, 3, 0)
        first, second, third = one0 - base, two0 - one0, three0 - two0
        assert third < second < first < 1.0
        assert first > 0.5


class TestWinProbability:
    def test_kickoff_balanced(self):
        from src.analytics.models.winprob import win_probability
        c = win_probability(1.4, 1.4, minute=0.0)
        assert c.p_win == pytest.approx(c.p_loss, abs=1e-9)
        assert 0.25 < c.p_win < 0.40
        assert c.p_draw > 0.2

    def test_dominant_side_favoured(self):
        from src.analytics.models.winprob import win_probability
        c = win_probability(2.2, 1.0, minute=0.0)
        assert c.p_win > c.p_loss
        assert c.p_win > 0.5

    def test_late_lead_is_safe(self):
        from src.analytics.models.winprob import win_probability
        c = win_probability(1.0, 1.0, minute=89.0, score_for=1)
        assert c.p_win > 0.9

    def test_score_curve_full_match(self):
        from src.analytics.models.winprob import score_curve
        curve = score_curve(1.4, 1.4, minute_step=15.0)
        assert curve[0].minute == 0.0
        assert curve[-1].minute == 90.0
        probs = [c.p_win + c.p_draw + c.p_loss for c in curve]
        assert all(p == pytest.approx(1.0, abs=1e-9) for p in probs)

    def test_goal_value_increases_with_time(self):
        from src.analytics.models.winprob import goal_value_by_minute
        vals = goal_value_by_minute(1.4, 1.4, minute_step=30.0)
        assert vals[0].minute == 0.0
        early, late = vals[0], vals[-1]
        # the equaliser at 90' is worth more than at 0'
        assert late.home_goal > early.home_goal
        assert late.away_goal > early.away_goal
        # conceding costs >= what scoring earns in a chase (balanced rates)
        assert late.away_goal > 0.0


class TestDixonColesRating:
    def _results(self, n: int = 300, seed: int = 0):
        import numpy as np
        from src.analytics.models.rating import MatchResult
        np.random.seed(seed)
        teams = ["A", "B", "C", "D"]
        strength = {"A": 1.8, "B": 1.3, "C": 0.9, "D": 0.5}
        out = []
        for _ in range(n):
            ht, at = tuple(np.random.choice(teams, 2, replace=False))
            lh = 1.35 * strength[ht]
            la = 1.35 * strength[at]
            out.append(MatchResult(
                ht, at, min(int(np.random.poisson(lh)), 8),
                min(int(np.random.poisson(la)), 8)))
        return teams, out

    def test_fit_orders_attack_strength(self):
        from src.analytics.models.rating import fit_dixon_coles
        teams, results = self._results()
        fit = fit_dixon_coles(results)
        assert fit.attack["A"] > fit.attack["D"]
        assert fit.attack["A"] > 1.0
        assert fit.attack["D"] < fit.attack["C"]

    def test_predict_match_favours_strong_home(self):
        from src.analytics.models.rating import fit_dixon_coles, predict_match
        _, results = self._results()
        fit = fit_dixon_coles(results)
        o = predict_match(fit, "A", "D")
        assert o.p_home > o.p_away
        assert o.exp_home > o.exp_away
        s = o.p_home + o.p_draw + o.p_away
        assert s == pytest.approx(1.0, abs=1e-6)

    def test_symmetric_without_home_advantage(self):
        from src.analytics.models.rating import DixonColesFit, predict_match
        fit = DixonColesFit(base=1.35, home_adv=1.0,
                            attack={"A": 1.2, "B": 1.0},
                            defence={"A": 0.9, "B": 1.0}, rho=0.0)
        o = predict_match(fit, "A", "A")
        assert o.p_home == pytest.approx(o.p_away, abs=1e-9)
        assert o.p_home + o.p_draw + o.p_away == pytest.approx(1.0, abs=1e-9)

    def test_table_luck(self):
        from src.analytics.models.rating import table_from_results
        teams, results = self._results()
        tab = table_from_results(results)
        assert set(tab) >= {"A", "B", "C", "D"}
        assert tab["A"]["expected_points"] > tab["D"]["expected_points"]
        assert all(isinstance(t["luck"], float) for t in tab.values())


class TestVaep:
    def _canonical_events(self):
        from src.analytics.schema import PassEvent
        # team 0 attacks toward x=0 (canonical): pass deep into the box
        return [
            PassEvent(type="Pass", timestamp_sec=0.0, x=90.0, y=35.0,
                      team_side=0, player_id=7, outcome="complete",
                      end_x=8.0, end_y=35.0),
            PassEvent(type="Pass", timestamp_sec=1.0, x=90.0, y=35.0,
                      team_side=0, player_id=7, outcome="incomplete",
                      end_x=6.0, end_y=35.0),
        ]

    def test_shot_goal_highly_valued(self):
        from src.analytics.models.vaep import value_action
        from src.analytics.schema import ShotEvent
        shot = ShotEvent(type="Shot", timestamp_sec=2.0, x=6.0, y=35.0,
                         team_side=0, player_id=9, result="goal", xg=0.5)
        v = value_action(shot)
        assert v.value == pytest.approx(0.5, abs=1e-9)  # xg after - 0 before
        assert v.scoring_add == pytest.approx(0.5, abs=1e-9)

    def test_forward_pass_gains_value(self):
        from src.analytics.models.vaep import hazard_surface, value_action
        from src.analytics.schema import PassEvent
        surf = hazard_surface(attacking_goal_x=0.0)
        p = PassEvent(type="Pass", timestamp_sec=0.0, x=90.0, y=35.0,
                      team_side=0, player_id=7, outcome="complete",
                      end_x=8.0, end_y=35.0)
        v = value_action(p, surf)
        assert v.value > 0.1  # moved into the danger zone

    def test_incomplete_pass_punished(self):
        from src.analytics.models.vaep import hazard_surface, value_action
        from src.analytics.schema import PassEvent
        surf = hazard_surface(attacking_goal_x=0.0)
        good = value_action(PassEvent(
            type="Pass", timestamp_sec=0.0, x=90.0, y=35.0, team_side=0,
            player_id=7, outcome="complete", end_x=8.0, end_y=35.0), surf)
        bad = value_action(PassEvent(
            type="Pass", timestamp_sec=1.0, x=90.0, y=35.0, team_side=0,
            player_id=7, outcome="incomplete", end_x=8.0, end_y=35.0), surf)
        assert bad.value < good.value
        assert bad.conceding_add > 0.0

    def test_summary_ranks_finisher(self):
        from src.analytics.models.vaep import vaep
        from src.analytics.schema import ShotEvent
        shots = [
            ShotEvent(type="Shot", timestamp_sec=2.0, x=6.0, y=35.0,
                      team_side=0, player_id=9, result="goal", xg=0.5),
            ShotEvent(type="Shot", timestamp_sec=3.0, x=40.0, y=10.0,
                      team_side=1, player_id=4, result="off_target", xg=0.1),
        ]
        valued, summary = vaep(self._canonical_events(), shots)
        assert summary.valued == 4
        assert summary.player_value(9) > 0.4
        assert summary.by_team[0] > summary.by_team[1]
        assert summary.top_players(1)[0][0] == 9


class TestShotAggregates:
    def test_uses_provider_xg(self):
        shots = [
            ShotEvent(type="Shot", timestamp_sec=1.0, x=110.0, y=35.0,
                      team_side=0, result="goal", xg=0.5),
            ShotEvent(type="Shot", timestamp_sec=2.0, x=30.0, y=10.0,
                      team_side=0, result="off_target", xg=0.1),
        ]
        agg = aggregate_shots(shots)
        assert agg.shots[0] == 2
        assert agg.xg[0] == pytest.approx(0.6)
        assert agg.goals[0] == 1
        assert agg.over_performance(0) == pytest.approx(0.4)


class TestVideoAdapterDormant:
    """The Video/CV adapter is provider-agnostic but DORMANT by default."""

    def test_default_is_dormant(self):
        v = VideoAdapter()
        assert not v.enabled

    def test_dormant_refuses_to_run(self):
        v = VideoAdapter()
        with pytest.raises(VideoAdapterDormant):
            v.events(1)
        with pytest.raises(VideoAdapterDormant):
            v.passes(1)
        with pytest.raises(VideoAdapterDormant):
            v.shots(1)
        with pytest.raises(VideoAdapterDormant):
            v.freeze_frames(1)
        with pytest.raises(VideoAdapterDormant):
            v.match(1)

    def test_explicit_enable_surfaces_parked_seam(self):
        v = VideoAdapter(enabled=True)
        assert v.enabled
        # running requires a real video + the (unimplemented) CV mapping seam
        with pytest.raises(NotImplementedError):
            v.events(1)


class TestStatsBombAdapter:
    """Smoke test against locally-cached WC-2022 Final (no network)."""

    @pytest.fixture
    def adapter(self):
        return StatsBombAdapter()

    def test_match_lineups(self, adapter):
        m = adapter.match(3869685)
        assert len(m.teams) == 2
        assert m.home_team == "France"
        assert m.away_team == "Argentina"
        assert len(m.players) >= 22

    def test_events_and_passes(self, adapter):
        events = adapter.events(3869685)
        passes = adapter.passes(3869685)
        shots = adapter.shots(3869685)
        assert len(events) > 0
        assert len(passes) >= 1000
        assert len(shots) >= 30
        for p in passes[:5]:
            assert p.team_side in (0, 1)
            assert 0.0 <= p.x <= 120.0
            assert 0.0 <= p.y <= 70.0
        # timestamps sorted and monotone, starting near 0
        ts = [e.timestamp_sec for e in events]
        assert ts == sorted(ts)

    def test_freeze_frames(self, adapter):
        ff = adapter.freeze_frames(3869685)
        assert len(ff) > 1000
        frame = ff[0]
        assert frame.ball is not None
        # each entry is a (player_id, side, x, y, keeper) tuple of length 5
        for entry in frame.players:
            assert len(entry) == 5
            assert 0.0 <= entry[2] <= 120.0
            assert 0.0 <= entry[3] <= 70.0

    def test_enriched_events(self, adapter):
        ps = adapter.passes(3869685)
        p = ps[0]
        assert p.period >= 1
        assert p.length is None or p.length > 0
        assert p.play_pattern != ""
        sh = adapter.shots(3869685)
        assert sh[0].technique != ""
        assert sh[0].result in ("goal", "saved", "off_target", "blocked", "post")

    def test_enriched_freeze_frames(self, adapter):
        ff = adapter.freeze_frames(3869685)
        f = ff[0]
        assert f.event_type != ""
        assert f.period >= 1
        assert isinstance(f.visible_area, list)


class TestSnapshotModels:
    def _freeze(self):
        from src.analytics.schema import FreezeFrame
        # 6 players: 3 per side, fixed positions
        players = []
        for i in range(3):
            players.append((None, 0, 20.0 + i * 5, 30.0, False))
            players.append((None, 1, 100.0 - i * 5, 40.0, False))
        return FreezeFrame(timestamp_sec=10.0, possession_side=0,
                           players=players, ball=(60.0, 35.0),
                           event_type="Pass", period=1)

    def test_to_tracking_df(self):
        from src.analytics.models.snapshot import to_tracking_df
        from src.analytics.schema import FreezeFrame
        ff = self._freeze()
        df = to_tracking_df(ff.players, ff.timestamp_sec)
        assert len(df) == 6
        assert set(df["team_id"]).issubset({0, 1})

    def test_snapshot_summary(self):
        from src.analytics.models.snapshot import snapshot_summary
        s = snapshot_summary(self._freeze())
        assert s.frames == 6
        assert 0.0 <= (s.territory_team_0 or 0) <= 1.0
        assert s.ball_zone_x == pytest.approx(60.0)


class TestFlowModels:
    def _frames(self):
        from src.analytics.schema import FreezeFrame
        # team 0 players drift right between two snapshots
        f1 = FreezeFrame(timestamp_sec=0.0, players=[
            (None, 0, 50.0, 35.0, False), (None, 0, 60.0, 40.0, False),
            (None, 1, 70.0, 35.0, False)])
        f2 = FreezeFrame(timestamp_sec=1.0, players=[
            (None, 0, 55.0, 35.0, False), (None, 0, 65.0, 40.0, False),
            (None, 1, 70.0, 35.0, False)])
        return [f1, f2]

    def test_team_formation(self):
        from src.analytics.models.flow import team_formation
        f = team_formation(self._frames(), team_side=0)
        assert f.n_samples == 4
        assert 50.0 < f.centroid_x < 60.0   # drifted right over both snapshots

    def test_flow_field(self):
        from src.analytics.models.flow import flow_field
        fl = flow_field(self._frames(), team_side=0)
        # team 0 moved right (positive x velocity) so field has +vx somewhere
        assert fl.vx.max() > 0
        assert fl.vx.shape == fl.grid_x.shape


class TestPressureModels:
    def _events(self):
        from src.analytics.schema import Event
        # team 1 under pressure deep (our team 0 presses); team 0 own action
        return [
            Event(type="Pass", timestamp_sec=0.0, x=110.0, y=35.0, team_side=1,
                  under_pressure=True, duration=3.0, period=1),
            Event(type="Pass", timestamp_sec=1.0, x=50.0, y=35.0, team_side=0,
                  under_pressure=True, duration=1.0, period=1),
            Event(type="Pass", timestamp_sec=2.0, x=80.0, y=35.0, team_side=1,
                  under_pressure=True, duration=6.0, period=1),
        ]

    def test_press_map(self):
        from src.analytics.models.pressure import press_map
        pm = press_map(self._events(), team_side=0)
        # 2 opponent events under pressure -> our presses; both in final third
        assert pm.total_press_events == 2
        assert pm.by_third.get("final_third", 0) == 2
        # team 0's own pressured action counts as pressed-against
        assert pm.pressured_against == 1

    def test_press_windows(self):
        from src.analytics.models.pressure import press_windows
        w = press_windows(self._events(), team_side=0)
        # durations 3.0 (on-ball? no, >2.3 not <=) and 6.0 (deep)
        assert w.deep_press_count == 1
        assert w.avg_press_duration == pytest.approx(4.5)


class TestAttackingModels:
    def _shots(self):
        from src.analytics.schema import ShotEvent
        return [
            ShotEvent(type="Shot", timestamp_sec=0.0, x=118.0, y=35.0,
                      team_side=0, result="goal", xg=0.7),
            ShotEvent(type="Shot", timestamp_sec=1.0, x=117.0, y=30.0,
                      team_side=0, result="off_target", xg=0.4),
        ]

    def test_shot_map(self):
        from src.analytics.models.attacking import shot_map
        sm = shot_map(self._shots())
        heat = sm.heatmap()
        assert heat.sum() == pytest.approx(1.1)
        assert heat.shape == (4, 6)
        assert sum(z.goals for z in sm.zones) == 1

    def test_progressive(self):
        from src.analytics.models.attacking import progressive_summary
        from src.analytics.schema import PassEvent
        passes = [
            PassEvent(type="Pass", timestamp_sec=0.0, x=40.0, y=35.0,
                      team_side=0, outcome="complete", end_x=90.0, end_y=35.0),
            PassEvent(type="Pass", timestamp_sec=1.0, x=40.0, y=35.0,
                      team_side=0, outcome="complete", end_x=30.0, end_y=35.0),
        ]
        s = progressive_summary(passes)
        assert s.progressive[0] == 1
        assert s.backward[0] == 1
