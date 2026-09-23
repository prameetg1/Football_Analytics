"""Statistical models over normalized analytics data.

Each model consumes `src.analytics.schema` types only — never provider JSON —
so they work identically over any adapter. Models follow the Soccermatics /
The Numbers Game findings: passing networks, possession & territory
(Soccermatics ch.7), Poisson expected points & prevention value (Numbers Game
ch.3-4), and xG aggregation (reusing the fitted shot model).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from src.analytics.schema import Event, Match, PassEvent, ShotEvent

PITCH_LENGTH = 120.0
PITCH_WIDTH = 70.0

# Soccermatics ch.7: links only count above a pass-count threshold.
PASS_NETWORK_MIN_LINKS = 3


def _possession_chains(
    events: Iterable[Event],
) -> list[list[Event]]:
    """Split a sorted event stream into possession chains (team changes).

    A chain is a run of events belonging to one side; it ends when the side
    changes or a turnover event (Ball Recovery / Dispossessed / Interception /
    Goal) occurs. The turnover event itself opens the next chain. Mirrors the
    "slippage funnel" concept (Numbers Game §1).
    """
    chains: list[list[Event]] = []
    cur: list[Event] = []
    last_side: int | None = None
    for e in events:
        if e.team_side is None:
            continue
        if e.type in {"Ball Recovery", "Dispossessed", "Interception",
                      "Goal", "Own Goal Against"}:
            if cur:
                chains.append(cur)
            cur = [e]
            last_side = e.team_side
            continue
        if last_side is not None and e.team_side != last_side:
            if cur:
                chains.append(cur)
            cur = []
        cur.append(e)
        last_side = e.team_side
    if cur:
        chains.append(cur)
    return chains


@dataclass
class PassingNetwork:
    """weighted directed pass network (nodes = player ids)."""
    edges: dict[tuple[int, int], int] = field(default_factory=dict)

    def add(self, passer: int, receiver: int) -> None:
        key = (passer, receiver)
        self.edges[key] = self.edges.get(key, 0) + 1

    def as_matrix(self, player_ids: list[int]) -> np.ndarray:
        """(n, n) non-negative adjacency over `player_ids` ordering."""
        n = len(player_ids)
        idx = {pid: i for i, pid in enumerate(player_ids)}
        m = np.zeros((n, n))
        for (a, b), w in self.edges.items():
            if a in idx and b in idx:
                m[idx[a], idx[b]] = w
        return m

    def degree_top(self, k: int = 5) -> list[tuple[int, int]]:
        """Top `k` players by total connected pass volume (in+out)."""
        vol: dict[int, int] = {}
        for (a, b), w in self.edges.items():
            vol[a] = vol.get(a, 0) + w
            vol[b] = vol.get(b, 0) + w
        return sorted(vol.items(), key=lambda kv: -kv[1])[:k]

    def _adjacency(self) -> dict[int, dict[int, float]]:
        """Undirected adjacency map (node -> neighbours -> pass counts)."""
        adj: dict[int, dict[int, float]] = {}
        for (a, b), w in self.edges.items():
            adj.setdefault(a, {})
            adj.setdefault(b, {})
            adj[a][b] = adj[a].get(b, 0.0) + w
            adj[b][a] = adj[b].get(a, 0.0) + w
        return adj

    def betweenness(self, weights: bool = True) -> dict[int, float]:
        """Betweenness centrality per player on the pass network.

        Fraction of shortest pass-sequences (fewest intervening players, volume
        as a small tiebreak) that run through each player. High-betweenness
        passers sit at the junction of otherwise-disconnected groups — the hub
        a decentralized side avoids (Soccermatics §3).
        """
        adj = self._adjacency()
        nodes = list(adj)
        bc = {n: 0.0 for n in nodes}
        for s in nodes:
            dist = {n: float("inf") for n in nodes}
            ways = {n: 0 for n in nodes}
            prev: dict[int, list[int]] = {n: [] for n in nodes}
            dist[s] = 0.0
            ways[s] = 1
            seen: set[int] = set()
            for _ in range(len(nodes)):
                cur = min((n for n in nodes if n not in seen),
                          key=lambda n: dist[n])
                seen.add(cur)
                if dist[cur] == float("inf"):
                    break
                for nb, w in adj[cur].items():
                    step = 1.0 if not weights else 1.0 + 1.0 / max(w, 1.0)
                    nd = dist[cur] + step
                    if nd < dist[nb] - 1e-12:
                        dist[nb] = nd
                        ways[nb] = ways[cur]
                        prev[nb] = [cur]
                    elif abs(nd - dist[nb]) < 1e-12:
                        ways[nb] += ways[cur]
                        prev[nb].append(cur)
            dep = {n: 0.0 for n in nodes}
            for v in sorted(nodes, key=lambda n: -dist[n]):
                if v == s or dist[v] == float("inf"):
                    continue
                for p in prev[v]:
                    dep[p] += (ways[p] / ways[v]) * (1.0 + dep[v])
            for v, d in dep.items():
                if v != s:
                    bc[v] += d
        two = float((len(nodes) - 1) * (len(nodes) - 2))
        if two > 0:
            for n in nodes:
                bc[n] /= two
        return bc

    def decentralization(self, weights: bool = True) -> float:
        """Centralization index in [0, 1] (Freeman centralization).

        1.0 = a single hub dominates all shortest paths; 0.0 = everyone equally
        central. Lower values mark the decentralized build-up Soccermatics §3
        links to ~8% higher scoring rates.
        """
        bc = self.betweenness(weights=weights)
        if not bc:
            return 0.0
        top = max(bc.values())
        if top <= 0:
            return 0.0
        n = len(bc)
        denom = (n - 1) * top if n > 1 else top
        return float(sum(top - v for v in bc.values()) / denom)


@dataclass
class PassNetworkSummary:
    """Aggregate centrality read-out for a pass network."""
    degree_top: list[tuple[int, int]]
    betweenness: dict[int, float]
    decentralization: float

    @property
    def most_central(self) -> int | None:
        if not self.betweenness:
            return None
        return max(self.betweenness, key=self.betweenness.get)


def summarize_network(
    net: PassingNetwork,
    player_ids: list[int] | None = None,
    weights: bool = True,
) -> PassNetworkSummary:
    """Combine degree + betweenness + centralization for a whole side network.

    `player_ids` restricts centrality to a declared lineup (unknown nodes are
    ignored); default uses every node that appears in the edge table.
    """
    if player_ids is not None:
        cut = {k: v for k, v in net.edges.items()
               if k[0] in player_ids and k[1] in player_ids}
        return summarize_network(PassingNetwork(edges=cut), weights=weights)
    bc = net.betweenness(weights=weights)
    return PassNetworkSummary(
        degree_top=net.degree_top(),
        betweenness=bc,
        decentralization=net.decentralization(weights=weights),
    )


@dataclass
class Possession:
    """time (s) and share of possession + passing rate per side."""
    seconds: dict[int, float] = field(default_factory=lambda: {0: 0.0, 1: 0.0})
    passes: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})

    def share(self, side: int) -> float:
        tot = sum(self.seconds.values())
        return (self.seconds.get(side, 0.0) / tot) if tot > 0 else 0.0

    def passing_rate(self, side: int) -> float:
        """successful passes per minute of possession (Soccermatics ch.7)."""
        sec = self.seconds.get(side, 0.0)
        return (self.passes.get(side, 0) / (sec / 60.0)) if sec > 0 else 0.0


def possession_analysis(
    events: Iterable[Event],
    passes: Iterable[PassEvent],
) -> Possession:
    """Possession seconds + passing rate by side from a sorted event stream.

    `events` and `passes` must be for the same match. Time-of-possession is
    attributed to the side holding at each event by charging every inter-event
    gap to the side of the event that opens it (the standard possession clock).
    """
    evs = sorted(events, key=lambda e: e.timestamp_sec)
    pos = Possession()
    for i, e in enumerate(evs):
        if e.team_side is None:
            continue
        nxt = evs[i + 1].timestamp_sec if i + 1 < len(evs) else e.timestamp_sec
        gap = max(float(nxt - e.timestamp_sec), 0.0)
        pos.seconds[e.team_side] = pos.seconds.get(e.team_side, 0.0) + gap
    for p in passes:
        if p.team_side is not None and p.outcome == "complete":
            pos.passes[p.team_side] = pos.passes.get(p.team_side, 0) + 1
    return pos


@dataclass
class ExpectedPoints:
    """Poisson-model expected points + goals (Numbers Game §3)."""
    lam_for: dict[int, float]
    lam_against: dict[int, float]
    expected_points: dict[int, float]
    expected_goals_for: dict[int, float]
    expected_goals_against: dict[int, float]

    @classmethod
    def from_rates(cls, lam_for: dict[int, float],
                   lam_against: dict[int, float]) -> "ExpectedPoints":
        """Build from per-side scoring/conceding goal rates per match."""
        n = len(lam_for)
        ep = {}
        for side, lf in lam_for.items():
            la = lam_against[side]
            p_win = p_draw = 0.0
            for k in range(0, 12):
                pf = _pois(lf, k)
                # opponent scores exactly k -> draw; < k -> we win
                p_draw += pf * _pois(la, k)
                cdf_below = sum(_pois(la, j) for j in range(0, k))
                p_win += pf * cdf_below
            ep[side] = 3.0 * p_win + 1.0 * p_draw
        return cls(lam_for=dict(lam_for), lam_against=dict(lam_against),
                   expected_points=ep, expected_goals_for=dict(lam_for),
                   expected_goals_against=dict(lam_against))

    def luck(self, actual_points: dict[int, float]) -> dict[int, float]:
        """actual - expected points per side (positive = outperformed)."""
        return {s: actual_points[s] - self.expected_points[s]
                for s in self.expected_points}


def _pois(lam: float, k: int) -> float:
    from math import factorial, exp
    return exp(-lam) * (lam ** k) / factorial(k)


def expected_points_from_goals(
    goals_for: dict[int, float],
    goals_against: dict[int, float],
) -> ExpectedPoints:
    """Expected points from per-side season goal for/against rates."""
    return ExpectedPoints.from_rates(goals_for, goals_against)


def _expected_points_diff(lam_for: float, lam_against: float,
                          score_for: int, score_ag: int) -> float:
    """E[pts] for a side under a Poisson model given the current scoreline.

    Uses the remaining-match goal expectation scaled by the odds of the goals
    occurring; for a season-level valuation the goal rates are the per-match
    rates and the "remaining" assumption is that one more goal shifts the
    expected difference. This is the engine behind the Numbers Game finding
    that the second goal is the most valuable.
    """
    # remaining goals for/against are Poisson draws shifted by the current score
    tot = 0.0
    for kf in range(0, 6):
        for ka in range(0, 6):
            p = _pois(lam_for, kf) * _pois(lam_against, ka)
            final_f = score_for + kf
            final_a = score_ag + ka
            if final_f > final_a:
                tot += 3.0 * p
            elif final_f == final_a:
                tot += p
    return tot


@dataclass
class GoalLineValuation:
    """Marginal expected-points value of scoring vs conceding a goal.

    For a side with scoring rate `lam_for` and conceding rate `lam_against`,
    valuations are averaged over the plausible current scoreline states
    (0-0, 1-x, etc.) per the Poisson model. Mirrors the Numbers Game "0 > 1"
    argument: a goal prevented is usually at least as valuable as a goal scored,
    and the second goal is the most valuable of all.
    """
    score_goal_avg: dict[int, float] = field(default_factory=dict)
    concede_goal_avg: dict[int, float] = field(default_factory=dict)
    value_ratio: dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_rates(cls, lam_for: dict[int, float],
                   lam_against: dict[int, float]) -> "GoalLineValuation":
        score_avg: dict[int, float] = {}
        concede_avg: dict[int, float] = {}
        ratio: dict[int, float] = {}
        for side, lf in lam_for.items():
            la = lam_against[side]
            # baseline expected points at 0-0
            base = _expected_points_diff(lf, la, 0, 0)
            # scoring the next goal moves us to 1-0; conceding to 0-1
            score_val = _expected_points_diff(lf, la, 1, 0) - base
            concede_val = base - _expected_points_diff(lf, la, 0, 1)
            score_avg[side] = float(score_val)
            concede_avg[side] = float(concede_val)
            ratio[side] = float(concede_val / score_val) if score_val != 0 else 0.0
        return cls(score_goal_avg=score_avg, concede_goal_avg=concede_avg,
                   value_ratio=ratio)


@dataclass
class Funnel:
    """Slippage funnel (Numbers Game §1): possessions -> shots -> goals."""
    possessions: int = 0
    shots: int = 0
    shots_on_target: int = 0
    goals: int = 0

    @property
    def possessions_to_shots(self) -> float:
        return self.shots / self.possessions if self.possessions else 0.0

    @property
    def possessions_to_goal(self) -> float:
        return self.goals / self.possessions if self.possessions else 0.0

    @property
    def shots_to_goal(self) -> float:
        return self.goals / self.shots if self.shots else 0.0


def funnel(events: Iterable[Event],
           shots: Iterable[ShotEvent]) -> Funnel:
    """Possession chains -> shots -> on-target -> goals (per side aggregated)."""
    chains = _possession_chains(events)
    f = Funnel(possessions=len(chains),
               shots=len(list(shots)))
    f.shots_on_target = sum(1 for s in shots if s.result != "off_target")
    f.goals = sum(1 for s in shots if s.result == "goal")
    return f
