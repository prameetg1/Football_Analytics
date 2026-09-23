"""In-play win-probability: the "score curve" (Soccermatics ch.3).

A match's win probability is a Markov race between two Poisson goal
processes (scoring hazard `lam` per side per minute) truncated at 90
minutes. At any minute with a known scoreline we can solve the
distribution of final goals and read off win / draw / loss probabilities.

The Numbers Game goal-value finding (ch.3-4) falls out naturally: the
marginal expected-points impact of the *next* goal changes with the
scoreline *and* the time remaining — an equaliser at 90' is worth more
than a 3-0 icing goal at 60'. `goal_value_by_minute` reports that
marginal per minute of the match.

All inputs are plain floats/ints (per-minute goal rates derived upstream
by `expected_points_from_goals` or a Dixon-Coles fit); no schema types
are consumed here so the model can price any match state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MATCH_MINUTES = 90.0
_MAX_GOALS = 30

# Weighting used for goal-value: the share of a single goal's marginal
# expected points compared with a full decisive point swing (so 1 point is
# "worth" ~ one full goal's first-order contribution).
_PTS_PER_GOAL_SCALE = 1.0


def _pois(lam: float, k: int) -> float:
    from math import exp, factorial
    return exp(-lam) * (lam ** k) / factorial(k)


@dataclass
class ScoreCurve:
    """Win-probability read-out at one moment of a match."""
    minute: float
    score_for: int
    score_against: int
    p_win: float = 0.0
    p_draw: float = 0.0
    p_loss: float = 0.0

    @property
    def expected_points(self) -> float:
        return 3.0 * self.p_win + 1.0 * self.p_draw


def _remaining(lam: float, minutes: float) -> float:
    """Mean goals a Poisson process scores in `minutes` remaining."""
    return max(lam * minutes / MATCH_MINUTES, 0.0)


def win_probability(
    lam_for: float,
    lam_against: float,
    minute: float,
    score_for: int = 0,
    score_against: int = 0,
) -> ScoreCurve:
    """P(win/draw/loss) at `minute` for a side scoring/conceding at lambdas.

    The remaining goal counts are independent Poisson draws; final totals =
    current score + draws, and the result is read from the comparison.
    """
    t_rem = max(MATCH_MINUTES - minute, 0.0)
    lf = _remaining(lam_for, t_rem)
    la = _remaining(lam_against, t_rem)
    p_win = p_draw = p_loss = 0.0
    for kf in range(0, _MAX_GOALS):
        pf = _pois(lf, kf)
        if pf == 0:
            continue
        for ka in range(0, _MAX_GOALS):
            p = pf * _pois(la, ka)
            final_f = score_for + kf
            final_a = score_against + ka
            if final_f > final_a:
                p_win += p
            elif final_f == final_a:
                p_draw += p
            else:
                p_loss += p
    return ScoreCurve(
        minute=minute, score_for=score_for, score_against=score_against,
        p_win=p_win, p_draw=p_draw, p_loss=p_loss,
    )


def score_curve(
    lam_for: float,
    lam_against: float,
    minute_step: float = 1.0,
) -> list[ScoreCurve]:
    """The full score-curve (Soccermatics ch.3): WP every `minute_step` from
    kickoff to full time at the current scoreline as the away-goals factor.
    """
    return [
        win_probability(lam_for, lam_against, m, 0, 0)
        for m in np.arange(0.0, MATCH_MINUTES + 1e-9, minute_step)
    ]


@dataclass
class GoalValueByMinute:
    """Marginal expected-points value of each goal at each minute."""
    minute: float
    home_goal: float
    away_goal: float


def goal_value_by_minute(
    lam_for: float,
    lam_against: float,
    score_for: int = 0,
    score_against: int = 0,
    minute_step: float = 5.0,
) -> list[GoalValueByMinute]:
    """Marginal E[pts] of scoring vs conceding the next goal, by minute.

    At each minute we price the scoreline one goal up (for) and one goal
    down (against) against the current state. The Numbers Game reading:
    the goal that wins the game is worth more the later it arrives, so the
    curve rises toward full time; conceding early is cheaper than conceding
    late.
    """
    out: list[GoalValueByMinute] = []
    for m in np.arange(0.0, MATCH_MINUTES + 1e-9, minute_step):
        base = win_probability(lam_for, lam_against, m, score_for,
                               score_against)
        up = win_probability(lam_for, lam_against, m, score_for + 1,
                             score_against)
        down = win_probability(lam_for, lam_against, m, score_for,
                               score_against + 1)
        out.append(GoalValueByMinute(
            minute=float(m),
            home_goal=_PTS_PER_GOAL_SCALE * (up.expected_points - base.expected_points),
            away_goal=_PTS_PER_GOAL_SCALE * (base.expected_points - down.expected_points),
        ))
    return out