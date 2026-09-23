"""Dixon-Coles double-Poisson team ratings + match prediction.

Teams are rated by attack (`att`) and defence (`def`) strengths against a
league baseline. P(home scores h, away scores a) is the product of two
Poissons in the expected-goal rates

    λ_home = base * att_home * def_away * home_adv
    λ_away = base * att_away * def_home

multiplied by the Dixon-Coles low-score correction τ(h, a) for the
0-0/1-0/0-1/1-1 cells, which a plain independent-Poisson model under-fits.

`fit_dixon_coles` estimates all strengths + the home-advantage factor from
a list of `(lam_base, home_goals, away_goals)` observed results (per-season
average goals per match as `lam_base`). `predict_match` returns the outcome
probabilities; ratings are simply the fitted strengths and need no iterative
updating beyond re-fitting as results stream in (a Poisson-GLM/EM-lite
update per fixture is the recursive analogue used in ratings products).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

DEFAULT_BASE = 1.35  # league-average goals per team per match


@dataclass(frozen=True)
class MatchResult:
    """One observed match result (provider-agnostic goal pair)."""
    home: str
    away: str
    home_goals: int
    away_goals: int

    @property
    def outcome(self) -> str:
        return ("home" if self.home_goals > self.away_goals
                else "away" if self.home_goals < self.away_goals else "draw")


@dataclass
class DixonColesFit:
    """Estimated ratings + calibration from observed results."""
    base: float
    home_adv: float = 1.0
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    rho: float = 0.0  # Dixon-Coles low-score correlation

    def team_attack(self, team: str) -> float:
        return self.attack.get(team, 1.0)

    def team_defence(self, team: str) -> float:
        return self.defence.get(team, 1.0)

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """λ_home / λ_away for a fixture under the model."""
        lh = self.base * self.team_attack(home) * self.team_defence(away) * self.home_adv
        la = self.base * self.team_attack(away) * self.team_defence(home)
        return float(lh), float(la)


def _pois(lam: float, k: int) -> float:
    from math import exp, factorial
    return exp(-lam) * (lam ** k) / factorial(k)


def _tau(lam_h: float, lam_a: float, h: int, a: int, rho: float) -> float:
    """Dixon-Coles τ correction for the low-score cells."""
    if h == 0 and a == 0:
        return 1.0 - lam_h * lam_a * rho
    if h == 0 and a == 1:
        return 1.0 + lam_h * rho
    if h == 1 and a == 0:
        return 1.0 + lam_a * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def _dc_loglike(
    params: np.ndarray,
    base: float,
    teams: list[str],
    results: list[MatchResult],
) -> float:
    """Negative log-likelihood of the DC model over observed results.

    params layout: [att(t0..tn-1), def(t0..tn-1), log(home_adv), rho].
    Attacks are anchored at mean 1 (soft normal prior) to break the
    scale/shift degeneracy.
    """
    n = len(teams)
    att = np.exp(params[:n])
    deff = np.exp(params[n:2 * n])
    home_adv = np.exp(params[2 * n])
    rho = params[2 * n + 1]
    ll = 0.0
    jac = 0.0
    for r in results:
        ah = att[teams.index(r.home)]
        aa = att[teams.index(r.away)]
        dh = deff[teams.index(r.home)]
        da = deff[teams.index(r.away)]
        lh = base * ah * da * home_adv
        la = base * aa * dh
        tau = _tau(lh, la, r.home_goals, r.away_goals, rho)
        pmf = tau * _pois(lh, r.home_goals) * _pois(la, r.away_goals)
        ll += np.log(max(pmf, 1e-300))
    penalty = 0.0
    for a in att:
        penalty -= (np.log(a) - np.log(np.mean(att))) ** 2 * 0.5
    return -(ll + penalty)


def fit_dixon_coles(
    results: list[MatchResult],
    base: float | None = None,
    max_iter: int = 5000,
) -> DixonColesFit:
    """Maximum-likelihood fit of attack/defence/home-adv/rho over results.

    Teams with a single game are under-identified; with ≥4 matches the fit
    converges to sensible league-normalised strengths. `base` = league mean
    goals per team per match when the caller has it.
    """
    teams = sorted({r.home for r in results} | {r.away for r in results})
    if not teams:
        raise ValueError("fit_dixon_coles needs at least one result")
    if base is None:
        base = float(np.mean([(r.home_goals + r.away_goals) / 2.0
                              for r in results]))
    base = max(base, 0.1)
    n = len(teams)
    x0 = np.concatenate([
        np.zeros(n), np.zeros(n), np.array([0.0]), np.array([0.0]),
    ])
    from scipy.optimize import minimize
    res = minimize(_dc_loglike, x0, args=(base, teams, results),
                   method="BFGS", options={"maxiter": max_iter,
                                           "gtol": 1e-6})
    params = res.x
    att = np.exp(params[:n])
    deff = np.exp(params[n:2 * n])
    # anchor attacks to mean 1 so strengths are comparable across leagues
    mean_att = float(np.mean(att))
    att = att / mean_att
    deff = deff * mean_att
    return DixonColesFit(
        base=base,
        home_adv=float(np.exp(params[2 * n])),
        attack=dict(zip(teams, att)),
        defence=dict(zip(teams, deff)),
        rho=float(np.clip(params[2 * n + 1], -0.5, 0.5)),
    )


@dataclass
class MatchOdds:
    """Outcome probabilities + expected goals for a fixture."""
    home: str
    away: str
    p_home: float
    p_draw: float
    p_away: float
    exp_home: float
    exp_away: float

    @property
    def favourite(self) -> str:
        if self.p_home >= self.p_away and self.p_home >= self.p_draw:
            return self.home
        if self.p_away >= self.p_draw:
            return self.away
        return "draw"


def predict_match(fit: DixonColesFit, home: str, away: str,
                  max_goals: int = 20) -> MatchOdds:
    """Outcome distribution + expected goals for a fixture under the fit."""
    lh, la = fit.expected_goals(home, away)
    p_home = p_draw = p_away = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = _tau(lh, la, h, a, fit.rho) * _pois(lh, h) * _pois(la, a)
            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total
    return MatchOdds(home=home, away=away,
                     p_home=p_home, p_draw=p_draw, p_away=p_away,
                     exp_home=float(lh), exp_away=float(la))


def table_from_results(results: list[MatchResult],
                       fit: DixonColesFit | None = None) -> dict[str, dict]:
    """League table from the fitted model: expected + actual points per team.

    Expected points use `predict_match` outcomes; actual points use result
    outcomes. A team outperforming its xPoints is over-rated (luck, per the
    Numbers Game season-of-half-luck note).
    """
    if fit is None:
        fit = fit_dixon_coles(results)
    actual: dict[str, float] = {}
    expected: dict[str, float] = {}
    played: dict[str, int] = {}
    for r in results:
        odds = predict_match(fit, r.home, r.away)
        actual[r.home] = actual.get(r.home, 0.0) + {
            "home": 3, "draw": 1, "away": 0}[r.outcome]
        actual[r.away] = actual.get(r.away, 0.0) + {
            "home": 0, "draw": 1, "away": 3}[r.outcome]
        expected[r.home] = expected.get(r.home, 0.0) + odds.p_home * 3 + odds.p_draw
        expected[r.away] = expected.get(r.away, 0.0) + odds.p_away * 3 + odds.p_draw
        played[r.home] = played.get(r.home, 0) + 1
        played[r.away] = played.get(r.away, 0) + 1
    return {
        t: {
            "played": played.get(t, 0),
            "actual_points": actual.get(t, 0.0),
            "expected_points": expected.get(t, 0.0),
            "luck": actual.get(t, 0.0) - expected.get(t, 0.0),
        }
        for t in sorted(set(actual) | set(expected))
    }