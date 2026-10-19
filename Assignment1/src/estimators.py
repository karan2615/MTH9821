"""Monte Carlo estimators: plain, control variate (CV), antithetic (AV),
combined (AV+CV), and the bonus conditional estimator.

Statistics are accumulated batch by batch (Chan et al. pairwise update of the
mean vector and co-moment matrix), so no path arrays are stored.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from model import (Scheme, simulate_Y, variance_from_Y, terminal_log_stock,
                   control_log_stock, put_payoffs, bs_put_implied_var_price,
                   conditional_put)

MASTER_SEED = 9821
BATCH_PATHS = 10_000          # path evaluations per batch (pairs: 5,000 per batch)


def stream(*key):
    """Independent named random stream: PCG64 seeded by SeedSequence(MASTER_SEED,
    spawn_key=key). Distinct keys give statistically independent streams."""
    return np.random.Generator(np.random.PCG64(
        np.random.SeedSequence(MASTER_SEED, spawn_key=tuple(key))))


def gaussian_inputs(rng, m, n):
    """Three mutually independent (m, n) arrays of N(0,1): G, U, D (in that order)."""
    return (rng.standard_normal((m, n)), rng.standard_normal((m, n)),
            rng.standard_normal((m, n)))


class Moments:
    """Running mean vector and co-moment matrix of k variables."""

    def __init__(self, k):
        self.n, self.mean, self.M2 = 0, np.zeros(k), np.zeros((k, k))

    def add(self, x):                      # x: (m, k)
        m = x.shape[0]
        mb = x.mean(axis=0)
        d = x - mb
        M2b = d.T @ d
        delta = mb - self.mean
        tot = self.n + m
        self.M2 += M2b + np.outer(delta, delta) * self.n * m / tot
        self.mean += delta * m / tot
        self.n = tot

    def cov(self):
        return self.M2 / (self.n - 1)

    def var(self):
        return np.diag(self.cov())

    def corr(self, i, j):
        c = self.cov()
        den = np.sqrt(c[i, i] * c[j, j])
        return c[i, j] / den if den > 0 else np.nan


def _batches(total, size):
    while total > 0:
        b = min(size, total)
        yield b
        total -= b


def _coef(mom, iy, ix):
    """Pilot regression coefficient Cov(Y, X)/Var(X); 0 if Var(X) = 0
    (the control then has no effect and the estimator reduces to plain MC)."""
    c = mom.cov()
    return c[iy, ix] / c[ix, ix] if c[ix, ix] > 0 else 0.0


def _summary(mom, col, m):
    est = mom.mean[col]
    se = np.sqrt(mom.var()[col] / m)
    return dict(price=est, se=se, ci_lo=est - 1.96 * se, ci_hi=est + 1.96 * se, m=m)


@dataclass
class MethodResult:
    method: str
    strikes: tuple
    per_strike: list                  # dicts: price, se, ci_lo, ci_hi, m, coef, corr...
    pilot_time: float
    prod_time: float
    path_evals_prod: int
    path_evals_pilot: int
    extra: dict = field(default_factory=dict)

    @property
    def total_time(self):
        return self.pilot_time + self.prod_time


# ------------------------------------------------------------------ path helpers
def _paths(sc, G, U, D, with_control):
    V = variance_from_Y(sc, simulate_Y(sc, G, U))
    S = np.exp(terminal_log_stock(sc, V, G, D))
    if not with_control:
        return S, None
    return S, np.exp(control_log_stock(sc, G, D))


# ------------------------------------------------------------------ the methods
def plain(sc: Scheme, strikes, n_prod, key):
    k = len(strikes)
    rng = stream(*key, 1)
    t0 = time.perf_counter()
    mom = Moments(k)
    for m in _batches(n_prod, BATCH_PATHS):
        S, _ = _paths(sc, *gaussian_inputs(rng, m, sc.p.n), with_control=False)
        mom.add(put_payoffs(S, strikes))
    res = [_summary(mom, i, mom.n) for i in range(k)]
    t_prod = time.perf_counter() - t0
    return MethodResult("Plain", tuple(strikes), res, 0.0, t_prod, n_prod, 0)


def control_variate(sc: Scheme, strikes, n_prod, n_pilot, key):
    k = len(strikes)
    cK = bs_put_implied_var_price(sc.p, np.asarray(strikes))
    # pilot: independent paths, coefficient per strike
    rng_pilot = stream(*key, 0)
    t0 = time.perf_counter()
    pm = Moments(2 * k)                                   # [X(k), C(k)]
    for m in _batches(n_pilot, BATCH_PATHS):
        S, St = _paths(sc, *gaussian_inputs(rng_pilot, m, sc.p.n), with_control=True)
        pm.add(np.hstack([put_payoffs(S, strikes), put_payoffs(St, strikes)]))
    beta = np.array([_coef(pm, i, k + i) for i in range(k)])
    t_pilot = time.perf_counter() - t0
    # production: frozen beta
    rng = stream(*key, 1)
    t0 = time.perf_counter()
    mom = Moments(3 * k)                                  # [X(k), C(k), R(k)]
    for m in _batches(n_prod, BATCH_PATHS):
        S, St = _paths(sc, *gaussian_inputs(rng, m, sc.p.n), with_control=True)
        X, C = put_payoffs(S, strikes), put_payoffs(St, strikes)
        mom.add(np.hstack([X, C, X - beta[None, :] * (C - cK[None, :])]))
    res = []
    for i in range(k):
        d = _summary(mom, 2 * k + i, mom.n)
        d.update(coef=beta[i], corr_XC=mom.corr(i, k + i), corr_XC_pilot=pm.corr(i, k + i),
                 ctrl_z=(mom.mean[k + i] - cK[i]) / np.sqrt(mom.var()[k + i] / mom.n))
        res.append(d)
    t_prod = time.perf_counter() - t0
    return MethodResult("CV", tuple(strikes), res, t_pilot, t_prod, n_prod, n_pilot)


def antithetic(sc: Scheme, strikes, n_prod, key):
    k = len(strikes)
    M = n_prod // 2
    rng = stream(*key, 1)
    t0 = time.perf_counter()
    mom = Moments(3 * k)                                  # [X+(k), X-(k), A(k)]
    for m in _batches(M, BATCH_PATHS // 2):
        G, U, D = gaussian_inputs(rng, m, sc.p.n)
        Sp, _ = _paths(sc, G, U, D, with_control=False)
        Sm, _ = _paths(sc, -G, -U, -D, with_control=False)
        Xp, Xm = put_payoffs(Sp, strikes), put_payoffs(Sm, strikes)
        mom.add(np.hstack([Xp, Xm, 0.5 * (Xp + Xm)]))
    res = []
    for i in range(k):
        d = _summary(mom, 2 * k + i, mom.n)               # observation = pair average
        d.update(corr_pair=mom.corr(i, k + i))
        res.append(d)
    t_prod = time.perf_counter() - t0
    return MethodResult("AV", tuple(strikes), res, 0.0, t_prod, 2 * M, 0)


def combined(sc: Scheme, strikes, n_prod, n_pilot, key):
    k = len(strikes)
    cK = bs_put_implied_var_price(sc.p, np.asarray(strikes))

    def pair_obs(rng, m):
        G, U, D = gaussian_inputs(rng, m, sc.p.n)
        Sp, Stp = _paths(sc, G, U, D, with_control=True)
        Sm, Stm = _paths(sc, -G, -U, -D, with_control=True)
        Xp, Xm = put_payoffs(Sp, strikes), put_payoffs(Sm, strikes)
        CA = 0.5 * (put_payoffs(Stp, strikes) + put_payoffs(Stm, strikes))
        return Xp, Xm, 0.5 * (Xp + Xm), CA

    # pilot: independent antithetic pairs
    rng_pilot = stream(*key, 0)
    t0 = time.perf_counter()
    pm = Moments(2 * k)                                   # [A(k), CA(k)]
    for m in _batches(n_pilot // 2, BATCH_PATHS // 2):
        _, _, A, CA = pair_obs(rng_pilot, m)
        pm.add(np.hstack([A, CA]))
    gamma = np.array([_coef(pm, i, k + i) for i in range(k)])
    t_pilot = time.perf_counter() - t0
    # production
    rng = stream(*key, 1)
    t0 = time.perf_counter()
    mom = Moments(5 * k)                                  # [X+, X-, A, CA, Rpair]
    for m in _batches(n_prod // 2, BATCH_PATHS // 2):
        Xp, Xm, A, CA = pair_obs(rng, m)
        mom.add(np.hstack([Xp, Xm, A, CA, A - gamma[None, :] * (CA - cK[None, :])]))
    res = []
    for i in range(k):
        d = _summary(mom, 4 * k + i, mom.n)
        d.update(coef=gamma[i], corr_pair=mom.corr(i, k + i),
                 corr_ACA=mom.corr(2 * k + i, 3 * k + i),
                 corr_ACA_pilot=pm.corr(i, k + i),
                 ctrl_z=(mom.mean[3 * k + i] - cK[i]) / np.sqrt(mom.var()[3 * k + i] / mom.n))
        res.append(d)
    t_prod = time.perf_counter() - t0
    return MethodResult("AV+CV", tuple(strikes), res, t_pilot, t_prod,
                        2 * (n_prod // 2), 2 * (n_pilot // 2))


def conditional(sc: Scheme, strikes, n_samples, key):
    """Bonus: average of Q_K over independent (G, U) volatility-driver samples."""
    k = len(strikes)
    rng = stream(*key, 1)
    t0 = time.perf_counter()
    mom = Moments(k)
    for m in _batches(n_samples, BATCH_PATHS):
        G = rng.standard_normal((m, sc.p.n))
        U = rng.standard_normal((m, sc.p.n))
        mom.add(conditional_put(sc, G, U, strikes))
    res = [_summary(mom, i, mom.n) for i in range(k)]
    t_prod = time.perf_counter() - t0
    return MethodResult("Conditional", tuple(strikes), res, 0.0, t_prod, n_samples, 0)


def add_efficiency(results, base="Plain"):
    """VRF and Gain of (19) relative to plain MC, strike by strike."""
    b = next(r for r in results if r.method == base)
    for r in results:
        for i, d in enumerate(r.per_strike):
            sp, sm = b.per_strike[i]["se"], d["se"]
            d["VRF"] = sp ** 2 / sm ** 2
            d["Gain"] = (b.total_time * sp ** 2) / (r.total_time * sm ** 2)
    return results
