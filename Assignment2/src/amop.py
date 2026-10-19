"""Bermudan put: shared simulator, regression routine, policy, forward evaluator,
and the two learners (Longstaff-Schwartz and backward fitted Q-learning).

Conventions (assignment eqs. 1-7):
  S0 = K = 100, r = 0.05, sigma = 0.20, T = 1, N = 50, h = T/N, t_n = n h.
  g(s) = max(K - s, 0), beta = exp(-r h).
  Exercise allowed at n = 0..N; maturity forces settlement.
  Interior rule (7): exercise iff g(s) > 0 and g(s) >= C_hat_n(s).
"""
from dataclasses import dataclass, field

import numpy as np

S0, K, R, SIGMA, T, N = 100.0, 100.0, 0.05, 0.20, 1.0, 50
H = T / N
BETA = np.exp(-R * H)


def payoff(s):
    return np.maximum(K - s, 0.0)


# ------------------------------------------------------------------ simulator (3)
def simulate(seed, n_paths, s0=S0, n_steps=N):
    """Paths S_0..S_N, shape (n_paths, N+1), from default_rng(seed).
    Shocks drawn as one (n_paths, N) standard-normal array: row = path, column = step."""
    z = np.random.default_rng(seed).standard_normal((n_paths, n_steps))
    inc = (R - 0.5 * SIGMA ** 2) * H + SIGMA * np.sqrt(H) * z
    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = s0
    S[:, 1:] = s0 * np.exp(np.cumsum(inc, axis=1))
    return S


# ------------------------------------------------------------------ regression (5)-(6)
def basis(s, d):
    z = s / K - 1.0
    return np.vander(z, d + 1, increasing=True)          # 1, z, ..., z^d


def fit(s, y, d):
    """Least squares by SVD (numpy.linalg.lstsq); no normal-equation inverse."""
    coef, *_ = np.linalg.lstsq(basis(s, d), y, rcond=None)
    return coef


def cont_value(coef, s):
    """Clipped continuation prediction (6): min(K, max(0, phi(s) b))."""
    d = len(coef) - 1
    return np.clip(basis(s, d) @ coef, 0.0, K)


# ------------------------------------------------------------------ fitted policy
@dataclass
class Policy:
    method: str
    degree: int
    seed: int
    coefs: list = field(default_factory=list)       # coefs[n] for n = 1..N-1 (index 0 unused)
    exercise_at_0: bool = False
    fitted_value: float = np.nan
    fallbacks: list = field(default_factory=list)   # (date, kind)

    def continuation(self, n, s):
        return cont_value(self.coefs[n], s)

    def exercise(self, n, s):
        """Rule (7) at interior dates 0 < n < N."""
        g = payoff(s)
        return (g > 0) & (g >= self.continuation(n, s))


def _ls_step(policy, n, s_itm, y_itm, d):
    """LSM regression with the assignment's small-sample fallbacks."""
    m = len(s_itm)
    if m >= d + 1:
        return fit(s_itm, y_itm, d)
    coef = np.zeros(d + 1)
    if m > 0:
        coef[0] = y_itm.mean()
        policy.fallbacks.append((n, "mean"))
    else:
        coef[0] = K
        policy.fallbacks.append((n, "none_itm"))
    return coef


# ------------------------------------------------------------------ Longstaff-Schwartz (Part 2)
def train_lsm(S, d, seed=None):
    n_paths = S.shape[0]
    pol = Policy("LSM", d, seed, coefs=[None] * N)
    tau = np.full(n_paths, N)                       # planned exercise index, initialized to N
    idx = np.arange(n_paths)
    for n in range(N - 1, 0, -1):
        Y = BETA ** (tau - n) * payoff(S[idx, tau])  # (8): all responses before any update
        s = S[:, n]
        itm = payoff(s) > 0
        coef = _ls_step(pol, n, s[itm], Y[itm], d)
        pol.coefs[n] = coef
        ex = pol.exercise(n, s)                     # (7); only ITM paths can satisfy g > 0
        tau = np.where(ex, n, tau)
    Y0 = BETA ** tau * payoff(S[idx, tau])          # (8) with n = 0
    c0 = Y0.mean()
    g0 = float(payoff(np.array([S[0, 0]]))[0])
    pol.fitted_value = max(c0, g0)
    pol.exercise_at_0 = (g0 > 0) and (g0 >= c0)
    pol.c0 = c0
    return pol


# ------------------------------------------------------------------ fitted Q-learning (Part 3)
def train_fq(S, d, seed=None):
    pol = Policy("FQ", d, seed, coefs=[None] * N)
    V_next = payoff(S[:, N])                        # (10): V_N(s) = g(s) on the training paths
    for n in range(N - 1, 0, -1):
        Y = BETA * V_next                           # (11), all paths
        s = S[:, n]
        coef = fit(s, Y, d)
        pol.coefs[n] = coef
        V_next = np.maximum(payoff(s), cont_value(coef, s))   # (12) with clipped C_hat (6)
    c0 = float(np.clip((BETA * V_next).mean(), 0.0, K))       # date 0: clipped average
    g0 = float(payoff(np.array([S[0, 0]]))[0])
    pol.fitted_value = max(c0, g0)
    pol.exercise_at_0 = (g0 > 0) and (g0 >= c0)
    pol.c0 = c0
    return pol


# ------------------------------------------------------------------ forward evaluator (13)
def evaluate(S, exercise_rule, exercise_at_0, beta=BETA):
    """Discounted payoff H_i = beta^tau_i g(S_tau_i) of a frozen policy.

    exercise_rule(n, s) -> bool array, used at interior dates 0 < n < last;
    the last column is forced settlement. Shared by both methods."""
    n_paths, n_last = S.shape[0], S.shape[1] - 1
    if exercise_at_0:
        return np.full(n_paths, float(payoff(S[0:1, 0])[0])), np.zeros(n_paths, int)
    tau = np.full(n_paths, n_last)
    alive = np.ones(n_paths, bool)
    for n in range(1, n_last):
        ex = np.zeros(n_paths, bool)
        ex[alive] = exercise_rule(n, S[alive, n])
        tau[ex] = n
        alive &= ~ex
    H = beta ** tau * payoff(S[np.arange(n_paths), tau])
    return H, tau


def summarize(x):
    m = len(x)
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(m)
    return dict(mean=mean, se=se, ci_lo=mean - 1.96 * se, ci_hi=mean + 1.96 * se)
