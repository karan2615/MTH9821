"""Rough Bergomi model: hybrid simulation scheme (7)-(12), Black-Scholes put (6),
the Black-Scholes control (14) and the conditional quantities (20)-(21).

Indexing follows the assignment: Gaussian inputs G_j, U_j, D_j for j = 1..n are
stored in column j-1 of arrays of shape (paths, n). Arrays of grid values
Y_j, q_j, V_j for j = 0..n are stored in column j.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.special import ndtr


@dataclass(frozen=True)
class Params:
    S0: float = 100.0
    xi: float = 0.04
    T: float = 0.5
    H: float = 0.1
    eta: float = 1.5
    rho: float = -0.7
    n: int = 128

    def with_(self, **kw):
        return replace(self, **kw)


STRIKES = (90.0, 100.0, 110.0)


# ---------------------------------------------------------------- Black-Scholes
def bs_put(s, K, a):
    """BS_put(s, K, a) of (6); intrinsic value (K - s)^+ where a = 0."""
    scalar = all(np.ndim(x) == 0 for x in (s, K, a))
    s, K, a = (np.atleast_1d(np.asarray(x, float)) for x in (s, K, a))
    s, K, a = np.broadcast_arrays(s, K, a)
    out = np.maximum(K - s, 0.0)
    pos = a > 0
    if np.any(pos):
        sa = np.sqrt(a[pos])
        d1 = (np.log(s[pos] / K[pos]) + 0.5 * a[pos]) / sa
        d2 = d1 - sa
        out[pos] = K[pos] * ndtr(-d2) - s[pos] * ndtr(-d1)
    return float(out[0]) if scalar else out


def bs_put_implied_var_price(p: Params, K):
    """c_K = E[C_K] = BS_put(S0, K, xi*T)  (Problem 2a)."""
    return bs_put(p.S0, K, p.xi * p.T)


# ------------------------------------------------------- scheme precomputation
@dataclass(frozen=True)
class Scheme:
    """Deterministic quantities of the scheme: Delta, a_H, w_k, q_j, and the
    lower-triangular kernel matrix used for the sum in (9)."""
    p: Params
    dt: float
    aH: float
    w: np.ndarray        # w[k] for k = 0..n (entries 0, 1 unused = 0)
    q: np.ndarray        # q_j for j = 0..n
    Kmat: np.ndarray     # (n, n): Kmat[i-1, j-1] = w_{j-i+1} for i < j, else 0


def make_scheme(p: Params) -> Scheme:
    n, H = p.n, p.H
    dt = p.T / n
    aH = np.sqrt(2 * H) / (H + 0.5)                                    # (8)
    k = np.arange(n + 1, dtype=float)
    w = np.zeros(n + 1)
    kk = k[2:]
    w[2:] = (np.sqrt(2 * H) * dt ** (H - 0.5)
             * (kk ** (H + 0.5) - (kk - 1) ** (H + 0.5)) / (H + 0.5))   # (8)
    q = np.zeros(n + 1)                                                # (10)
    q[1:] = dt ** (2 * H) + dt * np.concatenate(([0.0], np.cumsum(w[2:] ** 2)))
    i = np.arange(1, n + 1)
    lag = i[None, :] - i[:, None] + 1          # j - i + 1 for row i, column j
    Kmat = np.where(lag >= 2, w[np.clip(lag, 0, n)], 0.0)
    return Scheme(p, dt, aH, w, q, Kmat)


# ------------------------------------------------------------------ simulation
def simulate_Y(sc: Scheme, G, U):
    """Y_j, j = 0..n, from (9). G, U: (m, n). Returns (m, n+1)."""
    dt, H, aH = sc.dt, sc.p.H, sc.aH
    dW = np.sqrt(dt) * G
    local = dt ** H * (aH * G + np.sqrt(1 - aH ** 2) * U)
    hist = dW @ sc.Kmat                       # sum_{k=2}^j w_k dW_{j-k+1}
    Y = np.zeros((G.shape[0], sc.p.n + 1))
    Y[:, 1:] = local + hist
    return Y


def variance_from_Y(sc: Scheme, Y):
    """V_j = xi exp(eta Y_j - eta^2 q_j / 2), (11)."""
    p = sc.p
    return p.xi * np.exp(p.eta * Y - 0.5 * p.eta ** 2 * sc.q[None, :])


def terminal_log_stock(sc: Scheme, V, G, D):
    """log S_n from the log-Euler update (12), left-endpoint variance V_j,
    j = 0..n-1, with increments dW_{j+1}, dB_{j+1}."""
    p, dt = sc.p, sc.dt
    Vl = V[:, :-1]
    dW = np.sqrt(dt) * G
    dB = np.sqrt(dt) * D
    inc = -0.5 * Vl * dt + np.sqrt(Vl) * (p.rho * dW + np.sqrt(1 - p.rho ** 2) * dB)
    return np.log(p.S0) + inc.sum(axis=1)


def control_log_stock(sc: Scheme, G, D):
    """log S~_T of (14): same increments, constant variance xi."""
    p, dt = sc.p, sc.dt
    dW = np.sqrt(dt) * G
    dB = np.sqrt(dt) * D
    return (np.log(p.S0) - 0.5 * p.xi * p.T
            + np.sqrt(p.xi) * (p.rho * dW + np.sqrt(1 - p.rho ** 2) * dB).sum(axis=1))


def simulate_terminal(sc: Scheme, G, U, D):
    """S_n and S~_T for one batch of Gaussian inputs."""
    V = variance_from_Y(sc, simulate_Y(sc, G, U))
    return np.exp(terminal_log_stock(sc, V, G, D)), np.exp(control_log_stock(sc, G, D))


def put_payoffs(S, strikes):
    """(K - S)^+ for each strike; returns (m, len(strikes))."""
    return np.maximum(np.asarray(strikes)[None, :] - S[:, None], 0.0)


# ------------------------------------------------- bonus: conditional MC (20)-(21)
def conditional_put(sc: Scheme, G, U, strikes):
    """Q_K of (21) given G, U (the D inputs are integrated out exactly)."""
    p, dt = sc.p, sc.dt
    V = variance_from_Y(sc, simulate_Y(sc, G, U))[:, :-1]          # V_0..V_{n-1}
    I = dt * V.sum(axis=1)                                          # (20)
    J = (np.sqrt(V) * np.sqrt(dt) * G).sum(axis=1)                  # (20)
    s = p.S0 * np.exp(p.rho * J - 0.5 * p.rho ** 2 * I)
    a = (1 - p.rho ** 2) * I
    return np.stack([bs_put(s, K, a) for K in strikes], axis=1)
