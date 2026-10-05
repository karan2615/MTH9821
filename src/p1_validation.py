"""Problem 1(c), 1(d) and 2(c): simulator validation.

Outputs
  results/p1c_moments.csv, .tex      moments of Y_j and V_j at j = 32, 64, 128
  figures/p1c_variance_paths.pdf     three variance paths, H = 0.1 vs H = 0.4
  results/p1d_eta0_bs.csv, .tex      eta = 0 put prices vs Black-Scholes, several n
  results/p2c_identity.csv, .tex     eta = 0, beta = 1: max |R - c_K|
"""
import csv
import os

import numpy as np

from common import RESULTS, FIGURES, COLORS, MARKERS, mpl_style, write_tex
from estimators import stream, gaussian_inputs, Moments, BATCH_PATHS, _batches
from model import (Params, STRIKES, make_scheme, simulate_Y, variance_from_Y,
                   simulate_terminal, put_payoffs, bs_put, bs_put_implied_var_price)


def p1c(n_paths=100_000, js=(32, 64, 128)):
    p = Params()                                      # n = 128, baseline
    sc = make_scheme(p)
    rng = stream(1, 3)
    js = np.array(js)
    mom = Moments(2 * len(js))                        # [Y_j..., V_j...]
    for m in _batches(n_paths, BATCH_PATHS):
        G, U, _ = gaussian_inputs(rng, m, p.n)
        Y = simulate_Y(sc, G, U)
        V = variance_from_Y(sc, Y)
        mom.add(np.hstack([Y[:, js], V[:, js]]))
    var = mom.var()
    rows = []
    for i, j in enumerate(js):
        k = len(js) + i
        rows.append(dict(j=int(j), t=j * sc.dt,
                         mean_Y=mom.mean[i], se_mean_Y=np.sqrt(var[i] / mom.n),
                         var_Y=var[i], q=sc.q[j], var_Y_over_q=var[i] / sc.q[j],
                         mean_V=mom.mean[k], se_mean_V=np.sqrt(var[k] / mom.n),
                         xi=p.xi))
    _csv("p1c_moments.csv", rows)
    lines = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"$j$ & $t_j$ & $\overline{Y}_j$ (SE) & $s^2(\widehat Y_j)$ & $q_j$ "
             r"& $s^2/q_j$ & $\overline{V}_j$ (SE) \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['j']} & {r['t']:.4f} & {r['mean_Y']:+.4f} ({r['se_mean_Y']:.4f}) & "
                     f"{r['var_Y']:.4f} & {r['q']:.4f} & {r['var_Y_over_q']:.4f} & "
                     f"{r['mean_V']:.5f} ({r['se_mean_V']:.5f}) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("p1c_moments.tex", "\n".join(lines) + "\n")
    return rows, n_paths


def p1c_plot(n_show=3):
    plt = mpl_style()
    rng = stream(1, 4)
    base = Params()
    G, U, _ = gaussian_inputs(rng, n_show, base.n)       # same inputs for both H
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, H in zip(axes, (0.1, 0.4)):
        sc = make_scheme(base.with_(H=H))
        V = variance_from_Y(sc, simulate_Y(sc, G, U))
        t = np.arange(base.n + 1) * sc.dt
        for i in range(n_show):
            ax.plot(t, V[i], color=COLORS[i], label=f"path {i + 1}")
        ax.axhline(base.xi, color="#888888", lw=0.8, ls="--", label=r"$\xi$")
        ax.set_title(f"H = {H}")
        ax.set_xlabel("t (years)")
    axes[0].set_ylabel(r"$\widehat V_j$")
    axes[1].legend(loc="upper right", fontsize=7)
    fig.savefig(os.path.join(FIGURES, "p1c_variance_paths.pdf"))
    plt.close(fig)


def p1d(n_paths=100_000, ns=(1, 8, 128)):
    rows = []
    for idx, n in enumerate(ns):
        p = Params(eta=0.0, n=n)
        sc = make_scheme(p)
        rng = stream(1, 5, idx)
        mom = Moments(len(STRIKES))
        for m in _batches(n_paths, BATCH_PATHS):
            S, _ = simulate_terminal(sc, *gaussian_inputs(rng, m, n))
            mom.add(put_payoffs(S, STRIKES))
        var = mom.var()
        for i, K in enumerate(STRIKES):
            bs = bs_put(p.S0, K, p.xi * p.T)
            se = np.sqrt(var[i] / mom.n)
            rows.append(dict(n=n, K=K, mc=mom.mean[i], se=se, bs=bs,
                             z=(mom.mean[i] - bs) / se))
    _csv("p1d_eta0_bs.csv", rows)
    lines = [r"\begin{tabular}{rrrrrr}", r"\toprule",
             r"$n$ & $K$ & MC price & SE & BS (6) & $z$ \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['n']} & {r['K']:.0f} & {r['mc']:.4f} & {r['se']:.4f} & "
                     f"{r['bs']:.4f} & {r['z']:+.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("p1d_eta0_bs.tex", "\n".join(lines) + "\n")
    return rows


def p2c(n_paths=10_000):
    p = Params(eta=0.0)
    sc = make_scheme(p)
    G, U, D = gaussian_inputs(stream(1, 6), n_paths, p.n)
    S, St = simulate_terminal(sc, G, U, D)
    X, C = put_payoffs(S, STRIKES), put_payoffs(St, STRIKES)
    cK = bs_put_implied_var_price(p, np.array(STRIKES))
    R = X - 1.0 * (C - cK[None, :])                       # beta = 1
    rows = [dict(K=K, c_K=cK[i], max_abs_R_minus_cK=np.abs(R[:, i] - cK[i]).max(),
                 sd_R=R[:, i].std(ddof=1)) for i, K in enumerate(STRIKES)]
    rel = np.abs(S / St - 1).max()
    _csv("p2c_identity.csv", rows)
    lines = [r"\begin{tabular}{rrrr}", r"\toprule",
             r"$K$ & $c_K$ & $\max_i |R_i - c_K|$ & $s(R)$ \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['K']:.0f} & {r['c_K']:.6f} & {r['max_abs_R_minus_cK']:.1e} & "
                     f"{r['sd_R']:.1e} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("p2c_identity.tex", "\n".join(lines) + "\n")
    return rows, rel


def _csv(name, rows):
    with open(os.path.join(RESULTS, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    rows, N = p1c()
    print(f"P1(c) moments, {N} paths:")
    for r in rows:
        print({k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()})
    p1c_plot()
    print("P1(d) eta = 0 vs BS:")
    for r in p1d():
        print({k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()})
    rows, rel = p2c()
    print("P2(c) eta = 0, beta = 1:", rows, "max |S_n/S~_T - 1| =", rel)
