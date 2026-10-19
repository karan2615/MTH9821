"""Reproduce every number, table and figure of the report.

    python src/run_all.py

Writes results.csv (12 rows), paired.csv (6 rows) at the repository root, and
supporting outputs in results/, figures/ and fitted/.
"""
import csv
import json
import os
import platform
import time

import numpy as np

import amop as A

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG, FIT = (os.path.join(ROOT, d) for d in ("results", "figures", "fitted"))
for d in (RES, FIG, FIT):
    os.makedirs(d, exist_ok=True)

TRAIN_SEEDS = (17, 29, 43)
N_TRAIN = 20_000
EVAL_SEED = 20261008
N_EVAL = 100_000
DEGREES = (2, 4)
METHODS = {"LSM": A.train_lsm, "FQ": A.train_fq}


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def reference():
    with open(os.path.join(ROOT, "reference_values.csv")) as f:
        rows = list(csv.DictReader(f))
    return rows


# ---------------------------------------------------------------- Part 1(3) and 4(1) check
def two_period():
    """Two-period binomial put: S0 = K = 100, u = 1.2, d = 0.8, gross rate 1.05."""
    s0, k, u, dn, R = 100.0, 100.0, 1.2, 0.8, 1.05
    q = (R - dn) / (u - dn)
    g = lambda s: max(k - s, 0.0)
    V2 = {"uu": g(s0 * u * u), "ud": g(s0 * u * dn), "dd": g(s0 * dn * dn)}
    C1u = (q * V2["uu"] + (1 - q) * V2["ud"]) / R
    C1d = (q * V2["ud"] + (1 - q) * V2["dd"]) / R
    V1u, V1d = max(g(s0 * u), C1u), max(g(s0 * dn), C1d)
    C0 = (q * V1u + (1 - q) * V1d) / R
    V0 = max(g(s0), C0)
    # Evaluator check: enumerate the four paths with risk-neutral weights.
    paths = np.array([[s0, s0 * u, s0 * u * u], [s0, s0 * u, s0 * u * dn],
                      [s0, s0 * dn, s0 * dn * u], [s0, s0 * dn, s0 * dn * dn]])
    w = np.array([q * q, q * (1 - q), (1 - q) * q, (1 - q) ** 2])
    C1 = {round(s0 * u, 8): C1u, round(s0 * dn, 8): C1d}
    rule = lambda n, s: np.array([(g(x) > 0) and (g(x) >= C1[round(x, 8)]) for x in s])
    H, tau = A.evaluate(paths, rule, exercise_at_0=(g(s0) > 0 and g(s0) >= C0), beta=1 / R)
    return dict(q=q, C1_up=C1u, C1_down=C1d, action_up="continue" if C1u >= g(s0 * u) else "exercise",
                action_down="exercise" if g(s0 * dn) >= C1d else "continue", C0=C0, V0=V0,
                evaluator_value=float(w @ H), path_tau=tau.tolist(), path_H=H.tolist(),
                path_weights=w.tolist())


# ---------------------------------------------------------------- independent reference check
def crr_bermudan(steps, n_ex=A.N):
    """Independent CRR tree: exercise only every steps/n_ex tree steps (Bermudan)."""
    dt = A.T / steps
    u = np.exp(A.SIGMA * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(A.R * dt) - d) / (u - d)
    disc = np.exp(-A.R * dt)
    j = np.arange(steps + 1)
    V = A.payoff(A.S0 * u ** (steps - 2 * j))
    stride = steps // n_ex
    for i in range(steps - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if i % stride == 0:
            jj = np.arange(i + 1)
            V = np.maximum(V, A.payoff(A.S0 * u ** (i - 2 * jj)))
    return float(V[0])


def main():
    ref = reference()
    ref_fine = float(ref[-1]["bermudan_put"])
    ref_change = float(ref[-1]["bermudan_change_from_previous"])
    bs_eur = float(ref[-1]["european_put_black_scholes"])

    checks = {"two_period": two_period()}
    t0 = time.perf_counter()
    checks["crr_2500_independent"] = crr_bermudan(2500)
    checks["crr_2500_supplied"] = float(ref[0]["bermudan_put"])
    checks["crr_time_s"] = time.perf_counter() - t0

    # ---------------- evaluation set (one, shared by all twelve policies)
    t0 = time.perf_counter()
    S_eval = A.simulate(EVAL_SEED, N_EVAL)
    eval_sim_time = time.perf_counter() - t0
    eur = A.summarize(A.BETA ** A.N * A.payoff(S_eval[:, -1]))
    eur.update(black_scholes=bs_eur, z=(eur["mean"] - bs_eur) / eur["se"])
    checks["european"] = eur

    results, H_store, policies, sim_times = [], {}, {}, {}
    for seed in TRAIN_SEEDS:
        t0 = time.perf_counter()
        S = A.simulate(seed, N_TRAIN)
        sim_times[seed] = time.perf_counter() - t0
        for d in DEGREES:
            for m, trainer in METHODS.items():
                t0 = time.perf_counter()
                pol = trainer(S, d, seed)
                t_train = time.perf_counter() - t0
                t0 = time.perf_counter()
                H, tau = A.evaluate(S_eval, pol.exercise, pol.exercise_at_0)
                t_eval = time.perf_counter() - t0
                st = A.summarize(H)
                H_store[(seed, d, m)] = H
                policies[(seed, d, m)] = pol
                np.savez(os.path.join(FIT, f"{m}_d{d}_seed{seed}.npz"),
                         coefs=np.array([np.full(d + 1, np.nan)] + pol.coefs[1:]),
                         c0=pol.c0, fitted_value=pol.fitted_value,
                         exercise_at_0=pol.exercise_at_0)
                results.append(dict(
                    seed=seed, method=m, degree=d, fitted_value=pol.fitted_value,
                    policy_value=st["mean"], se=st["se"], ci_lo=st["ci_lo"], ci_hi=st["ci_hi"],
                    gap_to_reference=st["mean"] - ref_fine,
                    train_time_s=t_train, train_path_sim_time_s=sim_times[seed],
                    eval_time_s=t_eval, eval_path_sim_time_s=eval_sim_time,
                    fallbacks=len(pol.fallbacks)))
    write_csv(os.path.join(ROOT, "results.csv"), results)

    # loop-based evaluator (path by path) vs the vectorized one, first 2,000 evaluation paths
    pol = policies[(17, 4, "LSM")]
    sub = S_eval[:2000]
    H_vec, _ = A.evaluate(sub, pol.exercise, pol.exercise_at_0)
    H_loop = np.empty(len(sub))
    for i, path in enumerate(sub):
        n = 1
        while n < A.N and not pol.exercise(n, path[n:n + 1])[0]:
            n += 1
        H_loop[i] = A.BETA ** n * A.payoff(path[n])
    checks["loop_evaluator_max_abs_diff"] = float(np.abs(H_vec - H_loop).max())

    paired = []
    for seed in TRAIN_SEEDS:
        for d in DEGREES:
            st = A.summarize(H_store[(seed, d, "FQ")] - H_store[(seed, d, "LSM")])
            paired.append(dict(seed=seed, degree=d, mean_diff_FQ_minus_LSM=st["mean"],
                               se=st["se"], ci_lo=st["ci_lo"], ci_hi=st["ci_hi"]))
    write_csv(os.path.join(ROOT, "paired.csv"), paired)

    # ---------------- Part 4(3): decision plot, seed 17, n = 25
    S17 = A.simulate(17, N_TRAIN)
    lo, hi = np.percentile(S17[:, 25], [5, 95])
    grid = np.linspace(lo, hi, 200)
    curves = {f"{m} d={d}": policies[(17, d, m)].continuation(25, grid)
              for d in DEGREES for m in METHODS}
    regions = {k: policies[(17, int(k[-1]), k.split()[0])].exercise(25, grid) for k in curves}
    plot_n25(grid, curves, regions)
    np.savez(os.path.join(RES, "n25_curves.npz"), grid=grid, **{k.replace(" ", "_"): v for k, v in curves.items()})
    region_summary = {}
    for k, ex in regions.items():
        edges = np.flatnonzero(np.diff(ex.astype(int)))          # switches along the grid
        region_summary[k] = dict(n_switches=int(len(edges)),
                                 exercise_up_to=float(grid[ex].max()) if ex.any() else None,
                                 n_exercise_points=int(ex.sum()))
    checks["n25_regions"] = region_summary
    checks["fallbacks"] = {f"{m}_d{d}_seed{s}": p.fallbacks for (s, d, m), p in policies.items()}
    checks["reference"] = dict(bermudan_fine=ref_fine, last_change=ref_change, european_bs=bs_eur)
    import scipy, matplotlib
    checks["environment"] = dict(python=platform.python_version(), numpy=np.__version__,
                                 scipy=scipy.__version__, matplotlib=matplotlib.__version__,
                                 platform=platform.platform())
    with open(os.path.join(RES, "checks.json"), "w") as f:
        json.dump(checks, f, indent=2, default=float)

    for r in results:
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    for r in paired:
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    print(json.dumps({k: checks[k] for k in ("two_period", "european", "crr_2500_independent",
                                             "crr_2500_supplied", "n25_regions")}, indent=1, default=float))


def plot_n25(grid, curves, regions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"LSM": "#2a78d6", "FQ": "#eb6834"}
    styles = {2: "-", 4: "--"}
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.6, 4.2), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3.2, 1]))
    ax.plot(grid, A.payoff(grid), color="#222222", lw=1.6, label="payoff g(s)")
    for k, c in curves.items():
        m, d = k.split()[0], int(k[-1])
        ax.plot(grid, c, color=colors[m], ls=styles[d], lw=1.4, label=f"{m}, degree {d}")
    ax.set_ylabel("value at n = 25")
    ax.set_ylim(-0.5, 25)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="#e5e5e5", lw=0.6)
    for i, (k, ex) in enumerate(regions.items()):
        m, d = k.split()[0], int(k[-1])
        axr.scatter(grid[ex], np.full(ex.sum(), i), s=4, color=colors[m], marker="s")
    axr.set_yticks(range(len(regions)), list(regions), fontsize=7)
    axr.set_xlabel("stock price s")
    axr.set_title("exercise points (rule 7)", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "n25_decisions.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "n25_decisions.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
