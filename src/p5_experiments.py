"""Problem 5 (a)-(d) and Bonus (c).

Outputs
  results/p5a_baseline.csv, p5a_prices.tex, p5a_efficiency.tex
  figures/p5b_ci.pdf
  results/p5c_sensitivity.csv, p5c_sensitivity.tex
  results/bonus_conditional.csv, bonus_conditional.tex
  results/environment.json
"""
import csv
import os

import numpy as np

from common import (RESULTS, FIGURES, COLORS, MARKERS, mpl_style, write_tex,
                    save_json, environment, warm_up)
from estimators import (plain, control_variate, antithetic, combined, conditional,
                        add_efficiency, MASTER_SEED, BATCH_PATHS)
from model import Params, STRIKES, make_scheme

N_PROD = 100_000          # production path evaluations per method, Problem 5(a)
N_PILOT = 10_000          # pilot path evaluations (CV: paths; AV+CV: 5,000 pairs)
N_PROD_SENS = 100_000     # Problem 5(c): "at least 50,000"
N_BONUS = 100_000         # Bonus (c): independent volatility-driver samples
METHOD_IDS = {"Plain": 0, "CV": 1, "AV": 2, "AV+CV": 3}
TIMING_REPS = 5           # each timed run is repeated with the same seeds; the
                          # estimates are identical, the reported time is the median


def timed(fn, *args):
    """Run fn REPS times (identical seeds -> identical estimates). Keep the first
    result, replace its pilot/production times by the medians, record the spread."""
    runs = [fn(*args) for _ in range(TIMING_REPS)]
    r = runs[0]
    tot = np.array([x.total_time for x in runs])
    r.pilot_time = float(np.median([x.pilot_time for x in runs]))
    r.prod_time = float(np.median([x.prod_time for x in runs]))
    r.extra.update(t_min=float(tot.min()), t_max=float(tot.max()))
    return r


def run_four(p, strikes, n_prod, experiment):
    """All four methods; each method has its own pilot/production streams with
    keys (experiment, method_id, stage), stage 0 = pilot, 1 = production."""
    sc = make_scheme(p)
    out = [timed(plain, sc, strikes, n_prod, (experiment, 0)),
           timed(control_variate, sc, strikes, n_prod, N_PILOT, (experiment, 1)),
           timed(antithetic, sc, strikes, n_prod, (experiment, 2)),
           timed(combined, sc, strikes, n_prod, N_PILOT, (experiment, 3))]
    return add_efficiency(out)


def predicted_vrf(method, d):
    """Large-sample VRF implied by the relevant correlations (coefficient
    estimation error ignored): CV 1/(1-r^2), AV 1/(1+r_pair),
    AV+CV 1/((1+r_pair)(1-r_{A,C_A}^2))."""
    if method == "CV":
        return 1 / (1 - d["corr_XC"] ** 2)
    if method == "AV":
        return 1 / (1 + d["corr_pair"])
    if method == "AV+CV":
        return 1 / ((1 + d["corr_pair"]) * (1 - d["corr_ACA"] ** 2))
    return 1.0


def flatten(results, label=""):
    rows = []
    for r in results:
        for K, d in zip(r.strikes, r.per_strike):
            rows.append(dict(case=label, method=r.method, K=K, price=d["price"], se=d["se"],
                             ci_lo=d["ci_lo"], ci_hi=d["ci_hi"], m_obs=d["m"],
                             path_evals_prod=r.path_evals_prod,
                             path_evals_pilot=r.path_evals_pilot,
                             pilot_time_s=r.pilot_time, prod_time_s=r.prod_time,
                             total_time_s=r.total_time,
                             t_min_s=r.extra["t_min"], t_max_s=r.extra["t_max"],
                             coef=d.get("coef", np.nan),
                             corr_XC=d.get("corr_XC", np.nan),
                             corr_pair=d.get("corr_pair", np.nan),
                             corr_ACA=d.get("corr_ACA", np.nan),
                             ctrl_z=d.get("ctrl_z", np.nan),
                             VRF=d["VRF"], Gain=d["Gain"],
                             VRF_pred=predicted_vrf(r.method, d)))
    return rows


def write_csv(name, rows):
    with open(os.path.join(RESULTS, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def f(x, fmt):
    return "--" if (x is None or (isinstance(x, float) and np.isnan(x))) else format(x, fmt)


def tex_prices(rows):
    L = [r"\begin{tabular}{llrrcr}", r"\toprule",
         r"$K$ & Method & Price & SE & 95\% CI & $m$ \\", r"\midrule"]
    for K in STRIKES:
        for r in (x for x in rows if x["K"] == K):
            L.append(f"{K:.0f} & {r['method']} & {r['price']:.4f} & {r['se']:.4f} & "
                     f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] & {r['m_obs']:,} \\\\")
        L.append(r"\midrule" if K != STRIKES[-1] else r"\bottomrule")
    L.append(r"\end{tabular}")
    return "\n".join(L) + "\n"


def tex_efficiency(rows):
    L = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r"$K$ & Method & coef. & $\widehat\rho_{X,C}$ & $\widehat\rho_{X^+,X^-}$ & "
         r"$\widehat\rho_{A,C_A}$ & VRF & VRF$_{\rm pred}$ & Gain \\", r"\midrule"]
    for K in STRIKES:
        for r in (x for x in rows if x["K"] == K):
            L.append(f"{K:.0f} & {r['method']} & {f(r['coef'], '.4f')} & "
                     f"{f(r['corr_XC'], '.4f')} & {f(r['corr_pair'], '+.4f')} & "
                     f"{f(r['corr_ACA'], '.4f')} & {r['VRF']:.2f} & "
                     f"{r['VRF_pred']:.2f} & {r['Gain']:.2f} \\\\")
        L.append(r"\midrule" if K != STRIKES[-1] else r"\bottomrule")
    L.append(r"\end{tabular}")
    return "\n".join(L) + "\n"


def tex_timing(results):
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Method & pilot evals & prod.\ evals & pilot (s) & prod.\ (s) & total (s) "
         r"& total range (s) \\", r"\midrule"]
    for r in results:
        L.append(f"{r.method} & {r.path_evals_pilot:,} & {r.path_evals_prod:,} & "
                 f"{r.pilot_time:.3f} & {r.prod_time:.3f} & {r.total_time:.3f} & "
                 f"{r.extra['t_min']:.3f}--{r.extra['t_max']:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def ci_plot(rows):
    plt = mpl_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    methods = list(METHOD_IDS)
    for ax, K in zip(axes, STRIKES):
        for i, mth in enumerate(methods):
            r = next(x for x in rows if x["K"] == K and x["method"] == mth)
            ax.errorbar(i, r["price"], yerr=1.96 * r["se"], fmt=MARKERS[i], color=COLORS[i],
                        ms=5, capsize=3, lw=1.4)
        ax.set_xticks(range(4), methods, fontsize=7)
        ax.set_xlim(-0.5, 3.5)
        ax.set_title(f"K = {K:.0f}")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("price estimate, 95% CI")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "p5b_ci.pdf"))
    plt.close(fig)


def tex_sensitivity(rows):
    L = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
         r"Case & Method & Price & SE & coef. & $\widehat\rho_{X,C}$ & "
         r"$\widehat\rho_{X^+,X^-}$ & $\widehat\rho_{A,C_A}$ & VRF & Gain \\", r"\midrule"]
    cases = list(dict.fromkeys(r["case"] for r in rows))
    for c in cases:
        for r in (x for x in rows if x["case"] == c):
            L.append(f"{c} & {r['method']} & {r['price']:.4f} & {r['se']:.4f} & "
                     f"{f(r['coef'], '.4f')} & {f(r['corr_XC'], '.4f')} & "
                     f"{f(r['corr_pair'], '+.4f')} & {f(r['corr_ACA'], '.4f')} & "
                     f"{r['VRF']:.2f} & {r['Gain']:.2f} \\\\")
        L.append(r"\midrule" if c != cases[-1] else r"\bottomrule")
    L.append(r"\end{tabular}")
    return "\n".join(L) + "\n"


MCODE = {"Plain": "PL", "CV": "CV", "AV": "AV", "AV+CV": "BOTH", "Conditional": "COND"}
KCODE = {90.0: "a", 100.0: "b", 110.0: "c"}


def write_macros(rows, res, sens_rows, bonus_row):
    """LaTeX macros for every number quoted in the report text, so the prose
    always matches the tables of the same run."""
    out = []

    def mac(name, val):
        out.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    for r in rows:
        m, k = MCODE[r["method"]], KCODE[r["K"]]
        mac(f"VRF{m}{k}", f"{r['VRF']:.2f}")
        mac(f"Gain{m}{k}", f"{r['Gain']:.2f}")
        for c, nm in (("corr_XC", "RhoXC"), ("corr_pair", "RhoPair"), ("corr_ACA", "RhoACA"),
                      ("coef", "Coef"), ("ctrl_z", "CtrlZ")):
            if not np.isnan(r[c]):
                mac(f"{nm}{m}{k}", f"{r[c]:.2f}")
    for r in res:
        mac(f"Time{MCODE[r.method]}", f"{r.total_time:.2f}")
    for k in ("c",):
        g = [r["Gain"] for r in rows if KCODE[r["K"]] == k and r["method"] != "Plain"]
        mac(f"GainSpread{k}", f"{100 * (max(g) / min(g) - 1):.0f}")
    for r in sens_rows:
        cs = "E" if "eta" in r["case"] else "H"
        m = MCODE[r["method"]]
        mac(f"S{cs}VRF{m}", f"{r['VRF']:.1f}")
        mac(f"S{cs}Gain{m}", f"{r['Gain']:.1f}")
        for c, nm in (("corr_XC", "RhoXC"), ("corr_pair", "RhoPair"), ("corr_ACA", "RhoACA")):
            if not np.isnan(r[c]):
                mac(f"S{cs}{nm}{m}", f"{r[c]:.3f}" if nm != "RhoPair" else f"{r[c]:.2f}")
    mac("TimeCOND", f"{bonus_row['total_time_s']:.2f}")
    mac("VRFCOND", f"{bonus_row['VRF']:.2f}")
    mac("GainCOND", f"{bonus_row['Gain']:.2f}")
    write_tex("numbers.tex", "\n".join(out) + "\n")


def main():
    warm_up()
    base = Params()

    # ---------------- 5(a) baseline: all three strikes from the same paths per method
    res = run_four(base, STRIKES, N_PROD, experiment=5)
    rows = flatten(res, "baseline")
    write_csv("p5a_baseline.csv", rows)
    write_tex("p5a_prices.tex", tex_prices(rows))
    write_tex("p5a_efficiency.tex", tex_efficiency(rows))
    write_tex("p5a_timing.tex", tex_timing(res))
    ci_plot(rows)

    # ---------------- Bonus (c): conditional MC at K = 100 (all strikes computed
    # from the same samples; only K = 100 is reported)
    sc = make_scheme(base)
    cond = timed(conditional, sc, STRIKES, N_BONUS, (7, 0))
    pl = next(r for r in res if r.method == "Plain")
    i100 = STRIKES.index(100.0)
    d = cond.per_strike[i100]
    sp = pl.per_strike[i100]["se"]
    d["VRF"] = sp ** 2 / d["se"] ** 2
    d["Gain"] = pl.total_time * sp ** 2 / (cond.total_time * d["se"] ** 2)
    bonus_rows = [x for x in rows if x["K"] == 100.0] + [dict(
        case="baseline", method="Conditional", K=100.0, price=d["price"], se=d["se"],
        ci_lo=d["ci_lo"], ci_hi=d["ci_hi"], m_obs=d["m"], path_evals_prod=N_BONUS,
        path_evals_pilot=0, pilot_time_s=0.0, prod_time_s=cond.prod_time,
        total_time_s=cond.total_time, t_min_s=cond.extra["t_min"],
        t_max_s=cond.extra["t_max"], coef=np.nan, corr_XC=np.nan, corr_pair=np.nan,
        corr_ACA=np.nan, ctrl_z=np.nan, VRF=d["VRF"], Gain=d["Gain"], VRF_pred=np.nan)]
    write_csv("bonus_conditional.csv", bonus_rows)
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Method & Price & SE & time (s) & VRF & Gain \\", r"\midrule"]
    for r in bonus_rows:
        L.append(f"{r['method']} & {r['price']:.4f} & {r['se']:.4f} & "
                 f"{r['total_time_s']:.3f} & {r['VRF']:.2f} & {r['Gain']:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write_tex("bonus_conditional.tex", "\n".join(L) + "\n")

    # ---------------- 5(c) sensitivity at K = 100: change one parameter at a time
    sens_rows = []
    for idx, (label, p) in enumerate([(r"$\eta=0.5$", base.with_(eta=0.5)),
                                      (r"$H=0.4$", base.with_(H=0.4))]):
        r = run_four(p, (100.0,), N_PROD_SENS, experiment=60 + idx)
        sens_rows += flatten(r, label)
    write_csv("p5c_sensitivity.csv", sens_rows)
    write_tex("p5c_sensitivity.tex", tex_sensitivity(sens_rows))

    write_macros(rows, res, sens_rows, bonus_rows[-1])

    # ---------------- 5(d) reproducibility record
    env = environment()
    env.update(timing_reps=TIMING_REPS, master_seed=MASTER_SEED, batch_paths=BATCH_PATHS, n_prod=N_PROD,
               n_pilot=N_PILOT, n_prod_sensitivity=N_PROD_SENS, n_bonus=N_BONUS)
    save_json("environment.json", env)

    for r in rows + bonus_rows[-1:] + sens_rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(env)


if __name__ == "__main__":
    main()
