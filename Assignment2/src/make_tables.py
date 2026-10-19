"""LaTeX tables for the report from results.csv and paired.csv."""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rd = lambda p: list(csv.DictReader(open(os.path.join(ROOT, p))))
f = lambda x, k=4: f"{float(x):.{k}f}"

res = rd("results.csv")
L = [r"\begin{tabular}{rllrrrcrr}", r"\toprule",
     r"seed & method & $d$ & fitted $V_0$ & policy value & SE & 95\% CI & train (s) & eval (s) \\", r"\midrule"]
for i, r in enumerate(res):
    L.append(f"{r['seed']} & {r['method']} & {r['degree']} & {f(r['fitted_value'])} & {f(r['policy_value'])} & "
             f"{f(r['se'])} & [{f(r['ci_lo'])}, {f(r['ci_hi'])}] & {f(r['train_time_s'],2)} & {f(r['eval_time_s'],2)} \\\\")
    if i % 4 == 3 and i < len(res) - 1:
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "results", "tab_results.tex"), "w").write("\n".join(L) + "\n")

pr = rd("paired.csv")
L = [r"\begin{tabular}{rlrrc}", r"\toprule",
     r"seed & $d$ & FQ $-$ LSM & SE & 95\% CI \\", r"\midrule"]
for r in pr:
    L.append(f"{r['seed']} & {r['degree']} & {f(r['mean_diff_FQ_minus_LSM'])} & {f(r['se'])} & "
             f"[{f(r['ci_lo'])}, {f(r['ci_hi'])}] \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "results", "tab_paired.tex"), "w").write("\n".join(L) + "\n")
print("ok")
