# MTH 9821 HW3: American options, policy evaluation and fitted Q-learning

Team 9: Jaskaran Kalra, William McDonnell

Bermudan put (S0 = K = 100, r = 0.05, sigma = 0.2, T = 1, N = 50). Longstaff–Schwartz and backward
fitted Q-learning with polynomial bases of degree 2 and 4, sharing one simulator, one regression
routine and one forward evaluator. Each frozen policy is evaluated on an independent path set.

## Run

```bash
pip install -r requirements.txt
python src/run_all.py && python src/make_tables.py     # about 10 s on one CPU
cd report && latexmk -pdf report.tex                    # optional: rebuild the report
```

## Package versions used

Python 3.11.15, NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.9 (also recorded in `results/checks.json`).

## Outputs

| Path | Contents |
|---|---|
| `results.csv` | 12 rows: seed, method, degree, fitted initial value, policy value, SE, 95% CI, gap to reference, train/eval times, fallback count |
| `paired.csv` | 6 rows: FQ − LSM per seed and degree (mean, SE, 95% CI) on common evaluation paths |
| `results/checks.json` | two-period check, European vs Black–Scholes, independent CRR check, loop vs vectorized evaluator, n = 25 exercise regions, fallbacks, environment |
| `fitted/` | regression coefficients per date, method, degree and training seed (`.npz`) |
| `figures/n25_decisions.pdf` | continuation curves and exercise points at n = 25, seed 17 |
| `report/report.pdf` | report |
| `slides/` | 5-minute presentation |

## Code

| File | Contents |
|---|---|
| `src/amop.py` | simulator (3), basis (5), least squares by `lstsq`, clipping (6), rule (7), LSM (8), fitted Q (10)–(12), forward evaluator (13) |
| `src/run_all.py` | Part 1(3) and all of Part 4 |
| `src/make_tables.py` | LaTeX tables for the report |

## Reproducibility

Training paths: `numpy.random.default_rng(seed)` with seeds 17, 29, 43 (20,000 paths each), drawn as one
(paths × 50) standard-normal array, row = path, column = step. Evaluation: seed 20261008, 100,000 paths,
same layout. All values are deterministic given the seeds; run times vary by machine.
