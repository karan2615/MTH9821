# MTH 9821 Homework: Variance reduction in a rough volatility model

Rough Bergomi simulator (hybrid scheme, assignment eqs. 7–12) with plain Monte Carlo,
a Black–Scholes control variate, antithetic sampling, their combination, and the bonus
conditional Monte Carlo estimator. Python 3.11 + NumPy/SciPy.

## Reproduce

```bash
pip install -r requirements.txt
python run_all.py            # all tables -> results/, all figures -> figures/  (~1.5 min, 2 CPUs)
python run_all.py --report   # also builds report/report.pdf (needs latexmk + pdflatex)
```

Individual parts:

```bash
cd src
python p1_validation.py      # Problems 1(c), 1(d), 2(c)
python p5_experiments.py     # Problem 5(a)-(d) and Bonus (c)
```

## Layout

| Path | Contents |
|---|---|
| `src/model.py` | Parameters, BS put (6), scheme weights/variances (8)–(10), simulation (9), (11), (12), control (14), conditional payoff (20)–(21) |
| `src/estimators.py` | Random streams, batched moment accumulation, the four estimators, conditional estimator, VRF/Gain (19) |
| `src/p1_validation.py` | Simulator checks: moments at j = 32, 64, 128; variance-path plot; η = 0 vs Black–Scholes; η = 0, β = 1 identity |
| `src/p5_experiments.py` | Baseline comparison, CI plot, sensitivity (η = 0.5, H = 0.4), bonus, environment record |
| `results/` | CSV and LaTeX tables, `numbers.tex` (values quoted in the report text), `environment.json` |
| `figures/` | `p1c_variance_paths.pdf`, `p5b_ci.pdf` |
| `report/report.tex`, `report.pdf` | Report |

## Reproducibility

- Every random stream is `numpy.random.PCG64(SeedSequence(9821, spawn_key=key))`, with a distinct
  key per experiment / method / stage (pilot = 0, production = 1). The keys are listed in report §5(d).
- Batches of 10,000 path evaluations (5,000 antithetic pairs). Statistics are accumulated per batch;
  no path arrays are written.
- Estimates are deterministic given the seeds. Run times are not. Each timed run is repeated 5 times
  with the same seeds, and the median is reported, so VRF is reproducible exactly while Gain varies
  slightly between machines and runs.
