"""Shared helpers: output paths, plotting style, environment record, warm-up."""
import json
import os
import platform
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# Categorical palette (fixed order): blue, orange, aqua, yellow.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]


def mpl_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6,
        "axes.edgecolor": "#888888", "lines.linewidth": 1.4, "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.dpi": 200,
    })
    return plt


def save_json(name, obj):
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=2, default=float)


def write_tex(name, text):
    with open(os.path.join(RESULTS, name), "w") as f:
        f.write(text)


def environment():
    import scipy
    try:
        cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        model = next((l.split(":", 1)[1].strip() for l in cpu.splitlines()
                      if l.startswith("Model name")), platform.processor())
    except Exception:
        model = platform.processor()
    return dict(python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, os=platform.platform(),
                cpu=model, logical_cpus=os.cpu_count())


def warm_up():
    """One-time library initialisation (BLAS threads, ufunc dispatch), excluded
    from all timings as the timing protocol requires."""
    from model import Params, make_scheme, simulate_terminal
    sc = make_scheme(Params())
    rng = np.random.default_rng(0)
    z = rng.standard_normal((3, 200, sc.p.n))
    simulate_terminal(sc, *z)
