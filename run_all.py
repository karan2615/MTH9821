"""Reproduce every table and figure, then (optionally) compile the report.

    python run_all.py            # tables -> results/, figures -> figures/
    python run_all.py --report   # also compiles report/report.pdf (needs latexmk)
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

for script in ("p1_validation.py", "p5_experiments.py"):
    print(f"== {script}")
    subprocess.run([sys.executable, script], cwd=SRC, check=True)

if "--report" in sys.argv:
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                    "report.tex"], cwd=os.path.join(ROOT, "report"), check=True)
