"""Collect the numbers shown on the slides into slides/deck_data.json."""
import csv, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rd = lambda p: list(csv.DictReader(open(os.path.join(ROOT, p))))
num = lambda r: {k: (float(v) if k not in ("method",) else v) for k, v in r.items()}
out = dict(results=[num(r) for r in rd("results.csv")],
           paired=[num(r) for r in rd("paired.csv")],
           checks=json.load(open(os.path.join(ROOT, "results/checks.json"))))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "deck_data.json"), "w"), indent=1)
print("ok")
