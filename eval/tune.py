"""Grid search on the TUNE split only (spec §2), objective = mean nDCG@10.
  uv run python eval/tune.py --config C6
Writes results/tune_<config>.csv (every grid point) and prints the best setting."""
import argparse
import csv
import itertools

import run_eval as E
from convsearch import data, pipeline

GRID = {
    "w_idf": [True, False],
    "beta": [0.5, 1.0, 2.0],
    "alpha": [0.0, 0.3],
    "lam": [0.0, 0.05, 0.1, 0.2],
    "tau_idf": [8.0, 12.0, 16.0],
    "tau_j": [0.1, 0.2],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="C6")
    a = ap.parse_args()
    ix, qrels = pipeline.index(), data.qrels()
    qtexts = {v: data.queries(v) for v in ("lastturn", "rewrite")}
    convs = data.split("tune")
    keys = list(GRID)
    if a.config in ("C4",):                         # no gate / no Boolean -> those knobs are inert
        keys = ["w_idf", "beta", "alpha"]
    rows = []
    for vals in itertools.product(*(GRID[k] for k in keys)):
        E.OVERRIDES.clear()
        E.OVERRIDES.update(dict(zip(keys, vals)))
        mean, _ = E.run_config(a.config, convs, ix, qrels, qtexts)
        rows.append({**E.OVERRIDES, **{k: round(v, 4) for k, v in mean.items()}})
        print(len(rows), E.OVERRIDES, f"nDCG@10 {mean['nDCG@10']:.4f}", flush=True)
    rows.sort(key=lambda r: -r["nDCG@10"])
    with open(E.RESULTS / f"tune_{a.config}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print("best:", rows[0])
