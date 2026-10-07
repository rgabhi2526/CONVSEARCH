"""Figures + significance from results/<split>_perturn.json (after run_eval.py).
  uv run python eval/report.py --split test --a C6 --b C2
Writes results/<split>_bar.png, results/<split>_depth.png, results/<split>_significance.txt."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from convsearch import data

RESULTS = Path(__file__).resolve().parents[1] / "results"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--a", default="C6")
    ap.add_argument("--b", default="C2")
    a = ap.parse_args()
    per = json.loads((RESULTS / f"{a.split}_perturn.json").read_text())
    cfgs = list(per)

    # bar chart: nDCG@10 and P@5 per config
    fig, ax = plt.subplots(figsize=(9, 3.5))
    w = 0.4
    for j, m in enumerate(("nDCG@10", "P@5")):
        vals = [sum(v[m] for v in per[c].values()) / len(per[c]) for c in cfgs]
        bars = ax.bar([i + j * w for i in range(len(cfgs))], vals, w, label=m)
        ax.bar_label(bars, fmt="%.3f", fontsize=7)
    ax.set_xticks([i + w / 2 for i in range(len(cfgs))], cfgs)
    ax.set_title(f"MTRAG ClapNQ, {a.split} split ({len(per[cfgs[0]])} turns)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / f"{a.split}_bar.png", dpi=150)

    # nDCG@10 by turn depth
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for c in [c for c in ("C0", "C2", "C3", "C6", "C8", "C9") if c in per]:   # key configs only, readable
        by = defaultdict(list)
        for qid, v in per[c].items():
            by[data.split_id(qid)[1]].append(v["nDCG@10"])
        xs = sorted(by)
        ax.plot(xs, [sum(by[x]) / len(by[x]) for x in xs], marker="o", label=c)
    ax.set_xlabel("turn depth"); ax.set_ylabel("nDCG@10"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{a.split}_depth.png", dpi=150)

    # paired t-test + win/loss, a vs b, and first turns vs follow-ups
    lines = []
    if a.a in per and a.b in per:
        qids = sorted(per[a.a])
        x = [per[a.a][q]["nDCG@10"] for q in qids]
        y = [per[a.b][q]["nDCG@10"] for q in qids]
        t, p = stats.ttest_rel(x, y)
        win = sum(i > j for i, j in zip(x, y)); loss = sum(i < j for i, j in zip(x, y))
        lines.append(f"{a.a} vs {a.b} nDCG@10 over {len(qids)} turns: mean {sum(x)/len(x):.4f} vs {sum(y)/len(y):.4f}, "
                     f"win/tie/loss {win}/{len(qids)-win-loss}/{loss}, paired t = {t:.3f}, p = {p:.4f}")
    for c in cfgs:
        first = [v["nDCG@10"] for q, v in per[c].items() if data.split_id(q)[1] == 1]
        foll = [v["nDCG@10"] for q, v in per[c].items() if data.split_id(q)[1] > 1]
        lines.append(f"{c:6} first turns {sum(first)/len(first):.4f} (n={len(first)})  follow-ups {sum(foll)/len(foll):.4f} (n={len(foll)})")
    (RESULTS / f"{a.split}_significance.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
