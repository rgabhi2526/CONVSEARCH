"""Self-judged query set (brief: "judged queries with P/R/P@k").
TREC-style pooling: the top-5 of every config in CONFIGS are merged into one pool per turn,
shuffled (judges don't see which system found a passage), and each judge labels the pool.

  uv run python eval/judge.py new   --name conv1           # type a conversation, builds the pools
  uv run python eval/judge.py label --name conv1 --judge A # label pooled passages y/n (A and B separately)
  uv run python eval/judge.py score                        # P@5 / nDCG@5 per config + Cohen's kappa

A passage is relevant only if BOTH judges said yes (strict); scores with either-judge are also printed.
Files: judgments/<name>.json (committed)."""
import argparse
import json
import math
import random
from pathlib import Path

from convsearch import pipeline

DIR = Path(__file__).resolve().parents[1] / "judgments"
DEPTH = 5
CONFIGS = {
    "C0": {"query": "turn"},
    "C2": {"query": "rewrite1"},
    "C6": {"query": "fusion", "gate": True, "boolean": True},
    "C8": {"query": "fusion", "gate": True, "boolean": True, "mu": 3.0, "nu": 0.1},
    "C9": {"query": "fusion", "llm": False, "gate": True, "boolean": True},
}


def new(name):
    ix = pipeline.index()
    print("Type the conversation one user turn per line; empty line to finish.")
    turns = []
    while (t := input(f"turn {len(turns) + 1}> ").strip()):
        turns.append(t)
    runs = {c: [] for c in CONFIGS}
    for c, cfg in CONFIGS.items():
        prev = []
        for i, t in enumerate(turns):                  # same in-order protocol as run_eval
            ranked, _ = pipeline.run(t, turns[:i], prev, cfg, ix)
            prev = [d for d, _ in ranked[:10]]
            runs[c].append([ix.doc_ids[d] for d, _ in ranked[:DEPTH]])
    pools = []
    for i in range(len(turns)):
        pool = sorted({p for c in CONFIGS for p in runs[c][i]})
        random.Random(i).shuffle(pool)
        pools.append(pool)
    DIR.mkdir(exist_ok=True)
    (DIR / f"{name}.json").write_text(json.dumps(
        {"turns": turns, "runs": runs, "pools": pools, "labels": {}}, indent=1))
    print(f"saved judgments/{name}.json: {sum(map(len, pools))} passages to judge")


def label(name, judge):
    ix = pipeline.index()
    pos = {d: i for i, d in enumerate(ix.doc_ids)}
    f = DIR / f"{name}.json"
    J = json.loads(f.read_text())
    lab = J["labels"].setdefault(judge, {})
    for i, (turn, pool) in enumerate(zip(J["turns"], J["pools"])):
        print(f"\n=== turn {i + 1}: {turn}\n    (earlier: {' | '.join(J['turns'][:i]) or '-'})")
        for p in pool:
            k = f"{i}|{p}"
            if k in lab:
                continue
            d = pos[p]
            print(f"\n[{ix.titles[d]}] {ix.texts[d][:700]}")
            while (a := input("relevant to this turn? [y/n/q] ").strip().lower()) not in ("y", "n", "q"):
                pass
            if a == "q":
                break
            lab[k] = a == "y"
            f.write_text(json.dumps(J, indent=1))       # save after every answer
        else:
            continue
        break
    print("saved")


def score():
    def ndcg(ranked, rel):
        dcg = sum(1 / math.log2(i + 2) for i, p in enumerate(ranked) if p in rel)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), DEPTH)))
        return dcg / idcg if idcg else 0.0
    per = {mode: {c: {"P@5": [], "nDCG@5": []} for c in CONFIGS} for mode in ("both", "either")}
    agree = n = yes_a = yes_b = 0
    for f in sorted(DIR.glob("*.json")):
        J = json.loads(f.read_text())
        judges = sorted(J["labels"])
        if len(judges) < 2:
            print(f"{f.name}: needs 2 judges (has {judges})"); continue
        A, B = J["labels"][judges[0]], J["labels"][judges[1]]
        for i, pool in enumerate(J["pools"]):
            keys = [f"{i}|{p}" for p in pool if f"{i}|{p}" in A and f"{i}|{p}" in B]
            for k in keys:
                n += 1; agree += A[k] == B[k]; yes_a += A[k]; yes_b += B[k]
            rel = {"both": {k.split("|", 1)[1] for k in keys if A[k] and B[k]},
                   "either": {k.split("|", 1)[1] for k in keys if A[k] or B[k]}}
            for mode in rel:
                for c in CONFIGS:
                    ranked = J["runs"][c][i]
                    per[mode][c]["P@5"].append(sum(p in rel[mode] for p in ranked) / DEPTH)
                    per[mode][c]["nDCG@5"].append(ndcg(ranked, rel[mode]))
    if not n:
        print("no doubly-judged passages yet"); return
    pe = (yes_a * yes_b + (n - yes_a) * (n - yes_b)) / n ** 2
    kappa = (agree / n - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"{n} doubly-judged passages, raw agreement {agree / n:.2f}, Cohen's kappa {kappa:.2f}")
    for mode in per:
        print(f"\nrelevant = {mode} judge(s) said yes")
        for c, m in per[mode].items():
            k = len(m["P@5"])
            print(f"  {c}  P@5 {sum(m['P@5']) / k:.3f}  nDCG@5 {sum(m['nDCG@5']) / k:.3f}  (turns {k})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["new", "label", "score"])
    ap.add_argument("--name")
    ap.add_argument("--judge")
    a = ap.parse_args()
    if a.cmd == "new":
        new(a.name)
    elif a.cmd == "label":
        label(a.name, a.judge)
    else:
        score()
