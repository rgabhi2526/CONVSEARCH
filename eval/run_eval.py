"""Evaluate configs on MTRAG ClapNQ. Usage:
  uv run python eval/run_eval.py --configs C0,C1,C3 --split test
  uv run python eval/run_eval.py --check-bm25s      # our BM25 vs bm25s on 20 queries
Writes results/<split>.csv (one row per config) and results/<split>_perturn.json."""
import argparse
import csv
import json
import math
import time
from pathlib import Path

from convsearch import data, pipeline

RESULTS = Path(__file__).resolve().parents[1] / "results"

# id -> (pipeline config, which query text to feed)
F = {"query": "fusion", "offline": True}       # eval never calls the API (E15)
CONFIGS = {
    "C0": ({"query": "turn"}, "lastturn"),
    "C1": ({"query": "concat"}, "lastturn"),
    "C2": ({"query": "rewrite1", "offline": True}, "lastturn"),
    "C3": ({"query": "turn"}, "rewrite"),     # oracle: MTRAG human rewrite, no history
    "C4": (F, "lastturn"),
    "C5": ({**F, "gate": True}, "lastturn"),
    "C6": ({**F, "gate": True, "boolean": True}, "lastturn"),
    "C7": ({**F, "gate": True, "boolean": True, "mu": 3.0}, "lastturn"),
    "C8": ({**F, "gate": True, "boolean": True, "mu": 3.0, "nu": 0.1}, "lastturn"),
    "C9": ({**F, "llm": False, "gate": True, "boolean": True}, "lastturn"),
    "C0cos": ({"query": "turn", "scorer": "cosine"}, "lastturn"),
    "C4n1": ({**F, "n": 1}, "lastturn"),
    "C4n3": ({**F, "n": 3}, "lastturn"),
}
OVERRIDES = {}                                # --set key=value, applied to every config


def metrics(ranked, rel):
    ids = [d for d, _ in ranked]
    hits = [1 if d in rel else 0 for d in ids]
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits[:10]))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), 10)))
    ap, n = 0.0, 0
    for i, h in enumerate(hits):
        if h:
            n += 1
            ap += n / (i + 1)
    return {"P@5": sum(hits[:5]) / 5, "P@10": sum(hits[:10]) / 10,
            "R@10": sum(hits[:10]) / len(rel), "nDCG@10": dcg / idcg, "MAP": ap / len(rel)}


def run_config(cid, convs, ix, qrels, qtexts):
    cfg, variant = CONFIGS[cid]
    cfg = {**cfg, **OVERRIDES}
    per_turn = {}
    for conv in convs:
        turns = data.user_turns(conv)
        qids = {t: qid for qid, t in data.conversations()[conv]}
        prev = []
        for i, turn in enumerate(turns, start=1):     # in order; prev_topk = own previous top-10
            qid = qids.get(i)
            text = qtexts[variant][qid] if (qid and variant != "lastturn") else turn
            history = [] if variant == "rewrite" else turns[:i - 1]
            ranked, _ = pipeline.run(text, history, prev, cfg, ix)
            prev = [d for d, _ in ranked[:10]]
            if qid:
                rel = {p for p, r in qrels[qid].items() if r > 0}
                per_turn[qid] = metrics([(ix.doc_ids[d], s) for d, s in ranked], rel)
    mean = {m: sum(v[m] for v in per_turn.values()) / len(per_turn) for m in next(iter(per_turn.values()))}
    return mean, per_turn


def check_bm25s(ix, n=20):
    import bm25s
    import numpy as np
    from convsearch import rank
    from convsearch.text import query_terms, tokens
    # same tokens as our body zone -> scores must match
    corpus_tokens = [tokens(t) for t in ix.texts]
    r = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
    vocab = {}
    ids = [[vocab.setdefault(w, len(vocab)) for w in doc] for doc in corpus_tokens]
    r.index(bm25s.tokenization.Tokenized(ids=ids, vocab=vocab), show_progress=False)
    qs = list(data.queries("rewrite").values())[:n]
    worst = 0.0
    for q in qs:
        terms = [t for t in query_terms(q) if t in vocab]
        # bm25s (Lucene) drops the constant (k1+1) numerator factor -> rank-equivalent; divide it out
        ours = rank.bm25(ix, rank.tf_vec(terms)) / (rank.K1 + 1)
        theirs = r.get_scores(terms)
        worst = max(worst, float(np.abs(ours - theirs).max()))
    print(f"bm25s check on {n} queries: max |ours - bm25s| = {worst:.2e}")
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="C0,C1,C3")
    ap.add_argument("--split", default="test", choices=["tune", "test", "all"])
    ap.add_argument("--check-bm25s", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. --set alpha=0.5")
    ap.add_argument("--out", default=None, help="results file stem (default: split name)")
    a = ap.parse_args()
    for kv in a.set:
        k, v = kv.split("=")
        OVERRIDES[k] = v.lower() == "true" if v.lower() in ("true", "false") else float(v)
    ix = pipeline.index()
    if a.check_bm25s:
        check_bm25s(ix)
        raise SystemExit
    convs = list(data.conversations()) if a.split == "all" else data.split(a.split)
    qrels = data.qrels()
    qtexts = {v: data.queries(v) for v in ("lastturn", "rewrite")}
    RESULTS.mkdir(exist_ok=True)
    rows, per = [], {}
    for cid in a.configs.split(","):
        t0 = time.time()
        mean, per[cid] = run_config(cid, convs, ix, qrels, qtexts)
        rows.append({"config": cid, "turns": len(per[cid]), **{k: round(v, 4) for k, v in mean.items()}})
        print(f"{cid:6} " + "  ".join(f"{k} {v:.4f}" for k, v in mean.items()) + f"  ({time.time() - t0:.1f}s)")
    from convsearch import rerank
    rerank.save_emb()
    stem = a.out or a.split
    with open(RESULTS / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    with open(RESULTS / f"{stem}_perturn.json", "w") as f:
        json.dump(per, f, indent=1)
