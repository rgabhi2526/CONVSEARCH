"""One conversational turn -> ranked passages + trace. The eval and the demo both call run().

Configs (spec §7): C0 {query: turn} · C1 {query: concat} · C2 {query: rewrite1}
C3 eval feeds the human rewrite with {query: turn} · C4 {query: fusion} · C5 +gate · C6 +boolean
C9 {query: fusion, llm: False}
"""
import numpy as np

from . import agent as agent_mod
from . import boolean, fuse, gate as gate_mod, rank, rerank, rewrite
from .index import Index
from .text import query_terms

DEFAULT = {                # tuned on the tune split (eval/tune.py, results/tune_C6.csv)
    "query": "turn",      # turn | concat | rewrite1 | fusion
    "n": 5,               # rewrites used by fusion (first n of the cached 5)
    "llm": True,          # False -> no-LLM fusion (C9)
    "offline": False,     # True -> rewrites from cache only (eval, E15)
    "gate": False,
    "boolean": False,
    "scorer": "bm25",     # bm25 | cosine (lnc.ltc)
    "w_idf": False,        # fusion w = agree*idf (True) or agree (False; BM25 already has idf)
    "alpha": 0.3,         # title-zone weight
    "beta": 0.5,          # extra weight for terms typed in the current turn
    "lam": 0.2,           # Boolean bonus, as a fraction of the top score
    "tau_idf": 16.0,
    "tau_j": 0.1,
    "tau_h": 0.4,         # gate also needs mean history-term agreement < tau_h (1.01 = off)
    "mu": 0.0,            # [stretch] dense e5 re-rank weight (C7)
    "nu": 0.0,            # [stretch] Personalized PageRank weight (C8)
    "k": 100,
}

_ix = None


def index():
    global _ix
    if _ix is None:
        _ix = Index.load()
    return _ix


def _score(ix, qvec, cfg):
    if cfg["scorer"] == "cosine":
        return rank.cosine(ix, qvec)
    return rank.bm25(ix, qvec, alpha=cfg["alpha"])


def _fused(ix, turn, history, cfg, trace):
    rw, info = (None, {"llm": "off"}) if not cfg["llm"] else rewrite.rewrites(turn, history, offline=cfg["offline"])
    if rw is not None:
        rw = rw[:cfg["n"]]
    trace["rewrites"], trace["rewrite_info"] = rw, info
    qvec, finfo = fuse.fuse(ix, turn, history, rw, beta=cfg["beta"])
    if not cfg["w_idf"]:
        for r in finfo["table"]:
            if r["w"] > 0:
                r["w"] = round(r["w"] / r["idf"], 4)
                qvec[r["term"]] = r["w"]
    if "llm_fallback" in info:
        finfo["llm_fallback"] = info["llm_fallback"]                       # E14
    return qvec, finfo


def run(turn, history, prev_topk, config, ix=None):
    """turn: current user utterance; history: previous user turns (oldest first);
    prev_topk: doc ids shown for the previous turn (gate). Returns ([(doc, score)], trace)."""
    ix = ix or index()
    cfg = {**DEFAULT, **config}
    trace = {"config": cfg, "turn": turn, "history": history}

    if boolean.is_boolean(turn):                                         # explicit user Boolean query
        groups = boolean.parse(turn)
        steps = []
        B = boolean.search(ix, groups, steps)
        qvec = rank.tf_vec([t for g in groups for neg, ts in g if not neg for t in ts])
        scores = _score(ix, qvec, cfg)
        mask = np.zeros(ix.N, dtype=bool); mask[B] = True
        scores[~mask] = 0                                                # user asked for a filter
        trace.update(mode="explicit boolean", parsed=boolean.to_str(groups), steps=steps, qvec=qvec, B_size=len(B))
        ranked = rank.topk(scores, cfg["k"])
        trace["top"] = ranked[:10]
        return ranked, trace

    if cfg["query"] in ("turn", "concat"):
        text = " ".join(history + [turn]) if cfg["query"] == "concat" else turn
        terms = query_terms(text)
        qvec = rank.ltc(ix, terms) if cfg["scorer"] == "cosine" else rank.tf_vec(terms)
        trace.update(mode=cfg["query"], query_text=text)
    elif cfg["query"] == "rewrite1":                                     # C2: framework-style single rewrite
        rw, info = rewrite.rewrites(turn, history, offline=cfg["offline"])
        text = rw[0] if rw else " ".join(history + [turn])
        qvec = rank.tf_vec(query_terms(text))
        trace.update(mode="rewrite1", query_text=text, rewrite_info=info)
    else:
        qvec, finfo = _fused(ix, turn, history, cfg, trace)
        trace["mode"] = "fusion"
        if not qvec and history:                                         # E8: reuse previous turn's vector
            qvec, finfo = _fused(ix, history[-1], history[:-1], cfg, {})
            finfo["e8_fallback"] = "empty query -> previous turn's fused vector"
        trace["fusion"] = finfo
        if cfg["gate"] and history:
            # agreement is only a signal with real rewrites (no-LLM: every history term has agree 1)
            ha = [] if finfo.get("no_llm") else [r["agree"] for r in finfo["table"] if r["origin"] == "history" and r["w"] > 0]
            g = gate_mod.gate(ix, turn, prev_topk, cfg["tau_idf"], cfg["tau_j"], cfg["alpha"],
                              sum(ha) / len(ha) if ha else 0.0, cfg["tau_h"])
            if g["shift"]:
                for r in finfo["table"]:
                    if r["origin"] == "history" and r["w"] > 0:
                        r["w"], r["note"] = 0.0, "zeroed: topic shift"
                        qvec.pop(r["term"], None)
            trace["gate"] = g

    trace["qvec"] = qvec
    trace["oov"] = [t for t in qvec if ix.tid(t) is None]
    if not qvec:
        trace["message"] = "query too vague"                             # E8 with no history
        trace["top"] = []
        return [], trace
    scores = _score(ix, qvec, cfg)

    if cfg["boolean"] and cfg["query"] == "fusion":
        B, attempts = agent_mod.agent(ix, trace["fusion"]["table"], trace["fusion"]["phrases"])
        trace["agent"] = {"attempts": attempts, "B_size": len(B)}
        if B:
            bonus = cfg["lam"] * float(scores.max())
            idx = np.fromiter(B, dtype=np.int64)
            scores[idx] += bonus                                         # bonus, not filter (D11)
            trace["agent"]["bonus"] = round(bonus, 3)

    ranked = rank.topk(scores, max(cfg["k"], 100))
    if cfg["mu"] or (cfg["nu"] and prev_topk):
        rw = trace.get("rewrites")
        dq = rw[0] if rw else " ".join(history[-1:] + [turn])          # e5 needs natural language
        ranked, trace["rerank"] = rerank.rerank(ix, ranked, dq, prev_topk, cfg["mu"], cfg["nu"])
    ranked = ranked[:cfg["k"]]
    trace["top"] = ranked[:10]
    return ranked, trace


if __name__ == "__main__":
    from .index import toy
    ix = toy()
    H = ["where do the arizona cardinals play"]
    r, tr = run("Do they play outside the US?", H, [], {"query": "concat"}, ix)
    assert r[0][0] == 0 and "arizona" in tr["qvec"] and tr["oov"] == ["outsid", "us"]
    r, tr = run("Do they play outside the US?", H, [], {}, ix)
    assert [d for d, _ in r] == [0]                                      # C0: only "play" -> history lost
    r, tr = run("And Chicago?", H, [0], {"query": "fusion", "llm": False, "gate": True, "boolean": True}, ix)
    assert r[0][0] == 1 and not tr["gate"]["shift"] and tr["agent"]["B_size"] >= 0       # E5, E16
    r, tr = run("what about that one?", [], [], {"query": "fusion", "llm": False}, ix)
    assert r == [] and tr["message"] == "query too vague"                                # E8, no history
    r, tr = run("what about that one?", H, [], {"query": "fusion", "llm": False}, ix)
    assert r and "arizona" in tr["qvec"]                                                 # E8, with history
    r, tr = run("arizona NOT chicago", H, [], {}, ix)
    assert [d for d, _ in r] == [0] and tr["mode"] == "explicit boolean"                # E12 explicit NOT
    r, tr = run("english rock band history", H, [0, 1], {"query": "fusion", "llm": False, "gate": True, "tau_idf": 1}, ix)
    assert tr["gate"]["shift"] and "arizona" not in tr["qvec"] and r[0][0] == 2         # E3
    print("pipeline ok")
