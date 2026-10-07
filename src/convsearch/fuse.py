"""Rewrite fusion: N rewrites -> one idf-weighted query vector.
  agree(t) = |{i : t in rewrite_i}| / N
  w(t)     = agree(t) * idf(t)  (+ beta * idf(t) if t is in the current turn)
  drop t   if agree(t) < min(2, N)/N and t not in the current turn   (E6: lone hallucinations)
"""
import math
from collections import Counter

from .rank import idf
from .text import STOPWORDS, query_terms, words


def phrases_of(rewrites, n):
    """Adjacent content-word bigrams appearing in >= ceil(N/2) rewrites (agent phrase candidates)."""
    c = Counter()
    for r in rewrites:
        toks = [(w, w in STOPWORDS) for w in words(r)]
        bigrams = {(query_terms(a)[0], query_terms(b)[0])
                   for (a, sa), (b, sb) in zip(toks, toks[1:]) if not sa and not sb}
        c.update(bigrams)
    return [list(p) for p, k in c.items() if k >= math.ceil(n / 2)]


def fuse(ix, turn, history, rewrites, beta=0.5):
    """rewrites=None -> no-LLM mode (C9 / E14 fallback): one pseudo-rewrite = turn + history.
    -> (qvec {term: w}, info) where info has the weight table and flags for the trace."""
    flags = {}
    if rewrites is None:
        rewrites = [" ".join(history + [turn])]
        flags["no_llm"] = True
    n = len(rewrites)
    sets = [set(query_terms(r)) for r in rewrites]
    if n > 1 and all(s == sets[0] for s in sets):
        flags["diversity"] = 0                                       # E7
    turn_t = set(query_terms(turn))
    hist_t = {t for h in history for t in query_terms(h)}
    agree = Counter(t for s in sets for t in s)
    table, qvec = [], {}
    for t in sorted(set(agree) | turn_t):
        a = agree[t] / n
        i = idf(ix, t)
        origin = "turn" if t in turn_t else "history" if t in hist_t else "llm-only"
        row = {"term": t, "agree": round(a, 2), "idf": round(i, 3), "origin": origin, "w": 0.0, "note": ""}
        if ix.tid(t) is None:
            row["note"] = "OOV"                                      # E9
        elif a < min(2, n) / n and t not in turn_t:
            row["note"] = "dropped: low agreement"                   # E6
        else:
            row["w"] = round(a * i + (beta * i if t in turn_t else 0.0), 4)
            qvec[t] = row["w"]
        table.append(row)
    table.sort(key=lambda r: -r["w"])
    return qvec, {"table": table, "phrases": phrases_of(rewrites, n), "n": n, **flags}


if __name__ == "__main__":
    from .index import toy
    ix = toy()
    rw = ["Do the Arizona Cardinals play outside the US?"] * 4 + ["Do the Arizona Cardinals play outside the US in the 2023 season?"]
    q, info = fuse(ix, "Do they play outside the US?", ["where do the arizona cardinals play"], rw)
    row = {r["term"]: r for r in info["table"]}
    assert row["arizona"]["agree"] == 1.0 and row["arizona"]["origin"] == "history"
    assert row["season"]["note"] in ("dropped: low agreement", "OOV") and "season" not in q   # E6
    assert row["play"]["origin"] == "turn" and abs(q["play"] - 1.5 * idf(ix, "play")) < 1e-3
    assert ["arizona", "cardin"] in info["phrases"]
    _, info = fuse(ix, "x", [], ["arizona cardinals"] * 5)
    assert info["diversity"] == 0                                                              # E7
    q, info = fuse(ix, "And Chicago?", ["where do the arizona cardinals play"], None)          # E16
    assert info["no_llm"] and q["chicago"] > q["arizona"] > 0                                  # beta boosts turn
    q, _ = fuse(ix, "what about that one?", [], None)
    assert q == {}                                                                             # E8 -> pipeline
    print("fuse ok")
