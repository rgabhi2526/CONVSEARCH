"""Topic-shift gate: decide whether to drop history terms for this turn.
  self_contained = no anaphor AND >= 3 content terms AND sum idf >= tau_idf
  overlap        = Jaccard(top-10 of the turn alone, top-10 shown for the previous turn)
  hist_agree     = mean agree(t) of history-origin terms in the fused table (did the LLM carry context?)
  shift          = self_contained AND overlap < tau_j AND hist_agree < tau_h
Jaccard alone is not enough: "what about its price?" also has ~0 overlap (E4, D10)."""
from . import rank
from .text import query_terms, words

ANAPHORS = frozenset("it its they them their that this those these there he she him her his one".split())


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def gate(ix, turn, prev_topk, tau_idf=12.0, tau_j=0.1, alpha=0.0, hist_agree=0.0, tau_h=1.01):
    ws = words(turn)
    terms = query_terms(turn)
    anaphors = sorted(set(ws) & ANAPHORS)
    sum_idf = sum(rank.idf(ix, t) for t in terms)
    self_contained = not anaphors and len(terms) >= 3 and sum_idf >= tau_idf
    alone = [d for d, _ in rank.topk(rank.bm25(ix, rank.tf_vec(terms), alpha=alpha), 10)]
    overlap = jaccard(alone, prev_topk[:10])
    return {"anaphors": anaphors, "content_terms": len(terms), "sum_idf": round(sum_idf, 2),
            "self_contained": self_contained, "overlap": round(overlap, 3),
            "hist_agree": round(hist_agree, 2),
            "shift": self_contained and overlap < tau_j and hist_agree < tau_h}


if __name__ == "__main__":
    from .index import toy
    ix = toy()
    prev = [0, 1]                                    # previous turn showed the Cardinals docs
    g = gate(ix, "Do they play outside the US?", prev, tau_idf=1)
    assert g["anaphors"] == ["they"] and not g["shift"]                       # E2
    g = gate(ix, "english rock band history", prev, tau_idf=1)
    assert g["self_contained"] and g["overlap"] == 0 and g["shift"]           # E3
    g = gate(ix, "what about its price?", prev, tau_idf=1)
    assert not g["self_contained"] and not g["shift"]                         # E4
    g = gate(ix, "And Chicago?", prev, tau_idf=1)
    assert g["content_terms"] == 1 and not g["shift"]                         # E5
    g = gate(ix, "chicago cardinals moved arizona", prev, tau_idf=1)
    assert g["self_contained"] and g["overlap"] > 0.1 and not g["shift"]     # same topic, no anaphor
    print("gate ok")
