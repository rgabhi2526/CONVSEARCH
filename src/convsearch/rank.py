"""Ranking over a weighted query vector {term: w}: BM25 (title/body zones) and lnc.ltc cosine.
Term-at-a-time scoring into a dense accumulator, then heap top-K."""
import heapq
import math

import numpy as np

K1, B = 1.2, 0.75


def idf(ix, term, zone="body"):
    """BM25 (Lucene) idf = ln(1 + (N - df + 0.5) / (df + 0.5)); 0 for OOV terms."""
    t = ix.tid(term)
    if t is None:
        return 0.0
    df = ix.zones[zone].df(t)
    return math.log(1 + (ix.N - df + 0.5) / (df + 0.5))


def bm25_zone(ix, qvec, zone, k1=K1, b=B):
    Z = ix.zones[zone]
    scores = np.zeros(ix.N, dtype=np.float32)
    for term, w in qvec.items():
        t = ix.tid(term)
        if t is None or w == 0:                       # E9: OOV term contributes nothing
            continue
        docs, tf = Z.postings(t)
        if len(docs) == 0:
            continue
        norm = k1 * (1 - b + b * Z.doc_len[docs] / Z.avgdl)
        scores[docs] += w * idf(ix, term, zone) * tf * (k1 + 1) / (tf + norm)
    return scores


def bm25(ix, qvec, alpha=0.0, k1=K1, b=B):
    """score(d) = sum_t w(t) * [BM25_body(t,d) + alpha * BM25_title(t,d)]"""
    s = bm25_zone(ix, qvec, "body", k1, b)
    if alpha:
        s += alpha * bm25_zone(ix, qvec, "title", k1, b)
    return s


def cosine(ix, qvec):
    """lnc.ltc: doc = (1 + log tf), cosine-normalised; query weight = w(t) (already idf-weighted
    by fusion; plain queries get ltc = (1 + log qtf) * idf), cosine-normalised."""
    Z = ix.zones["body"]
    scores = np.zeros(ix.N, dtype=np.float32)
    qnorm = math.sqrt(sum(w * w for w in qvec.values())) or 1.0
    for term, w in qvec.items():
        t = ix.tid(term)
        if t is None or w == 0:
            continue
        docs, tf = Z.postings(t)
        scores[docs] += (w / qnorm) * (1 + np.log(tf)) / Z.lnc_norm[docs]
    return scores


def ltc(ix, terms):
    """Plain query terms -> ltc weights (1 + log qtf) * idf."""
    q = {}
    for t in terms:
        q[t] = q.get(t, 0) + 1
    return {t: (1 + math.log(n)) * idf(ix, t) for t, n in q.items()}


def tf_vec(terms):
    """Plain query terms -> raw query-tf weights (standard BM25 query)."""
    q = {}
    for t in terms:
        q[t] = q.get(t, 0) + 1.0
    return q


def topk(scores, k=10):
    """Heap top-K over docs with non-zero score -> [(doc, score)] best first."""
    cand = np.flatnonzero(scores)
    return [(d, s) for s, d in heapq.nlargest(k, zip(scores[cand].tolist(), cand.tolist()))]


if __name__ == "__main__":
    from .index import toy
    from .text import query_terms
    ix = toy()
    q = tf_vec(query_terms("arizona cardinals"))
    s = bm25(ix, q)
    # hand-computed: N=3, df(arizona)=2, df(cardin)=2 -> idf = ln(1 + 1.5/2.5)
    i = math.log(1.6)
    avgdl = 8.0
    def bm(tf, dl): return i * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * dl / avgdl))
    assert abs(s[0] - (bm(2, 7) + bm(1, 7))) < 1e-5 and abs(s[1] - (bm(1, 10) + bm(1, 10))) < 1e-5
    assert s[2] == 0 and [d for d, _ in topk(s, 2)] == [0, 1]
    assert bm25(ix, {"zzz": 1.0}).sum() == 0                     # OOV
    st = bm25(ix, q, alpha=1.0)
    assert st[0] > s[0] and st[2] == 0                            # title zone adds
    c = cosine(ix, ltc(ix, query_terms("arizona cardinals")))
    assert c[0] > c[1] > c[2] == 0 and c.max() <= 1 + 1e-6
    print("rank ok")
