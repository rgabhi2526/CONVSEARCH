"""[stretch] Re-rank the top-100 with
  dense: cosine of e5-small-v2 embeddings (query vs passage)             -> + mu * minmax(dense)
  PPR:   Personalized PageRank on a kNN graph (tf-idf cosine, k=5) over
         top-100 ∪ previous turn's top-10, teleporting to the previous top-10 -> + nu * minmax(g)
final(d) = score(d)/max_score + mu*dense(d) + nu*g(d). PPR needs a previous turn (skipped on turn 1)."""
import math
from collections import Counter

import numpy as np

from . import data
from .rank import idf
from .text import query_terms

_model, _emb = None, {}
EMB = data.DATA / "e5_cache.npz"                             # passage embeddings by doc id (gitignored)


def save_emb():
    if _emb:
        ids = np.array(list(_emb), dtype=np.int64)
        np.savez(EMB, ids=ids, E=np.stack([_emb[i] for i in ids.tolist()]))


def _minmax(x):
    x = np.asarray(x, dtype=np.float32)
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


def dense(ix, query, docs):
    """e5 cosine(query, passage) for docs; passage embeddings memoised by doc id."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("intfloat/e5-small-v2", device="cpu")
        if EMB.exists() and not _emb:
            a = np.load(EMB)
            _emb.update(zip(a["ids"].tolist(), a["E"]))
    new = [d for d in docs if d not in _emb]
    if new:
        E = _model.encode([f"passage: {ix.titles[d]}. {ix.texts[d]}" for d in new],
                          normalize_embeddings=True, batch_size=64)
        _emb.update(zip(new, E))
    q = _model.encode([f"query: {query}"], normalize_embeddings=True)[0]
    return np.array([float(_emb[d] @ q) for d in docs], dtype=np.float32)


def tfidf_matrix(ix, docs):
    """Rows = l2-normalised (1 + log tf) * idf vectors of the docs (re-tokenised from text)."""
    rows, vocab = [], {}
    for d in docs:
        c = Counter(query_terms(ix.texts[d]))
        rows.append({vocab.setdefault(t, len(vocab)): (1 + math.log(n)) * idf(ix, t) for t, n in c.items()})
    M = np.zeros((len(docs), len(vocab)), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, w in r.items():
            M[i, j] = w
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M


def ppr(ix, docs, seeds, k=5, damping=0.85, iters=50):
    """Personalized PageRank over a symmetric kNN graph. seeds: doc ids to teleport to.
    -> {doc: g(d)}"""
    nodes = list(dict.fromkeys(list(docs) + list(seeds)))
    M = tfidf_matrix(ix, nodes)
    S = M @ M.T
    np.fill_diagonal(S, 0)
    A = np.zeros_like(S)
    nn = np.argsort(-S, axis=1)[:, :k]
    for i, js in enumerate(nn):
        A[i, js] = S[i, js]
    A = np.maximum(A, A.T)                                   # undirected
    deg = A.sum(axis=0)
    P = A / np.where(deg > 0, deg, 1)                        # column-stochastic
    p = np.array([1.0 if n in set(seeds) else 0.0 for n in nodes])
    p /= p.sum()
    r = p.copy()
    for _ in range(iters):
        r = (1 - damping) * p + damping * (P @ r)
    return dict(zip(nodes, r))


def rerank(ix, ranked, query, prev_topk, mu=0.0, nu=0.0, depth=100):
    """ranked: [(doc, score)] best first -> (re-ranked list, info)."""
    head, tail = ranked[:depth], ranked[depth:]
    if not head:
        return ranked, {}
    docs = [d for d, _ in head]
    s = np.array([sc for _, sc in head], dtype=np.float32)
    final = s / s.max()
    info = {}
    if mu:
        dn = _minmax(dense(ix, query, docs))
        final = final + mu * dn
        info["dense"] = dict(zip(docs[:10], np.round(dn[:10], 3).tolist()))
    if nu and prev_topk:
        g = ppr(ix, docs, prev_topk[:10])
        gn = _minmax([g[d] for d in docs])
        final = final + nu * gn
        info["ppr"] = dict(zip(docs[:10], np.round(gn[:10], 3).tolist()))
    order = np.argsort(-final, kind="stable")
    out = [(docs[i], float(final[i])) for i in order]
    # tail keeps its order below the re-ranked head
    return out + [(d, 0.0) for d, _ in tail], info


if __name__ == "__main__":
    from .index import toy
    ix = toy()
    g = ppr(ix, [0, 1, 2], [0])
    assert g[0] > g[2] and g[1] > g[2]                       # mass flows to Cardinals neighbours
    r, info = rerank(ix, [(2, 1.0), (1, 0.9), (0, 0.8)], "", [0], nu=2.0)
    assert r[0][0] != 2 and "ppr" in info                    # PPR pulls on-topic docs up
    print("rerank ok")
