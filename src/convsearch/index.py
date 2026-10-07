"""Positional inverted index with two zones (title, body), stored as flat numpy arrays.

Per zone, for term id t:
  postings of t     = post_doc[term_ptr[t]:term_ptr[t+1]]   (doc ids, ascending)
  tf of those docs  = post_tf[same slice]
  positions of posting p = positions[pos_ptr[p]:pos_ptr[p+1]]
df(t) = term_ptr[t+1] - term_ptr[t]. Doc lengths count all tokens (stopwords included).
"""
import json
import time
from array import array

import numpy as np

from . import data
from .text import tokens

INDEX_DIR = data.DATA / "index"
ZONES = ("title", "body")


class Zone:
    def __init__(self, term_ptr, post_doc, post_tf, pos_ptr, positions, doc_len):
        self.term_ptr, self.post_doc, self.post_tf = term_ptr, post_doc, post_tf
        self.pos_ptr, self.positions, self.doc_len = pos_ptr, positions, doc_len
        self.avgdl = float(doc_len.mean())
        # lnc doc-vector length for cosine: sqrt(sum_t (1 + log tf)^2) per doc
        w = 1 + np.log(post_tf.astype(np.float32))
        self.lnc_norm = np.sqrt(np.bincount(post_doc, weights=w * w, minlength=len(doc_len))).astype(np.float32)

    def df(self, t):
        return int(self.term_ptr[t + 1] - self.term_ptr[t])

    def postings(self, t):
        """(doc ids, tfs) for term id t; empty arrays if t is out of range."""
        if t is None or t + 1 >= len(self.term_ptr):
            return self.post_doc[:0], self.post_tf[:0]
        s, e = self.term_ptr[t], self.term_ptr[t + 1]
        return self.post_doc[s:e], self.post_tf[s:e]

    def positions_of(self, t, doc):
        """Positions of term t inside doc (binary search in t's postings)."""
        s, e = self.term_ptr[t], self.term_ptr[t + 1]
        i = s + np.searchsorted(self.post_doc[s:e], doc)
        if i == e or self.post_doc[i] != doc:
            return self.positions[:0]
        return self.positions[self.pos_ptr[i]:self.pos_ptr[i + 1]]


def _build_zone(term_ids, doc_ids, pos, n_terms, n_docs):
    """Group the (term, doc, position) token stream into postings.
    Tokens arrive in doc order and position order, so a STABLE sort by term id
    leaves each term's postings sorted by doc and each posting's positions sorted."""
    term_ids = np.frombuffer(term_ids, dtype=np.int32)
    doc_ids = np.frombuffer(doc_ids, dtype=np.int32)
    order = np.argsort(term_ids, kind="stable")
    t, d = term_ids[order], doc_ids[order]
    positions = np.frombuffer(pos, dtype=np.int32)[order]
    # a new posting starts wherever (term, doc) changes
    new = np.ones(len(t), dtype=bool)
    new[1:] = (t[1:] != t[:-1]) | (d[1:] != d[:-1])
    starts = np.flatnonzero(new)
    pos_ptr = np.append(starts, len(t)).astype(np.int64)
    post_doc = d[starts]
    post_tf = np.diff(pos_ptr).astype(np.int32)
    term_ptr = np.searchsorted(t[starts], np.arange(n_terms + 1)).astype(np.int64)
    doc_len = np.bincount(doc_ids, minlength=n_docs).astype(np.int32)
    return Zone(term_ptr, post_doc, post_tf, pos_ptr, positions, doc_len)


class Index:
    def __init__(self, vocab, doc_ids, titles, texts, zones):
        self.vocab = vocab                      # term -> term id
        self.doc_ids, self.titles, self.texts = doc_ids, titles, texts
        self.zones = zones                      # {"title": Zone, "body": Zone}
        self.N = len(doc_ids)

    @classmethod
    def build(cls, docs):
        """docs: iterable of (doc_id, title, text)."""
        vocab, doc_ids, titles, texts = {}, [], [], []
        streams = {z: (array("i"), array("i"), array("i")) for z in ZONES}
        for d, (did, title, text) in enumerate(docs):
            doc_ids.append(did); titles.append(title); texts.append(text)
            for z, s in (("title", title), ("body", text)):
                ids = [vocab.setdefault(w, len(vocab)) for w in tokens(s)]
                tt, dd, pp = streams[z]
                tt.extend(ids); dd.extend([d] * len(ids)); pp.extend(range(len(ids)))
        zones = {z: _build_zone(*streams[z], len(vocab), len(doc_ids)) for z in ZONES}
        return cls(vocab, doc_ids, titles, texts, zones)

    def tid(self, term):
        return self.vocab.get(term)

    def save(self, path=INDEX_DIR):
        path.mkdir(parents=True, exist_ok=True)
        for z, Z in self.zones.items():
            np.savez(path / f"{z}.npz", term_ptr=Z.term_ptr, post_doc=Z.post_doc, post_tf=Z.post_tf,
                     pos_ptr=Z.pos_ptr, positions=Z.positions, doc_len=Z.doc_len)
        terms = sorted(self.vocab, key=self.vocab.get)
        with open(path / "meta.json", "w") as f:
            json.dump({"terms": terms, "doc_ids": self.doc_ids, "titles": self.titles, "texts": self.texts}, f)

    @classmethod
    def load(cls, path=INDEX_DIR):
        with open(path / "meta.json") as f:
            m = json.load(f)
        zones = {}
        for z in ZONES:
            a = np.load(path / f"{z}.npz")
            zones[z] = Zone(a["term_ptr"], a["post_doc"], a["post_tf"], a["pos_ptr"], a["positions"], a["doc_len"])
        vocab = {t: i for i, t in enumerate(m["terms"])}
        return cls(vocab, m["doc_ids"], m["titles"], m["texts"], zones)


def toy():
    """3-doc corpus used by every module's self-test."""
    return Index.build([
        ("d0", "Arizona Cardinals", "The Arizona Cardinals play in Glendale Arizona"),
        ("d1", "Chicago Cardinals", "The Chicago Cardinals moved to St Louis then to Arizona"),
        ("d2", "The Who", "The Who are an English rock band"),
    ])


if __name__ == "__main__":
    import sys
    ix = toy()
    B = ix.zones["body"]
    t = ix.tid("arizona")
    assert B.df(t) == 2 and list(B.postings(t)[0]) == [0, 1] and list(B.postings(t)[1]) == [2, 1]
    assert list(B.positions_of(t, 0)) == [1, 6] and list(B.positions_of(t, 1)) == [9]
    assert len(B.positions_of(t, 2)) == 0
    assert B.df(ix.tid("the")) == 3                       # stopwords are indexed (D9)
    assert ix.zones["title"].df(ix.tid("cardin")) == 2
    assert list(B.doc_len) == [7, 10, 7]
    assert abs(B.lnc_norm[0] - np.sqrt(5 + (1 + np.log(2)) ** 2)) < 1e-5
    print("index ok")
    if "--build" in sys.argv:                            # full ClapNQ build
        t0 = time.time()
        ix = Index.build(data.corpus())
        print(f"built {ix.N} docs, {len(ix.vocab)} terms, "
              f"{len(ix.zones['body'].post_doc)} body postings in {time.time() - t0:.0f}s")
        ix.save()
