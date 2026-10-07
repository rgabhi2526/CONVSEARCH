"""Boolean retrieval: query parser (AND / OR / NOT / "phrase"), df-ordered postings
intersection, phrase matching via positions.

Parsed query = OR of AND-groups; each AND-group = list of items (neg, terms):
  terms has 1 stem for a word, >1 stems for a phrase. Precedence NOT > AND > OR,
  adjacent words are an implicit AND. No parentheses (ponytail: add if demo needs them).
"""
import re

import numpy as np

from .text import tokens

_tok = re.compile(r'"[^"]*"|\S+')


def parse(q):
    groups, group, neg = [], [], False
    for raw in _tok.findall(q):
        if raw == "OR":
            if group:
                groups.append(group)
            group, neg = [], False
        elif raw == "AND":
            continue
        elif raw == "NOT":
            neg = True
        else:
            terms = tokens(raw.strip('"'))
            if terms:
                group.append((neg, terms))
            neg = False
    if group:
        groups.append(group)
    return groups


def is_boolean(q):
    """User typed an explicit Boolean query (operators are case-sensitive)."""
    return '"' in q or re.search(r"\b(AND|OR|NOT)\b", q) is not None


def to_str(groups):
    def item(neg, terms):
        s = f'"{" ".join(terms)}"' if len(terms) > 1 else terms[0]
        return "NOT " + s if neg else s
    return " OR ".join(" AND ".join(item(*it) for it in g) for g in groups)


def term_docs(ix, term):
    t = ix.tid(term)
    return ix.zones["body"].postings(t)[0] if t is not None else np.empty(0, dtype=np.int32)


def phrase_docs(ix, terms, within=None):
    """Docs where terms occur at consecutive positions (positional index).
    within: docs already surviving the AND -> only those need a positional check."""
    Z = ix.zones["body"]
    tids = [ix.tid(t) for t in terms]
    if None in tids:
        return np.empty(0, dtype=np.int32)
    docs = term_docs(ix, terms[0]) if within is None else within
    for t in terms[(0 if within is not None else 1):]:
        docs = np.intersect1d(docs, term_docs(ix, t), assume_unique=True)
    keep = []
    for d in docs.tolist():
        start = set(Z.positions_of(tids[0], d).tolist())
        for i, t in enumerate(tids[1:], 1):
            start &= {p - i for p in Z.positions_of(t, d).tolist()}
            if not start:
                break
        if start:
            keep.append(d)
    return np.array(keep, dtype=np.int32)


def df_est(ix, terms):
    """df of a word; for a phrase the min df of its words (upper bound) - used for ordering."""
    return min(len(term_docs(ix, t)) for t in terms)


def and_group(ix, group, steps=None):
    """Intersect positive items in increasing-df order (smallest list first keeps
    intermediate results small), then subtract NOT items."""
    pos = sorted((it for it in group if not it[0]), key=lambda it: df_est(ix, it[1]))
    neg = [it for it in group if it[0]]
    if not pos:
        return np.empty(0, dtype=np.int32)
    res = None
    for _, terms in pos:
        docs = phrase_docs(ix, terms, res) if len(terms) > 1 else term_docs(ix, terms[0])
        res = docs if res is None else np.intersect1d(res, docs, assume_unique=True)
        if steps is not None:
            steps.append({"op": "AND", "item": " ".join(terms), "df": len(docs), "result": len(res)})
        if len(res) == 0:
            break                                      # early exit: AND with empty is empty
    for _, terms in neg:
        docs = phrase_docs(ix, terms) if len(terms) > 1 else term_docs(ix, terms[0])
        res = np.setdiff1d(res, docs, assume_unique=True)
        if steps is not None:
            steps.append({"op": "NOT", "item": " ".join(terms), "df": len(docs), "result": len(res)})
    return res


def search(ix, groups, steps=None):
    """Evaluate a parsed query -> sorted doc ids."""
    res = np.empty(0, dtype=np.int32)
    for g in groups:
        res = np.union1d(res, and_group(ix, g, steps)).astype(np.int32)
    return res


if __name__ == "__main__":
    from .index import toy
    ix = toy()
    assert parse('arizona AND NOT chicago OR "the who"') == [[(False, ["arizona"]), (True, ["chicago"])], [(False, ["the", "who"])]]
    assert to_str(parse('cardinals NOT chicago')) == "cardin AND NOT chicago"
    assert list(search(ix, parse("arizona cardinals"))) == [0, 1]
    assert list(search(ix, parse("arizona NOT chicago"))) == [0]               # E12 explicit NOT works
    assert list(search(ix, parse('"the who"'))) == [2]                         # E13 stopword phrase
    assert list(search(ix, parse('"cardinals arizona"'))) == []                # order matters
    assert list(search(ix, parse("rock OR glendale"))) == [0, 2]
    assert list(search(ix, parse("arizona zzz"))) == []                        # OOV in AND
    steps = []
    search(ix, parse("the arizona"), steps)
    assert [s["item"] for s in steps] == ["arizona", "the"]                    # df order: 2 before 3
    assert is_boolean('"the who"') and is_boolean("a NOT b") and not is_boolean("is it not raining")
    print("boolean ok")
