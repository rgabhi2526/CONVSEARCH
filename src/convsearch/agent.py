"""Boolean agent with df feedback: builds a conjunctive query from the fused weights,
reads the hit count h from the postings intersection, and refines:
  h == 0   -> drop the lowest-agreement term (tie -> highest idf: rarest terms over-constrain);
              terms typed in the current turn are dropped last
  h > MAX  -> add the next-weight term, else swap a word pair for its phrase
Result set B gives a score bonus in ranking (not a filter, D11)."""
from . import boolean

MAX_HITS, START, ITERS = 500, 4, 3


def agent(ix, table, phrases):
    """table: fusion weight table (sorted by w desc); phrases: [[t1, t2], ...]
    -> (B as a set of doc ids, attempts for the trace)"""
    rows = [r for r in table if r["w"] > 0]
    info = {r["term"]: r for r in rows}
    items = [[r["term"]] for r in rows[:START]]           # AND of top-4 weighted terms
    rest = [[r["term"]] for r in rows[START:]]
    attempts, B = [], set()
    for _ in range(ITERS):
        if not items:
            break
        group = [(False, it) for it in items]
        steps = []
        docs = boolean.and_group(ix, group, steps)
        h = len(docs)
        q = boolean.to_str([group])
        if h == 0:
            words_ = [it for it in items if len(it) == 1]
            if not words_:
                attempts.append({"query": q, "hits": h, "steps": steps, "action": "give up"}); break
            # terms the user typed this turn go last; then lowest agreement; tie -> highest idf
            drop = min(words_, key=lambda it: (info[it[0]].get("origin") == "turn",
                                               info[it[0]]["agree"], -info[it[0]]["idf"]))
            items.remove(drop)
            action = f"0 hits -> drop '{drop[0]}'"
        elif h > MAX_HITS:
            ph = next((p for p in phrases if [p[0]] in items and [p[1]] in items), None)
            if rest:
                items.append(rest.pop(0))
                action = f"{h} hits -> add '{items[-1][0]}'"
            elif ph:
                items = [it for it in items if it not in ([ph[0]], [ph[1]])] + [ph]
                action = f"{h} hits -> phrase \"{' '.join(ph)}\""
            else:
                attempts.append({"query": q, "hits": h, "steps": steps, "action": "too broad, nothing to add"}); break
        else:
            B = set(docs.tolist())
            attempts.append({"query": q, "hits": h, "steps": steps, "action": "accept"})
            break
        attempts.append({"query": q, "hits": h, "steps": steps, "action": action})
    return B, attempts                                     # E10/E11: B stays empty if never accepted


if __name__ == "__main__":
    from .index import toy
    def T(*rows):
        return [{"term": t, "w": w, "agree": a, "idf": i} for t, w, a, i in rows]
    ix = toy()
    B, att = agent(ix, T(("arizona", 1, 1, .5), ("cardin", .9, 1, .5)), [])
    assert B == {0, 1} and att[-1]["action"] == "accept"
    # E10: 'chicago' AND 'rock' -> 0 hits; drop lowest agree (rock) -> accept
    B, att = agent(ix, T(("chicago", 1, 1, 1), ("rock", .9, .4, 1)), [])
    assert "drop 'rock'" in att[0]["action"] and B == {1}
    # E10: nothing ever matches
    B, att = agent(ix, T(("glendal", 1, 1, 1), ("rock", 1, 1, 1), ("english", 1, 1, 1), ("chicago", 1, 1, 1)), [])
    assert B == set() and len(att) == 3
    # E11: too broad -> add / phrase (MAX_HITS lowered to exercise it on the toy corpus)
    MAX_HITS = 1
    B, att = agent(ix, T(("arizona", 1, 1, .5), ("cardin", .9, 1, .5)), [["arizona", "cardin"]])
    assert 'phrase "arizona cardin"' in att[0]["action"] and B == {0}
    print("agent ok")
