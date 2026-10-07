"""MTRAG ClapNQ download + loaders. Run `python -m convsearch.data` once to fetch into data/."""
import csv, io, json, urllib.request, zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BASE = "https://raw.githubusercontent.com/IBM/mt-rag-benchmark/main/"
TASKS = "mtrag-human/retrieval_tasks/clapnq/"
FILES = {
    "clapnq_lastturn.jsonl": TASKS + "clapnq_lastturn.jsonl",
    "clapnq_rewrite.jsonl": TASKS + "clapnq_rewrite.jsonl",
    "clapnq_questions.jsonl": TASKS + "clapnq_questions.jsonl",
    "qrels.tsv": TASKS + "qrels/dev.tsv",
}
CORPUS_ZIP = "corpora/passage_level/clapnq.jsonl.zip"


def download():
    DATA.mkdir(exist_ok=True)
    for name, path in FILES.items():
        if not (DATA / name).exists():
            urllib.request.urlretrieve(BASE + path, DATA / name)
    if not (DATA / "clapnq.jsonl").exists():
        with urllib.request.urlopen(BASE + CORPUS_ZIP) as r:              # 45 MB zip -> 162 MB jsonl
            zipfile.ZipFile(io.BytesIO(r.read())).extract("clapnq.jsonl", DATA)


def corpus():
    """Yield (doc_id, title, text) for every passage."""
    with open(DATA / "clapnq.jsonl") as f:
        for line in f:
            p = json.loads(line)
            yield p["_id"], p.get("title", ""), p["text"]


def split_id(qid):
    """'<conv><::><turn>' -> (conv, int turn)."""
    conv, turn = qid.split("<::>")
    return conv, int(turn)


def queries(variant="lastturn"):
    """{qid: text} with the '|user|: ' prefix stripped. variant: lastturn | rewrite | questions."""
    out = {}
    with open(DATA / f"clapnq_{variant}.jsonl") as f:
        for line in f:
            q = json.loads(line)
            out[q["_id"]] = q["text"].removeprefix("|user|: ").strip()
    return out


def qrels():
    """{qid: {doc_id: rel}}"""
    out = defaultdict(dict)
    with open(DATA / "qrels.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[row["query-id"]][row["corpus-id"]] = int(row["score"])
    return dict(out)


def conversations():
    """{conv: [(qid, turn_no), ...] sorted by turn}, convs sorted by id (spec §2 split)."""
    convs = defaultdict(list)
    for qid in queries("lastturn"):
        conv, t = split_id(qid)
        convs[conv].append((qid, t))
    return {c: sorted(v, key=lambda x: x[1]) for c, v in sorted(convs.items())}


def user_turns(conv):
    """All user turns of a conversation, in order, incl. turns that have no qrels.
    clapnq_questions holds every user turn up to that point, one per line (turn t = line t),
    so the last evaluated turn's text gives the whole history."""
    qid = conversations()[conv][-1][0]
    return [l.removeprefix("|user|: ").strip() for l in queries("questions")[qid].split("\n")]


def split(name):
    """'tune' = first 10 conversations by id, 'test' = remaining 19."""
    cs = list(conversations())
    return cs[:10] if name == "tune" else cs[10:]


if __name__ == "__main__":
    download()
    q, r, c = queries(), qrels(), conversations()
    assert len(q) == 208 and len(c) == 29, (len(q), len(c))
    assert sum(len(v) for v in r.values()) == 578
    assert len(split("tune")) == 10 and len(split("test")) == 19
    print("ok:", len(q), "turns,", len(c), "convs, turn depth", min(t for v in c.values() for _, t in v), "-", max(t for v in c.values() for _, t in v))
