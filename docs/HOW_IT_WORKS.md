# How our conversational search system works

A complete walkthrough of everything in this repo, written so either of us can explain any part in the video or a viva. It starts from IR basics and works up to our own ideas. Every number below comes from our real index or our real runs unless it is marked as a toy example.

**How to read this:** Part A is the big picture. Part B explains each building block in pipeline order, giving the concept, how our code does it, and a worked example. Part C covers evaluation, the demo and tooling. Part D is a viva cheat-sheet.

---

## Contents

- **Part A: Big picture**
  1. [The problem](#1-the-problem)
  2. [Our solution in one diagram](#2-our-solution-in-one-diagram)
  3. [The dataset: MTRAG ClapNQ](#3-the-dataset-mtrag-clapnq)
- **Part B: The building blocks**
  4. [Text processing (`text.py`)](#4-text-processing-textpy)
  5. [The inverted index (`index.py`)](#5-the-inverted-index-indexpy)
  6. [Ranking: BM25 and cosine (`rank.py`)](#6-ranking-bm25-and-cosine-rankpy)
  7. [Boolean retrieval (`boolean.py`)](#7-boolean-retrieval-booleanpy)
  8. [LLM query rewrites (`rewrite.py`)](#8-llm-query-rewrites-rewritepy)
  9. [Rewrite fusion (`fuse.py`)](#9-rewrite-fusion-fusepy)
  10. [Topic-shift gate (`gate.py`)](#10-topic-shift-gate-gatepy)
  11. [Boolean agent (`agent.py`)](#11-boolean-agent-agentpy)
  12. [Stretch: dense re-rank and PageRank (`rerank.py`)](#12-stretch-dense-re-rank-and-personalized-pagerank-rerankpy)
  13. [Putting it together (`pipeline.py`)](#13-putting-it-together-pipelinepy)
  14. [Edge cases E1–E16](#14-edge-cases-e1e16)
- **Part C: Evaluation, demo, tooling**
  15. [Evaluation](#15-evaluation)
  16. [The demo app (`app.py`)](#16-the-demo-app-apppy)
  17. [Libraries and tools](#17-libraries-and-tools)
  18. [Repo map and commands](#18-repo-map-and-commands)
- **Part D: Viva prep**
  19. [Honest limitations](#19-honest-limitations)
  20. [Likely viva questions](#20-likely-viva-questions)
  21. [Glossary](#21-glossary)

---

# Part A: Big picture

## 1. The problem

In **conversational search**, the user asks a chain of questions, and each later question leans on the earlier ones:

```
turn 1: where do the arizona cardinals play this week
turn 2: Do they play outside the US?
turn 3: And Chicago?
```

A normal search engine sees only the current text. For turn 2 it searches for `play outside us`: the word "they" means nothing to it, so it returns random sports passages. That is our **C0 baseline**. On follow-up turns its nDCG@10 is **0.229**, against **0.436** on first turns. The context is lost.

The obvious fixes each have problems:

| Fix | Problem |
|---|---|
| Glue all earlier turns onto the query (C1, "concat") | Old topics pollute the query. On the test set this scores **lower** than doing nothing (0.235 vs 0.256). |
| Ask an LLM to rewrite the question once (C2) | Works (0.305), but a single rewrite can hallucinate ("…in the 2023 season") or drop context, and we can't see why it failed. |
| Let an LLM do the search or answer | Not allowed: the assignment wants **our own IR engine**. |

**Our idea:** ask the LLM for **5** rewrites and treat them as **votes**. A word that most rewrites agree on is probably what the user meant. A word that only one rewrite added is probably a hallucination. Our own engine turns those votes into a weighted query, checks whether the topic changed, runs a Boolean "agent" over the postings, and ranks with our own BM25. The LLM **only proposes text**. It never ranks, filters or answers.

## 2. Our solution in one diagram

```
 user turn + earlier user turns
          │
          ▼
 ┌─────────────────┐   5 rewrites (cached on disk)
 │ rewrite.py      │──────────────────────────┐
 │ Gemini proposes │                          │
 └─────────────────┘                          ▼
                                     ┌──────────────────┐
                                     │ fuse.py          │  agree(t) = share of rewrites containing t
                                     │ votes → weights  │  w(t), origin = turn / history / llm-only
                                     └────────┬─────────┘  drop lone hallucinated terms
                                              ▼
                                     ┌──────────────────┐
                                     │ gate.py          │  topic shift? → zero the history terms
                                     └────────┬─────────┘
                                              ▼
                ┌─────────────────────────────┴───────────────────┐
                ▼                                                 ▼
       ┌──────────────────┐                             ┌───────────────────┐
       │ rank.py  BM25    │  over our inverted index    │ agent.py Boolean  │ AND of top terms,
       │ (title + body)   │◄────── index.py ───────────►│ df feedback loop  │ relax / tighten
       └────────┬─────────┘                             └─────────┬─────────┘
                │            docs in Boolean set B get + λ·max score │
                └───────────────────────┬──────────────────────────┘
                                        ▼
                              [stretch] rerank.py: + μ·dense(e5) + ν·PageRank
                                        ▼
                               top-10 passages + full trace (shown in app.py)
```

`report/pipeline.png` is the same diagram drawn properly, for the report and the video.

Every box writes what it did into a **trace** dictionary. The demo only *displays* the trace, so nothing on screen is faked.

## 3. The dataset: MTRAG ClapNQ

**MTRAG** is IBM's multi-turn RAG benchmark (Apache-2.0). We use its **ClapNQ** part: Wikipedia passages, with questions originally from Google's Natural Questions.

| Thing | Count | File (in `data/`, downloaded by `python -m convsearch.data`) |
|---|---|---|
| Passages | 183,408 | `clapnq.jsonl` (from a 45 MB zip) |
| Conversations | 29 | (derived from query IDs) |
| Judged user turns | 208 | `clapnq_lastturn.jsonl` (only the last turn's text) |
| All user turns | 222 | `clapnq_questions.jsonl` (each line = every user turn so far) |
| Human rewrites | 208 | `clapnq_rewrite.jsonl` (a person rewrote each turn as a standalone question) |
| Relevance judgments (qrels) | 578 | `qrels.tsv` (query id, passage id, score) |

**Query IDs** look like `<conversation id><::><turn number>`, e.g. `6a738cc0…<::>3`. `data.split_id` splits them.

**History.** `clapnq_questions.jsonl` stores, for each judged turn, all user turns up to that point, one per line and each prefixed with `|user|: `. `data.user_turns(conv)` reads the last judged turn of a conversation to get the whole list, stripping the prefix. 14 user turns have no judgments. We still **run** them, because the next turn needs the previous turn's top-10 (decision D20).

**We use previous user turns only as history** (D8), not the gold agent answers. Our system doesn't generate answers, so the demo and the eval see exactly the same kind of input.

**Train/test split (D12).** Conversations are sorted by ID:
- **tune:** the first 10. All parameter tuning happens here.
- **test:** the other 19, which have **141 judged turns**. Every number in the report comes from these.

This way we never report a score on data we tuned on.

---

# Part B: The building blocks

## 4. Text processing (`text.py`)

Text has to become **terms** before we can index or search it. The document side and the query side must use exactly the same steps, or terms won't match.

**Steps:**
1. **Case-fold** with `lower()`.
2. **Strip accents** with Unicode NFKD normalisation, then drop non-ASCII, so `café` → `cafe`. A user typing "cafe" still matches.
3. **Tokenise** with the regex `[a-z0-9]+`, so any non-alphanumeric character is a split point: `NFL's 2023-season!` → `nfl, s, 2023, season`.
4. **Stem** with the Porter stemmer from nltk (`cardinals` → `cardin`, `playing` → `play`, `infertility` → `infertil`). Stemming merges word forms so "adoption" matches "adopted". Stems look odd (`cardin`, `glendal`), which is fine: they are internal keys, not words. Results are memoised with `lru_cache`, since the same word gets stemmed millions of times while building the index.
5. **Stopwords** (`the`, `is`, `what`, …, plus chat filler like `please`, `thanks`, `tell`, `one`):
   - **Index side (`tokens`)** keeps them, with their positions.
   - **Query side (`query_terms`)** drops them before stemming.

**Why keep stopwords in the index (D9)?** Phrase queries need them: the band **"The Who"** is made entirely of stopwords. Because the index stores `the` and `who` with positions, `"the who"` works as a phrase (edge case E13). For *ranking* we drop them from the query, since `the` appears in 167,610 of 183,408 passages and its idf is only 0.09.

**Why is "one" a stopword?** Turns like *"what about that one?"* should reduce to an empty query (E8), which then triggers our fallback (§13).

```python
words("Café, NFL's 2023-season!")                 # ['cafe', 'nfl', 's', '2023', 'season']
tokens("The Arizona Cardinals are playing")       # ['the', 'arizona', 'cardin', 'are', 'play']
query_terms("Where do the Arizona Cardinals play?")  # ['arizona', 'cardin', 'play']
query_terms("what about that one?")               # []   ← E8
```

## 5. The inverted index (`index.py`)

### Concept

An **inverted index** maps each term to the list of documents that contain it, called its **postings list**. A query then only looks at the postings of its own few terms instead of scanning 183K passages. For each posting we store:
- the **doc id**;
- the **term frequency** `tf`: how often the term occurs in that doc;
- the **positions**: where in the doc it occurs, which phrase queries need.

`df(t)`, the **document frequency**, is the length of the postings list. It feeds idf.

### Two zones

We build **two separate indexes ("zones")** with the same structure: one over passage **titles** and one over passage **bodies**. A query word that appears in the title ("Arizona Cardinals") is strong evidence, so the ranker adds title-zone BM25 with weight α = 0.3 (§6).

### How it is stored: flat numpy arrays (CSR layout)

A Python dict of lists of tuples for 12M postings would need several GB of RAM. Our laptops have 8 GB. So each zone is **five flat numpy arrays**, the same "compressed sparse row" trick that sparse matrices use:

| Array | Meaning |
|---|---|
| `term_ptr` | `term_ptr[t] : term_ptr[t+1]` is the slice of postings for term id `t` |
| `post_doc` | doc id of each posting, ascending within a term |
| `post_tf` | tf of each posting |
| `pos_ptr` | `pos_ptr[p] : pos_ptr[p+1]` is the slice of `positions` for posting `p` |
| `positions` | every token position, grouped by posting |

Plus `doc_len` (tokens per doc, stopwords included) and `lnc_norm` (precomputed vector length for cosine).

**Toy example** (the 3-document corpus that every self-test uses, `index.toy()`):

```
d0  title "Arizona Cardinals"   body "The Arizona Cardinals play in Glendale Arizona"
d1  title "Chicago Cardinals"   body "The Chicago Cardinals moved to St Louis then to Arizona"
d2  title "The Who"             body "The Who are an English rock band"
```

The body zone's arrays (term ids: arizona = 0, cardin = 1, the = 2, play = 3, …):

```
term_ptr  = [0, 2, 4, 7, 8, ...]
post_doc  = [0, 1,   0, 1,   0, 1, 2,   0, ...]
post_tf   = [2, 1,   1, 1,   1, 1, 1,   1, ...]
positions = [1, 6,  9,  2,  2,  0,  0,  0, ...]
pos_ptr   = [0, 2, 3, 4, 5, 6, 7, 8, ...]
doc_len   = [7, 10, 7]
```

Reading term 0 (`arizona`):
- Its postings are `post_doc[0:2]` = docs **0 and 1**, so df = 2.
- tf is `post_tf[0:2]` = **2 and 1**.
- Posting 0 (d0) has positions `positions[0:2]` = **[1, 6]**: "The **Arizona** Cardinals play in Glendale **Arizona**" counting from 0.
- Posting 1 (d1) has position **[9]**.

### How the index is built (`Index.build`, `_build_zone`)

1. Stream through the corpus once. Give each new stem the next integer id (`vocab`), and append three numbers per token into compact `array('i')` buffers: **(term id, doc id, position)**.
2. **Stable-sort** the token stream by term id (`np.argsort(kind="stable")`). Tokens were appended in doc order and then position order, and a stable sort keeps that order within a term. So after one sort, each term's postings are already sorted by doc, and each posting's positions are sorted. No second sort is needed.
3. A new posting starts wherever (term, doc) changes. `np.flatnonzero` finds those starts, and **tf = gap between consecutive starts** (`np.diff`).
4. `term_ptr` comes from `np.searchsorted` over the term ids of the posting starts.

**Real numbers:** building the full index takes **26 s** and about 1.2 GB peak RAM. It holds **192,988 distinct terms** and **11.96 M body postings**, and takes 395 MB on disk (`data/index/*.npz` + `meta.json`). The demo and eval load it once (`pipeline.index()`, cached).

**Lookups:**
- `Zone.postings(t)` is an O(1) slice.
- `Zone.positions_of(t, doc)` uses `np.searchsorted` (binary search) inside t's postings to find the doc, then slices its positions.

## 6. Ranking: BM25 and cosine (`rank.py`)

### BM25: the main scorer

For a query vector `q = {term: w}` and a document `d`:

$$
\text{BM25}(d) = \sum_{t \in q} w(t)\cdot \text{idf}(t)\cdot \frac{tf_{t,d}\,(k_1+1)}{tf_{t,d} + k_1\left(1-b+b\frac{|d|}{\text{avgdl}}\right)}
$$

with **k1 = 1.2, b = 0.75** (fixed, standard values). What each piece does:

- **idf**: rare words matter more. We use the Lucene form `idf = ln(1 + (N − df + 0.5)/(df + 0.5))`, which is never negative. Real values in our corpus:

  | term | df | idf |
  |---|---|---|
  | `the` | 167,610 | **0.09**, useless for ranking |
  | `region` | 6,415 | 3.35 |
  | `adopt` | 2,713 | 4.21 |
  | `arizona` | 1,002 | 5.21 |
  | `cardin` | 516 | 5.87 |
  | `infertil` | 116 | **7.36**, very specific |

- **tf saturation**: the fraction `tf(k1+1)/(tf + k1·…)` rises with tf but levels off toward k1 + 1 = 2.2. Saying "Arizona" 10 times is not 10× better than saying it once. This is why BM25 beats raw tf-idf.
- **Length normalisation (b)**: `|d|/avgdl` compares the doc's length with the average (**103.3 tokens** for bodies here). Long passages contain more words by chance, so their tf is discounted. b = 0.75 means mostly normalised.
- **w(t)**: the query-term weight. For a plain query it is how often the term appears in the query (`tf_vec`). For our fused query it comes from the rewrite votes (§9).

**Title zone:** `score = BM25_body + α · BM25_title`, with α = 0.3. The title zone has its own avgdl (3.13 tokens) and its own idf.

**Worked toy example**, query "arizona cardinals" → terms `arizona`, `cardin`, each with w = 1:
- N = 3 and df = 2 for both terms, so idf = ln(1 + 1.5/2.5) = ln 1.6 = **0.470**. avgdl = (7 + 10 + 7)/3 = 8.
- **d0** (length 7, arizona tf = 2, cardin tf = 1). The length factor is 1.2 · (0.25 + 0.75 · 7/8) = 1.0875.
  - arizona: 0.470 · 2 · 2.2 / (2 + 1.0875) = 0.670
  - cardin: 0.470 · 2.2 / (1 + 1.0875) = 0.495
  - total **1.165**
- **d1** (length 10, tf = 1 each). The length factor is 1.2 · (0.25 + 0.75 · 10/8) = 1.425.
  - Each term gives 0.470 · 2.2 / 2.425 = 0.426, so the total is **0.853**.
- **d2** contains neither term: **0**.

Our code prints exactly `[1.1651, 0.8528, 0]`, and `rank.py`'s self-test checks these by hand. With α = 0.3 the title zone adds to both: `[1.600, 0.994, 0]`.

**How it is computed (term-at-a-time):** start a zero `float32` array of length N. For each query term, fetch its postings and add that term's BM25 contribution to those docs' slots in one vectorised numpy expression. Docs without any query term are never touched. An OOV term (not in the vocabulary, e.g. a typo, E9) is skipped.

**Top-K (`topk`):** take only the non-zero scores (`np.flatnonzero`), then `heapq.nlargest(k, …)`. That is O(n log k) instead of sorting all 183K.

**Correctness check against a library:** `eval/run_eval.py --check-bm25s` scores 20 queries with our code and with the `bm25s` library using the same tokens.
- bm25s leaves out the constant (k1 + 1) factor. That doesn't change the ranking, so we divide it out.
- The maximum difference is then **3.8 × 10⁻⁶**, which is float rounding.
- bm25s is a dev-only dependency used for this check. It never runs in the pipeline.

### Cosine (lnc.ltc): the alternative scorer

The classic vector-space model, available as `scorer: "cosine"`:
- **Documents (lnc):** weight `1 + log tf`, no idf, divided by the doc's vector length. `lnc_norm` is precomputed at index load with one `np.bincount`.
- **Queries (ltc):** weight `(1 + log qtf) · idf`, cosine-normalised.

It scores **0.149** nDCG@10 versus BM25's 0.256 for the same baseline (C0cos vs C0). It has no tf saturation and no length prior tuned like b, so long noisy passages and short title-like passages are handled worse. We keep it as an ablation that shows *why* BM25 is the default.

## 7. Boolean retrieval (`boolean.py`)

Boolean retrieval answers "which documents match exactly this condition?" with a yes or no, no scores. We use it in two places:
1. The user types an **explicit Boolean query**: `arizona NOT chicago`, `"the who" band`.
2. The **Boolean agent** (§11) builds AND queries automatically.

### Parser

- Operators `AND`, `OR`, `NOT` must be in **capitals**, so normal English like "is it not raining" isn't misread (`is_boolean`). Quotes mark phrases.
- Precedence is **NOT > AND > OR**, and adjacent words are an implicit AND. There are no parentheses (we haven't needed them).
- A parsed query is an **OR of AND-groups**. Each item is `(negated?, [stems])`; a phrase has more than one stem.

```
parse('arizona AND NOT chicago OR "the who"')
→ [[(False, ['arizona']), (True, ['chicago'])],     # group 1: arizona AND NOT chicago
   [(False, ['the', 'who'])]]                        # group 2: the phrase "the who"
```

### AND: intersect postings, smallest first

`and_group` sorts the positive items by **df, smallest first**, then intersects the doc-id arrays one by one (`np.intersect1d`; the postings are already sorted and unique). Starting from the rarest term keeps the intermediate result small. Once the running result is empty, it **stops early**, because an AND with an empty set stays empty. Every step is logged (`op`, `item`, `df`, running `result`) and shown in the demo.

Real example from our index (the agent's accepted query in §11):

```
AND adopt   df 2713  → result 2713
AND region  df 6415  → result  117
AND differ  df 8900  → result   15
```

### NOT and OR

- **NOT** subtracts with `np.setdiff1d`.
- **OR** unions the groups with `np.union1d`.

### Phrases, using positions

`phrase_docs` first intersects the words' postings to get the candidate docs. Then, for each candidate, it checks positions: a phrase `w0 w1 w2` matches if some start position `p` has `w0` at p, `w1` at p + 1 and `w2` at p + 2. The code keeps a set of candidate starts, and for word i it intersects with `{position − i}`.

**Speed fix:** a phrase like `"the who"` has huge postings (`the` is in 167K docs). Inside an AND we pass the docs that already survived (`within=`), so positions are checked only there. That brought a `"the who"` query from **1.19 s to 0.13 s**.

```
search(toy, parse('"the who"'))           → [d2]     E13: phrase made of stopwords works
search(toy, parse('"cardinals arizona"')) → []       word order matters
search(toy, parse('arizona NOT chicago')) → [d0]     E12: explicit NOT
```

### In the pipeline (explicit mode, D19)

When the user types Boolean syntax, the result set becomes a **hard filter**: we rank only the matching docs, by BM25 on the positive terms. The LLM is not used. The user asked for exact logic, so we give it to them.

## 8. LLM query rewrites (`rewrite.py`)

**What it does:** asks Google Gemini (`gemini-3.5-flash-lite`) to turn the current question plus earlier user questions into **N = 5 standalone search queries**.

**The prompt** (paraphrased):
- Resolve pronouns and ellipsis using the earlier questions.
- If the last question starts a new topic, don't carry the old topic over.
- Don't answer the question and don't invent facts.
- Return JSON `{"rewrites": [...]}`.

**Settings, and why:**
- **Temperature 0.7.** We *want* the 5 rewrites to differ: their disagreement is the signal fusion uses. At temperature 0 you'd get 5 near-copies (E7).
- **JSON response mode** (`response_mime_type="application/json"`), so the answer is reliably parseable.
- **flash-lite** takes about 1 s per call. The 3.1 version took 7–17 s per call (D14).

**Real example** (test conversation `6a738cc0…`):
```
history: "types and causes of male and female infertility", "How many years, think about adoption?"
turn:    "Regional differences"
rewrites:
  1 regional differences in adoption rates
  2 regional differences in infertility causes
  3 geographic variations in male and female infertility
  4 regional disparities in adoption statistics
  5 geographic differences regarding infertility and adoption
```
The rewrites **disagree** about whether the user means infertility or adoption. One rewrite alone would have committed to one of them. §9 shows how fusion keeps both.

### The cache: reproducibility and zero cost

- Every answer is saved in `cache/rewrites.json`, which is **committed to git**. The key is the SHA-1 of `[history, turn, n]`.
- `python -m convsearch.rewrite --all` filled it once for all 222 user turns.
- The eval runs with **`offline=True`**: it only reads the cache and never calls the API (E15). Anyone can clone the repo and get **exactly** our numbers with no API key. We checked this with a fresh clone.

### Failures (E14)

On a 429 rate limit, a network error or bad JSON, the code retries 3 times with **5 s, 10 s, 20 s** backoff. If that still fails, `rewrites()` returns `None` with an `llm_fallback` note, and fusion falls back to **no-LLM mode** (§9). Search never breaks because the LLM is down.

**The key rule:** the LLM's output is only *text*. Our code decides what to do with it.

## 9. Rewrite fusion (`fuse.py`)

This is the core idea of the project.

### The formulas

Given N rewrites:

- **agree(t)** = (number of rewrites containing term t) / N. It is a **vote**: 1.0 means every rewrite agreed, 0.2 means only one rewrite added it.
- **origin(t)**:
  - `turn` if the user typed t in this question;
  - else `history` if t appears in an earlier user turn;
  - else `llm-only` (the LLM introduced it).
- **Drop rule (E6):** if `agree(t) < min(2, N)/N` (with N = 5, that means only 1 of 5 rewrites) **and** the user didn't type it now, drop it. A word that only one rewrite thought of is likely a hallucination, like "2023 season".
- **Weight:** `w(t) = agree(t) · idf(t) + β · idf(t) if t is a turn term`, with **β = 0.5**. The β bonus means words the user actually typed always count extra: they are the most trustworthy evidence.
- **OOV terms** (not in the index) get weight 0 and the note `OOV` (E9).

**The `w_idf` switch (D16).** The spec multiplied by idf, but BM25 *already* multiplies by idf, so using both gives idf². Tuning chose `w_idf=False`, which divides the idf back out. So in practice:

```
w(t) = agree(t)  (+ β if typed this turn)
```

The fused query vector `{term: w}` is then scored by BM25 like any other query. Each rewrite isn't searched separately: we search **once** with the merged weights. That is why fusion costs no more than a single search (about 10 ms).

**Diversity flag (E7):** if all 5 rewrites produce the same term set, fusion flags `diversity: 0` in the trace. The votes then carry no information, and the result equals the single-rewrite case.

**Phrase candidates:** pairs of adjacent content words that appear together in at least ⌈N/2⌉ = 3 rewrites (e.g. `arizona cardin`). The Boolean agent can use them (§11).

### Real worked example: "Regional differences"

The fusion table that the trace and demo show (sorted by idf-weighted weight; `w` is after `w_idf=False`):

| term | agree | idf | origin | w | note |
|---|---|---|---|---|---|
| infertil | 0.6 | 7.36 | history | 0.60 | |
| region | 0.6 | 3.35 | **turn** | **1.10** | 0.6 + β |
| differ | 0.6 | 3.03 | **turn** | **1.10** | 0.6 + β |
| adopt | 0.6 | 4.21 | history | 0.60 | |
| geograph | 0.4 | 5.34 | llm-only | 0.40 | 2 of 5 rewrites, kept |
| caus | 0.2 | 3.16 | history | 0 | dropped: low agreement |
| dispar | 0.2 | 6.94 | llm-only | 0 | dropped: low agreement |
| statist | 0.2 | 5.22 | llm-only | 0 | dropped: low agreement |
| … | 0.2 | | | 0 | dropped |

What it shows:
- Both readings, infertility and adoption, survive at 0.6. Fusion **hedges** where a single rewrite would gamble.
- One-off words like "disparities" and "statistics" are dropped.
- The words the user actually typed get the highest weight.

### No-LLM mode (C9, E16, and the E14 fallback)

With `rewrites=None`, fusion makes **one pseudo-rewrite** = earlier turns + current turn concatenated. Then every term has agree = 1, so turn terms get 1.5 and history terms 1.0. This is "concat, but the current turn is boosted". It scores 0.295, well above C1's plain concat (0.235), with no LLM at all.

## 10. Topic-shift gate (`gate.py`)

**Problem:** sometimes the user really does change topic ("what causes the northern lights in norway" after the Cardinals). Then the history terms are noise.

**The gate decides `shift = yes/no` from three conditions, all of which must hold:**

1. **self_contained** = no anaphor **and** at least 3 content terms **and** Σ idf ≥ τ_idf (= 16).
   - Anaphors are words that point back: `it, its, they, them, their, that, this, those, these, there, he, she, him, her, his, one`.
   - "Do they play…?" contains "they", so it is not self-contained (E2).
   - "And Chicago?" has only 1 content term, so it is not self-contained (E5).
   - The idf sum makes sure the question is specific enough to stand alone.
2. **overlap < τ_j** (= 0.1). This is the Jaccard overlap (shared / union) between the top-10 that *this turn alone* would retrieve and the top-10 that was shown for the previous turn. Near 0 means "this question lives in a different part of the corpus".
3. **hist_agree < τ_h** (= 0.4). This is the mean agreement of the history-origin terms in the fusion table. If most rewrites carried the old topic over, the LLM thinks the context is needed (D21). It is used only when real rewrites exist; in no-LLM mode every history term has agree 1, so it would always block.

**If shift = yes:** every history-origin term gets `w = 0` and the note "zeroed: topic shift".

**Why not Jaccard alone (D10, E4)?** "what about its price?" also has about 0 overlap with the previous results, since its words alone retrieve random stuff, but it is clearly a follow-up. Condition 1 catches it: "its" is an anaphor.

In the "Regional differences" example the gate output is `content_terms 2, sum_idf 6.38 → self_contained False, hist_agree 0.6 → shift False`. History is kept, correctly.

**Honest finding (D21):**
- *With* an LLM, the gate adds nothing (C5 = C4 = 0.3484 on test). The rewrites already drop old topics, because the prompt tells them to.
- *Without* an LLM it helps: on the tune split, no-LLM goes from 0.276 to 0.293 with the gate.

We report this honestly instead of hiding it.

## 11. Boolean agent (`agent.py`)

**Idea:** BM25 ranks by soft evidence, so a passage that mentions "adoption" 10 times can beat one that mentions all of "regional", "differences" and "adoption". A Boolean AND finds the passages that contain **all** the key terms. The question is which terms to AND. Too many gives 0 hits; too few gives thousands.

**The agent is a small feedback loop driven by the postings sizes (df), up to 3 attempts:**

1. Start with the **AND of the top-4 weighted terms** from the fusion table.
2. Run it and get the hit count **h** from the intersection.
   - **h = 0, too strict (E10):** drop one term. The choice:
     - terms the user typed this turn are dropped **last** (D18);
     - otherwise the **lowest agree** goes first;
     - on a tie, the **highest idf** goes, because the rarest term over-constrains most.
   - **h > 500, too broad (E11):** add the next-weight term. If none is left, replace two adjacent words with their **phrase** (positional match). If neither is possible, stop.
   - **1 ≤ h ≤ 500:** accept. These docs are the set **B**.
3. If nothing is accepted after 3 attempts, B stays empty and there's no effect.

**B is a bonus, not a filter (D11):** every doc in B gets `+ λ · (max BM25 score)` with **λ = 0.2**. A hard filter would throw away good passages that happen to miss one word, which hurts recall badly. Because the bonus is relative to the top score, it means the same thing for every query, whatever the score scale (D17).

**Real trace, "Regional differences":**

```
attempt 1: infertil AND region AND differ AND adopt
           AND infertil df 116 → 116 · AND adopt df 2713 → 7 · AND region df 6415 → 0
           0 hits → drop 'infertil'   (agree tie 0.6 with adopt; region/differ are turn terms → kept;
                                       infertil has the higher idf → dropped)
attempt 2: region AND differ AND adopt → 15 hits → accept
           bonus = 0.2 × max score = 2.729 added to those 15 docs
```

C2, the single rewrite, scored nDCG@10 = **0** on this turn. Our C6 scored **0.5**.

**Why D18 exists:** in no-LLM mode every term has agree = 1. The old tie-break, highest idf first, dropped the one word the user typed, "chicago" in "And Chicago?", which is exactly the wrong word to lose.

**Where it helps:** with the LLM, Boolean takes C5 → C6 from 0.348 to **0.356**. In no-LLM mode it slightly hurts (0.276 → 0.259 on tune without the gate), because without votes the top-4 terms are noisier.

## 12. Stretch: dense re-rank and Personalized PageRank (`rerank.py`)

These run only in C7/C8 and only re-order the **top-100** from BM25. The core system (C6) is pure lexical IR.

**Final score:** `final(d) = score(d)/max_score + μ · minmax(dense(d)) + ν · minmax(ppr(d))`. Min-max scaling maps each signal to [0, 1] within this query's 100 docs, so the weights μ and ν mean the same thing for every query (D17).

### Dense (μ = 3)

- **e5-small-v2** is a small (33M-parameter) pretrained sentence-embedding model run with `sentence-transformers` on the CPU. It maps text to a 384-dimensional vector so that texts with similar meaning are close.
- We embed the query as `"query: <first rewrite>"` and each passage as `"passage: <title>. <text>"` (the prefixes e5 expects), and score with cosine similarity.
- Dense matching catches **synonyms and paraphrases** that BM25 misses ("geographic variations" ≈ "regional differences").
- Passage embeddings are cached in `data/e5_cache.npz`. The first query loads the model (about a minute), and later ones take about 1.6 s per 100 new passages.
- μ was tuned on the tune split. Gains rise and then **plateau at μ ≈ 3** (D22).

### Personalized PageRank (ν = 0.1)

- Build a small graph: the top-100 plus the previous turn's top-10. Each passage links to its **5 most similar passages** (tf-idf cosine; `tfidf_matrix`), and the graph is made undirected.
- Run PageRank with damping 0.85 for 50 iterations, but **teleport only to the previous turn's top-10**. Passages strongly connected to what the user was just reading score high. That is a conversational "stay near the current topic" signal computed from the corpus itself.
- It needs a previous turn, so it is skipped on turn 1.
- It adds only about **+0.002** (C7 0.527 → C8 0.529). The honest takeaway is that it is a minor effect.

**Results:** C7/C8 reach **0.527 / 0.529**, versus 0.356 for C6. We present this as a **stretch result, not credited to fusion**. Note also that e5 was trained on data that includes Natural Questions, which ClapNQ is built from, so part of this jump may be the model having seen similar question/passage pairs.

## 13. Putting it together (`pipeline.py`)

`pipeline.run(turn, history, prev_topk, config, ix)` → `(ranked, trace)`. The eval and the demo both call only this function. A `config` dictionary switches components on and off, so **every ablation is a config change, not a code change**.

**Order of operations:**

1. **Explicit Boolean?** (`is_boolean`) If yes: parse, run `search` to get the set B, score BM25 on the positive terms, zero every doc outside B, and return. This is the filter mode from §7.
2. **Build the query vector** according to `config["query"]`:
   - `turn`: the current turn only (C0).
   - `concat`: all turns glued together (C1).
   - `rewrite1`: the first cached rewrite as plain text (C2, D15).
   - `fusion`: rewrites, then `fuse`, then the `w_idf` adjustment (C4+). Then:
     - **E8 fallback:** if the vector is empty ("what about that one?") and there is history, reuse the previous turn's fused vector.
     - **Gate**, if on and there is history (§10).
3. **Empty query** with no history: return nothing, with the message "query too vague" (E8).
4. **Score** with BM25 (body + α·title), or with cosine if `scorer="cosine"`.
5. **Boolean agent**, if on (fusion only): add the λ bonus to set B.
6. **Top-100** via the heap, then the optional **rerank** (μ, ν).
7. Cut to `k`, and store `trace["top"]`.

**`DEFAULT` config** (tuned on the tune split; `eval/tune.py` writes `results/tune_C6.csv`):

| key | value | meaning |
|---|---|---|
| query | turn | turn / concat / rewrite1 / fusion |
| n | 5 | rewrites used (C4n1 and C4n3 test 1 and 3) |
| llm | True | False gives no-LLM fusion (C9) |
| offline | False | True means cache-only (the eval sets it) |
| gate, boolean | False | component switches |
| scorer | bm25 | or cosine |
| w_idf | False | fused weight agree·idf (True) or agree (False) |
| α | 0.3 | title-zone weight |
| β | 0.5 | bonus for terms typed this turn |
| λ | 0.2 | Boolean bonus × max score |
| τ_idf, τ_j, τ_h | 16, 0.1, 0.4 | gate thresholds |
| μ, ν | 0, 0 | dense / PPR weights (C7: μ 3; C8: μ 3, ν 0.1) |
| k | 100 | results returned |

**The trace** holds: config, turn, history, rewrites and their cache info, the fusion table (term, agree, idf, origin, w, note), phrases and flags, the gate numbers, every agent attempt with its intersection steps, the bonus, the rerank scores of the top-10, the final query vector, OOV terms and the top-10. The demo renders exactly this.

**Speed:** C6 takes **10.6 ms per turn** with cached rewrites, over all test turns. A live Gemini call adds about 1 s.

## 14. Edge cases E1–E16

Each one is covered by an `assert` in some module's `__main__` self-test (`uv run python -m convsearch.<module>`).

| # | Case | Example | How we handle it | Test |
|---|---|---|---|---|
| E1 | First turn | "where do the arizona cardinals play this week" | No history, so fusion is effectively over the rewrites; the gate is skipped | first-turn split in results |
| E2 | Pronoun follow-up | "Do they play outside the US?" | Anaphor "they" → not self-contained → history kept | gate |
| E3 | Real topic shift | "english rock band history" after Cardinals | Self-contained + 0 overlap → shift → history zeroed | gate, pipeline |
| E4 | Looks like a shift but isn't | "what about its price?" | Anaphor "its" blocks the gate (why Jaccard alone fails) | gate |
| E5 | Ellipsis | "And Chicago?" | 1 content term → not self-contained; "chicago" boosted by β | gate, pipeline |
| E6 | Hallucinated rewrite word | "+2023 season" in 1 of 5 | agree 0.2 < 0.4 → dropped | fuse |
| E7 | All rewrites identical | 5 copies | `diversity: 0` flag, behaves like a single rewrite | fuse |
| E8 | Empty after stopwords | "what about that one?" | Reuse the previous turn's vector; with no history, "query too vague" | pipeline |
| E9 | OOV / typo | "Cardnals" | Term skipped in scoring, `OOV` note in the table | rank |
| E10 | Boolean 0 hits | over-specific AND | Agent drops a term; may give up, leaving B empty | agent |
| E11 | Boolean too broad | "nfl teams" | Agent adds a term or switches to a phrase | agent |
| E12 | Natural-language negation | "not the Chicago one" | **Limitation:** BM25 can't negate, so "chicago" is a query word that *boosts* Chicago. Explicit `cardinals NOT chicago` works. | boolean, pipeline |
| E13 | Phrase made of stopwords | `"the who"` | Stopwords + positions are indexed | boolean |
| E14 | LLM failure | 429, bad JSON | Retry with backoff, then fall back to no-LLM fusion | rewrite |
| E15 | Eval rerun | — | `offline=True`, cache only, 0 API calls | eval configs |
| E16 | No LLM at all | C9 | One pseudo-rewrite = history + turn | fuse, pipeline |

---

# Part C: Evaluation, demo, tooling

## 15. Evaluation

### Protocol (`eval/run_eval.py`)

- For each test conversation, run **every user turn in order**. History = earlier user turns, and `prev_topk` = **this same config's** own top-10 from the previous turn, exactly what a live user would have seen.
- Score only turns that have qrels. A passage is relevant if its qrel score is > 0.
- Results go to `results/test.csv` (means) and `results/test_perturn.json` (every turn, used for significance tests).
- C3 is special: it feeds MTRAG's **human rewrite** as the query, with no history. That is the "what if a person rewrote it" oracle.

### Metrics

Use a toy ranked list `[R, N, R, N, N, …]`, where R is relevant, and suppose 3 relevant passages exist in total.

| Metric | Definition | Toy value |
|---|---|---|
| **P@5** | relevant in top-5 / 5 | 2/5 = 0.4 |
| **P@10** | relevant in top-10 / 10 | 0.2 |
| **R@10** | relevant in top-10 / all relevant | 2/3 = 0.67 |
| **nDCG@10** (main) | DCG = Σ rel_i / log₂(i+1) over the top 10; divided by the ideal DCG (all relevant at the top) | DCG = 1/1 + 1/2 = 1.5; IDCG = 1 + 0.631 + 0.5 = 2.131; **0.70** |
| **MAP** | mean over queries of AP; AP = average of precision at each relevant hit, divided by the number relevant | (1/1 + 2/3)/3 = 0.56 |

nDCG is the main metric because it rewards putting relevant passages **near the top**, which matters for a chat UI, and it is the metric MTRAG reports.

### Configs and test results (141 judged turns, 19 conversations)

| Config | What it is | nDCG@10 | P@5 | R@10 | MAP |
|---|---|---|---|---|---|
| C0 | current turn only (baseline) | 0.256 | 0.126 | 0.343 | 0.202 |
| C1 | concat all turns | 0.235 | 0.123 | 0.305 | 0.193 |
| C2 | single LLM rewrite | 0.305 | 0.152 | 0.391 | 0.243 |
| C3 | human rewrite (oracle) | 0.293 | 0.155 | 0.384 | 0.232 |
| C4 | fusion of 5 rewrites | 0.348 | 0.175 | 0.438 | 0.285 |
| C5 | + gate | 0.348 | 0.175 | 0.438 | 0.285 |
| **C6** | **+ Boolean agent (our core)** | **0.356** | **0.187** | **0.452** | **0.287** |
| C7 | C6 + dense (stretch) | 0.527 | 0.264 | 0.622 | 0.439 |
| C8 | C7 + PPR (stretch) | 0.529 | 0.267 | 0.628 | 0.439 |
| C9 | fusion + gate + Boolean, **no LLM** | 0.295 | 0.152 | 0.391 | 0.235 |
| C0cos | C0 with cosine instead of BM25 | 0.149 | 0.068 | 0.210 | 0.123 |
| C4n1 / C4n3 | fusion with 1 / 3 rewrites | 0.328 / 0.345 | | | |

**The story the numbers tell:**
- **Context matters:** C0 is 0.436 on first turns but 0.229 on follow-ups.
- **Naive context hurts:** C1 < C0.
- **Fusion beats a single rewrite:** C4 0.348 vs C2 0.305. More rewrites help (N = 1 → 3 → 5: 0.328 → 0.345 → 0.348). On follow-ups, C6 is 0.343 vs C0's 0.229.
- **Fusion even beats the human rewrite** (C3). A human writes one good query; votes over five give a weighted query with more useful terms.
- **No-LLM fusion (C9)** almost matches the LLM single rewrite (0.295 vs 0.305) at zero LLM cost.

### Significance

`eval/report.py` runs a **paired t-test** over the 141 per-turn nDCG scores of C6 vs C2. "Paired" because both systems see the same turns, so we test the per-turn differences.

```
C6 vs C2: mean 0.3563 vs 0.3051, win/tie/loss 54/65/22, t = 3.72, p = 0.0003
```

p < 0.001, so the improvement is very unlikely to be chance. The script also writes the bar chart (`results/test_bar.png`) and nDCG by turn depth (`results/test_depth.png`).

### Tuning (`eval/tune.py`)

A grid search **on the tune split only** over `w_idf × β × α × λ × τ_idf × τ_j`, with mean nDCG@10 as the objective. It writes every grid point to `results/tune_C6.csv`, and the best setting became `DEFAULT`. The μ/ν grid for the stretch was a separate run (D22).

### Self-judged set (`eval/judge.py`)

The brief also wants our own queries with our own judgments. MTRAG's qrels can't judge passages that no system in MTRAG ever retrieved, so we use **TREC-style pooling**:

1. `new --name conv1`: you type a conversation. It runs C0, C2, C6, C8 and C9, merges each one's top-5 into one **pool** per turn, and **shuffles** it, so judges can't tell which system found what (no bias).
2. `label --name conv1 --judge A`, then the same with `--judge B`: each of you labels each pooled passage y/n **independently**. It saves after every answer.
3. `score`: P@5 and nDCG@5 per config, under two definitions of relevance: "both judges said yes" (strict) and "either said yes" (lenient). It also reports **Cohen's kappa**, which measures how much we agree *beyond chance*: κ = (p_observed − p_chance)/(1 − p_chance). Above 0.6 is good agreement.

## 16. The demo app (`app.py`)

A Streamlit chat interface: `uv run streamlit run app.py`.

- **Sidebar:**
  - pick a preset (C8, C6, C4, C9, C2, C0);
  - tweak α, β, λ, τ_idf, τ_j, μ, ν live;
  - optionally show the **C0 baseline side by side**;
  - "New conversation" resets.
- **Each turn** shows your question, then expandable panels in pipeline order:
  1. the LLM rewrites;
  2. the **fusion weight table**;
  3. the gate decision and its numbers;
  4. each Boolean agent attempt with its intersection steps;
  5. rerank scores;
  6. the final query vector.

  Then the top-10 passages with scores.
- The index loads once (`st.cache_resource`). The question shows up immediately, with a spinner while the search runs. The first dense query loads e5 (about a minute).
- Typing `arizona NOT chicago` or `"the who" band` triggers explicit Boolean mode, and the trace shows the parsed query and the intersection steps.
- The app **computes nothing itself**: it calls `pipeline.run` and displays the trace.

**Good demo script:** "where do the arizona cardinals play" → "Do they play outside the US?" → "And Chicago?". Then show "not the Chicago one" failing (E12) and `cardinals NOT chicago` working.

## 17. Libraries and tools

| Tool | Used for | Why this one |
|---|---|---|
| **Python 3.14 + uv** | language and package manager | `uv` makes installs reproducible (`uv.lock`) and fast; `uv run` needs no venv activation |
| **numpy** | postings arrays, scoring, intersections | flat arrays fit 12M postings in RAM; vectorised scoring |
| **nltk** | Porter stemmer only | standard, well-known stemmer (we didn't write our own stemmer; that is allowed, but scoring is ours) |
| **google-genai** | Gemini API client | free tier; only used to fill the cache |
| **python-dotenv** | reads `GEMINI_API_KEY` from `.env` | the key stays out of git (`.env` is gitignored) |
| **sentence-transformers** | runs e5-small-v2 | stretch only |
| **streamlit** | demo UI | about an hour to build, looks clean on video |
| **bm25s** (dev only) | BM25 correctness cross-check | never in the pipeline |
| **matplotlib, scipy** (dev only) | plots, paired t-test | |
| **pandoc + xelatex** | `report/report.md` → PDF | write in Markdown, get a LaTeX-quality PDF |
| **graphviz** | `report/pipeline.dot` → PNG | the diagram is text, so it can be diffed |
| **git + GitHub** (`rgabhi2526/ir-convsearch`, private) | version control | |

**What is ours (hand-written):** tokenisation, the index, BM25, cosine, Boolean parsing and intersection, phrase matching, fusion, the gate, the agent, PPR, the evaluation metrics and the pooling tool. **Not ours:** the stemmer, the LLM and the e5 model. All are declared in the report.

## 18. Repo map and commands

```
src/convsearch/
  data.py      download + loaders, splits
  text.py      normalise, tokenise, stem, stopwords
  index.py     positional inverted index (numpy CSR), build/save/load, toy corpus
  rank.py      BM25 (zones), cosine lnc.ltc, top-K
  boolean.py   parser, df-ordered intersection, phrases, NOT/OR
  rewrite.py   Gemini rewrites + committed cache
  fuse.py      votes → weighted query
  gate.py      topic-shift gate
  agent.py     Boolean agent with df feedback
  rerank.py    [stretch] dense e5 + Personalized PageRank
  pipeline.py  run() — the one entry point
eval/  run_eval.py · tune.py · report.py · judge.py
app.py                 Streamlit demo
cache/rewrites.json    all LLM rewrites (committed)
results/               CSVs, per-turn JSON, plots, significance
report/                report.md → report.pdf, pipeline diagram
DEVLOG.md              decisions D1–D23, progress, AI-use log
```

```bash
uv sync                                                # install
uv run python -m convsearch.data                       # download data (~45 MB)
uv run python -m convsearch.index --build              # build index (26 s)
uv run python -m convsearch.pipeline                   # self-test (also: text, index, rank, boolean, fuse, gate, agent, rerank)
uv run python eval/run_eval.py --configs C0,C2,C6 --split test
uv run python eval/run_eval.py --check-bm25s
uv run python eval/report.py --split test --a C6 --b C2
uv run streamlit run app.py
```

---

# Part D: Viva prep

## 19. Honest limitations

1. **Natural-language negation (E12):** "not the Chicago one" adds "chicago" as a query term, which boosts what the user wanted to exclude. Only explicit `NOT` works.
2. **The gate is redundant when the LLM is on** (C5 = C4). It earns its place only in no-LLM mode.
3. **Our fusion gains are modest in absolute terms:** 0.356 is still far from perfect. Many MTRAG passages are judged only for answers the original agent gave, so sparse qrels under-count correct results.
4. **Dense gains come from a pretrained model.** That is why C7/C8 are reported separately from our core. e5 may also have seen Natural Questions data.
5. **One loss pattern:** history terms can hurt when the follow-up narrows the topic. Example: Cardinals turn 2, where "week" was carried over from turn 1. C6 scores 0.531 against C0's 0.671 there.
6. **Small test set:** 141 turns. The C6 vs C2 difference is significant, but smaller differences (C5 vs C6) are not.
7. **LLM rewrites are cached from one model at temperature 0.7.** A rerun with a fresh API call would give different rewrites. The cache is what makes our numbers reproducible.

## 20. Likely viva questions

**Q: Why not just use the LLM to search or answer?**
The assignment is about IR. More importantly, an LLM can't see a 183K-passage corpus. Retrieval has to come from an index. We use the LLM for the one thing it is good at here, understanding conversational language, and keep it out of ranking.

**Q: Why 5 rewrites instead of 1?**
Voting. One rewrite either hallucinates or misses context, and you can't tell which. With five, agreement tells us how sure the LLM is about each term. Results: N = 1 → 0.328, N = 3 → 0.345, N = 5 → 0.348, against a single plain rewrite at 0.305.

**Q: Isn't searching 5 rewrites 5× slower?**
We don't search them separately. We merge them into **one** weighted query vector and search once, in about 10 ms.

**Q: Why BM25 over tf-idf cosine?**
tf saturation (k1) and length normalisation (b). Measured: 0.256 vs 0.149 on the same queries.

**Q: Why keep stopwords in the index?**
For phrases: `"the who"`. The cost is index size. They are still removed from ranked queries.

**Q: How do you know your BM25 is correct?**
We compared it with the bm25s library on 20 queries: the maximum difference is 3.8e-6, after accounting for bm25s leaving out the constant (k1 + 1) factor. The toy self-test also checks values computed by hand.

**Q: Why is the Boolean result a bonus and not a filter?**
A filter throws away good passages that miss one word, which kills recall. A bonus of λ · max score lifts the all-terms-present docs without excluding the rest. When the *user* types Boolean, it is a filter, because they asked for exact logic.

**Q: Why order AND intersections by df?**
The intermediate result can never be larger than the smallest list, so starting small keeps every intersection cheap, and we stop as soon as it is empty.

**Q: How did you avoid overfitting your parameters?**
Tuning used only the 10 tune conversations. All reported numbers are from the 19 test conversations, which we never tuned on.

**Q: Can someone reproduce your numbers?**
Yes. Rewrites are cached in git, and the eval never calls the API. We did a fresh clone with no API key and got identical numbers.

**Q: Why does fusion beat even the human rewrite (C3)?**
The human writes one clean question, so BM25 gets one set of words. Fusion gives a *weighted* set that includes paraphrase variants from several rewrites, and it boosts the user's own words. That is more matching evidence for BM25.

**Q: What does the gate actually do, if C5 = C4?**
With the LLM, nothing measurable: the prompt already makes rewrites drop old topics, and our third condition (hist_agree) keeps the gate from firing falsely. In no-LLM mode, where nothing else can detect a topic change, it improves the tune split from 0.276 to 0.293.

**Q: What is Personalized PageRank doing here?**
It is PageRank where the random surfer teleports only to the passages from the previous turn. Passages similar to (linked with) what the user was just reading get higher scores, a "stay on topic" signal. The effect is small (+0.002).

**Q: What is Cohen's kappa and why does it matter?**
It is agreement between our two judges, corrected for chance agreement. Without it, our self-judged scores could just reflect one person's opinion.

## 21. Glossary

| Term | Meaning |
|---|---|
| **Posting / postings list** | one (doc, tf, positions) entry; all of a term's entries |
| **df / idf** | number of docs containing the term; log-scaled rarity weight |
| **tf** | occurrences of a term in a doc |
| **Zone** | a separately indexed field (title, body) |
| **CSR** | compressed sparse row: flat arrays plus pointer array |
| **Stemming** | cutting words to a root (`cardinals` → `cardin`) |
| **OOV** | out of vocabulary: the term never occurs in the corpus |
| **Anaphor** | a word pointing back to earlier context ("they", "it") |
| **Ellipsis** | an omitted part of the question ("And Chicago?") |
| **Query rewrite** | turning a context-dependent question into a standalone one |
| **agree(t)** | fraction of rewrites containing term t |
| **Jaccard** | \|A ∩ B\| / \|A ∪ B\| |
| **qrels** | relevance judgments (query, passage, score) |
| **nDCG@10** | ranking quality of the top 10, 1.0 = ideal order |
| **Ablation** | turning off one component to measure its contribution |
| **Pooling** | judging only the union of several systems' top results |
| **Paired t-test** | significance test on per-query differences between two systems |
| **Dense retrieval** | matching by neural embeddings instead of words |
| **PPR** | Personalized PageRank: PageRank with teleport to chosen seed nodes |
| **Tune / test split** | conversations used to set parameters vs to report scores |
