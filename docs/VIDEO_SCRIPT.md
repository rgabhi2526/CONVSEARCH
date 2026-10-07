# Demo video script (2 people, ~7 min, no slides)

**A** = owns text, index, rank, boolean, agent, eval. **B** = owns rewrite, fuse, gate, rerank, pipeline, app.
Replace A/B with your names. Everything on screen is live: the app, the terminal and the code. There are no slides (brief rule).

Covers every "video must show" item: problem + track fit (≤ 1 min) · end-to-end run on real queries · a limitation · code + intermediate output (postings, weights, scores) · results vs baseline · each member explains their own part.

Every query below was dry-run on 2026-10-07. The results quoted are what the screen will show. Their rewrites are in `cache/rewrites.json`, so the demo works even if Gemini is slow. **Type them exactly as written, with the same case and punctuation and in the same order.** The cache key is the exact turn plus its history, so any change means a live API call and possibly different rewrites.

---

## Before you hit record

- [ ] `uv run streamlit run app.py`, open it, and run one throwaway query so the index is loaded.
- [ ] In the sidebar: **C6 core**, with "Show C0 baseline next to it" **on**. (Don't use C8. Its first query loads e5 for about a minute.)
- [ ] The editor has these tabs open: `src/convsearch/index.py`, `rank.py`, `agent.py`, `fuse.py`, `gate.py`, `pipeline.py`.
- [ ] The terminal is in the repo root, with a large font (≥ 18 pt).
- [ ] Open `results/test_bar.png` and `results/test_depth.png` in an image viewer.
- [ ] Turn off notifications. Check both mics. Each person records their own screen share, or one person drives while the other narrates.
- [ ] Do one full rehearsal with a timer. Aim for 7:00; 5–8 min is allowed.

---

## 0:00–0:50 · Problem and track fit (B)

**SCREEN:** app, empty chat.

**B:**
> "Hi, we're A and B, and this is our Track 2 project: conversational search. In a conversation people don't repeat themselves. They ask 'where do the Arizona Cardinals play', then 'do *they* play outside the US?' A normal search engine sees only 'play outside US' and has no idea who 'they' are.
> Our system takes the whole conversation and finds the right Wikipedia passages, from 183,000 of them in IBM's MTRAG ClapNQ benchmark. The key rule we set ourselves: an LLM may *suggest* rewrites, but all indexing, retrieval and ranking is our own IR code."

---

## 0:50–2:40 · Live end-to-end demo (B drives, A reacts)

**DO:** type `where do the arizona cardinals play this week`.

**B:**
> "First turn. Ours is on the left, the plain baseline on the right. Both find the 2017 Arizona Cardinals season. Easy."

**DO:** type `Do they play outside the US?`

**B:**
> "Now the follow-up. Look at the baseline on the right: 'Outside Edge', 'Wembley Stadium'. It lost the topic. Ours on the left still shows the Arizona Cardinals season."

**DO:** expand **1 · LLM rewrites**, then **2 · Fusion**.

**B:**
> "Here's why. We ask Gemini for five rewrites, not one. Then we treat them as votes. This table is the fusion: 'cardin' and 'arizona' appear in all five rewrites, so agreement is 1.0, and the origin column says they came from history. 'outside', which the user actually typed, gets a bonus. Words only one rewrite invented get dropped. The LLM proposes text; this table is our code deciding."

**DO:** expand **3 · Topic-shift gate**.

**B:**
> "The gate checks whether the topic changed. 'They' is a pronoun, so this isn't self-contained: keep history."

**DO:** type `who wrote the novel pride and prejudice`, then expand the gate.

**B:**
> "Now a real topic change. No pronoun, four specific words, and zero overlap with the previous results, so the gate says SHIFT and zeroes the Cardinals terms. The results are Pride and Prejudice."

**DO:** click **New conversation**. Type `where do the arizona cardinals play this week`, `Do they play outside the US?`, then `not the Chicago one`.

**A:**
> "And here's a limitation we want to be honest about. 'Not the Chicago one': ranking can't do negation. 'chicago' becomes the *highest*-weighted term, see the table, so we get Illinois and Rhode Island. Ranked retrieval only adds evidence; it can't subtract."

**DO:** type `arizona cardinals NOT chicago`, then expand the Boolean panel.

**A:**
> "If you write it as Boolean, our parser handles it: 'cardin' has 516 docs, AND 'arizona' leaves 138, NOT 'chicago' removes 33, leaving 105. The top results are all Arizona Cardinals pages. That intersection comes from our own postings lists, which brings us to my part."

---

## 2:40–4:20 · Index, BM25, Boolean (A)

**SCREEN:** `index.py`, the class docstring at the top.

**A:**
> "I built the index. It's a positional inverted index over titles and bodies, and to fit 12 million postings in 8 GB of RAM it's five flat numpy arrays, not Python lists. term_ptr says where a term's postings start; post_doc, post_tf and positions hold the rest. We build it with one stable sort of the token stream, which takes 26 seconds for the whole corpus."

**DO:** in the terminal (≈ 4 s):

```bash
uv run python -c "
from convsearch import pipeline, rank
ix = pipeline.index(); B = ix.zones['body']; t = ix.tid('cardin')
docs, tf = B.postings(t); d = int(docs[tf.argmax()])
print('df =', B.df(t), ' idf =', round(rank.idf(ix, 'cardin'), 2))
print('first postings (doc, tf):', list(zip(docs[:5].tolist(), tf[:5].tolist())))
print('positions in', ix.titles[d], ':', B.positions_of(t, d).tolist())
"
```

**A:**
> "This is the live postings list for 'cardinals', stemmed to 'cardin': document frequency 516, idf 5.87, then doc ids with term frequencies, and the exact word positions inside one doc. Positions are what make phrase search work. We even keep stopwords so that a query for the band "The Who" works."

**SCREEN:** `rank.py`, the `bm25_zone` function.

**A:**
> "Ranking is BM25, hand-written: idf times a saturating tf, normalised by document length, with k1 = 1.2 and b = 0.75. Plus 0.3 times the same score on the title zone. To prove it's correct we compared it with the bm25s library on 20 queries: the maximum difference is 4 in a million, which is just float rounding."

**SCREEN:** `boolean.py`, the `and_group` function.

**A:**
> "For Boolean we intersect postings smallest-df first, so the intermediate list stays small, and we stop as soon as it's empty. Phrases are checked with positions, only inside docs that already survived the AND."

---

## 4:20–5:50 · Rewrites, fusion, gate (B), then the agent (A)

**DO:** in the app, **New conversation**. Type these three turns, with C0 shown next to them:
1. `types and causes of male and female infertility`
2. `How many years, think about adoption?`
3. `Regional differences`

**B:**
> "This is a real test conversation from MTRAG. 'Regional differences': differences in *what*? Look at the five rewrites. Two say infertility, two say adoption, one says both. The LLM itself isn't sure."

**SCREEN:** the fusion table expanded.

**B:**
> "A single rewrite would pick one reading and gamble. Fusion keeps both at agreement 0.6. 'region' and 'differ' are what the user typed, so they get the top weight, 1.1. 'disparities' and 'statistics' appear in only one rewrite, so they're dropped as likely hallucinations. All five rewrites become *one* weighted query, so we search once, in about 10 ms."

**SCREEN:** `fuse.py`, the `fuse` function, the agree/w lines. Then `gate.py`, the `shift =` line.

**B:**
> "In code it's a few lines: agree is the share of rewrites containing the term; weight is agree plus beta if typed now. The gate needs three things: no pronoun with enough rare words, near-zero result overlap with the last turn, and low agreement on history terms. Jaccard alone would wrongly flag 'what about its price?'."

**DO:** expand **4 · Boolean agent**.

**A:**
> "Then my Boolean agent. BM25 can rank a page high just for repeating 'adoption'. The agent ANDs the top four terms: infertility AND region AND differ AND adopt. You can see the intersection: 116, then 7, then zero. Zero hits, too strict. So it drops a term: never what the user typed, and among ties the rarest, infertility. Region AND differ AND adopt gives 15 hits, and we accept. Those 15 get a bonus of 20% of the top score. It's a bonus, not a filter, so we don't kill recall. On this turn the single-rewrite system scores zero nDCG; ours scores 0.5."

---

## 5:50–7:00 · Evaluation (A), stretch and wrap-up (B)

**DO:** in the terminal (≈ 10 s; `--out` keeps the committed results untouched):

```bash
uv run python eval/run_eval.py --configs C0,C2,C6 --split test --out video_demo
```

**A:**
> "Evaluation runs live on the 19 test conversations, 141 judged turns. We tuned only on 10 other conversations. The baseline gets nDCG@10 0.256; a single LLM rewrite 0.305; our fusion with gate and Boolean agent 0.356. P@5 goes from 0.126 to 0.187, and recall@10 from 0.34 to 0.45. Rewrites come from a committed cache, so this makes zero API calls and anyone can reproduce it exactly."

**SCREEN:** `results/test_bar.png`, then `cat results/test_significance.txt | head -1`.

**A:**
> "Against the single rewrite: 54 wins, 22 losses, paired t-test p = 0.0003. Interesting detail: fusion even beats the *human-written* rewrite, 0.293, because five votes give BM25 more of the right words. And with no LLM at all, our fusion gets 0.295."

**SCREEN:** `results/test_depth.png`.

**B:**
> "As a stretch we re-rank the top 100 with a small dense model, e5, and a personalized PageRank anchored on the last turn's results. That reaches 0.53. But it's a pretrained model, so we report it separately and don't credit it to our core.
> Limitations: no natural-language negation, which you saw, and the topic gate only helps when there's no LLM. With the LLM, the rewrites already handle topic shifts. Next we'd parse 'not the X one' into a Boolean NOT. Thanks for watching. The code, report and every number are in the repo."

**END** ≈ 7:00

---

## Fallbacks during recording

| If… | Do |
|---|---|
| A live query hangs (Gemini slow or rate-limited) | Retype one of the scripted queries. They are cached and answer instantly. |
| You type a new query and the API fails | The trace shows `llm_fallback`, and the system silently uses no-LLM fusion. Say so; it's a feature (E14). |
| You overrun 8 min | Cut the pride-and-prejudice topic shift (≈ 25 s) and the depth plot (≈ 15 s). |
| You're under 5 min | Add `uv run python -m convsearch.pipeline` (self-tests for edge cases E3, E5, E8, E12 print "pipeline ok"), or show `"the who" band` in the app. |

## After recording

- [ ] Length is 5–8 min. Upload unlisted to YouTube or Drive, and test the link in a private window.
- [ ] Put the link in `report/report.md` (VIDEO-LINK) and the README, then rebuild the PDF.
- [ ] Tick the Phase 5 boxes in `ASSIGNMENT.md`.
