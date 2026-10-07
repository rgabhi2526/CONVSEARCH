# convsearch: T2 conversational search (CSD358 IR)

Multi-turn search over MTRAG ClapNQ (183K Wikipedia passages). An LLM proposes rewrites; our own positional inverted index + BM25 does retrieval. Design: `docs/superpowers/specs/2026-10-06-t2-convsearch-design.md`. Progress: `DEVLOG.md`. Full explainer of every component: `docs/HOW_IT_WORKS.md`.

## Setup
```bash
uv sync
uv run python -m convsearch.data          # download MTRAG ClapNQ into data/
uv run python -m convsearch.index --build # build index (~30 s, ~1.2 GB RAM)
cp .env.example .env                      # add GEMINI_API_KEY (only needed for new rewrites)
```

## Run
```bash
# reproduce every number (rewrites come from the committed cache/rewrites.json: zero API calls)
uv run python eval/run_eval.py --configs C0,C1,C2,C3,C4,C5,C6,C7,C8,C9,C0cos,C4n1,C4n3 --split test
uv run python eval/report.py --split test --a C6 --b C2     # plots + paired t-test
uv run python eval/tune.py --config C6                      # grid search on the tune split
uv run python eval/run_eval.py --check-bm25s                # our BM25 vs bm25s

# module self-tests
for m in text index rank boolean fuse gate agent rerank pipeline; do uv run python -m convsearch.$m; done

# demo (new questions call Gemini; needs GEMINI_API_KEY in .env)
uv run streamlit run app.py
```

## Results (test split: 19 conversations, 141 judged turns)
| Config | nDCG@10 | P@5 | R@10 |
|---|---|---|---|
| C0 last turn (BM25) | 0.256 | 0.126 | 0.343 |
| C2 single LLM rewrite | 0.305 | 0.152 | 0.391 |
| C3 human rewrite | 0.293 | 0.155 | 0.384 |
| C4 rewrite fusion (N=5) | 0.348 | 0.175 | 0.438 |
| **C6 core: fusion + gate + Boolean agent** | **0.356** | **0.187** | **0.452** |
| C8 + dense e5 + PPR re-rank (stretch) | 0.529 | 0.267 | 0.628 |
| C9 no-LLM | 0.295 | 0.152 | 0.391 |

C6 vs C2: paired t-test p = 0.0003. Full table, ablations and decisions: `DEVLOG.md`.

## What works / what's planned
Works: everything in configs C0–C9, the explicit Boolean query syntax (`AND`, `OR`, `NOT`, `"phrase"`) and the Streamlit demo.
Not handled: spelling correction, natural-language negation ("not the Chicago one"), answer generation, other MTRAG domains.

## Report
```bash
cd report && pandoc report.md -o report.pdf --pdf-engine=xelatex -V mainfont="Helvetica Neue" -V monofont=Menlo
```

## Self-judged queries
```bash
uv run python eval/judge.py new --name conv1            # type a conversation
uv run python eval/judge.py label --name conv1 --judge A  # each member labels separately
uv run python eval/judge.py score
```

## Data credits
MTRAG benchmark (IBM, Apache-2.0): https://github.com/IBM/mt-rag-benchmark (ClapNQ passages from Wikipedia).
