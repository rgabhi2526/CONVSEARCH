"""Streamlit demo: uv run streamlit run app.py
Chat-style multi-turn search. Every intermediate value comes from pipeline.run's trace."""
import streamlit as st

from convsearch import pipeline

PRESETS = {
    "C8 full (core + dense + PPR)": {"query": "fusion", "gate": True, "boolean": True, "mu": 3.0, "nu": 0.1},
    "C6 core (fusion + gate + Boolean)": {"query": "fusion", "gate": True, "boolean": True},
    "C4 fusion only": {"query": "fusion"},
    "C9 no-LLM (fusion + gate + Boolean)": {"query": "fusion", "llm": False, "gate": True, "boolean": True},
    "C2 single LLM rewrite": {"query": "rewrite1"},
    "C0 last turn only (baseline)": {"query": "turn"},
}

st.set_page_config(page_title="ConvSearch", layout="wide")
ix = st.cache_resource(pipeline.index)()

with st.sidebar:
    preset = st.selectbox("Config", list(PRESETS))
    cfg = dict(PRESETS[preset])
    with st.expander("Parameters"):
        for k in ("alpha", "beta", "lam", "tau_idf", "tau_j", "mu", "nu"):
            cfg[k] = st.number_input(k, value=float(cfg.get(k, pipeline.DEFAULT[k])), step=0.05)
    compare = st.checkbox("Show C0 baseline next to it", value=True)
    if st.button("New conversation"):
        st.session_state.clear()
        st.rerun()
    st.caption("Explicit Boolean works too: `arizona NOT chicago`, `\"the who\" band`")

ss = st.session_state
ss.setdefault("turns", [])          # [(turn, ranked, trace, base_ranked)]


def show_results(ranked, n=10):
    for i, (d, s) in enumerate(ranked[:n], 1):
        st.markdown(f"**{i}. {ix.titles[d]}** · `{ix.doc_ids[d]}` · score {s:.2f}")
        st.caption(ix.texts[d][:300] + ("…" if len(ix.texts[d]) > 300 else ""))


def show_trace(tr):
    if tr.get("mode") == "explicit boolean":
        with st.expander(f"Boolean query: {tr['parsed']} → {tr['B_size']} docs", expanded=True):
            st.table(tr["steps"])
        return
    if tr.get("rewrites") is not None or "rewrite_info" in tr:
        with st.expander("1 · LLM rewrites (proposals only)"):
            st.write(tr.get("rewrites") or tr.get("query_text"))
            st.caption(str(tr.get("rewrite_info")))
    if "fusion" in tr:
        f = tr["fusion"]
        flags = {k: v for k, v in f.items() if k not in ("table", "phrases")}
        with st.expander("2 · Fusion: w(t) = agree·idf (+β·idf if typed now)", expanded=True):
            st.dataframe(f["table"], hide_index=True)
            st.caption(f"phrases: {f['phrases']} · flags: {flags}")
    if "gate" in tr:
        g = tr["gate"]
        with st.expander(f"3 · Topic-shift gate → {'SHIFT: history dropped' if g['shift'] else 'keep history'}"):
            st.json(g)
    if "agent" in tr:
        a = tr["agent"]
        with st.expander(f"4 · Boolean agent → |B| = {a['B_size']}" + (f", bonus {a.get('bonus')}" if a.get("bonus") else "")):
            for at in a["attempts"]:
                st.markdown(f"`{at['query']}` → **{at['hits']}** hits · {at['action']}")
                st.table(at["steps"])
    if "rerank" in tr:
        with st.expander("5 · Re-rank of top-100 (dense e5 / PPR g(d), min-max normalised)"):
            st.json(tr["rerank"])
    with st.expander("Query vector sent to BM25"):
        st.json(tr["qvec"])


for turn, ranked, tr, base in ss.turns:
    with st.chat_message("user"):
        st.write(turn)
    with st.chat_message("assistant"):
        if not ranked:
            st.warning(tr.get("message", "no results"))
        show_trace(tr)
        cols = st.columns(2) if base is not None else [st.container()]
        with cols[0]:
            st.subheader(preset.split(" ")[0])
            show_results(ranked)
        if base is not None:
            with cols[1]:
                st.subheader("C0 baseline")
                show_results(base)

if q := st.chat_input("Ask a question, then follow up…"):
    history = [t for t, *_ in ss.turns]
    prev = [d for d, _ in ss.turns[-1][1][:10]] if ss.turns else []
    with st.chat_message("user"):   # echo the query now; results replace this on rerun
        st.write(q)
    with st.chat_message("assistant"), st.spinner("Rewriting, fusing, searching… (first dense query loads e5, ~1 min)"):
        ranked, tr = pipeline.run(q, history, prev, cfg, ix)
        base = pipeline.run(q, history, [], {"query": "turn"}, ix)[0] if compare else None
    ss.turns.append((q, ranked, tr, base))
    st.rerun()
