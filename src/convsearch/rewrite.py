"""LLM query rewrites (Gemini) with a committed disk cache.
The LLM only PROPOSES standalone rewrites; all retrieval and ranking stays in our IR code.
`python -m convsearch.rewrite --all` fills the cache for every MTRAG user turn."""
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache" / "rewrites.json"
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
N = 5

PROMPT = """You rewrite the LAST user question of a conversation into standalone search queries.
Resolve pronouns and ellipsis using the earlier questions. If the last question starts a new
topic, do not carry the old topic over. Do not answer the question. Do not invent facts.
Return JSON: {{"rewrites": [ ... exactly {n} different standalone rewrites ... ]}}

Earlier questions (oldest first):
{history}

Last question: {turn}"""

_cache = None
_client = None


def _load():
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    return _cache


def _save():
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(_cache, indent=1, sort_keys=True))


def key(turn, history, n=N):
    return hashlib.sha1(json.dumps([history, turn, n]).encode()).hexdigest()


def _call(turn, history, n):
    global _client
    from google import genai
    from google.genai import types
    if _client is None:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    hist = "\n".join(f"- {h}" for h in history) or "(none)"
    r = _client.models.generate_content(
        model=MODEL, contents=PROMPT.format(n=n, history=hist, turn=turn),
        config=types.GenerateContentConfig(temperature=0.7, response_mime_type="application/json"))
    rw = [s.strip() for s in json.loads(r.text)["rewrites"] if isinstance(s, str) and s.strip()]
    if not rw:
        raise ValueError("empty rewrites")
    return rw[:n]


def rewrites(turn, history, n=N, offline=False, retries=3):
    """-> (list of rewrites or None, info). None means fall back to no-LLM fusion (E14).
    offline=True: cache only, never call the API (eval reproducibility, E15)."""
    c = _load()
    k = key(turn, history, n)
    if k in c:
        return c[k]["rewrites"], {"cached": True}
    if offline:
        return None, {"cached": False, "llm_fallback": "not in cache (offline)"}
    err = None
    for attempt in range(retries):
        try:
            rw = _call(turn, history, n)
            c[k] = {"turn": turn, "history": history, "model": MODEL, "rewrites": rw}
            _save()
            return rw, {"cached": False}
        except Exception as e:                       # 429 / network / bad JSON
            err = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(2 ** attempt * 5)             # 5s, 10s, 20s backoff
    return None, {"cached": False, "llm_fallback": err}


if __name__ == "__main__":
    import sys
    if "--all" in sys.argv:
        from . import data
        todo = [(t, ts[:i]) for c in data.conversations() for ts in [data.user_turns(c)] for i, t in enumerate(ts)]
        done = fail = 0
        for turn, hist in todo:
            if key(turn, hist) in _load():
                continue
            rw, info = rewrites(turn, hist)
            done += rw is not None
            fail += rw is None
            print(f"[{done + fail}] {'ok ' if rw else 'ERR'} {turn[:60]!r} {info.get('llm_fallback', '')}", flush=True)
            time.sleep(float(os.environ.get("REWRITE_SLEEP", "4")))   # free-tier RPM
        print(f"cache: {len(_load())} entries, new {done}, failed {fail}")
    else:
        rw, info = rewrites("Do they play outside the US?", ["where do the arizona cardinals play this week"])
        print(info, json.dumps(rw, indent=1))
        assert rw and len(rw) == N
        assert rewrites("zz never cached zz", [], offline=True)[0] is None
        print("rewrite ok")
