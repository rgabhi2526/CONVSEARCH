"""Text processing: normalisation, tokenisation, Porter stemming, stopwords (query side only)."""
import re
import unicodedata
from functools import lru_cache

from nltk.stem import PorterStemmer

_porter = PorterStemmer()
_word = re.compile(r"[a-z0-9]+")

# Small hand-picked list. The index KEEPS stopwords (with positions) so phrase
# queries like "the who" work (D9). Only ranked queries drop them.
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further
had has have having how i if in into is it its itself just me more most my no nor not of off
on once only or other our out over own same should so some such than that the their them then
there these they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours tell know please thanks thank ok also one
""".split())


@lru_cache(maxsize=None)
def stem(word):
    return _porter.stem(word)


def words(s):
    """Case-fold, strip accents (café -> cafe), split on non-alphanumerics."""
    s = unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode()
    return _word.findall(s)


def tokens(s):
    """Index-side terms: every word stemmed, stopwords kept, order kept (positions)."""
    return [stem(w) for w in words(s)]


def query_terms(s):
    """Query-side terms for ranking: stopwords dropped, then stemmed."""
    return [stem(w) for w in words(s) if w not in STOPWORDS]


if __name__ == "__main__":
    assert words("Café, NFL's 2023-season!") == ["cafe", "nfl", "s", "2023", "season"]
    assert tokens("The Arizona Cardinals are playing") == ["the", "arizona", "cardin", "are", "play"]
    assert query_terms("Where do the Arizona Cardinals play?") == ["arizona", "cardin", "play"]
    assert query_terms("what about that one?") == []  # E8: empty query
    print("text ok")
