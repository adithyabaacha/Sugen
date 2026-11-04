"""CPU-only offline lexical retrieval: word/character TF-IDF plus BM25."""
from __future__ import annotations
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .clinical import search_queries, clean_text

TOKEN_RE = re.compile(r"[a-z][a-z0-9]+", re.I)


class BM25:
    """Sparse BM25 with precomputed posting lists (no network/dependencies)."""
    def __init__(self, docs, k1=1.5, b=0.75):
        self.n = len(docs)
        tokens = [TOKEN_RE.findall(d) for d in docs]
        self.avgdl = sum(len(t) for t in tokens) / max(1, self.n)
        self.doclen = np.array([len(t) for t in tokens], dtype=np.float32)
        self.postings = defaultdict(list)
        for i, words in enumerate(tokens):
            for word, tf in Counter(words).items():
                self.postings[word].append((i, tf))
        self.idf = {word: math.log(1.0 + (self.n - len(p) + 0.5)/(len(p) + 0.5)) for word, p in self.postings.items()}
        self.k1 = k1
        self.b = b

    def scores(self, query):
        scores = np.zeros(self.n, dtype=np.float32)
        for word in set(TOKEN_RE.findall(query)):
            idf = self.idf.get(word)
            if idf is None:
                continue
            for idx, tf in self.postings[word]:
                denominator = tf + self.k1 * (1.0 - self.b + self.b * self.doclen[idx]/max(self.avgdl, 1.0))
                scores[idx] += idf * (tf * (self.k1 + 1)) / denominator
        return scores


def rrf_lists(score_arrays, weights, k=60):
    result = np.zeros(len(score_arrays[0]), dtype=np.float32)
    for arr, weight in zip(score_arrays, weights):
        # Rank all documents, even if all zero; limit contribution to nonzero matches.
        order = np.argsort(-arr, kind="stable")
        ranks = np.empty(len(arr), dtype=np.int32)
        ranks[order] = np.arange(len(arr))
        result += np.where(arr > 0, weight / (k + ranks + 1), 0.0)
    return result


class LexicalRetriever:
    def __init__(self, catalog):
        self.catalog = catalog
        self.records = catalog.records
        self.codes = [r["code"] for r in self.records]
        self.docs = [clean_text(r["title"]) for r in self.records]
        self.word = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True,
                                    token_pattern=r"(?u)\b\w\w+\b", dtype=np.float32)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1,
                                    strip_accents="unicode", sublinear_tf=True, dtype=np.float32)
        self.word_index = self.word.fit_transform(self.docs)
        self.char_index = self.char.fit_transform(self.docs)
        self.bm25 = BM25(self.docs)

    def _tfidf_score(self, text):
        return (
            (self.word.transform([text]) @ self.word_index.T).toarray().ravel(),
            (self.char.transform([text]) @ self.char_index.T).toarray().ravel(),
        )

    def rank(self, text: str, limit: int = 50) -> list[str]:
        if not isinstance(text, str) or not text.strip():
            return [r["code"] for r in self.records[:limit]]
        full, focus = search_queries(text)
        word_full, char_full = self._tfidf_score(full)
        bm_full = self.bm25.scores(full)
        arrays = [word_full, char_full, bm_full]
        weights = [1.0, 1.0, 0.8]
        if focus:
            word_focus, char_focus = self._tfidf_score(focus)
            bm_focus = self.bm25.scores(focus)
            arrays += [word_focus, char_focus, bm_focus]
            weights += [2.5, 2.0, 1.8]
        scores = rrf_lists(arrays, weights)
        # Slight preference for the terminal code when lexical evidence is similar.
        for i, r in enumerate(self.records):
            if r.get("leaf") and scores[i] > 0:
                scores[i] *= 1.04
        order = np.argsort(-scores, kind="stable")[:limit]
        return [self.codes[i] for i in order]

    def predict(self, text: str, k: int = 3) -> list[str]:
        return self.rank(text, limit=k)
