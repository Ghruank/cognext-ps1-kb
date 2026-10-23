"""Per-collection indexes over Typed Evidence Units: dense vectors (bge-small, CPU) and BM25.

Units are embedded with a short contextual header (document + section) so that
otherwise-identical paragraphs from different documents don't collapse onto one vector.
"""

import pickle
import re
from pathlib import Path

import bm25s
import numpy as np

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def unit_header(u: dict) -> str:
    return f"{Path(u['doc']).stem} | {u.get('section', '')}"


def _tok(texts: list[str]) -> list[list[str]]:
    return [re.findall(r"[a-z0-9][a-z0-9.%]*", t.lower()) for t in texts]


class Index:
    def __init__(self, units: list[dict]):
        self.units = units
        self.pos = {u["id"]: i for i, u in enumerate(units)}
        self.vecs: np.ndarray | None = None
        self.bm25: bm25s.BM25 | None = None

    def build(self, embed: bool = True, batch_size: int = 256) -> "Index":
        docs = [f"{unit_header(u)}\n{u['text']}" for u in self.units]
        self.bm25 = bm25s.BM25()
        self.bm25.index(_tok(docs), show_progress=False)
        if embed:
            from fastembed import TextEmbedding
            model = TextEmbedding(EMBED_MODEL)
            # long tables are truncated for the vector (BM25 still sees all of it); embedding in
            # slices keeps peak memory low enough for a laptop-sized server
            short = [" ".join(d.split()[:300]) for d in docs]
            parts = []
            for s in range(0, len(short), 2048):
                parts.append(np.asarray(list(model.embed(short[s:s + 2048], batch_size=batch_size)),
                                        dtype=np.float32))
            self.vecs = np.vstack(parts)
        return self

    def bm25_scores(self, query: str) -> np.ndarray:
        return self.bm25.get_scores(_tok([query])[0]).astype(np.float32)

    def dense_scores(self, qvec: np.ndarray) -> np.ndarray:
        return self.vecs @ qvec

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "Index":
        with open(path, "rb") as f:
            return pickle.load(f)


_qmodel = None


def embed_query(q: str) -> np.ndarray:
    global _qmodel
    if _qmodel is None:
        from fastembed import TextEmbedding
        _qmodel = TextEmbedding(EMBED_MODEL)
    return np.asarray(next(iter(_qmodel.embed([QUERY_PREFIX + q]))), dtype=np.float32)
