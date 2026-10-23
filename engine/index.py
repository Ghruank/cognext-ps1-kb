"""Per-collection indexes over Typed Evidence Units: dense vectors (bge-small, CPU) and BM25.

Units are embedded with a short contextual header (document + section) so that
otherwise-identical paragraphs from different documents don't collapse onto one vector.
"""

import pickle
import re
from pathlib import Path

import bm25s
import numpy as np

# Static (Model2Vec) embeddings: ~23k paragraphs/s on CPU, so a full corpus embeds in
# seconds. bge-small is ~20/s on the same CPU (35 min for the training corpus).
EMBED_MODEL = "minishlab/potion-retrieval-32M"


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

    def build(self, embed: bool = True, batch_size: int = 1024) -> "Index":
        docs = [f"{unit_header(u)}\n{u['text']}" for u in self.units]
        self.bm25 = bm25s.BM25()
        self.bm25.index(_tok(docs), show_progress=False)
        if embed:
            # long tables are truncated for the vector; BM25 still sees all of it
            short = [" ".join(d.split()[:300]) for d in docs]
            self.vecs = _normalize(_model().encode(short, batch_size=batch_size))
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


_m = None


def _model():
    global _m
    if _m is None:
        from model2vec import StaticModel
        _m = StaticModel.from_pretrained(EMBED_MODEL)
    return _m


def _normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def embed_query(q: str) -> np.ndarray:
    return _normalize(_model().encode([q]))[0]
