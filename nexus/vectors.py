"""Small exact page-vector side index; no vector database or ANN service.

Use the same cached static encoder as NEO's unit vectors so query embeddings work
offline and remain in the same space as the page vectors.
"""

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

MODEL = "minishlab/potion-retrieval-32M"
PAGE_WORDS = 380
_loaded: dict[str, tuple[list[tuple[str, int]], np.ndarray]] = {}


def page_texts(docs: list[dict]) -> tuple[list[tuple[str, int]], list[str]]:
    pages: dict[tuple[str, int], list[str]] = defaultdict(list)
    for d in docs:
        for u in d.get("units", []):
            pages[(d["doc"], int((u.get("pages") or [1])[0]))].append(u.get("text", ""))
    keys = sorted(pages)
    texts = [" ".join(" ".join(pages[k]).split()[:PAGE_WORDS]) for k in keys]
    return keys, texts


def create_page_vectors(docs: list[dict], existing: str | Path | None = None) -> tuple[list[tuple[str, int]], np.ndarray] | None:
    keys, texts = page_texts(docs)
    if existing and Path(existing).exists():
        with open(existing, "rb") as f:
            saved = pickle.load(f)
        pos = {tuple(k): i for i, k in enumerate(saved["keys"])}
        if saved.get("model") == MODEL and all(k in pos for k in keys):
            vecs = np.asarray([saved["vecs"][pos[k]] for k in keys], dtype=np.float32)
            return keys, vecs
    try:
        from engine.index import _model
        from neo.tree import describe

        page_units: dict[tuple[str, int], list[str]] = defaultdict(list)
        for d in docs:
            for u in d.get("units", []):
                page = int((u.get("pages") or [1])[0])
                text = describe(u)
                page_units[(d["doc"], page)].append(f"{Path(d['doc']).stem} | {text}")
        unit_texts = [text for key in keys for text in page_units.get(key, [])]
        unit_vecs = _model().encode([" ".join(t.split()[:300]) for t in unit_texts], batch_size=1024)
        unit_vecs = np.asarray(unit_vecs, dtype=np.float32)
        unit_vecs /= np.linalg.norm(unit_vecs, axis=1, keepdims=True) + 1e-9
        vecs = np.zeros((len(keys), unit_vecs.shape[1]), dtype=np.float32)
        offset = 0
        for i, key in enumerate(keys):
            n = len(page_units.get(key, []))
            if n:
                vecs[i] = unit_vecs[offset:offset + n].mean(axis=0)
                vecs[i] /= np.linalg.norm(vecs[i]) + 1e-9
            offset += n
    except Exception:
        return None
    return keys, vecs


def save_page_vectors(path: str | Path, keys: list[tuple[str, int]], vecs: np.ndarray) -> None:
    with open(path, "wb") as f:
        pickle.dump({"model": MODEL, "keys": keys, "vecs": np.asarray(vecs, dtype=np.float32)}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)


def load_page_vectors(path: str | Path) -> tuple[list[tuple[str, int]], np.ndarray]:
    path = str(Path(path).resolve())
    if path not in _loaded:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        _loaded[path] = ([tuple(k) for k in obj["keys"]], np.asarray(obj["vecs"], dtype=np.float32))
    return _loaded[path]


def query_vectors(texts: list[str]) -> np.ndarray | None:
    try:
        from engine.index import _model

        vecs = np.asarray(_model().encode(texts), dtype=np.float32)
    except Exception:
        return None
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs
