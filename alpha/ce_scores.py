"""Cross-encoder relevance for every candidate unit (Self-Healing-RAG's precision stage), cached per question.

    .venv/bin/python -m alpha.ce_scores
"""

import json
import pickle
from pathlib import Path

import numpy as np

STORE = Path(__file__).resolve().parent / "storage"
CE_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
CACHE = STORE / "ce_scores.pkl"
_ce = None
WORDS = 40
TOP = 100


def ce_score(question: str, texts: list[str], only: list[int] | None = None) -> np.ndarray:
    """Scores texts[k] for k in `only` (all if None); the rest get the lowest score seen."""
    global _ce
    if _ce is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _ce = TextCrossEncoder(CE_MODEL, threads=8)
    only = list(range(len(texts))) if only is None else only
    out = np.full(len(texts), np.nan, dtype=np.float32)
    if only:
        out[only] = list(_ce.rerank(question, [" ".join(texts[k].split()[:WORDS]) for k in only], batch_size=64))
    return np.nan_to_num(out, nan=float(np.nanmin(out)) if only else 0.0)


def shortlist(tree, question: str, positions: list[int], verified: set[int]) -> list[int]:
    """Indices into positions worth a cross-encoder call: verified units plus the top by embedding similarity."""
    from engine.index import embed_query
    sims = tree.unit_vecs[positions] @ embed_query(question)
    return sorted(verified | set(np.argsort(-sims)[:TOP].tolist()))


def main() -> None:
    from eval import bench
    from alpha.pipeline import load
    _, tree = load("train")
    index = json.load(open(STORE / "claude" / "verify_index.json"))
    cache = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}
    for q in bench.load():
        pos = index[q["id"]]["positions"]
        if q["id"] in cache or not pos:
            continue
        ids = json.load(open(STORE / "claude" / "verdict" / f"{q['id']}.json")).get("relevant_ids", []) \
            if (STORE / "claude" / "verdict" / f"{q['id']}.json").exists() else []
        ver = {int(k) for k in ids if 0 <= int(k) < len(pos)}
        cache[q["id"]] = ce_score(q["question"], [tree.units[p]["text"] for p in pos], shortlist(tree, q["question"], pos, ver))
        pickle.dump(cache, open(CACHE, "wb"))
        print(q["id"], len(pos), flush=True)


if __name__ == "__main__":
    main()
