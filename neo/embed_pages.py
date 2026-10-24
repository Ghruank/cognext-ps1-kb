"""Page-level dense vectors with a stronger local embedder (fastembed ONNX, CPU, no LLM).

A page is ~16 units, so there are 16x fewer page vectors than unit vectors; that is
what makes bge-class models affordable on CPU where per-unit embedding is not.

    python -m neo.embed_pages train BAAI/bge-small-en-v1.5
"""

import pickle
import sys
import time

import numpy as np

from neo.pipeline import STORE

PAGE_WORDS = 380   # bge models read 512 tokens; the page head carries its topic


def page_texts(docs: list[dict]) -> tuple[list[tuple[str, int]], list[str]]:
    pages: dict[tuple[str, int], list[str]] = {}
    for d in docs:
        for u in d["units"]:
            pages.setdefault((d["doc"], u["pages"][0]), []).append(u["text"])
    keys = sorted(pages)
    return keys, [" ".join(" ".join(pages[k]).split()[:PAGE_WORDS]) for k in keys]


def main() -> None:
    from fastembed import TextEmbedding
    collection, model = sys.argv[1], sys.argv[2]
    docs = pickle.load(open(STORE / f"{collection}_docs.pkl", "rb"))
    keys, texts = page_texts(docs)
    m = TextEmbedding(model, threads=12)
    t = time.time()
    vecs = np.asarray(list(m.embed(texts, batch_size=16)), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    print(f"{len(keys)} pages embedded with {model} in {time.time() - t:.0f}s")
    pickle.dump({"model": model, "keys": keys, "vecs": vecs},
                open(STORE / f"{collection}_pagevecs_{model.split('/')[-1]}.pkl", "wb"))


if __name__ == "__main__":
    main()
