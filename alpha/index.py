"""Index a folder of PDFs as a named ALPHA collection. One LLM call per document (catalogue card).

    .venv/bin/python -m alpha.index data/Documents train

PDFs -> layout-block units (alpha.ingest) -> page vectors (alpha.embed_pages) -> catalogue cards
-> tree + NeoRAG graph (alpha.pipeline.build). Everything lands in alpha/storage/<collection>_*.
"""

import json
import pickle
import sys
import time

import numpy as np

import engine.catalog
from alpha import llm
from alpha.embed_pages import page_texts
from alpha.ingest import parse_folder
from alpha.pipeline import PAGE_MODEL, STORE, build

engine.catalog.chat = llm.chat  # catalogue cards through Groq; engine/ stays untouched


def index_folder(folder: str, collection: str, workers: int = 8) -> dict:
    t0 = time.time()
    docs = parse_folder(folder, workers=workers)
    if not docs:
        raise ValueError("no PDF documents found")
    pickle.dump(docs, open(STORE / f"{collection}_docs.pkl", "wb"))

    from fastembed import TextEmbedding
    keys, texts = page_texts(docs)
    vecs = np.asarray(list(TextEmbedding(PAGE_MODEL, threads=workers).embed(texts, batch_size=16)), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    pickle.dump({"model": PAGE_MODEL, "keys": keys, "vecs": vecs},
                open(STORE / f"{collection}_pagevecs_{PAGE_MODEL.split('/')[-1]}.pkl", "wb"))

    catalog = engine.catalog.build_catalog(docs, workers=4)
    json.dump(catalog, open(STORE / f"{collection}_catalog.json", "w"))
    graph, tree = build(collection, catalog)
    return {"documents": len(docs), "pages": len(keys), "units": len(tree.units),
            "categories": len(graph.categories), "seconds": round(time.time() - t0)}


if __name__ == "__main__":
    print(index_folder(sys.argv[1], sys.argv[2]))
