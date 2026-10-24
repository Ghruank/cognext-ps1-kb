"""Gold-page recall with query-time concept expansion (oracle documents).

    python -m neo.exp_expand 2          # smoke test on 2 questions
    python -m neo.exp_expand subset     # the 25-question tuning subset
"""

import json
import pickle
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastembed import TextEmbedding

from engine.index import embed_query
from eval import bench
from neo.expand import expand
from neo.pipeline import STORE

NS = (20, 40, 80, 120)


def main() -> None:
    arg = sys.argv[1]
    qs = [q for q in bench.load() if q["gold"]]
    qs = [q for q in qs if q["id"] in set(bench.subset())] if arg == "subset" else qs[:int(arg)]
    tree = pickle.load(open(STORE / "train_tree.pkl", "rb"))
    pv = pickle.load(open(STORE / "train_pagevecs_bge-small-en-v1.5.pkl", "rb"))
    pos = {k: i for i, k in enumerate(pv["keys"])}
    tree.page_vecs = np.stack([pv["vecs"][pos[k]] if k in pos else np.zeros(pv["vecs"].shape[1], np.float32)
                               for k in tree.pages])
    enc = TextEmbedding(pv["model"], threads=12)

    cache_path = STORE / "expand_cache.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    todo = [q for q in qs if q["id"] not in cache]
    with ThreadPoolExecutor(3) as pool:
        for q, (terms, _) in zip(todo, pool.map(lambda q: expand(q["question"]), todo)):
            cache[q["id"]] = terms
    json.dump(cache, open(cache_path, "w"), indent=1)

    variants = {"question only": lambda q: q["question"],
                "question + expansion": lambda q: q["question"] + " " + " ; ".join(cache[q["id"]])}
    for name, text in variants.items():
        rec = {n: [] for n in NS}
        for q in qs:
            t = text(q)
            gold = {(c["doc"], p) for c in q["gold"] for p in c["pages"]}
            docs = sorted({c["doc"] for c in q["gold"]})
            qp = np.asarray(list(enc.query_embed([t]))[0], dtype=np.float32)
            s = tree.page_scores(t, embed_query(t), docs, contiguity=0, section_weight=0, page_vec_weight=1.0, qvec_page=qp)
            ranked = [tree.pages[p] for p in sorted(s, key=lambda p: -s[p])]
            for n in NS:
                rec[n].append(len(set(ranked[:n]) & gold) / len(gold))
        print(f"{name:24} n={len(qs):3d} " + "  ".join(f"R@{n}={st.mean(v):.3f}" for n, v in rec.items()), flush=True)
    if len(qs) <= 3:
        for q in qs:
            print(q["id"], cache[q["id"]])


if __name__ == "__main__":
    main()
