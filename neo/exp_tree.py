"""Gold-page recall of the LLM-free tree scorers, documents given (oracle routing). No LLM.

    python -m neo.exp_tree train
"""

import pickle
import statistics as st
import sys
import time

from engine.index import embed_query
from eval import bench
from neo.pipeline import STORE
from neo.tree import Tree

NS = (20, 40, 80, 120)


def run(tree: Tree, qs: list[dict], name: str, _qenc=None, **kw) -> None:
    rec = {n: [] for n in NS}
    for q in qs:
        gold = {(c["doc"], p) for c in q["gold"] for p in c["pages"]}
        docs = sorted({c["doc"] for c in q["gold"]})
        extra = {"qvec_page": _qenc(q["question"])} if _qenc else {}
        s = tree.page_scores(q["question"], embed_query(q["question"]), docs, **kw, **extra)
        ranked = [tree.pages[p] for p in sorted(s, key=lambda p: -s[p])]
        for n in NS:
            rec[n].append(len(set(ranked[:n]) & gold) / len(gold))
    print(f"{name:44} " + "  ".join(f"R@{n}={st.mean(v):.3f}" for n, v in rec.items()), flush=True)


def main() -> None:
    collection = sys.argv[1]
    path = STORE / f"{collection}_tree.pkl"
    if path.exists():
        tree = pickle.load(open(path, "rb"))
    else:
        t = time.time()
        tree = Tree.build(pickle.load(open(STORE / f"{collection}_docs.pkl", "rb")))
        pickle.dump(tree, open(path, "wb"))
        print(f"tree: {len(tree.pages)} pages, {len(tree.sections)} sections, built in {time.time() - t:.0f}s")
    qs = [q for q in bench.load() if q["gold"]]
    run(tree, qs, "unit max only", contiguity=0, section_weight=0)
    run(tree, qs, "+ contiguity 0.3", contiguity=0.3, section_weight=0)
    run(tree, qs, "+ contiguity 0.6", contiguity=0.6, section_weight=0)
    run(tree, qs, "+ contiguity 0.3 + section 0.3", contiguity=0.3, section_weight=0.3)
    run(tree, qs, "+ contiguity 0.6 + section 0.6", contiguity=0.6, section_weight=0.6)


if __name__ == "__main__":
    main()
