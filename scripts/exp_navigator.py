"""Architecture 1 on the tuning subset: LLM-Navigator at several page budgets (costs credits).

    python -m scripts.exp_navigator 10 30 60
"""

import pickle
import statistics as st
import sys

import numpy as np

from engine.index import Index
from engine.navigator import navigate
from eval import bench
from scripts.exp_retrieval import proxy, summarize


def main() -> None:
    budgets = [int(a) for a in sys.argv[1:]] or [30]
    index = Index.load("storage/train_index.pkl")
    pools = {p["id"]: p for p in pickle.load(open("storage/train_pools.pkl", "rb"))}
    qs = {q["id"]: q for q in bench.load()}
    subset = bench.subset()
    results = {}
    for n in budgets:
        rows, cost, sel_sizes = [], [], []
        for qid in subset:
            p = pools[qid]
            score = p["X"][:, 1] + 0.5 * p["X"][:, 3]  # dense_z + 0.5 * bm25_z within the pool
            chosen, usage = navigate(qs[qid]["question"], index.units, p["idx"], score, n_pages=n)
            rows.append(proxy(p["y"][chosen].astype(float), p["n_gold"]))
            cost.append(usage["usd"])
            sel_sizes.append(len(chosen))
            results[(n, qid)] = {"chosen": chosen, "usage": usage}
        summarize(f"1  LLM-Navigator, {n} pages", rows)
        print(f"{'':42} median selected {st.median(sel_sizes):.0f}, cost/query ${st.mean(cost):.4f}")
    pickle.dump(results, open("storage/navigator_runs.pkl", "wb"))


if __name__ == "__main__":
    main()
