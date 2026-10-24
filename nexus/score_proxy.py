"""Score NEXUS output against the existing text-overlap retrieval proxy.

This is a fast diagnostic only. Checkpoints must still be scored with eval.run_eval,
which wraps the unmodified organizer evaluator.
"""

import argparse
import json
import pickle
import statistics
from pathlib import Path

from eval import bench
from eval.gold_map import label_units

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs")
    # NEXUS's default index is built from NEO's parser-native units. Keep proxy
    # labels on those same unit IDs; the older STAR pickle has incompatible IDs.
    ap.add_argument("--docs-pickle", default=str(ROOT / "neo" / "storage" / "train_docs.pkl"))
    args = ap.parse_args()
    with open(args.docs_pickle, "rb") as f:
        docs = pickle.load(f)
    labels = label_units(docs)
    gold = {q["id"]: q for q in bench.load()}
    rows = [json.loads(line) for line in open(args.outputs)]
    pvals, rvals = [], []
    for row in rows:
        qid = row["id"]
        chosen = {c["id"] for c in row.get("chunk_meta", []) if c.get("id")}
        relevant = chosen & labels[qid]
        g = len(gold[qid]["gold"])
        if g == 0:
            p = r = 1.0 if not chosen else 0.0
        else:
            p = len(relevant) / len(chosen) if chosen else 0.0
            r = len(relevant) / g
        pvals.append(p)
        rvals.append(r)
        print(f"{qid} P={p:.3f} R={r:.3f} selected={len(chosen)} proxy_hits={len(relevant)} gold={g}")
    p, r = statistics.mean(pvals), statistics.mean(rvals)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    print(f"\n{len(rows)} questions: P={p:.3f} R={r:.3f} F1={f1:.3f} (gold-overlap proxy)")


if __name__ == "__main__":
    main()
