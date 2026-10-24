"""Run NEO over benchmark questions and write an output file for eval.run_eval.

    python -m neo.run outputs/neo_subset.jsonl subset      # the 25-question tuning subset
    python -m neo.run outputs/neo_smoke.jsonl A06          # given ids

Resumable: questions already in the output file are skipped.
"""

import json
import sys
import time
from pathlib import Path

from eval import bench
from neo.pipeline import STORE, build, load


def main() -> None:
    out, which = Path(sys.argv[1]), sys.argv[2]
    ids = bench.subset() if which == "subset" else which.split(",")
    rows = {r["id"]: r for r in bench.load()}
    if not (STORE / "train_graph.pkl").exists():
        t = time.time()
        graph, tree = build("train", json.load(open("storage/train_catalog.json")))
        print(f"graph: {len(graph.categories)} categories, {len(graph.change_links)} change links, "
              f"built in {time.time() - t:.0f}s", flush=True)
    graph, tree = load("train")
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    out.parent.mkdir(parents=True, exist_ok=True)
    for qid in ids:
        if qid in done:
            continue
        r = rows[qid]
        try:
            res = __import__("neo.pipeline", fromlist=["query"]).query(r["question"], graph, tree)
        except Exception as e:  # one failed question must not stop a long run
            print(f"{qid}: FAILED {e!r}", flush=True)
            continue
        with out.open("a") as f:
            f.write(json.dumps({"id": qid, "answer": res["answer"], "chunks": [c["text"] for c in res["chunks"]],
                                "chunk_meta": [{k: c[k] for k in ("doc", "pages", "id")} for c in res["chunks"]],
                                "trace": res["trace"], "tokens": res["tokens"], "latency_s": res["latency_s"]},
                               default=str) + "\n")
        plan = res["trace"][0]["output"]
        print(f"{qid}: {plan['retrieval']}, docs {len(plan['documents'])} "
              f"(gold {len({c['doc'] for c in r['gold']})}), chunks {len(res['chunks'])} (gold {len(r['gold'])}), "
              f"{res['tokens']['prompt'] + res['tokens']['completion']} tokens, {res['latency_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
