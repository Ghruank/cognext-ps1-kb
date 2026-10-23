"""E1 oracle: write answers from the gold chunks and return the gold chunks as context.

Context metrics here show how often the judge accepts gold; answer metrics show the
writer's ceiling when retrieval is perfect.

    python -m eval.oracle outputs/e1_oracle.jsonl [--subset]
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from engine.writer import write
from eval import bench


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--subset", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    rows = bench.load()
    if args.subset:
        keep = set(bench.subset())
        rows = [r for r in rows if r["id"] in keep]

    def run(r: dict) -> dict:
        answer, usage = write(r["question"], r["gold"])
        print(f"{r['id']}: {len(r['gold'])} units -> {len(answer.split())} words "
              f"(gold {len(r['answer'].split())}), ${usage.get('usd', 0):.4f}", flush=True)
        return {"id": r["id"], "answer": answer, "chunks": [c["text"] for c in r["gold"]], "usage": usage}

    with ThreadPoolExecutor(args.workers) as pool:
        outs = list(pool.map(run, rows))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for o in outs:
            f.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    main()
