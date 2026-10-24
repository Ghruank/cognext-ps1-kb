"""Re-write answers for an existing output file from its saved chunks (no retrieval rerun).

    python -m neo.rewrite outputs/neo_smoke.jsonl
"""

import json
import sys

from eval import bench
from neo.pipeline import load
from neo.writer import write


def main() -> None:
    path = sys.argv[1]
    rows = [json.loads(line) for line in open(path)]
    qs = {r["id"]: r for r in bench.load()}
    _, tree = load("train")
    pos = {u["id"]: i for i, u in enumerate(tree.units)}
    for o in rows:
        units = [tree.units[pos[m["id"]]] for m in o["chunk_meta"]]
        o["answer"], usage = write(qs[o["id"]]["question"], units)
        print(f"{o['id']}: {usage.get('calls')} calls, {len(o['answer'].split())} words", flush=True)
    with open(path, "w") as f:
        for o in rows:
            f.write(json.dumps(o, default=str) + "\n")


if __name__ == "__main__":
    main()
