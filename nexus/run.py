"""Build a local NEXUS index and run benchmark questions.

Examples:
    python -m nexus.run --build --ids A01,A02 --offline --no-answer
    python -m nexus.run --ids A01 --outputs outputs/nexus.jsonl

Without --offline, NEXUS makes one final answer-framing call. Planning, retrieval,
indexing, and evidence selection are deterministic.
"""

import argparse
import json
from pathlib import Path

from eval import bench
from nexus.index import build_from_pickle
from nexus.pipeline import query

ROOT = Path(__file__).resolve().parents[1]
STORE = Path(__file__).resolve().parent / "storage"
DEFAULT_INDEX = STORE / "train.sqlite"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="(re)build the local SQLite source map")
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    ap.add_argument("--docs-pickle", default=str(ROOT / "neo" / "storage" / "train_docs.pkl"))
    ap.add_argument("--ids", default="", help="comma-separated ids; default: all 100 questions")
    ap.add_argument("--outputs", default=str(ROOT / "outputs" / "nexus.jsonl"))
    ap.add_argument("--offline", action="store_true", help="disable query-time LLM calls")
    ap.add_argument("--no-answer", action="store_true", help="retrieve only")
    ap.add_argument("--no-verify", action="store_true", help="skip the query-time evidence filter")
    ap.add_argument("--limit-per-facet", type=int, default=0,
                    help="optional candidate cap; default is unlimited for recall")
    ap.add_argument("--no-dense", action="store_true",
                    help="skip page-vector routing when the local query encoder is unavailable")
    ap.add_argument("--llm-metadata", action="store_true",
                    help="ablation: index saved LLM-generated document metadata and page cards")
    ap.add_argument("--max-blocks-per-facet", type=int, default=24,
                    help="deterministic evidence budget per query facet (default: 24)")
    ap.add_argument("--dense-threshold", type=float, default=0.42,
                    help="minimum cosine score for flat page-vector routes (default: 0.42)")
    args = ap.parse_args()

    index_path = Path(args.index)
    if args.build or not index_path.exists():
        print("building source map...", flush=True)
        print(build_from_pickle(args.docs_pickle, index_path, use_dense=not args.no_dense,
                                use_llm_metadata=args.llm_metadata), flush=True)

    rows = {r["id"]: r for r in bench.load()}
    wanted = args.ids.split(",") if args.ids else list(rows)
    out = Path(args.outputs)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    for qid in wanted:
        if qid in done:
            continue
        r = rows[qid]
        result = query(r["question"], index_path, use_llm=not args.offline,
                       verify_relevance=not args.no_verify,
                       generate_answer=not args.no_answer,
                       per_facet_limit=args.limit_per_facet or None, observe=not args.offline,
                       use_dense=not args.no_dense, max_per_facet=args.max_blocks_per_facet,
                       vector_threshold=args.dense_threshold)
        record = {"id": qid, "answer": result["answer"],
                  "chunks": [c["text"] for c in result["chunks"]],
                  "chunk_meta": [{k: c[k] for k in ("id", "doc", "pages")} for c in result["chunks"]],
                  "trace": result["trace"], "usage": result["usage"],
                  "latency_s": result["latency_s"], "retrieval": result["retrieval"]}
        with out.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{qid}: {len(result['chunks'])} chunks, {result['retrieval']['candidate_blocks']} candidates, "
              f"{result['usage']['calls']} calls, {result['latency_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
