"""Score a system-output file with the official evaluator.

Usage:
    python -m eval.run_eval outputs/e1_oracle.jsonl [--subset] [--ids A01,B05] [--workers 6]

The output file is JSONL with {"id", "answer", "chunks": [str, ...]} per question.
Each question is scored by calling the official evaluate_dataset on a one-item list,
so the scoring code is exactly the organisers'; questions just run in parallel.
"""

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval import bench
from eval.official import evaluate_dataset, make_evaluator, spent_usd

METRICS = ("answer_precision", "answer_recall", "context_precision", "context_recall")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs")
    ap.add_argument("--subset", action="store_true", help="only the fixed 25-question tuning subset")
    ap.add_argument("--ids", default="")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--official", action="store_true",
                    help="exact notebook payload (thinking on); default is the cheap dev judge")
    args = ap.parse_args()

    gold = {r["id"]: r for r in bench.load()}
    outs = [json.loads(line) for line in open(args.outputs)]
    wanted = set(args.ids.split(",")) if args.ids else set(bench.subset()) if args.subset else None
    if wanted:
        outs = [o for o in outs if o["id"] in wanted]

    evaluator = make_evaluator("official" if args.official else "dev")
    before = spent_usd()

    def score(o: dict) -> dict:
        g = gold[o["id"]]
        item = {"query": g["question"], "ai_answer": o["answer"], "benchmark_answer": g["answer"],
                "retrieved_chunks": o["chunks"], "benchmark_chunks": [c["text"] for c in g["gold"]]}
        try:
            res = evaluate_dataset(evaluator, [item], verbose=False)["per_question"][0]
        except RuntimeError as e:
            # the judge itself failed (e.g. ran out of tokens decomposing a long answer)
            print(f"{o['id']}  JUDGE FAILED: {e}", flush=True)
            res = {m: 0.0 for m in METRICS} | {"error": str(e)}
        res["id"] = o["id"]
        print(f"{o['id']}  AnsP={res['answer_precision']:.2f} AnsR={res['answer_recall']:.2f} "
              f"CtxP={res['context_precision']:.2f} CtxR={res['context_recall']:.2f}  "
              f"(returned {len(o['chunks'])}, gold {len(g['gold'])})", flush=True)
        return res

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(score, outs))

    print(f"\n{len(results)} questions")
    for fam in ("A", "B", "C", None):
        rs = [r for r in results if fam is None or r["id"].startswith(fam)]
        if rs:
            print(f"{fam or 'ALL'}: " + "  ".join(
                f"{m.split('_')[0][:3]}{m.split('_')[1][0].upper()}={statistics.mean(r[m] for r in rs):.3f}"
                for m in METRICS))
    print(f"judge spend this run: ${spent_usd() - before:.3f}   total ledger: ${spent_usd():.3f}")

    report = Path(args.outputs).with_suffix(".official.json" if args.official else ".dev.json")
    report.write_text(json.dumps(results, indent=1))
    print(f"per-question detail: {report}")


if __name__ == "__main__":
    main()
