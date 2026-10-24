"""NEO vs STAR on STAR's offline proxy (same labels rule, same P/R/F definitions). Retrieval only.

    NEO_LLM=gemini python -m alpha.exp_proxy 2      # smoke test on 2 questions
    NEO_LLM=gemini python -m alpha.exp_proxy all    # all 100 questions

P = selected units carrying gold text / selected units, R = those units / gold chunk count,
F = 2*mean(P)*mean(R)/(mean(P)+mean(R)), exactly as scripts.exp_retrieval.summarize.
STAR is re-scored per question from its saved out-of-fold run (SIGN + smoothing a=0.5, f1
selector), the 0.199 / 0.673 / 0.307 row.
"""

import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import alpha.verify
from eval import bench
from eval.gold_map import label_units
from engine.graph import norm_adj, smooth
from engine.select import select
from alpha import pipeline
from alpha.pipeline import STORE, load
from scripts.exp_retrieval import proxy, summarize

OUT = STORE / "proxy_runs.jsonl"


def star_rows(ids: list[str]) -> dict[str, tuple]:
    pools = pickle.load(open("storage/train_pools.pkl", "rb"))
    gprobs = pickle.load(open("storage/oof_graph.pkl", "rb"))
    out = {}
    for p, pr in zip(pools, gprobs):
        if p["id"] in ids:
            sm = smooth(pr, norm_adj(p["edges"], len(p["y"])), alpha=0.5)
            chosen, _ = select(sm, utility="f1")
            out[p["id"]] = proxy(p["y"][chosen].astype(float), p["n_gold"]) + (len(chosen),)
    return out


def main() -> None:
    import os
    if os.environ.get("NEO_LLM") == "gemini":
        neo.verify.BATCH_WORDS = 200000  # Gemini's long context: one verification call per question (free tier is ~20 calls/day/model)
    arg = sys.argv[1]
    qs = bench.load() if arg == "all" else bench.load()[:int(arg)]
    graph, tree = load("train")
    docs = pickle.load(open(STORE / "train_docs.pkl", "rb"))
    labels = label_units(docs)
    pipeline._page_query_vec("warm up")

    done = {}
    if OUT.exists():
        for line in OUT.open():
            r = json.loads(line)
            done[r["id"]] = r

    def run(q: dict) -> dict:
        if q["id"] in done:
            return done[q["id"]]
        res = pipeline.query(q["question"], graph, tree, write_answer=False)
        if not res["tokens"]["prompt"]:
            raise RuntimeError(f"{q['id']}: no LLM call succeeded")
        r = {"id": q["id"], "selected": [c["id"] for c in res["chunks"]], "docs": res["trace"][0]["output"]["documents"],
             "candidates": res["trace"][-1]["input"]["candidate_units"] if res["chunks"] or len(res["trace"]) > 2 else 0,
             "tokens": res["tokens"], "latency_s": res["latency_s"]}
        with OUT.open("a") as f:
            f.write(json.dumps(r) + "\n")
        return r

    def safe(q: dict) -> dict | None:
        try:
            return run(q)
        except Exception as e:  # not recorded, so a rerun retries it
            print(f"{q['id']}: FAILED {e!r}"[:300], flush=True)
            return None

    with ThreadPoolExecutor(3) as pool:
        runs = list(pool.map(safe, qs))
    kept = [(q, r) for q, r in zip(qs, runs) if r is not None]
    print(f"{len(kept)}/{len(qs)} questions completed")
    qs, runs = [q for q, _ in kept], [r for _, r in kept]

    star = star_rows([q["id"] for q in qs])
    neo_rows, star_list = [], []
    for q, r in zip(qs, runs):
        hits = np.array([u in labels[q["id"]] for u in r["selected"]], dtype=float)
        neo_rows.append(proxy(hits, len(q["gold"])))
        star_list.append(star[q["id"]][:2])
        gd = {c["doc"] for c in q["gold"]}
        print(f"{q['id']}: NEO P={neo_rows[-1][0]:.2f} R={neo_rows[-1][1]:.2f} sel={len(r['selected']):3d} | "
              f"STAR P={star[q['id']][0]:.2f} R={star[q['id']][1]:.2f} sel={star[q['id']][2]:3d} | gold={len(q['gold']):3d} "
              f"docs routed {len(gd & set(r['docs']))}/{len(gd)} (+{len(set(r['docs']) - gd)} extra) | {r['latency_s']:.0f}s", flush=True)
    print()
    summarize(f"STAR (SIGN + smoothing, f1), n={len(qs)}", star_list)
    summarize(f"NEO (NeoRAG + tree + verify), n={len(qs)}", neo_rows)


if __name__ == "__main__":
    main()
