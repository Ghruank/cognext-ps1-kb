"""Offline retrieval comparison on the free proxy (no LLM calls).

    python -m scripts.exp_retrieval [--rebuild]

Proxy metrics per question, mirroring the official definitions:
    P = selected units that carry gold text / selected units
    R = selected units that carry gold text / number of gold chunks
All learned models are evaluated out-of-fold (5 folds grouped by question).
"""

import json
import pickle
import statistics as st
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from engine.features import FEATURES, Router, candidate_pool, pool_edges
from engine.graph import norm_adj, sign_features, smooth
from engine.index import Index, embed_query
from engine.select import select
from eval import bench

POOLS = Path("storage/train_pools.pkl")


def build_pools() -> list[dict]:
    index = Index.load("storage/train_index.pkl")
    catalog = json.load(open("storage/train_catalog.json"))
    labels = pickle.load(open("storage/train_labels.pkl", "rb"))
    router = Router(index, catalog, pickle.load(open("storage/train_lattice.pkl", "rb")))
    pools = []
    for q in bench.load():
        qv = embed_query(q["question"])
        dense, bm25 = index.dense_scores(qv), index.bm25_scores(q["question"])
        idx, feats, route = candidate_pool(router, q["question"], dense, bm25)
        pos = labels[q["id"]]
        y = np.array([index.units[i]["id"] in pos for i in idx], dtype=np.int8)
        gold_docs = {c["doc"] for c in q["gold"]}
        pool_docs = {index.units[i]["doc"] for i in idx}
        # global hybrid ranking for the banned-baseline reference (reciprocal rank fusion)
        rrf = 1 / (60 + np.argsort(np.argsort(-dense))) + 1 / (60 + np.argsort(np.argsort(-bm25)))
        top = np.argsort(-rrf)[:100]
        pools.append({"id": q["id"], "idx": idx, "X": feats, "y": y, "n_gold": len(q["gold"]),
                      "edges": pool_edges(index, idx),
                      "n_pos_total": len(pos), "route_recall": len(gold_docs & pool_docs) / max(1, len(gold_docs)),
                      "baseline_top": [index.units[i]["id"] in pos for i in top]})
        print(f"{q['id']}: pool {len(idx):5d}, positives in pool {int(y.sum()):3d}/{len(pos):3d}, "
              f"gold docs routed {len(gold_docs & pool_docs)}/{len(gold_docs)}", flush=True)
    pickle.dump(pools, open(POOLS, "wb"))
    return pools


def proxy(sel_hits: np.ndarray, n_gold: int) -> tuple[float, float]:
    if n_gold == 0:
        return (1.0, 1.0) if len(sel_hits) == 0 else (0.0, 0.0)
    if len(sel_hits) == 0:
        return 0.0, 0.0
    return float(sel_hits.mean()), float(sel_hits.sum() / n_gold)


def summarize(name: str, rows: list[tuple]) -> None:
    p, r = [x[0] for x in rows], [x[1] for x in rows]
    print(f"{name:42} P={st.mean(p):.3f}  R={st.mean(r):.3f}  F={2*st.mean(p)*st.mean(r)/(st.mean(p)+st.mean(r)+1e-9):.3f}")


def oof_lgbm(pools: list[dict], key: str = "X") -> list[np.ndarray]:
    groups = np.concatenate([[k] * len(p["y"]) for k, p in enumerate(pools)])
    X = np.concatenate([p[key] for p in pools])
    y = np.concatenate([p["y"] for p in pools])
    raw = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
        m.fit(X[tr], y[tr])
        raw[te] = m.predict_proba(X[te])[:, 1]
    # Platt calibration, itself out-of-fold over questions
    logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1)).reshape(-1, 1)
    cal = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(logit, y, groups):
        lr = LogisticRegression(C=1.0).fit(logit[tr], y[tr])
        cal[te] = lr.predict_proba(logit[te])[:, 1]
    out, start = [], 0
    for p in pools:
        out.append(cal[start:start + len(p["y"])])
        start += len(p["y"])
    return out


def evaluate_selector(name: str, pools: list[dict], probs: list[np.ndarray], **kw) -> None:
    rows, ks, sums = [], [], []
    for p, pr in zip(pools, probs):
        chosen, g = select(pr, **kw)
        rows.append(proxy(p["y"][chosen].astype(float), p["n_gold"]))
        ks.append(len(chosen))
        sums.append((g, p["n_gold"]))
    summarize(name, rows)
    corr = np.corrcoef([a for a, _ in sums], [b for _, b in sums])[0, 1]
    print(f"{'':42} median k={st.median(ks):.0f}  corr(G_hat, gold count)={corr:.2f}")


def main() -> None:
    pools = pickle.load(open(POOLS, "rb")) if POOLS.exists() and "--rebuild" not in sys.argv else build_pools()
    print(f"\nrouting: gold docs inside the pool, mean {st.mean(p['route_recall'] for p in pools):.3f}")
    print(f"ceiling: positives reachable in pool / gold count, mean "
          f"{st.mean(min(1, p['y'].sum() / p['n_gold']) for p in pools if p['n_gold']):.3f}\n")

    for k in (10, 20, 50, 100):
        summarize(f"R  banned baseline: hybrid RRF top-{k}",
                  [proxy(np.array(p["baseline_top"][:k], dtype=float), p["n_gold"]) for p in pools])
    print()
    probs = oof_lgbm(pools)
    pickle.dump(probs, open("storage/oof_lgbm.pkl", "wb"))
    for util in ("f1", "mean"):
        evaluate_selector(f"2  calibrated selector (LightGBM, {util})", pools, probs, utility=util)
    for k in (20, 50, 100):
        summarize(f"   LightGBM scores, fixed top-{k}",
                  [proxy(p["y"][np.argsort(-pr)[:k]].astype(float), p["n_gold"]) for p, pr in zip(pools, probs)])
    print()
    # Architecture 3: evidence graph. SIGN multi-hop features, then Correct & Smooth style smoothing.
    for p in pools:
        p["adj"] = norm_adj(p["edges"], len(p["y"]))
        p["Xg"] = sign_features(p["X"], p["adj"], hops=2)
    gprobs = oof_lgbm(pools, key="Xg")
    pickle.dump(gprobs, open("storage/oof_graph.pkl", "wb"))
    for util in ("f1", "mean"):
        evaluate_selector(f"3  evidence graph (SIGN + LightGBM, {util})", pools, gprobs, utility=util)
    for alpha in (0.3, 0.5):
        sm = [smooth(pr, p["adj"], alpha=alpha) for p, pr in zip(pools, gprobs)]
        evaluate_selector(f"3+ SIGN + smoothing a={alpha} (f1)", pools, sm, utility="f1")
    sm = [smooth(pr, p["adj"], alpha=0.5) for p, pr in zip(pools, probs)]
    evaluate_selector("   ablation: no SIGN, smoothing only (f1)", pools, sm, utility="f1")


if __name__ == "__main__":
    main()
