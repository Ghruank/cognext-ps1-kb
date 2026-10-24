"""Page-coherence refinement on top of NEO's unit verifier. Offline: saved Claude verdicts, no LLM calls.

    .venv/bin/python -m alpha.exp_coherence

Gold chunks cluster: an annotator marks runs of blocks on the same pages. The verifier judges
each unit alone, so it keeps isolated look-alikes (precision loss) and drops short lines inside
gold runs (recall loss). A small classifier over verdict-pattern features (how much of the page
was verified, verified neighbours, length, type) re-decides every candidate unit. Scored with
leave-question-group-out CV on the same proxy as alpha.claude_tasks.score.
"""

import json
import os
import re
import pickle
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from eval import bench
from eval.gold_map import label_units
from engine.index import embed_query
from pathlib import Path
from scripts.exp_retrieval import proxy, summarize

STORE = Path(__file__).resolve().parent / "storage"
DIR = STORE / "claude"
MODEL = STORE / "coherence.pkl"
THRESHOLD = 0.2
_model = None
USE_CE = (STORE / "ce_scores.pkl").exists() and os.environ.get("ALPHA_CE", "0") == "1"  # A/B: no gain, off by default


_TOK = re.compile(r"[a-z0-9]+")
STOP = set("the a an of and or to in for on by with is are was were be what which how does do as at from that this its it their".split())


def toks(t: str) -> set[str]:
    return {w for w in _TOK.findall(t.lower()) if w not in STOP and len(w) > 1}


def features(tree, pos: list[int], verified: set[int], question: str = "", facets: list[str] = (), ce=None) -> np.ndarray:
    """One row per candidate (index into pos)."""
    units = [tree.units[p] for p in pos]
    page_of = [(u["doc"], int(str(u["pages"]).strip("[]").split(",")[0])) for u in units]
    by_page = defaultdict(list)
    for k, pg in enumerate(page_of):
        by_page[pg].append(k)
    doc_ver = defaultdict(int)
    doc_all = defaultdict(int)
    for k, pg in enumerate(page_of):
        doc_all[pg[0]] += 1
        doc_ver[pg[0]] += k in verified
    n_ver = max(len(verified), 1)
    qv = embed_query(question + " " + " ; ".join(facets))
    sims = tree.unit_vecs[pos] @ qv
    qt = toks(question)
    ft = [toks(f) for f in facets] or [qt]
    lex = [len(toks(u["text"]) & qt) / max(len(qt), 1) for u in units]
    flex = [max(len(toks(u["text"]) & f) / max(len(f), 1) for f in ft) for u in units]
    page_sim = defaultdict(float)
    for k, pg in enumerate(page_of):
        page_sim[pg] = max(page_sim[pg], sims[k])
    rank = np.argsort(np.argsort(-sims)) / len(units)
    has_ce = ce is not None
    ce = np.zeros(len(units)) if ce is None else np.asarray(ce, dtype=float)
    ce_rank = np.argsort(np.argsort(-ce)) / len(units)
    page_ce = defaultdict(lambda: -1e9)
    for k, pg in enumerate(page_of):
        page_ce[pg] = max(page_ce[pg], ce[k])
    rows = []
    for k, u in enumerate(units):
        same = by_page[page_of[k]]
        pv = sum(j in verified for j in same)
        near = [j for j in range(k - 3, k + 4) if 0 <= j < len(units) and j != k and page_of[j][0] == page_of[k][0]]
        nb1 = sum(j in verified for j in (k - 1, k + 1) if j in near)
        nb3 = sum(j in verified for j in near)
        words = len(u["text"].split())
        rows.append([
            k in verified, pv, pv / len(same), len(same), (pv - (k in verified)) / max(len(same) - 1, 1),
            nb1, nb3, np.log1p(words), u["type"] == "table", u["type"] == "heading" if "type" in u else 0,
            doc_ver[page_of[k][0]] / n_ver, doc_ver[page_of[k][0]] / doc_all[page_of[k][0]],
            len(verified) / len(units), np.log1p(len(units)),
            u["text"].strip().endswith(":"), sum(c.isdigit() for c in u["text"]) / max(len(u["text"]), 1),
            sims[k], rank[k], page_sim[page_of[k]], lex[k], flex[k],
            np.mean([lex[j] for j in near]) if near else 0, np.mean([sims[j] for j in near]) if near else 0,
        ] + ([ce[k], ce_rank[k], page_ce[page_of[k]], np.mean([ce[j] for j in near]) if near else 0,
              ce[k] - ce.max()] if has_ce else []))
    return np.array(rows, dtype=float)


def main() -> None:
    from alpha.pipeline import load
    _, tree = load("train")
    index = json.load(open(DIR / "verify_index.json"))
    labels = label_units(pickle.load(open(STORE / "train_docs.pkl", "rb")))
    qs = [q for q in bench.load() if (DIR / "verdict" / f"{q['id']}.json").exists() and index[q["id"]]["positions"]]
    ce = pickle.load(open(STORE / "ce_scores.pkl", "rb")) if (STORE / "ce_scores.pkl").exists() else {}
    if ce:
        qs = [q for q in qs if q["id"] in ce]
    if not USE_CE:
        ce = {}
    if os.environ.get("ALPHA_NOSAVE"):
        global MODEL
        MODEL = STORE / "coherence_ab.pkl"
        print(f"cross-encoder cache covers {len(qs)} questions; CE features {'on' if USE_CE else 'off'}")
    X, y, g, meta = [], [], [], []
    for gi, q in enumerate(qs):
        pos = index[q["id"]]["positions"]
        ids = json.load(open(DIR / "verdict" / f"{q['id']}.json")).get("relevant_ids", [])
        ver = {int(k) for k in ids if 0 <= int(k) < len(pos)}
        plan = index[q["id"]]["plan"]
        f = features(tree, pos, ver, q["question"], plan["facets"], ce.get(q["id"]))
        lab = np.array([tree.units[p]["id"] in labels[q["id"]] for p in pos], dtype=float)
        X.append(f); y.append(lab); g += [gi] * len(pos); meta.append((q, ver, lab))
    Xa, ya, ga = np.vstack(X), np.concatenate(y), np.array(g)

    ceiling = [proxy(lab, len(q["gold"])) for q, _, lab in meta]
    summarize("candidate pool (all units kept)", ceiling)
    summarize("NEO verifier as is", [proxy(lab[sorted(v)], len(q["gold"])) for q, v, lab in meta])
    summarize("oracle selection (pool ceiling)", [proxy(lab[lab > 0], len(q["gold"])) for q, _, lab in meta])

    prob = np.zeros(len(ya))
    for tr, te in GroupKFold(5).split(Xa, ya, ga):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0)
        m.fit(Xa[tr], ya[tr])
        prob[te] = m.predict_proba(Xa[te])[:, 1]
    off = np.cumsum([0] + [len(l) for _, _, l in meta])
    for th in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        rows = []
        for i, (q, _, lab) in enumerate(meta):
            pq = prob[off[i]:off[i + 1]]
            rows.append(proxy(lab[pq >= th], len(q["gold"])))
        summarize(f"coherence classifier th={th}", rows)
    best = None
    for th in (0.15, 0.2, 0.25, 0.3):
        for a in (0.3, 0.4, 0.5, 0.6):
            rows = []
            for i, (q, _, lab) in enumerate(meta):
                pq = prob[off[i]:off[i + 1]]
                rows.append(proxy(lab[pq >= max(th, a * pq.max())], len(q["gold"])))
            p, r = np.mean([x[0] for x in rows]), np.mean([x[1] for x in rows])
            summarize(f"relative th={th} a={a}", rows)
            if best is None or 2 * p * r / (p + r) > best[0]:
                best = (2 * p * r / (p + r), th, a)
    print("best", best)
    np.save(DIR / "coherence_oof.npy", prob)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0)
    model.fit(Xa, ya)
    pickle.dump({"model": model, "threshold": THRESHOLD, "use_ce": USE_CE}, open(MODEL, "wb"))
    print(f"saved {MODEL}")


def refine(tree, question: str, facets: list[str], positions: list[int], chosen: list[int]) -> list[int]:
    """Pipeline stage: re-decide every candidate from the verdict pattern; returns positions kept."""
    global _model
    if not positions:
        return []
    if _model is None:
        _model = pickle.load(open(MODEL, "rb"))
    k_of = {p: k for k, p in enumerate(positions)}
    ver = {k_of[c] for c in chosen if c in k_of}
    ce = None
    if _model.get("use_ce"):
        from alpha.ce_scores import ce_score, shortlist
        ce = ce_score(question, [tree.units[p]["text"] for p in positions], shortlist(tree, question, positions, ver))
    prob = _model["model"].predict_proba(features(tree, positions, ver, question, facets, ce))[:, 1]
    return [p for p, pr in zip(positions, prob) if pr >= _model["threshold"]]


if __name__ == "__main__":
    main()
