"""Layer 3 inputs: document routing scores, per-unit features and the evidence graph.

Every feature is relative to the query's own candidate pool (ranks, z-scores,
percentiles) so a model trained on one corpus transfers to a corpus it has never seen.
"""

import re

import numpy as np

from engine.catalog import card_text
from engine.index import Index, _tok

N_ROUTE_DOCS = 10  # documents kept in the candidate pool, softly weighted by route score


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


def _pct(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)) / max(1, len(x) - 1)


class Router:
    """Scores documents for a query from catalogue cards plus paragraph-level evidence."""

    def __init__(self, index: Index, catalog: dict[str, dict], lattice=None):
        import bm25s
        self.index = index
        self.lattice = lattice
        self.docs = list(catalog)
        self.cards = catalog
        self.doc_units = {d: [] for d in self.docs}
        for i, u in enumerate(index.units):
            self.doc_units.setdefault(u["doc"], []).append(i)
        texts = [card_text(catalog[d]) for d in self.docs]
        self.card_bm25 = bm25s.BM25()
        self.card_bm25.index(_tok(texts), show_progress=False)
        self.card_names = [[n.lower() for n in [catalog[d].get("title", "")] + catalog[d].get("entities", []) if len(n) > 2]
                           for d in self.docs]

    def score(self, query: str, dense: np.ndarray, bm25: np.ndarray) -> dict[str, dict]:
        ql = query.lower()
        card_bm = self.card_bm25.get_scores(_tok([query])[0])
        out = {}
        for j, d in enumerate(self.docs):
            ids = self.doc_units.get(d, [])
            if not ids:
                continue
            top_dense = np.sort(dense[ids])[-10:]
            top_bm = np.sort(bm25[ids])[-10:]
            name_hit = max((len(n) for n in self.card_names[j] if n in ql), default=0)
            out[d] = {"card_bm25": float(card_bm[j]), "unit_dense_top": float(top_dense.mean()),
                      "unit_bm25_top": float(top_bm.mean()), "name_hit": float(name_hit > 0),
                      "name_len": float(name_hit)}
        keys = ["card_bm25", "unit_dense_top", "unit_bm25_top"]
        for k in keys:
            vals = _z(np.array([v[k] for v in out.values()]))
            for v, z in zip(out.values(), vals):
                v[k + "_z"] = float(z)
        for v in out.values():
            v["route"] = v["unit_dense_top_z"] + 0.5 * v["unit_bm25_top_z"] + 0.5 * v["card_bm25_z"] + 2.0 * v["name_hit"]
        ranked = sorted(out, key=lambda d: -out[d]["route"])
        for r, d in enumerate(ranked):
            out[d]["route_rank"] = r
        return out


FEATURES = ["dense", "dense_z", "dense_pct", "bm25_z", "bm25_pct", "doc_route", "doc_rank", "doc_name_hit",
            "is_table", "n_words", "nb_dense_max", "nb_bm25_max", "sec_dense_mean", "sec_dense_max",
            "doc_pos", "num_overlap", "dense_doc_z",
            # version-family lattice
            "family_size", "is_consolidated", "period_match", "is_latest", "unit_shared"]


def candidate_pool(router: Router, query: str, dense: np.ndarray, bm25: np.ndarray):
    """Returns (unit indices in the pool, feature matrix, doc route info)."""
    route = router.score(query, dense, bm25)
    docs = sorted(route, key=lambda d: route[d]["route_rank"])[:N_ROUTE_DOCS]
    idx = np.array([i for d in docs for i in router.doc_units[d]])
    units = router.index.units
    pd, pb = dense[idx], bm25[idx]
    dz, dp, bz, bp = _z(pd), _pct(pd), _z(pb), _pct(pb)
    qnums = set(re.findall(r"\d[\d.,]*", query))

    feats = np.zeros((len(idx), len(FEATURES)), dtype=np.float32)
    pos_in_pool = {u: k for k, u in enumerate(idx)}
    sec_groups: dict[tuple, list[int]] = {}
    doc_groups: dict[str, list[int]] = {}
    for k, i in enumerate(idx):
        u = units[i]
        sec_groups.setdefault((u["doc"], u.get("section", "")), []).append(k)
        doc_groups.setdefault(u["doc"], []).append(k)
    sec_mean, sec_max, dense_doc_z = np.zeros(len(idx)), np.zeros(len(idx)), np.zeros(len(idx))
    for ks in sec_groups.values():
        sec_mean[ks], sec_max[ks] = pd[ks].mean(), pd[ks].max()
    for ks in doc_groups.values():
        dense_doc_z[ks] = _z(pd[ks])

    lat = router.lattice
    vfeat = {d: (lat.doc_features(d, query) if lat else {}) for d in docs}
    for k, i in enumerate(idx):
        u = units[i]
        r = route[u["doc"]]
        v = vfeat[u["doc"]]
        nb = [pos_in_pool[j] for j in (i - 1, i + 1) if j in pos_in_pool and units[j]["doc"] == u["doc"]]
        nums = set(re.findall(r"\d[\d.,]*", u["text"]))
        feats[k] = [pd[k], dz[k], dp[k], bz[k], bp[k], r["route"], r["route_rank"], r["name_hit"],
                    u["type"] == "table", min(len(u["text"].split()), 400),
                    max((pd[j] for j in nb), default=pd[k]), max((pb[j] for j in nb), default=pb[k]),
                    sec_mean[k], sec_max[k], i / max(1, len(units)), len(qnums & nums), dense_doc_z[k],
                    v.get("family_size", 1), v.get("is_consolidated", 0), v.get("period_match", 0),
                    v.get("is_latest", 0), float(lat is not None and lat.unit_role.get(u["id"]) == "shared")]
    return idx, feats, route


def pool_edges(index: Index, idx: np.ndarray, vec_knn: int = 5) -> np.ndarray:
    """Evidence graph over a pool: reading order, same section, and semantic k-NN. Returns [2, E]."""
    units = index.units
    local = {int(u): k for k, u in enumerate(idx)}
    src, dst = [], []
    for k, i in enumerate(idx):
        j = int(i) + 1
        if j in local and units[j]["doc"] == units[int(i)]["doc"]:
            src += [k, local[j]]
            dst += [local[j], k]
    sec: dict[tuple, list[int]] = {}
    for k, i in enumerate(idx):
        sec.setdefault((units[int(i)]["doc"], units[int(i)].get("section", "")), []).append(k)
    for ks in sec.values():
        if 1 < len(ks) <= 40:
            for a in ks:
                for b in ks:
                    if a != b:
                        src.append(a)
                        dst.append(b)
    if vec_knn and index.vecs is not None:
        v = index.vecs[idx]
        for start in range(0, len(idx), 2048):  # blocked so a 20k-unit pool stays in memory
            sims = v[start:start + 2048] @ v.T
            sims[np.arange(len(sims)), np.arange(start, start + len(sims))] = -1
            nn = np.argpartition(-sims, vec_knn, axis=1)[:, :vec_knn]
            for a, row in enumerate(nn, start):
                src += [a] * vec_knn
                dst += row.tolist()
    return np.array([src, dst], dtype=np.int64)
