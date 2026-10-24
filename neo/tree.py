"""The retrieval tree, built without an LLM: Document -> Section -> Page -> Unit.

Sections are contiguous page ranges. A document's PDF outline is used where it exists;
otherwise pages are cut where the topic shifts (TextTiling over page vectors: a cut at
every local minimum of adjacent-page similarity that falls below the document's mean).

Scoring is bottom-up so it works for any document size: unit scores (dense + BM25)
aggregate to pages, pages borrow from their neighbours (relevant pages cluster: on the
training set the page after a gold page is gold far more often than the base rate), and
sections score as the strength of their best pages.
"""

import re
from dataclasses import dataclass, field

import bm25s
import numpy as np

from engine.index import _model, _normalize, _tok


def describe(u: dict) -> str:
    """What a unit is about, for scoring. A table is reduced to its caption, column headers
    and row labels: numbers and markdown pipes otherwise dominate both BM25 and the vector,
    and an analyst's question names line items ("borrowings", "finance costs"), not values."""
    head = u.get("section", "")
    if u.get("type") != "table":
        return f"{head}\n{u['text']}"
    lines = u["text"].split("\n")
    caption = lines[0][len("Table:"):].strip() if lines and lines[0].startswith("Table:") else ""
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines if ln.strip().startswith("|")]
    rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
    header = " ".join(c for c in (rows[0] if rows else []) if c and not re.fullmatch(r"[\d.,()%\s-]+", c))
    labels = [r[0] for r in rows[1:] if r and r[0] and not re.fullmatch(r"[\d.,()%\s-]+", r[0])]
    return f"{head}\ntable: {caption}. columns: {header}. rows: {'; '.join(labels)}"


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else np.zeros_like(x)


@dataclass
class Tree:
    units: list[dict]
    pages: list[tuple[str, int]] = field(default_factory=list)          # page keys, document order
    page_of_unit: np.ndarray | None = None                              # unit -> page index
    page_units: list[list[int]] = field(default_factory=list)
    doc_pages: dict[str, list[int]] = field(default_factory=dict)       # doc -> page indices
    sections: list[tuple[str, int, int]] = field(default_factory=list)  # (doc, first page idx, last page idx)
    section_of_page: np.ndarray | None = None
    unit_vecs: np.ndarray | None = None
    page_vecs: np.ndarray | None = None                                 # optional stronger page embedder
    unit_bm25: object = None
    page_bm25: object = None

    # ---------- build ----------

    @staticmethod
    def build(docs: list[dict]) -> "Tree":
        units = [u for d in docs for u in d["units"]]
        t = Tree(units=units)
        key_idx: dict[tuple[str, int], int] = {}
        pou = np.zeros(len(units), dtype=np.int64)
        for i, u in enumerate(units):
            k = (u["doc"], u["pages"][0])
            if k not in key_idx:
                key_idx[k] = len(t.pages)
                t.pages.append(k)
                t.page_units.append([])
                t.doc_pages.setdefault(u["doc"], []).append(key_idx[k])
            pou[i] = key_idx[k]
            t.page_units[key_idx[k]].append(i)
        t.page_of_unit = pou
        texts = [describe(u) for u in units]
        t.unit_bm25 = bm25s.BM25()
        t.unit_bm25.index(_tok(texts), show_progress=False)
        t.unit_vecs = _normalize(_model().encode([" ".join(x.split()[:300]) for x in texts], batch_size=1024))
        page_text = [" ".join(texts[i] for i in us) for us in t.page_units]
        t.page_bm25 = bm25s.BM25()
        t.page_bm25.index(_tok(page_text), show_progress=False)
        t._segment(docs)
        return t

    def _segment(self, docs: list[dict]) -> None:
        pv = np.zeros((len(self.pages), self.unit_vecs.shape[1]), dtype=np.float32)
        np.add.at(pv, self.page_of_unit, self.unit_vecs)
        pv = _normalize(pv)
        toc_starts = {d["doc"]: {p for lvl, _, p in d["toc"] if lvl <= 2} for d in docs}
        self.section_of_page = np.zeros(len(self.pages), dtype=np.int64)
        for doc, idx in self.doc_pages.items():
            starts = {0}
            pnum = [self.pages[i][1] for i in idx]
            toc = toc_starts.get(doc, set())
            if len(toc) >= 3:
                starts |= {k for k, p in enumerate(pnum) if p in toc}
            else:
                sim = np.array([float(pv[idx[k]] @ pv[idx[k + 1]]) for k in range(len(idx) - 1)])
                if len(sim) > 2:
                    thr = sim.mean()
                    for k in range(1, len(sim) - 1):
                        if sim[k] < thr and sim[k] <= sim[k - 1] and sim[k] <= sim[k + 1]:
                            starts.add(k + 1)
            starts = sorted(starts)
            for a, b in zip(starts, starts[1:] + [len(idx)]):
                sid = len(self.sections)
                self.sections.append((doc, idx[a], idx[b - 1]))
                self.section_of_page[idx[a]:idx[b - 1] + 1] = sid

    # ---------- score ----------

    def page_scores(self, query: str, qvec: np.ndarray, docs: list[str], *, contiguity: float = 0.3,
                    section_weight: float = 0.3, page_vec_weight: float = 0.0, qvec_page=None) -> dict[int, float]:
        """Score every page of `docs` for the query; returns {page index: score}."""
        pidx = np.array([i for d in docs for i in self.doc_pages.get(d, [])])
        if not len(pidx):
            return {}
        uidx = np.array([u for p in pidx for u in self.page_units[p]])
        owner = np.searchsorted(pidx, self.page_of_unit[uidx]) if np.all(np.diff(pidx) > 0) else \
            np.array([{p: k for k, p in enumerate(pidx)}[self.page_of_unit[u]] for u in uidx])
        dense = self.unit_vecs[uidx] @ qvec
        bm = self.unit_bm25.get_scores(_tok([query])[0])[uidx]
        unit = _z(dense) + 0.5 * _z(bm)
        best = np.full(len(pidx), -1e9)
        np.maximum.at(best, owner, unit)
        page_bm = _z(self.page_bm25.get_scores(_tok([query])[0])[pidx])
        s = _z(best) + 0.5 * page_bm
        if page_vec_weight and self.page_vecs is not None and qvec_page is not None:
            s = s + page_vec_weight * _z(self.page_vecs[pidx] @ qvec_page)
        # contiguity: neighbours on either side within the same document
        if contiguity:
            pos = {p: k for k, p in enumerate(pidx)}
            nb = np.array([np.mean([s[pos[q]] for q in (p - 1, p + 1)
                                    if q in pos and self.pages[q][0] == self.pages[p][0]] or [s[k]])
                           for k, p in enumerate(pidx)])
            s = s + contiguity * nb
        if section_weight:
            sec = self.section_of_page[pidx]
            top = {}
            for k, sid in enumerate(sec):
                top.setdefault(sid, []).append(s[k])
            sec_score = {sid: float(np.mean(sorted(v)[-3:])) for sid, v in top.items()}
            s = s + section_weight * np.array([sec_score[x] for x in sec])
        return {int(p): float(v) for p, v in zip(pidx, s)}
