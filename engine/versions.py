"""Layer 2b, Version-Family Lattice: group near-duplicate documents and mark what differs.

Idea after VersionRAG (Huwiler et al., arXiv 2510.08109): documents that are versions of
one another (FY2025 vs FY2026 report, Basel 2017 text vs its consolidation) form a family;
retrieval should know which version a query asks for and which paragraphs exist only in
one version. Our implementation:

  - shingle overlap between documents gives containment |A & B| / |A|, which flags a
    consolidated text (it contains most of a smaller document) better than Jaccard
  - families = union-find over pairs with containment above a threshold, or the same
    entity and document type in the catalogue
  - each unit is "shared" (its text appears in another family member) or "delta"
  - query years / fiscal years are matched against each document's catalogue period
"""

import re
from itertools import combinations

FAMILY_CONTAINMENT = 0.25
SHARED_UNIT = 0.7
CHANGE_SIM = 0.85  # cosine between version-specific paragraphs that are the same item in two versions

_COMPARE = re.compile(r"\b(compar\w*|chang\w*|evolv\w*|versus|vs\.?|differ\w*|shift\w*|trend\w*|over the (?:past|last)|"
                      r"year[- ]on[- ]year|between (?:the|its) (?:two|last)|across (?:the|its|both)|both (?:of )?(?:its|the)|"
                      r"from \S+ to \S+|swing\w*|trace|before and after|relative to (?:a|last|the prior) year)\b", re.I)


def query_intent(query: str) -> str:
    """'compare' when the question spans versions or periods, else 'single'."""
    return "compare" if _COMPARE.search(query) or len(_years(query)) >= 2 else "single"


def _shingles(text: str, n: int = 5) -> set[int]:
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {hash(" ".join(w[i:i + n])) for i in range(max(1, len(w) - n + 1))}


def _years(text: str) -> set[int]:
    """Calendar years mentioned, with fiscal-year forms normalised: FY2026, FY 2025-26, 2025-26 -> 2026."""
    ys = set()
    for a, b in re.findall(r"(?:FY\s?)?((?:19|20)\d\d)\s?[-/–]\s?(\d\d)\b", text):
        ys.add(int(a[:2] + b))
    for y in re.findall(r"FY\s?((?:19|20)\d\d)\b", text):
        ys.add(int(y))
    ys |= {int(y) for y in re.findall(r"\b((?:19|20)\d\d)\b", text)}
    return ys


class Lattice:
    def __init__(self, docs: list[dict], catalog: dict[str, dict]):
        self.catalog = catalog
        unit_sh = {u["id"]: _shingles(u["text"]) for d in docs for u in d["units"]}
        doc_sh = {d["doc"]: set().union(*(unit_sh[u["id"]] for u in d["units"])) if d["units"] else set()
                  for d in docs}
        names = list(doc_sh)
        parent = {n: n for n in names}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        self.containment: dict[tuple[str, str], float] = {}
        for a, b in combinations(names, 2):
            inter = len(doc_sh[a] & doc_sh[b])
            ca, cb = inter / max(1, len(doc_sh[a])), inter / max(1, len(doc_sh[b]))
            self.containment[(a, b)], self.containment[(b, a)] = ca, cb
            ea = {e.lower() for e in catalog.get(a, {}).get("entities", [])[:2]}
            eb = {e.lower() for e in catalog.get(b, {}).get("entities", [])[:2]}
            same_series = (catalog.get(a, {}).get("doc_type") == catalog.get(b, {}).get("doc_type")
                           and catalog.get(a, {}).get("doc_type") == "annual_report" and ea & eb)
            if max(ca, cb) >= FAMILY_CONTAINMENT or same_series:
                parent[find(a)] = find(b)

        self.family = {n: find(n) for n in names}
        members: dict[str, list[str]] = {}
        for n, f in self.family.items():
            members.setdefault(f, []).append(n)
        self.members = members

        # a consolidated document contains most of a smaller family member
        self.consolidated = {a for (a, b), c in self.containment.items()
                             if self.family[a] == self.family[b] and len(doc_sh[a]) > 1.5 * len(doc_sh[b])
                             and self.containment[(b, a)] >= 0.5}
        self.doc_years = {n: _years(f"{catalog.get(n, {}).get('period', '')} {catalog.get(n, {}).get('date', '')} {n}")
                          for n in names}

        self.unit_role: dict[str, str] = {}
        for d in docs:
            others = [doc_sh[m] for m in members[self.family[d["doc"]]] if m != d["doc"]]
            for u in d["units"]:
                sh = unit_sh[u["id"]]
                shared = any(len(sh & o) / max(1, len(sh)) >= SHARED_UNIT for o in others) if others else False
                self.unit_role[u["id"]] = "shared" if shared else "delta"

    def build_change_links(self, index) -> None:
        """Pair version-specific paragraphs that are the same item in two versions of a family
        (e.g. one financial-highlights paragraph with different figures). These become
        graph edges, so evidence found in one version pulls in its counterpart."""
        import numpy as np
        pos = index.pos
        self.change_links: list[tuple[str, str]] = []
        for fam in self.members.values():
            docs = [d for d in fam if d not in self.consolidated]
            for a, b in combinations(docs, 2):
                ua = [u["id"] for u in index.units if u["doc"] == a and self.unit_role.get(u["id"]) == "delta"]
                ub = [u["id"] for u in index.units if u["doc"] == b and self.unit_role.get(u["id"]) == "delta"]
                if not ua or not ub:
                    continue
                va, vb = index.vecs[[pos[i] for i in ua]], index.vecs[[pos[i] for i in ub]]
                for s0 in range(0, len(ua), 2048):
                    sims = va[s0:s0 + 2048] @ vb.T
                    best = sims.argmax(1)
                    for r, j in enumerate(best):
                        if sims[r, j] >= CHANGE_SIM:
                            self.change_links.append((ua[s0 + r], ub[int(j)]))

    def doc_features(self, doc: str, query: str) -> dict[str, float]:
        qy = _years(query)
        dy = self.doc_years.get(doc, set())
        fam = self.members[self.family[doc]]
        if not qy or not dy:
            period = 0.0
        else:
            period = 1.0 if qy & dy else -1.0
        latest = max((max(self.doc_years[m], default=0) for m in fam), default=0)
        compare = query_intent(query) == "compare"
        # comparative questions want every version in the family; single-version ones the matching period
        return {"family_size": float(len(fam)), "is_consolidated": float(doc in self.consolidated),
                "period_match": period, "is_latest": float(max(dy, default=0) == latest and len(fam) > 1),
                "q_compare": float(compare), "compare_x_family": float(compare and len(fam) > 1),
                "single_x_mismatch": float(not compare and period < 0)}

    def summary(self) -> list[list[str]]:
        return [m for m in self.members.values() if len(m) > 1]
