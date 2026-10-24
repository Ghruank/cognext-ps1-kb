"""Index time: the NeoRAG graph, Category -> Document -> Version -> Page -> Unit, plus Change edges.

Ported from NeoRAG (github.com/Ghruank/neorag), with these changes:
  - Category nodes: NeoRAG clusters documentation descriptions with one LLM call; same
    here over the STAR catalogue cards (engine.catalog), so no category is assumed in advance.
  - Version nodes: NeoRAG reads version strings from filenames and changelogs. Financial
    corpora have no changelogs, so versions come from the catalogue period/date and the
    version families in engine.versions.Lattice (shingle containment + same issuer series).
  - Change nodes: NeoRAG's "Differ" origin. Paragraphs that exist in only one version and
    pair with a paragraph in another version (Lattice.change_links) are change edges.
  - Content nodes are pages, each with an LLM card (neo.pagecards), not raw chunks.
"""

import json
import re
from dataclasses import dataclass, field

from alpha.llm import chat
from engine.versions import Lattice, _years

CLUSTER_SYSTEM = "You organise a document library into categories. Respond only with JSON."

CLUSTER_PROMPT = """Group these documents into categories by subject matter (e.g. one company's reports, one regulatory framework, one accounting standard, one family of quantitative models). Different years or versions of the same document series always go in the same category. Every document index must appear in exactly one category.

{docs}

Return JSON: {{"categories": [{{"name": str, "description": str, "docs": [indices]}}]}}"""


@dataclass
class NeoGraph:
    catalog: dict[str, dict]
    categories: dict[str, dict] = field(default_factory=dict)   # name -> {"description", "docs"}
    doc_category: dict[str, str] = field(default_factory=dict)
    family: dict[str, str] = field(default_factory=dict)        # doc -> family root
    family_members: dict[str, list[str]] = field(default_factory=dict)
    doc_years: dict[str, list[int]] = field(default_factory=dict)
    consolidated: set[str] = field(default_factory=set)
    change_links: list[tuple[str, str]] = field(default_factory=list)  # (unit id, unit id) across versions
    page_cards: dict[tuple[str, int], dict] = field(default_factory=dict)
    page_units: dict[tuple[str, int], list[int]] = field(default_factory=dict)  # (doc, page) -> unit positions

    def versions(self, doc: str) -> list[str]:
        """Other versions of this document, oldest first."""
        fam = self.family_members.get(self.family.get(doc, doc), [doc])
        return sorted(fam, key=lambda d: max(self.doc_years.get(d) or [0]))

    def describe(self) -> str:
        """Category -> document -> version listing the query parser reads."""
        lines = []
        for name, cat in self.categories.items():
            lines.append(f"## {name}: {cat['description']}")
            for d in cat["docs"]:
                c = self.catalog.get(d, {})
                ver = " / ".join(x for x in (c.get("period", ""), c.get("date", ""), c.get("version", "")) if x)
                fam = [m for m in self.versions(d) if m != d]
                lines.append(f"- {d} | {c.get('title', '')} | {c.get('doc_type', '')} | {c.get('jurisdiction', '')}"
                             + (f" | version: {ver}" if ver else "")
                             + (f" | other versions: {', '.join(fam)}" if fam else "")
                             + (" | consolidated text" if d in self.consolidated else "")
                             + f" | topics: {', '.join(c.get('topics', [])[:8])}")
        return "\n".join(lines)


def cluster_categories(catalog: dict[str, dict], family: dict[str, str]) -> dict[str, dict]:
    names = list(catalog)
    listing = "\n".join(f"{i}. {d} | {catalog[d].get('title', '')} | {catalog[d].get('issuer', '')} | "
                        f"{catalog[d].get('doc_type', '')} | {catalog[d].get('summary', '')[:200]}"
                        for i, d in enumerate(names))
    try:
        content, _ = chat(CLUSTER_SYSTEM, CLUSTER_PROMPT.format(docs=listing), max_tokens=8000,
                          json_mode=True, kind="neo-cluster")
        cats = json.loads(content).get("categories", [])
    except (RuntimeError, ValueError):
        cats = []
    out, seen = {}, set()
    for c in cats:
        docs = [names[i] for i in c.get("docs", []) if isinstance(i, int) and 0 <= i < len(names) and names[i] not in seen]
        if docs:
            seen.update(docs)
            out[str(c.get("name", f"category {len(out)}"))] = {"description": str(c.get("description", "")), "docs": docs}
    # version families must not be split across categories; anything unplaced gets its own category
    for d in names:
        if d in seen:
            continue
        home = next((n for n, c in out.items() if any(family.get(m) == family.get(d) for m in c["docs"])), None)
        if home:
            out[home]["docs"].append(d)
        else:
            out[catalog[d].get("title") or d] = {"description": catalog[d].get("summary", ""), "docs": [d]}
        seen.add(d)
    return out


def build_graph(docs: list[dict], catalog: dict[str, dict], index, page_cards: dict) -> NeoGraph:
    lattice = Lattice(docs, catalog)
    lattice.build_change_links(index)
    g = NeoGraph(catalog=catalog, family=lattice.family, family_members=lattice.members,
                 doc_years={d: sorted(y) for d, y in lattice.doc_years.items()},
                 consolidated=set(lattice.consolidated), change_links=list(lattice.change_links),
                 page_cards=page_cards)
    g.categories = cluster_categories(catalog, lattice.family)
    g.doc_category = {d: n for n, c in g.categories.items() for d in c["docs"]}
    for i, u in enumerate(index.units):
        g.page_units.setdefault((u["doc"], u["pages"][0]), []).append(i)
    return g


def query_years(q: str) -> set[int]:
    return _years(q)


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())
