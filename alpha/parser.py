"""Query time, stage 1: NeoRAG retrieval-mode parser, extended for financial corpora.

NeoRAG classifies a query as VersionRetrieval / ChangeRetrieval / ContentRetrieval and
extracts category, documentation and version from the graph listing. Here the parser
also returns the exact documents in scope (a hard filter, not a soft score, which is what
separates FY2025 from FY2026 or a proposal from the final rule) and the parts of the
question that need evidence, which the page navigator uses for coverage.
"""

import json
from dataclasses import dataclass, field

from alpha.llm import chat
from alpha.graph import NeoGraph, norm_name

SYSTEM = "You route questions to documents in a bank's library. Respond only with JSON."

PROMPT = """Library (category -> documents, with version/period):
{library}

Question: {question}

1. retrieval: "ChangeRetrieval" if the question compares versions, years or periods of the same document series or asks what changed; "VersionRetrieval" if it only asks which versions/editions exist; otherwise "ContentRetrieval".
2. documents: every filename (exactly as listed) whose content is needed to answer. Pick the version that matches the year/period/edition the question names; include several versions only for ChangeRetrieval or when the question spans periods. Include companion documents (e.g. a summary memo, FAQ, implementation guide or the standard it explains) only when the question's topic is covered by them too. Use [] if no document in the library covers the question.
3. facets: 2-8 short phrases naming the distinct pieces of information a complete answer needs.

Return JSON: {{"retrieval": str, "documents": [str], "facets": [str], "reason": str}}"""


@dataclass
class Plan:
    retrieval: str = "ContentRetrieval"
    documents: list[str] = field(default_factory=list)
    facets: list[str] = field(default_factory=list)
    reason: str = ""
    usage: dict = field(default_factory=dict)


def _resolve(name: str, docs: list[str]) -> str | None:
    """Exact filename, else the unique filename the LLM's string is a normalised substring of."""
    if name in docs:
        return name
    n = norm_name(name)
    hits = [d for d in docs if n and (n in norm_name(d) or norm_name(d) in n)]
    return hits[0] if len(hits) == 1 else None


def parse(question: str, graph: NeoGraph) -> Plan:
    docs = list(graph.catalog)
    try:
        content, usage = chat(SYSTEM, PROMPT.format(library=graph.describe(), question=question),
                              max_tokens=1200, json_mode=True, kind="neo-parse")
        raw = json.loads(content)
    except (RuntimeError, ValueError):
        return Plan(documents=docs, reason="parser failed; no document filter")
    plan = Plan(retrieval=str(raw.get("retrieval", "ContentRetrieval")),
                facets=[str(f) for f in raw.get("facets", [])][:8], reason=str(raw.get("reason", "")), usage=usage)
    for name in raw.get("documents", []):
        d = _resolve(str(name), docs)
        if d and d not in plan.documents:
            plan.documents.append(d)
    if plan.retrieval == "ChangeRetrieval":
        # a change question needs every version in the family of each chosen document
        for d in list(plan.documents):
            for v in graph.versions(d):
                if v not in plan.documents and v not in graph.consolidated:
                    plan.documents.append(v)
    return plan
