"""Query time, stage 2: LLM page navigation over the page-card layer of the graph.

The parser has already fixed which documents (and versions) are in scope. Here the LLM
reads every page card of those documents and picks the pages that hold evidence. Cards
are short, so even a 400-page annual report fits in a single call; long documents are
split into slices that run in parallel.
"""

import json
from concurrent.futures import ThreadPoolExecutor

from alpha.llm import chat

SLICE_PAGES = 70   # ~70 cards per call keeps a call under Groq's 8k tokens/minute

SYSTEM = "You navigate documents for a bank analyst. Respond only with JSON."

PROMPT = """Question: {question}
{facets}
Below are page cards from "{doc}" ({title}). Each card summarises one page. Select every page that likely contains evidence needed for a complete answer: facts, figures, rules, explanations, tables or notes covering any part of the question. Be inclusive across all parts of the question, since a page that is missed cannot be recovered later. Skip filler pages and pages about unrelated topics.

{cards}

Return JSON: {{"pages": [page numbers]}}"""


def _slice_call(question: str, facets: list[str], doc: str, title: str,
                items: list[tuple[int, str]]) -> tuple[list[int], dict]:
    cards = "\n".join(f"p{p}: {c}" for p, c in items)
    fac = ("Parts of the question to cover:\n" + "\n".join(f"- {f}" for f in facets) + "\n") if facets else ""
    try:
        content, usage = chat(SYSTEM, PROMPT.format(question=question, facets=fac, doc=doc, title=title, cards=cards),
                              max_tokens=1200, json_mode=True, kind="neo-navigate")
        valid = {p for p, _ in items}
        return sorted({int(p) for p in json.loads(content).get("pages", []) if str(p).isdigit() and int(p) in valid}), usage
    except (RuntimeError, ValueError, TypeError):
        return [], {}


def select_pages(question: str, facets: list[str], docs: list[str], cards: dict, catalog: dict,
                 workers: int = 6) -> tuple[dict[str, list[int]], dict]:
    """Returns ({doc: [pages]}, usage totals)."""
    jobs = []
    for d in docs:
        items = sorted((p, c["card"]) for (dd, p), c in cards.items() if dd == d and c.get("kind") != "filler")
        for i in range(0, len(items), SLICE_PAGES):
            jobs.append((d, catalog.get(d, {}).get("title", ""), items[i:i + SLICE_PAGES]))
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda j: _slice_call(question, facets, *j), jobs))
    out: dict[str, list[int]] = {}
    for (d, _, _), (pages, _) in zip(jobs, results):
        out.setdefault(d, []).extend(pages)
    usage = {"prompt_tokens": sum(u.get("prompt_tokens", 0) for _, u in results),
             "completion_tokens": sum(u.get("completion_tokens", 0) for _, u in results),
             "usd": sum(u.get("usd", 0) for _, u in results), "calls": len(jobs)}
    return {d: sorted(set(p)) for d, p in out.items()}, usage
