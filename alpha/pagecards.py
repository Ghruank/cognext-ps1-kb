"""Index time: one short LLM-written card per page, the lowest content node of the NeoRAG graph.

NeoRAG describes every category, documentation and version node so an LLM can route a
query through the graph by reading descriptions instead of comparing embeddings. We push
that down one level: each page gets a card saying what it covers (topics, metric names,
periods, entities, kind of content). At query time the LLM picks pages by reading cards,
which finds evidence that similarity search misses: on the training set, with the right
documents already known, hybrid similarity reaches only 52% of gold pages in its top 80.
"""

import json
from concurrent.futures import ThreadPoolExecutor

from alpha.llm import chat

MAX_PAGE_WORDS = 600
BATCH_WORDS = 2600   # input per call; with the reply this stays under Groq's 8k tokens/minute
MAX_PAGES_PER_CALL = 6

SYSTEM = "You index pages of financial and regulatory documents for a retrieval system. Respond only with JSON."

PROMPT = """Document: {doc} ({title})

For each page below write a card of 25-60 words saying what the page covers, so a reader can decide from the card alone whether the page helps answer a question. Name the specific topics, metrics and line items, periods/years, entities, rules or paragraph numbers, and the kind of content (narrative, table, note to accounts, definition, example, list). Do not copy numbers. Also write 2-4 questions an analyst could ask that this page answers. Mark pages that are only a cover, table of contents, blank or boilerplate with "kind": "filler" (no questions).

{pages}

Return JSON: {{"cards": [{{"page": int, "kind": "content" or "filler", "card": str, "questions": [str]}}]}}"""


def page_texts(units: list[dict]) -> dict[tuple[str, int], str]:
    """(doc, page) -> concatenated unit text on that page."""
    out: dict[tuple[str, int], list[str]] = {}
    for u in units:
        out.setdefault((u["doc"], u["pages"][0]), []).append(u["text"])
    return {k: " ".join(" ".join(v).split()[:MAX_PAGE_WORDS]) for k, v in out.items()}


def _pack(pages: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    """Consecutive pages packed into calls of at most BATCH_WORDS words."""
    out, cur, n = [], [], 0
    for p, t in pages:
        w = len(t.split())
        if cur and (n + w > BATCH_WORDS or len(cur) >= MAX_PAGES_PER_CALL):
            out.append(cur)
            cur, n = [], 0
        cur.append((p, t))
        n += w
    return out + ([cur] if cur else [])


def _card_batch(doc: str, title: str, batch: list[tuple[int, str]]) -> tuple[dict[int, dict], float]:
    pages = "\n\n".join(f"=== Page {p} ===\n{t}" for p, t in batch)
    try:
        content, usage = chat(SYSTEM, PROMPT.format(doc=doc, title=title, pages=pages),
                              max_tokens=220 * len(batch) + 300, json_mode=True, kind="pagecards")
        cards = {int(c["page"]): c for c in json.loads(content).get("cards", []) if "page" in c}
        return cards, usage.get("usd", 0.0)
    except (RuntimeError, ValueError, TypeError, KeyError):
        return {}, 0.0


def build_page_cards(units: list[dict], catalog: dict[str, dict], docs: set[str] | None = None,
                     existing: dict | None = None, workers: int = 6) -> tuple[dict, float]:
    """Returns ({(doc, page): {"kind", "card"}}, usd). Pages already in `existing` are skipped;
    a page the LLM fails on falls back to its first words, so no page is ever unreachable."""
    texts = page_texts(units)
    cards = dict(existing or {})
    todo: dict[str, list[tuple[int, str]]] = {}
    for (d, p), t in sorted(texts.items()):
        if (docs is None or d in docs) and (d, p) not in cards:
            todo.setdefault(d, []).append((p, t))
    jobs = [(d, catalog.get(d, {}).get("title", ""), b) for d, ps in todo.items() for b in _pack(ps)]
    usd = 0.0
    with ThreadPoolExecutor(workers) as pool:
        for (d, _, batch), (got, cost) in zip(jobs, pool.map(lambda j: _card_batch(*j), jobs)):
            usd += cost
            for p, t in batch:
                c = got.get(p)
                cards[(d, p)] = ({"kind": c.get("kind", "content"), "card": str(c.get("card", "")),
                                  "questions": [str(x) for x in c.get("questions", [])][:4]} if c
                                 else {"kind": "content", "card": " ".join(t.split()[:40]), "questions": []})
    return cards, usd
