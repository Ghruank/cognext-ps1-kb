"""Layer 2, document catalogue: one typed card per document, built at ingestion.

The card records what embeddings can't separate reliably: which entity, which period
or version, which issuer and jurisdiction. Query routing scores documents against
these fields before any paragraph is scored.
"""

import json
from concurrent.futures import ThreadPoolExecutor

from engine.llm import chat

CARD_SYSTEM = "You catalogue documents for a bank's research library. Respond only with JSON."

CARD_PROMPT = """From the opening pages and outline of this document, fill in its catalogue card.

Filename: {name}
Page count: {n_pages}
Outline (first entries): {toc}

Opening text:
{text}

Return JSON with exactly these keys:
{{"title": str, "issuer": str (organisation that published it), "entities": [str] (companies, banks or regulators it is mainly about, include common short names and abbreviations),
"doc_type": one of ["annual_report", "regulation_or_standard", "proposed_rule", "guidance_or_faq", "supervisory_handbook", "accounting_standard", "board_memo_or_summary", "academic_paper", "technical_document", "other"],
"jurisdiction": str (country/region or "international"), "period": str (fiscal year or reporting period it covers, or ""), "date": str (publication or issue date, or ""),
"version": str (edition, revision or amendment info, or ""), "topics": [str] (5-12 specific topics it covers), "summary": str (2-3 sentences)}}"""


def build_card(d: dict) -> dict:
    toc = "; ".join(t for _, t, _ in d["toc"][:40])[:1500]
    user = CARD_PROMPT.format(name=d["doc"], n_pages=d["n_pages"], toc=toc or "(none)",
                              text=d["first_pages"][:5000])
    content, _ = chat(CARD_SYSTEM, user, max_tokens=900, json_mode=True, kind="catalog")
    card = json.loads(content)
    card["doc"] = d["doc"]
    card["n_pages"] = d["n_pages"]
    return card


def build_catalog(docs: list[dict], workers: int = 8) -> dict[str, dict]:
    with ThreadPoolExecutor(workers) as pool:
        cards = list(pool.map(build_card, docs))
    return {c["doc"]: c for c in cards}


def card_text(c: dict) -> str:
    return (f"{c.get('title', '')}. Issuer: {c.get('issuer', '')}. Entities: {', '.join(c.get('entities', []))}. "
            f"Type: {c.get('doc_type', '')}. Jurisdiction: {c.get('jurisdiction', '')}. Period: {c.get('period', '')}. "
            f"Date: {c.get('date', '')}. {c.get('version', '')}. Topics: {', '.join(c.get('topics', []))}. {c.get('summary', '')}")
