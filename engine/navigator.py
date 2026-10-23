"""Architecture 1, LLM-Navigator: the LLM reads shortlisted pages and picks the evidence.

Follows the Enterprise RAG Challenge winner (IlyaRice/RAG-Challenge-2, MIT): route to the
right documents, shortlist pages by retrieval score, then let the LLM judge the text.
Adapted here to return paragraph IDs rather than whole pages, because the judge scores
paragraph-sized chunks. `n_pages` is the cost dial.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from engine.llm import chat

SYSTEM = "You select evidence for a bank analyst. Respond only with JSON."

PROMPT = """Question: {question}

Below are numbered paragraphs from {n_pages} pages of the bank's documents. Select every paragraph that is genuinely relevant and useful for answering the question: it states a fact, figure, rule, event or explanation the answer needs. Exclude paragraphs that are only loosely on-topic, boilerplate, tables of contents, or about a different entity, year or version than the question asks about.

{paragraphs}

Return JSON: {{"relevant_ids": [ids of relevant paragraphs]}}"""


def shortlist_pages(units: list[dict], idx: np.ndarray, score: np.ndarray, n_pages: int) -> list[tuple]:
    """Top pages by their best paragraph score within the candidate pool."""
    best: dict[tuple, float] = {}
    for k, i in enumerate(idx):
        u = units[i]
        key = (u["doc"], u["pages"][0])
        best[key] = max(best.get(key, -1e9), float(score[k]))
    return sorted(best, key=lambda p: -best[p])[:n_pages]


def navigate(question: str, units: list[dict], idx: np.ndarray, score: np.ndarray,
             n_pages: int = 30, pages_per_call: int = 6, workers: int = 8) -> tuple[list[int], dict]:
    """Returns (pool positions the LLM selected, usage totals)."""
    pages = shortlist_pages(units, idx, score, n_pages)
    by_page: dict[tuple, list[int]] = {}
    for k, i in enumerate(idx):
        u = units[i]
        key = (u["doc"], u["pages"][0])
        if key in pages:
            by_page.setdefault(key, []).append(k)
    batches = [pages[s:s + pages_per_call] for s in range(0, len(pages), pages_per_call)]

    def run(batch: list[tuple]) -> tuple[list[int], dict]:
        ks = [k for p in batch for k in by_page.get(p, [])]
        if not ks:
            return [], {}
        paras = "\n\n".join(f"[{k}] ({units[idx[k]]['doc']} p{units[idx[k]]['pages'][0]}) "
                            f"{units[idx[k]]['text'][:1500]}" for k in ks)
        content, usage = chat(SYSTEM, PROMPT.format(question=question, n_pages=len(batch), paragraphs=paras),
                              max_tokens=800, json_mode=True, kind="navigator")
        chosen = [int(x) for x in json.loads(content).get("relevant_ids", []) if int(x) in set(ks)]
        return chosen, usage

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(run, batches))
    selected = sorted({k for ks, _ in results for k in ks})
    usage = {"prompt_tokens": sum(u.get("prompt_tokens", 0) for _, u in results),
             "completion_tokens": sum(u.get("completion_tokens", 0) for _, u in results),
             "usd": sum(u.get("usd", 0) for _, u in results), "calls": len(batches)}
    return selected, usage
