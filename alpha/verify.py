"""Query time, stage 3: unit verification on the navigated pages.

Every unit on a selected page is shown to the LLM with the same criterion the context
judge applies ("genuinely relevant and useful for answering the query"), so what we
return is what the judge will accept. Units are shown in reading order with their page,
so a short line is judged in the context of the paragraph around it.
"""

import json
from concurrent.futures import ThreadPoolExecutor

from alpha.llm import chat

BATCH_WORDS = 2800   # input per call, under Groq's 8k tokens/minute
MAX_UNIT_WORDS = 100  # same truncation as the Claude verdicts the coherence model was trained on

SYSTEM = "You are a strict evaluator judging whether text chunks are relevant to a user's query. Respond only with JSON."

PROMPT = """User query:
{question}
{facets}
Chunks:
{chunks}

For each chunk, decide if it is genuinely relevant and useful for answering the query: it states a fact, figure, rule, condition, event or explanation that belongs in a complete answer. Chunks about a different entity, year, version or topic than the query asks about are not relevant, even if they look similar.

Return JSON: {{"relevant_ids": [ids of relevant chunks]}}"""


def _call(question: str, facets: list[str], items: list[tuple[int, dict]]) -> tuple[list[int], dict]:
    chunks = "\n\n".join(f"[{k}] ({u['doc']} p{u['pages'][0]}) " + " ".join(u["text"].split()[:MAX_UNIT_WORDS])
                         for k, u in items)
    fac = ("Parts of the query: " + "; ".join(facets) + "\n") if facets else ""
    try:
        content, usage = chat(SYSTEM, PROMPT.format(question=question, facets=fac, chunks=chunks),
                              max_tokens=1000, json_mode=True, kind="neo-verify")
        valid = {k for k, _ in items}
        return [int(k) for k in json.loads(content).get("relevant_ids", []) if str(k).isdigit() and int(k) in valid], usage
    except (RuntimeError, ValueError, TypeError) as e:
        print(f"verify call failed: {e!r}"[:300], flush=True)
        return [], {}


def verify(question: str, facets: list[str], units: list[dict], positions: list[int],
           workers: int = 6) -> tuple[list[int], dict]:
    """positions index into `units`; returns the subset judged relevant (reading order) and usage."""
    items = [(k, units[k]) for k in positions]
    batches, cur, n = [], [], 0
    for k, u in items:
        w = min(len(u["text"].split()), MAX_UNIT_WORDS) + 8
        if cur and n + w > BATCH_WORDS:
            batches.append(cur)
            cur, n = [], 0
        cur.append((k, u))
        n += w
    if cur:
        batches.append(cur)
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda b: _call(question, facets, b), batches))
    if batches and not any(u for _, u in results):
        raise RuntimeError(f"all {len(batches)} verification calls failed")
    chosen = sorted({k for ks, _ in results for k in ks})
    usage = {"prompt_tokens": sum(u.get("prompt_tokens", 0) for _, u in results),
             "completion_tokens": sum(u.get("completion_tokens", 0) for _, u in results),
             "usd": sum(u.get("usd", 0) for _, u in results), "calls": len(batches)}
    return chosen, usage
