"""Answer writer that fits Groq's 8k tokens/minute per request: map over evidence, then write.

Map: each batch of verified units (~2.5k words) is reduced to the atomic cited facts it
states that bear on the question. Write: STAR's answer rules (engine.writer.SYSTEM) over
the collected facts. When all evidence fits in one call, the map step is skipped.
"""

import json
from concurrent.futures import ThreadPoolExecutor

from engine.writer import ABSTAIN, SYSTEM, format_evidence
from neo.llm import chat

BATCH_WORDS = 1900   # with the 1.2k-token reply this stays under the 8k tokens/minute cap
FACTS_WORDS = 2000   # facts handed to the final call (fits with the rules and a 2.5k-token reply)

MAP_SYSTEM = "You extract facts for a bank analyst. Respond only with JSON."

MAP_PROMPT = """Question: {question}

Excerpts:
{evidence}

List every fact in these excerpts that a complete answer to the question needs: each figure, date, rating, condition, event, rule or explanation, as one short statement copied faithfully (numbers, units and names exactly as written), ending with its citation [filename pN]. Skip facts that do not bear on the question.

Return JSON: {{"facts": [str]}}"""


def _batches(units: list[dict]) -> list[list[dict]]:
    out, cur, n = [], [], 0
    for u in units:
        w = min(len(u["text"].split()), 300)
        if cur and n + w > BATCH_WORDS:
            out.append(cur)
            cur, n = [], 0
        cur.append(u)
        n += w
    return out + ([cur] if cur else [])


def _trim(units: list[dict]) -> list[dict]:
    return [u | {"text": " ".join(u["text"].split()[:300])} for u in units]


def write(question: str, units: list[dict]) -> tuple[str, dict]:
    if not units:
        return ABSTAIN, {}
    batches = _batches(units)
    calls = []
    if len(batches) == 1:
        evidence = format_evidence(_trim(batches[0]))
    else:
        def extract(b: list[dict]) -> tuple[list[str], dict]:
            try:
                content, usage = chat(MAP_SYSTEM, MAP_PROMPT.format(question=question, evidence=format_evidence(_trim(b))),
                                      max_tokens=1200, json_mode=True, kind="neo-write-map")
                return [str(f) for f in json.loads(content).get("facts", [])], usage
            except (RuntimeError, ValueError):
                return [], {}
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(extract, batches))
        calls = [u for _, u in results]
        facts, n = [], 0
        for fs, _ in results:
            for f in fs:
                n += len(f.split())
                if n > FACTS_WORDS:
                    break
                facts.append(f)
        evidence = "\n".join(f"- {f}" for f in facts)
    user = f"Excerpts:\n\n{evidence}\n\nQuestion: {question}\n"
    content, usage = chat(SYSTEM, user, max_tokens=2500, kind="neo-write")
    calls.append(usage)
    return content, {"calls": len(calls), "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in calls),
                     "completion_tokens": sum(c.get("completion_tokens", 0) for c in calls), "usd": 0.0}
