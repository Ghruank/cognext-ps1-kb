"""Query time, chunk elimination: remove candidates that are clearly not relevant, keep the rest.

Selection asks "which chunks are relevant?" and drops anything the model is unsure about,
which costs recall. Elimination asks the opposite question and only removes a chunk on
positive evidence that it is useless, so a borderline chunk survives. Two passes:

  E0 structural (no model): chunk shapes the context judge never accepts on their own,
     e.g. equation fragments, bibliography entries, contents lines, bylines, bare titles.
  E1 LLM elimination: the judge's own criterion, inverted. The model names the chunks
     that are clearly not useful for answering; everything it does not name is kept.

Each rule is calibrated against the judge's verdicts (neo.exp_elim) so it removes
almost no chunk the judge would accept.
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from neo.llm import chat

# ---------- E0: structural ----------

_WORD = re.compile(r"[A-Za-z]{3,}")
_REF = re.compile(r"^\s*(\[\d{1,3}\]|\d{1,3}\.\s+[A-Z][a-z]+,?\s+[A-Z]\.)")
_TOC = re.compile(r"(\.\s*){4,}\s*\d+|\s\.{3,}\s*\d+\s*$")
_FIG_ONLY = re.compile(r"^(figure|fig\.|chart|exhibit|source|note|table)\s*[\d.:]*\s*\S{0,60}$", re.I)
_BOILER = re.compile(r"(all rights reserved|intentionally left blank|this page is left blank|^page \d+( of \d+)?$|"
                     r"forward-looking statements? (are|is) (based|subject))", re.I)


_NUM = re.compile(r"^[(\[]?[-+]?[\d.,%]+[)\]]?$|^[-–—]$")


def _math_ratio(t: str) -> float:
    """Share of tokens that are lone symbols or letters (equation debris); numbers and dashes in tables don't count."""
    toks = t.split()
    return sum(1 for w in toks if not _WORD.search(w) and not _NUM.match(w)) / max(1, len(toks))


def structural_reason(u: dict) -> str | None:
    """Why a unit is structurally useless on its own, or None to keep it."""
    t = u["text"].strip()
    words = t.split()
    if u.get("type") == "table":
        return None
    if len(words) <= 2:
        return "fragment"
    if _BOILER.search(t):
        return "boilerplate"
    if _TOC.search(t) and len(words) < 60:
        return "contents"
    if _REF.match(t) and re.search(r"\b(19|20)\d{2}\b", t):
        return "reference"
    if len(words) >= 8 and _math_ratio(t) > 0.5:
        return "equation"
    if _FIG_ONLY.match(t):
        return "caption"
    return None


# ---------- E1: LLM elimination ----------

BATCH_WORDS = 2800
# deepseek (OpenCode) can spend thousands of tokens on hidden reasoning; Groq's budget counts against 8k tokens/minute
MAX_OUT = 6000 if os.environ.get("NEO_LLM", "groq") == "opencode" else 1500
MAX_UNIT_WORDS = 300

SYSTEM = "You are a strict evaluator judging whether text chunks are relevant to a user's query. Respond only with JSON."

PROMPTS = {
    "strict": """User query:
{question}
{facets}
Chunks:
{chunks}

A chunk is relevant if it is genuinely relevant and useful for answering the query. Identify the chunks that are clearly NOT relevant: about a different entity, period, version or topic; generic or introductory text with no fact, figure, rule or explanation the answer would use; or text that only shares keywords with the query. If a chunk could reasonably support any part of the answer, do not list it.

Return JSON: {{"irrelevant_ids": [ids]}}""",
    # broad, multi-part questions: the judge accepts background and supporting text, so only drop the unrelated
    "lenient": """User query:
{question}
{facets}
Chunks:
{chunks}

Another evaluator will later decide, for each chunk, whether it is genuinely relevant and useful for answering the query. It accepts chunks that support any part of the query, including definitions, background, conditions, procedures and examples on the query's topics. Your job is only to remove chunks it would certainly reject: text about a different entity, period, version or subject; boilerplate, navigation, legal or administrative filler; or text that merely shares a keyword. When in doubt, keep the chunk.

Return JSON: {{"irrelevant_ids": [ids of chunks to remove]}}""",
}


def _call(question: str, facets: list[str], items: list[tuple[int, dict]], kind: str, prompt: str,
          depth: int = 0) -> tuple[set[int] | None, dict]:
    chunks = "\n\n".join(f"[{k}] " + " ".join(u["text"].split()[:MAX_UNIT_WORDS]) for k, u in items)
    fac = ("Parts of the query: " + "; ".join(facets) + "\n") if facets else ""
    try:
        content, usage = chat(SYSTEM, PROMPTS[prompt].format(question=question, facets=fac, chunks=chunks),
                              max_tokens=MAX_OUT, json_mode=True, kind=kind)
        valid = {k for k, _ in items}
        return {int(k) for k in json.loads(content).get("irrelevant_ids", []) if str(k).isdigit() and int(k) in valid}, usage
    except (RuntimeError, ValueError, TypeError):
        if depth or len(items) < 4:
            return None, {}   # a failed batch eliminates nothing
        # long batches are where hidden reasoning runs away; halves usually succeed
        a, ua = _call(question, facets, items[:len(items) // 2], kind, prompt, 1)
        b, ub = _call(question, facets, items[len(items) // 2:], kind, prompt, 1)
        usage = {"usd": ua.get("usd", 0) + ub.get("usd", 0)}
        return (None if a is None and b is None else (a or set()) | (b or set())), usage


def _batches(items: list[tuple[int, dict]]) -> list[list[tuple[int, dict]]]:
    out, cur, n = [], [], 0
    for k, u in items:
        w = min(len(u["text"].split()), MAX_UNIT_WORDS) + 6
        if cur and n + w > BATCH_WORDS:
            out.append(cur)
            cur, n = [], 0
        cur.append((k, u))
        n += w
    return out + ([cur] if cur else [])


def _llm_pass(question, facets, units, keep, kind, prompt, workers, order) -> tuple[set[int], dict]:
    batches = _batches([(i, units[i]) for i in order(keep)])
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda b: _call(question, facets, b, kind, prompt), batches))
    drop = set().union(*(r for r, _ in results if r))
    return drop, {"usd": sum(u.get("usd", 0) for _, u in results), "calls": len(batches),
                  "failed": sum(r is None for r, _ in results)}


def eliminate(question: str, facets: list[str], units: list[dict], *, structural: bool = True,
              llm: bool = True, prompt: str = "lenient", votes: int = 1, workers: int = 6,
              kind: str = "neo-eliminate") -> tuple[list[int], dict]:
    """Returns (indices into `units` that survive, info with per-pass removals and usage).

    votes=2 runs a second pass with the chunks interleaved into different batches and
    removes a chunk only if both passes name it.
    """
    keep = list(range(len(units)))
    info = {"structural": {}, "llm_removed": 0, "usd": 0.0, "calls": 0, "failed_batches": 0}
    if structural:
        survivors = []
        for i in keep:
            r = structural_reason(units[i])
            if r:
                info["structural"][r] = info["structural"].get(r, 0) + 1
            else:
                survivors.append(i)
        keep = survivors
    if llm and keep:
        orders = [lambda k: k, lambda k: k[::2] + k[1::2]][:votes]
        drop = None
        for order in orders:
            d, st = _llm_pass(question, facets, units, keep, kind, prompt, workers, order)
            drop = d if drop is None else drop & d
            info["usd"] += st["usd"]
            info["calls"] += st["calls"]
            info["failed_batches"] += st["failed"]
        info["llm_removed"] = len(drop)
        keep = [i for i in keep if i not in drop]
    return keep, info
