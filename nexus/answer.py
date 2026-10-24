"""Single-call answer framing over deterministic, verified evidence."""

import os
import re

from neo.llm import chat

ABSTAIN = "No information was retrieved for this question."
MAX_ANSWER_WORDS = 30000


def _bounded_context(evidence: list[dict], word_budget: int = MAX_ANSWER_WORDS) -> list[str]:
    """Fit the evidence into one request while giving every facet an early slot."""
    ordered: list[dict] = []
    facets = sorted({i for e in evidence for i in e.get("facets", [])})
    used: set[str] = set()
    by_facet = {}
    for facet in facets:
        by_facet[facet] = sorted((e for e in evidence if facet in e.get("facets", [])),
                                 key=lambda e: (-float(e.get("score", 0)), e["doc"], e["pages"][0]))
    # Round-robin the best ranked evidence from each facet before filling remaining space.
    for rank in range(max((len(rows) for rows in by_facet.values()), default=0)):
        for facet in facets:
            matches = by_facet[facet]
            if len(matches) > rank:
                chosen = matches[rank]
                if chosen["id"] not in used:
                    ordered.append(chosen)
                    used.add(chosen["id"])
    ordered.extend(e for e in sorted(evidence, key=lambda e: -float(e.get("score", 0))) if e["id"] not in used)

    passages, consumed = [], 0
    for e in ordered:
        if consumed >= word_budget:
            break
        words = " ".join(e["text"].split()).split()
        words = words[:min(220, word_budget - consumed)]
        if not words:
            continue
        consumed += len(words)
        passages.append(f"[{e['doc']} p{e['pages'][0]}] ({e.get('section', '')})\n{' '.join(words)}")
    return passages


def write_answer(question: str, evidence: list[dict]) -> tuple[str, dict]:
    """Make one final LLM call; no map/extract/reduce calls happen before it."""
    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    if not evidence:
        return ABSTAIN, usage
    passages = _bounded_context(evidence)
    system = "You write concise, well-framed answers grounded only in supplied source evidence."
    prompt = (f"Question: {question}\n\nRetrieved evidence:\n" + "\n\n".join(passages) +
              "\n\nAnswer the question directly and cover every requested part. Preserve source values, units, dates, "
              "and conditions. Do not add facts from memory or infer unsupported causes. Cite every factual claim "
              "with the supplied [filename pN] marker. If evidence is insufficient, say what is missing.")
    provider = os.getenv("NEXUS_ANSWER_PROVIDER", "gemini")
    text, call_usage = chat(system, prompt, max_tokens=3000, kind="nexus-answer", provider=provider)
    usage["calls"] = 1
    usage["prompt_tokens"] = call_usage.get("prompt_tokens", 0)
    usage["completion_tokens"] = call_usage.get("completion_tokens", 0)
    return text.strip(), usage


def write_extractive_answer(question: str, evidence: list[dict], facets: list[dict] | None = None) -> tuple[str, dict]:
    """Create a cited answer digest using only source text and deterministic ranking."""
    from nexus.index import tokens

    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    if not evidence:
        return ABSTAIN, usage
    obligations = facets or [{"name": "answer", "terms": tokens(question)}]
    lines: list[str] = []
    seen: set[str] = set()
    for facet in obligations:
        terms = set(tokens(" ".join(facet.get("terms", []))))
        sentences = []
        for ev in evidence:
            source = f"[{ev['doc']} p{ev['pages'][0]}]"
            pieces = re.split(r"(?<=[.!?])\s+|\n+", ev["text"])
            for piece in pieces:
                piece = " ".join(piece.split())
                if len(piece) < 24:
                    continue
                overlap = len(terms & set(tokens(piece)))
                if overlap:
                    sentences.append((overlap, len(piece), source, piece))
        sentences.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        added = 0
        for _, _, source, piece in sentences:
            key = re.sub(r"\W+", " ", piece.lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {source} {piece}")
            added += 1
            if added >= 3:
                break
    if not lines:
        for ev in evidence[:min(3, len(evidence))]:
            excerpt = " ".join(ev["text"].split())[:500]
            if excerpt:
                lines.append(f"- [{ev['doc']} p{ev['pages'][0]}] {excerpt}")
    return "\n".join(lines) if lines else ABSTAIN, usage
