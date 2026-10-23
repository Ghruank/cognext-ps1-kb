"""Free proxy labels: which of our units carry a gold chunk's text.

A unit is gold-relevant for a question when most of its word 4-grams appear in some
gold chunk of the same document (gold pages +/-1). No LLM calls; used for training
labels, the offline proxy metric, and the homophily analysis.
"""

import re

from eval import bench


def _shingles(text: str, n: int = 4) -> set:
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def label_units(docs: list[dict], threshold: float = 0.5) -> dict[str, set[str]]:
    """Returns {question_id: set(unit_id)} of units whose text is covered by gold."""
    by_doc = {d["doc"]: d["units"] for d in docs}
    labels = {}
    for q in bench.load():
        gold_by_doc: dict[str, list] = {}
        for c in q["gold"]:
            gold_by_doc.setdefault(c["doc"], []).append(c)
        pos = set()
        for doc, chunks in gold_by_doc.items():
            pages = {p + d for c in chunks for p in c["pages"] for d in (-1, 0, 1)}
            gold_sh = set().union(*(_shingles(c["text"]) for c in chunks))
            for u in by_doc.get(doc, []):
                if not pages.intersection(u["pages"]):
                    continue
                sh = _shingles(u["text"])
                if sh and len(sh & gold_sh) / len(sh) >= threshold:
                    pos.add(u["id"])
        labels[q["id"]] = pos
    return labels
