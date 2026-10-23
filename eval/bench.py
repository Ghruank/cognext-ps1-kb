"""Training benchmark loader, filename aliases and the fixed tuning subset."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "Benchmarks" / "benchmark_100.jsonl"
DOCS = ROOT / "data" / "Documents"

# Gold metadata names three documents differently from the corpus files.
ALIASES = {
    "fact-sheet-20260319.pdf": "fact-sheet-SUMMARY.pdf",
    "2026-05960.pdf": "91fr14952.pdf",
    "Guidelines on Management of Problem Loans.pdf": "SAMA_EN_3575_VER1.pdf",
}


def load() -> list[dict]:
    rows = []
    for line in BENCH.open():
        r = json.loads(line)
        chunks = []
        for doc, items in r["benchmark_chunks_w_metadata"].items():
            for c in items:
                chunks.append({"doc": ALIASES.get(doc, doc), "pages": c["metadata"]["page_numbers"],
                               "section": c["section_name"], "text": c["content"]})
        rows.append({"id": r["id"], "family": r["id"][0], "question": r["question"],
                     "answer": r["benchmark_answer"], "gold": chunks})
    return rows


def subset(n_per_family: int = 8) -> list[str]:
    """Deterministic stratified subset: per family, evenly spaced along gold-set size.

    Always includes the zero-gold (unanswerable) question so abstention is exercised.
    """
    rows = load()
    ids = []
    for fam in "ABC":
        fam_rows = sorted((r for r in rows if r["family"] == fam), key=lambda r: (len(r["gold"]), r["id"]))
        step = (len(fam_rows) - 1) / (n_per_family - 1)
        ids += [fam_rows[round(i * step)]["id"] for i in range(n_per_family)]
    ids += [r["id"] for r in rows if not r["gold"] and r["id"] not in ids]
    return ids
