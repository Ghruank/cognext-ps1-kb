"""Which LLM-free parser lines up best with the benchmark's chunks? (no LLM calls)

For sampled gold pages, every gold chunk is matched to the best chunk a parser produced
on the same page (word 4-gram Jaccard). High Jaccard means the parser cuts where the
benchmark cuts, so judged-relevant chunks count one-for-one against the gold count.

    python -m neo.exp_parsers
"""

import pickle
import random
import re
import statistics as st
import tempfile
import time
from pathlib import Path

import fitz

from eval import bench

DOCS = Path("data/Documents")
SAMPLE_DOCS = ["CEATIntegratedAnnualReport2526Re.pdf", "Metal Coatings_Annual Report FY 2026.pdf",
               "OCC ACL Handbook.pdf", "SAMA_EN_3575_VER1.pdf", "BCBS-Basel-Endgame.pdf"]
PAGES_PER_DOC = 8


def sh(t: str) -> set:
    w = re.findall(r"[a-z0-9]+", t.lower())
    return {" ".join(w[i:i + 4]) for i in range(max(1, len(w) - 3))}


def jac(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def sample_pages() -> dict[str, list[int]]:
    gold_pages: dict[str, set] = {}
    for q in bench.load():
        for c in q["gold"]:
            if c["doc"] in SAMPLE_DOCS:
                gold_pages.setdefault(c["doc"], set()).update(c["pages"])
    rng = random.Random(0)
    return {d: sorted(rng.sample(sorted(ps), min(PAGES_PER_DOC, len(ps)))) for d, ps in gold_pages.items()}


def gold_chunks(pages: dict[str, list[int]]) -> dict[tuple, list[str]]:
    out: dict[tuple, set] = {}
    for q in bench.load():
        for c in q["gold"]:
            for p in c["pages"]:
                if p in pages.get(c["doc"], []):
                    out.setdefault((c["doc"], p), set()).add(c["text"])
    return {k: list(v) for k, v in out.items()}


def pymupdf_chunks(pages) -> tuple[dict, float]:
    docs = {d["doc"]: d for d in pickle.load(open("storage/train_docs.pkl", "rb"))}
    out = {}
    for d, ps in pages.items():
        for u in docs[d]["units"]:
            for p in u["pages"]:
                if p in ps:
                    out.setdefault((d, p), []).append(u["text"])
    return out, float("nan")


def docling_chunks(pages, chunker: str) -> tuple[dict, float]:
    from docling.chunking import HierarchicalChunker, HybridChunker
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    ch = HierarchicalChunker() if chunker == "hierarchical" else HybridChunker(max_tokens=256, merge_peers=True)
    out, n_pages, secs = {}, 0, 0.0
    for d, ps in pages.items():
        src = fitz.open(DOCS / d)
        mini = fitz.open()
        for p in ps:
            mini.insert_pdf(src, from_page=p - 1, to_page=p - 1)
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            mini.save(f.name)
            t0 = time.perf_counter()
            doc = conv.convert(f.name).document
            secs += time.perf_counter() - t0
        n_pages += len(ps)
        for c in ch.chunk(doc):
            mini_pages = {pr.page_no for it in c.meta.doc_items for pr in it.prov}
            for mp in mini_pages:
                out.setdefault((d, ps[mp - 1]), []).append(c.text)
    return out, secs / max(1, n_pages)


def score(name: str, chunks: dict, gold: dict, spp: float) -> None:
    best, per_page = [], []
    for k, gs in gold.items():
        cand = [sh(t) for t in chunks.get(k, [])]
        per_page.append(len(cand))
        for g in gs:
            gsh = sh(g)
            best.append(max((jac(gsh, c) for c in cand), default=0.0))
    print(f"{name:28} gold-chunk match: mean J={st.mean(best):.3f}  J>=0.5: {sum(b >= 0.5 for b in best) / len(best):.2f}  "
          f"J>=0.8: {sum(b >= 0.8 for b in best) / len(best):.2f}  chunks/page={st.mean(per_page):.1f}  s/page={spp:.2f}", flush=True)


def main() -> None:
    pages = sample_pages()
    gold = gold_chunks(pages)
    print(f"{sum(len(p) for p in pages.values())} pages, {sum(len(v) for v in gold.values())} gold chunks")
    score("pymupdf units (STAR)", *pymupdf_chunks(pages)[:1], gold, float("nan"))
    for c in ("hierarchical", "hybrid"):
        ch, spp = docling_chunks(pages, c)
        score(f"docling {c}", ch, gold, spp)


if __name__ == "__main__":
    main()
