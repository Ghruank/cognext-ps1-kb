"""Layout-block ingestion, no LLM: one unit per PyMuPDF text block, one per table.

Measured against the benchmark's own chunks on sampled gold pages (neo.exp_parsers):
raw layout blocks match the gold chunk boundaries far better than merged paragraphs
(STAR's engine.ingest) or Docling's chunkers, and parse at ~0.01 s/page vs ~4 s/page for
Docling on CPU. Tables are detected on every page and emitted as one markdown unit
prefixed with "Table: <caption>", the format the benchmark uses; text blocks inside a
table's area are dropped so a table is never counted twice.

Each unit keeps its page, its block position on the page and the heading path in force,
which the hierarchy layer (neo.tree) builds on.
"""

import re
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import fitz

BAND = 0.08          # top/bottom fraction of the page treated as header/footer territory
MAX_WORDS = 400      # longer blocks are split on sentence boundaries


def _clean(t: str) -> str:
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)
    return re.sub(r"\s+", " ", t).strip()


def _norm(s: str) -> str:
    return re.sub(r"\d+", "#", s).lower()


def _split(text: str) -> list[str]:
    w = text.split()
    if len(w) <= MAX_WORDS:
        return [text]
    sents = re.split(r"(?<=[.;:])\s+(?=[A-Z(\[\d])", text)
    out, cur = [], []
    for s in sents:
        if cur and len(" ".join(cur + [s]).split()) > MAX_WORDS // 2:
            out.append(" ".join(cur))
            cur = []
        cur.append(s)
    return out + ([" ".join(cur)] if cur else [])


def _tables(page: "fitz.Page", blocks: list[tuple]) -> list[tuple[fitz.Rect, str]]:
    try:
        found = page.find_tables(use_layout=False).tables  # the ONNX layout path segfaults in pymupdf 2.0
    except Exception:
        return []
    out = []
    for t in found:
        md = t.to_markdown().strip()
        if md.count("\n") < 2 or len(md.split()) < 6:
            continue
        r = fitz.Rect(t.bbox)
        # caption: the nearest short text block ending just above the table
        above = [b for b in blocks if b[3] <= r.y0 + 2 and r.y0 - b[3] < 40 and len(b[4].split()) <= 40
                 and not fitz.Rect(b[:4]).intersects(r)]
        cap = _clean(max(above, key=lambda b: b[3])[4]) if above else ""
        out.append((r, f"Table: {cap}\n{md}" if cap else md))
    return out


def _ocr(page: "fitz.Page") -> list[str]:
    import io

    import pytesseract
    from PIL import Image
    text = pytesseract.image_to_string(Image.open(io.BytesIO(page.get_pixmap(dpi=200).tobytes("png"))))
    return [_clean(p) for p in re.split(r"\n\s*\n", text) if p.strip()]


def parse_pdf(path: str) -> dict:
    doc = fitz.open(path)
    name = Path(path).name
    raw = []
    for page in doc:
        h = page.rect.height or 1
        blocks = [b for b in page.get_text("blocks", sort=True) if b[6] == 0 and b[4].strip()]
        raw.append((page, h, blocks))

    edge = Counter()
    for _, h, blocks in raw:
        edge.update({_norm(_clean(b[4])) for b in blocks if b[3] / h < BAND or b[1] / h > 1 - BAND})
    boiler = {t for t, n in edge.items() if n >= max(3, int(0.25 * len(raw))) and len(t) < 200}

    # heading detection from font size, using span-level info only where it is cheap
    sizes = []
    for page, _, _ in raw[:: max(1, len(raw) // 40)]:
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for s in line["spans"]:
                    if len(s["text"].split()) > 3:
                        sizes.append(round(s["size"], 1))
    body = statistics.median(sizes) if sizes else 10.0

    toc = doc.get_toc()
    toc_by_page: dict[int, list] = {}
    for lvl, title, pno in toc:
        toc_by_page.setdefault(pno, []).append((lvl, title.strip()))
    path_: list[str] = []
    units, ocr_pages = [], 0
    for pno, (page, h, blocks) in enumerate(raw, 1):
        for lvl, title in toc_by_page.get(pno, []):
            path_[lvl - 1:] = [title]
        if sum(len(b[4]) for b in blocks) < 30 and page.get_images():
            texts = _ocr(page)
            ocr_pages += 1
            for k, t in enumerate(texts):
                units.append({"type": "prose", "pages": [pno], "block": k, "text": t, "section": " > ".join(path_[-3:])})
            continue
        tables = _tables(page, blocks)
        items = []
        for r, md in tables:
            items.append((r.y0, r.x0, "table", md))
        for b in blocks:
            t = _clean(b[4])
            if not t or _norm(t) in boiler:
                continue
            if any(fitz.Rect(b[:4]).intersects(r) for r, _ in tables):
                continue
            if tables and any(t in md for _, md in tables) and len(t.split()) <= 40:
                continue  # the table caption, already folded into its table
            items.append((b[1], b[0], "prose", t))
        items.sort(key=lambda x: (round(x[0] / 4), x[1]))
        heading_sizes = {}
        if items:
            for b in page.get_text("dict")["blocks"]:
                spans = [s for line in b.get("lines", []) for s in line["spans"] if s["text"].strip()]
                if spans:
                    heading_sizes[round(b["bbox"][1])] = (max(s["size"] for s in spans),
                                                          all(s["flags"] & 16 for s in spans))
        for k, (y, _, kind, t) in enumerate(items):
            size, bold = heading_sizes.get(round(y), (body, False))
            is_heading = kind == "prose" and len(t.split()) <= 14 and (size > body * 1.15 or bold) and not t.endswith(".")
            if is_heading and not toc:
                path_[:] = (path_[:1] if size > body * 1.4 else path_[:2])[:2] + [t]
            for part in (_split(t) if kind == "prose" else [t]):
                units.append({"type": "heading" if is_heading else kind, "pages": [pno], "block": k,
                              "text": part, "section": " > ".join(path_[-3:])})

    meta = doc.metadata or {}
    first = " ".join(" ".join(_clean(b[4]) for b in blocks) for _, _, blocks in raw[:3])[:6000]
    return {"doc": name, "n_pages": len(doc), "ocr_pages": ocr_pages, "units": units,
            "pdf_meta": {k: meta.get(k) for k in ("title", "author", "subject", "creationDate")},
            "first_pages": first, "toc": [(lvl, t, p) for lvl, t, p in toc[:200]]}


def parse_folder(folder: str, workers: int = 10) -> list[dict]:
    paths = sorted(str(p) for p in Path(folder).rglob("*") if p.suffix.lower() == ".pdf")
    paths.sort(key=lambda p: -Path(p).stat().st_size)
    with ProcessPoolExecutor(workers) as pool:
        docs = list(pool.map(parse_pdf, paths))
    for d in docs:
        for i, u in enumerate(d["units"]):
            u["id"] = f"{d['doc']}#{i}"
            u["doc"] = d["doc"]
    repair_ligatures(docs)
    return docs


LIGATURES = ("fi", "fl", "ff", "ffi", "ffl")


def repair_ligatures(docs: list[dict]) -> int:
    """Some PDFs map the fi/fl/ff ligature glyphs to nothing ("nancial") or to a bare "f"
    ("proft", "fnancial"). A rare token is replaced when re-inserting a
    ligature at one position gives a word the corpus uses far more often."""
    counts = Counter(w for d in docs for u in d["units"] for w in re.findall(r"[a-z]{3,}", u["text"].lower()))
    fix: dict[str, str] = {}
    for w, n in counts.items():
        best, best_n = None, max(5, 5 * n)
        cands = [w[:i] + lig + w[i:] for i in range(len(w) + 1) for lig in LIGATURES]
        # other PDFs keep the "f" and drop only the ligature's second letter: "proft", "fnancial"
        cands += [w[:i + 1] + c + w[i + 1:] for i, ch in enumerate(w) if ch == "f" for c in "il"]
        for cand in cands:
            if counts.get(cand, 0) > best_n:
                best, best_n = cand, counts[cand]
        if best:
            fix[w] = best

    def sub(m: re.Match) -> str:
        w = m.group(0)
        r = fix.get(w.lower())
        if r is None:
            return w
        return r.capitalize() if w[0].isupper() else r

    n_fixed = 0
    for d in docs:
        for u in d["units"]:
            new = re.sub(r"[A-Za-z]{3,}", sub, u["text"])
            n_fixed += new != u["text"]
            u["text"] = new
    return n_fixed
