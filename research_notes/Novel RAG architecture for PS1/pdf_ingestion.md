# Fast, robust ingestion of messy financial/regulatory PDFs into paragraph units (PS1)

Scope: parser/OCR/structure tool comparison, doc typing + metadata extraction, CPU embedding throughput, recommended pipeline + time estimate for ~8,600 pages / 35 PDFs (1 scanned 92-page PDF). Research date 2026-10-23. Note: the search tool was rate-limited for part of this session, so several numbers come from a small set of primary sources; gaps are flagged explicitly.

## Q1. Parser comparison: speed, tables/multi-column quality, page numbers, license

### Takeaway
Only the MuPDF/PDFium/Poppler-class extractors (PyMuPDF/pymupdf4llm, pypdfium2, pdftotext) are fast enough on CPU to ingest 8,600 pages in minutes; ML layout pipelines (Docling, MinerU, Marker-with-models, Unstructured hi_res) run at roughly 0.06-0.8 pages/s on CPU, i.e. 3-40 hours for this corpus, so they are only viable as targeted fallbacks (e.g. table-heavy pages) or with a GPU. Licensing is the main catch for PyMuPDF (AGPL) and Marker/MinerU model weights.

### Cited Findings
**Raw text-extraction speed and quality (py-pdf/benchmarks, 14 test PDFs, versions current as of Oct 2026)**
- Average extraction time per test file: PyMuPDF 0.1 s, pypdfium2 0.1 s, Tika 0.2 s, pdftotext 0.3 s, pypdf 3.2 s, pdfminer.six 5.3 s, pdfplumber 8.6 s; on the largest file PyMuPDF took 0.4 s vs pypdf 21.6 s vs pdfplumber 56.5 s — [py-pdf/benchmarks](https://github.com/py-pdf/benchmarks)
- Average text-extraction quality score: pypdfium2 95%, pypdf 95%, PyMuPDF 94%, Tika 93%, pdftotext 90%, pdfminer.six 88%, pdfplumber 74% — [py-pdf/benchmarks](https://github.com/py-pdf/benchmarks)
- Licenses / versions listed: PyMuPDF 1.28.2 = GNU AGPL 3.0 or commercial; pypdf 6.21.1 = BSD-3; pypdfium2 5.14.0 = Apache-2.0 or BSD-3; pdfplumber 0.11.10 = MIT; pdfminer.six = MIT; pdftotext (Python binding, poppler) = GPL — [py-pdf/benchmarks](https://github.com/py-pdf/benchmarks)

**ML layout pipelines on CPU (Docling technical report, arXiv 2501.17887)**
- Seconds/page on x86 CPU (AWS g6.xlarge, 8 vCPU AMD EPYC 7R13, GPU hidden, 8 threads): Docling 2.5.2 = 3.1 s; MinerU 0.9.3 = 3.3 s; Marker 0.3.10 = >16 s; Unstructured 0.16.5 = 4.2 s. On M3 Max: Docling 1.27, Marker 4.2, Unstructured 2.7, MinerU did not complete. On Nvidia L4 GPU: Docling 0.49, MinerU 0.21, Marker 0.86; Unstructured gets no GPU benefit — [Docling Technical Report v2](https://arxiv.org/html/2501.17887)
- Earlier Docling report (no OCR, 225 pages): on 16-core Intel Xeon E5-2690 with 16 threads, 0.64 s/page with the pypdfium backend vs 1.08 s/page with native docling-parse; peak RSS ~2.4 GB (pypdfium) vs ~6.2 GB (docling-parse). EasyOCR in Docling "runs fairly slow on CPU (upwards of 30 seconds per page)" — [Docling Technical Report v1](https://arxiv.org/html/2408.09869)

**Marker (self-reported, current README)**
- Throughput on one B200 host at sustained concurrency: "Fast, no OCR (CPU)" 23.7 pages/s; "Fast (GPU-assisted)" 7.4 pages/s; "Balanced (GPU)" 2.9 pages/s (≈0.3 pages/s single-stream) — [datalab-to/marker](https://github.com/datalab-to/marker)
- olmOCR-bench overall (self-reported by Datalab): Marker balanced 76.0; MinerU pipeline (GPU) 72.7 at 0.54 pg/s; Marker fast 66.6; Docling (GPU) 50.3 at 2.1 pg/s; Marker fast no-OCR CPU 43.6 — [datalab-to/marker](https://github.com/datalab-to/marker)
- License: code Apache-2.0; model weights under a modified AI Pubs Open Rail-M license, free for research/personal use and startups under $5M funding/revenue — [datalab-to/marker](https://github.com/datalab-to/marker)

**OmniDocBench (v1.6_full leaderboard on the official repo; repo notes a move to v1.7 on 2026-04-30)**
- Pipeline tools: MinerU-Pipeline overall 86.47, text edit distance 0.055, table TEDS 81.88; Marker overall 78.44, text edit 0.157, table TEDS 65.77. Docling, PP-StructureV3 and pymupdf4llm have no end-to-end scores in that table — [OmniDocBench](https://github.com/opendatalab/OmniDocBench)
- VLM parsers dominate the top: TeleOCR 96.91, PaddleOCR-VL-1.6 96.34, MinerU2.5-Pro 95.75, Gemini 3 Flash 92.62 (TEDS 89.29), GPT-5.2 86.59, olmOCR 85.74, Mistral OCR 85.66 — [OmniDocBench](https://github.com/opendatalab/OmniDocBench)
- Original paper notes MinerU and Marker lose accuracy on mixed Chinese/English and Marker loses points to missed detections — [OmniDocBench CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Ouyang_OmniDocBench_Benchmarking_Diverse_CVPR_2025_paper.pdf)

**MinerU (current README)**
- License is now the "MinerU Open Source License", "based on Apache 2.0 with additional conditions" (check LICENSE.md before relying on it). Quality tiers flash/basic/standard/advanced; `basic` runs on CPU via ONNX (~0.8 GB models, 2 GB RAM); standard/advanced need ONNX + llama.cpp VLM (8 GB RAM, Vulkan recommended); PyTorch engines need GPU/MPS. No speed numbers published — [opendatalab/MinerU](https://github.com/opendatalab/MinerU)

**pymupdf4llm**
- `to_markdown()` header detection: default builds `IdentifyHeaders` (font-size scan, `body_limit` default 11, up to 6 levels), or pass `TocHeaders(doc)` to take header levels from `Document.get_toc()` (bookmarks) without scanning page content. `page_chunks=True` returns per-page dicts including `page_boxes` with classes text, picture, table, caption, title, section-header, page-header, page-footer, list-item, footnote, formula; there is a `use_layout()` switch for the GPU-free layout analysis mode — [pymupdf4llm API docs](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html); [pymupdf4llm overview](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)

**LlamaParse (hosted)**
- v2 tiers per page: Fast 1 credit, Cost-effective 3, Agentic 10, Agentic Plus 45; credits $1.25 per 1,000 (NA/EU); layout extraction +3 credits/page — [LlamaParse pricing](https://developers.llamaindex.ai/llamaparse/general/pricing/). An older pricing page lists Agentic Plus at 90 credits and $1.00/1k credits (conflict; likely outdated) — [LlamaIndex pricing (older)](https://developers.llamaindex.ai/python/cloud/pricing)

### Inferences
- Cost of LlamaParse for 8,600 pages: Cost-effective ≈ 8,600 × 3 × $1.25/1000 ≈ $32; Agentic ≈ $108. Plus network latency and upload of large PDFs inside a timed /index call; and the hidden test corpus would be sent to a third party. Not recommended as the primary path; possibly acceptable offline for tables.
- At Docling's 3.1 s/page (8 vCPU), 8,600 pages ≈ 7.4 h; even at the best CPU figure (0.64 s/page, 16 threads, no OCR, no table model tuning) ≈ 1.5 h. MinerU CPU ≈ 3.3 s/page → ≈ 7.9 h. Marker with models on CPU >16 s/page → >38 h. All exceed a 20-30 min ingestion budget on a laptop. Marker's "fast no-OCR CPU" mode (23.7 pg/s) would fit, but its score (43.6 olmOCR-bench) shows it is essentially text-layer extraction and the figure was measured on a very large B200 host, not a laptop.
- PyMuPDF's per-file averages (0.1 s on the benchmark docs) imply hundreds of pages/s for plain `get_text`; pymupdf4llm adds table finding and header analysis and is substantially slower than raw `get_text` (not quantified in sources found). With `multiprocessing` over pages/documents, 8,600 pages should be a few minutes on a 4-8 core machine (estimate, measure locally).
- pdfplumber is the best-known tool for table geometry but is ~80x slower than PyMuPDF and scored lowest on text quality; use it (or PyMuPDF `page.find_tables()`) only on pages flagged as table-heavy.
- Federal Register 3-column pages: PyMuPDF `get_text("blocks")`/`sort=True` returns blocks in reading order by default for most born-digital multi-column layouts; pdftotext without `-layout` also handles columns well. pdfplumber/pypdf are weaker on column order. (Engineering experience; no benchmark source found specific to 3-column FR layout.)
- AGPL for PyMuPDF only matters if the system is distributed/networked as a product; for a hackathon submission it is generally acceptable, but note it in the README. pypdfium2 (Apache/BSD) is the permissively-licensed speed-equivalent fallback, but it lacks PyMuPDF's TOC/font/table APIs.

### Gaps
- No independent, same-hardware benchmark of pymupdf4llm pages/s (with tables and header detection) was found; py-pdf/benchmarks reports per-file times, not per-page.
- No OmniDocBench score for Docling or pymupdf4llm in the current official leaderboard.
- Unstructured's current (2026) `fast` vs `hi_res` throughput and license details not re-verified (2024 figure only: 4.2 s/page on x86 CPU). Unstructured OSS is Apache-2.0 per general knowledge, not re-verified here.
- No source found that specifically evaluates parsers on Federal Register 3-column notices or FASB ASUs.

## Q2. OCR fallback: Tesseract vs PaddleOCR vs docTR vs RapidOCR; detecting pages without a text layer

### Takeaway
On CPU, Tesseract and RapidOCR (ONNX port of PaddleOCR models) are the fastest (~0.2-0.5 s per small page image in tests); PaddleOCR proper is several times slower on CPU and docTR/OnnxTR sits in between. For one 92-page scanned English PDF, any of them finishes in roughly 1-3 minutes on CPU, so pick the one easiest to install (Tesseract via PyMuPDF's built-in OCR, or RapidOCR as a pip-only option).

### Cited Findings
- Codesota live test (March 2026, Apple M-series CPU, one 800x600 invoice): Tesseract ≈ 0.16 s, RapidOCR ≈ 0.21 s, PaddleOCR ≈ 4.85 s (older test), docTR ≈ 1.8 s (community figure) — [Codesota OCR for Python](https://codesota.com/ocr/best-for-python)
- OnnxTR project table (CPU, older versions): docTR v0.8.1 ≈ 1.29 s/page on FUNSD and 0.60 s on CORD; pytesseract ≈ 0.50 / 0.52 s; PaddleOCR v2.7.3 (no classifier) ≈ 1.27 / 0.38 s; OnnxTR CPU ≈ 0.57 / 0.25 s — [onnxtr on PyPI](https://pypi.org/project/onnxtr/0.3.2)
- WebPII paper OCR-only latency: Tesseract 453 ms, EasyOCR 715 ms, PaddleOCR 2,143 ms — [WebPII, arXiv 2603.17357](https://arxiv.org/pdf/2603.17357)
- Docling's EasyOCR default is "upwards of 30 seconds per page" on CPU — [Docling Technical Report v1](https://arxiv.org/html/2408.09869)
- Accuracy ranking from the OmniDocBench leaderboard: VLM OCR (PaddleOCR-VL, MinerU2.5, Gemini 3 Flash) is far ahead of pipeline OCR on full-page parsing; PaddleOCR-VL-1.6 overall 96.34 — [OmniDocBench](https://github.com/opendatalab/OmniDocBench)

### Inferences
- Detection of no-text-layer pages (standard PyMuPDF practice, not benchmarked): per page, `len(page.get_text().strip()) < ~20-50 chars` AND (`page.get_images()` non-empty, ideally one image covering most of the page bbox). Additional signals: absence of fonts (`page.get_fonts()` empty), or text that is mostly garbage (low ratio of dictionary words/alphanumerics → broken font encoding, OCR it too). Check per page, not per document; annual reports often mix scanned signature/auditor pages into digital docs.
- PyMuPDF exposes Tesseract directly (`page.get_textpage_ocr(dpi=300, full=True)`), keeping output in the same span/block format so the downstream chunker is unchanged. Requires the tesseract binary + tessdata installed.
- Estimate for the 92-page scanned PDF: render at 200-300 dpi (~0.1-0.3 s/page) + Tesseract ~0.5-1.5 s per full A4 page at 300 dpi → roughly 1-3 min single-threaded, <1 min with 4 processes. A full A4 page is larger than the invoice/FUNSD test images above, so scale those numbers up by ~2-4x (estimate).
- Cheap-LLM vision OCR is an option for the scanned doc only if the API supports images; at 92 pages it is cheap, but adds latency and nondeterminism. Not needed.

### Gaps
- No single controlled 2026 benchmark comparing all four OCR engines' accuracy (CER/WER) on scanned A4 English financial pages was found.
- Tesseract on scanned financial tables (column alignment) quality is not quantified in the sources found.

## Q3. Recovering structure and producing ~60-word paragraph units

### Takeaway
Use the PDF outline (`doc.get_toc()`) when present as the authoritative section tree, fall back to font-size/bold heading detection (pymupdf4llm `IdentifyHeaders`) plus regex for numbered paragraphs, and build evidence units from PyMuPDF text blocks merged/split to ~40-120 words, always carrying (filename, page_start, page_end, section_path).

### Cited Findings
- pymupdf4llm can take header levels from `Document.get_toc()` via `TocHeaders` or infer them from font sizes via `IdentifyHeaders` (body_limit=11, max 6 levels); `page_chunks=True` yields per-page output with layout boxes classified as title, section-header, page-header, page-footer, list-item, table, footnote, formula, etc. — [pymupdf4llm API](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html)
- Docling's outputs are a structured document model (DoclingDocument) with layout + reading order + TableFormer table structure, but at 0.6-3 s/page on CPU — [Docling Technical Report v2](https://arxiv.org/html/2501.17887)
- A 2026 paper proposes "evidence units" as parser-independent document organization for retrieval, evaluated with MinerU and Docling on OmniDocBench — [Evidence Units, arXiv 2604.00500](https://arxiv.org/pdf/2604.00500)

### Inferences
- Recommended unitization algorithm (design, not sourced):
  1. Per page, `page.get_text("dict", sort=True)` → blocks/lines/spans with font size, bold flag, bbox.
  2. Drop running headers/footers: lines that repeat (normalized, digits stripped) at the same y-band on >30% of pages of a doc; drop bare page numbers.
  3. Heading detection: TOC titles fuzzy-matched on their page (best), else spans with size > body_size + 1pt or bold-and-short (<12 words, no terminal period).
  4. Numbered paragraph detection for Basel/BCBS/FASB: regex at line start like `^\d{1,3}\.\s`, `^\d+\.\d+(\.\d+)*\s`, `^\(?[a-z]\)\s`, `^\(?[ivx]+\)\s`, FASB codification refs `^\d{3}-\d{2}-\d{2}-\d+`, and Federal Register `§\s?\d+\.\d+`. Each match starts a new unit and its number is stored as `para_id` (huge win for queries like "paragraph 52 of BCBS 239").
  5. Merge consecutive blocks within a section until ≥ ~40 words; split blocks > ~150 words at sentence boundaries with ~1 sentence overlap. Target median ≈ 60-80 words to match the gold units (median 63).
  6. Allow units to cross a page break only when a sentence is cut mid-way (no terminal punctuation at page end); record `pages=[p, p+1]`.
  7. Prepend context for embedding only (not for display): `"{doc_title} > {section_path} > {para_id}: "` — contextual headers measurably help on regulatory docs with repeated boilerplate.
- Store page numbers as physical 1-based PDF page index (what gold labels almost certainly use) plus, optionally, the printed page label via `page.get_label()` for display.
- Hyphenation: join `-\n` line breaks; PyMuPDF flag `TEXT_DEHYPHENATE` handles this.
- Equations in quant papers: text-layer extraction yields garbled math; acceptable since retrieval targets prose; don't spend OCR/VLM budget on formulas.

### Gaps
- No benchmark found comparing chunking strategies specifically on numbered-paragraph regulatory text; recommendation is design reasoning.
- Proportion of corpus PDFs with usable bookmarks (get_toc) unknown; must be measured on the 35 PDFs (quick: `len(doc.get_toc())`).

## Q4. Tables in annual reports: markdown vs row-level linearization

### Takeaway
Do both: keep each table as one markdown unit (with caption/section and page) for display/LLM answering, and additionally emit row-level linearized units ("{table title} | {row label} | {col header}: {value}; ...") for retrieval, because numeric queries match a single row far better than a 40-row grid. Pipeline table detectors on CPU are weak (Marker TEDS 65.8; MinerU-Pipeline 81.9) and slow, so use PyMuPDF `find_tables()` with a text-layer fallback.

### Cited Findings
- Table structure accuracy (TEDS) on OmniDocBench: MinerU-Pipeline 81.88, Marker 65.77, Gemini 3 Flash 89.29, PaddleOCR-VL-1.6 94.76 — [OmniDocBench](https://github.com/opendatalab/OmniDocBench)
- pymupdf4llm marks table boxes (class "table") in page_chunks output — [pymupdf4llm API](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html)
- pdfplumber is the slowest extractor in py-pdf benchmarks (8.6 s average per file vs 0.1 s PyMuPDF) — [py-pdf/benchmarks](https://github.com/py-pdf/benchmarks)

### Inferences
- Many annual-report "tables" are borderless (whitespace-aligned); PyMuPDF `find_tables(strategy="text")` or plain line-based reconstruction (group spans by y, split by x-gaps) is often good enough. Line-strategy works for ruled tables.
- Gold evidence units are paragraph-sized; if gold units come from tables they are likely a few rows of text as extracted by a text-layer tool. So ensure the plain text of table regions is ALSO indexed as normal blocks (don't remove it when you create markdown); deduplicate at answer time.
- Detect "numbers-heavy" pages (ratio of numeric tokens > ~40%) and route only those to the table extractor; caps time spent.
- Optional: LLM table cleanup only for top-retrieved tables at query time (lazy), not at ingestion.

### Gaps
- No source found with quantitative retrieval gains from row-level linearization vs whole-table markdown on financial reports specifically.
- PyMuPDF `find_tables()` speed/accuracy not benchmarked in sources found.

## Q5. Automatic document typing and metadata extraction

### Takeaway
A two-tier approach is cheap and robust: (1) heuristics from PDF metadata, filename, TOC and first-page text (regex for fiscal year, "Basel Committee", "Federal Register / Vol. / No.", "Accounting Standards Update No. 20XX-XX", "arXiv", "Annual Report"), then (2) a single cheap-LLM JSON-extraction call per document on the first ~2-3 pages (≈2-4k tokens). For 35 docs this is ~100-150k input tokens total — cents and under a minute if run concurrently.

### Cited Findings
- No primary source found with measured accuracy for first-page heuristics vs LLM metadata extraction on regulatory/financial PDFs (see Gaps). Hosted parsers price per page (LlamaParse Cost-effective 3 credits/page = ~$0.00375/page) which is far more than an LLM call on 3 pages of already-extracted text — [LlamaParse pricing](https://developers.llamaindex.ai/llamaparse/general/pricing/)

### Inferences
- Heuristic signals (all cheap, available via PyMuPDF): `doc.metadata` (title, author, subject, creator — often populated by publishers like BIS/FASB but often junk in annual reports), filename tokens, first TOC entry, largest-font text on page 1, first-page regexes:
  - Fiscal year: `FY\s?20\d\d`, `20\d\d[-–/]\d\d`, `year ended \w+ \d{1,2},? 20\d\d`, "Annual Report 2024-25".
  - Federal Register: `Federal Register\s*/\s*Vol\.\s*\d+,\s*No\.\s*\d+` + date + agency + `RIN`/`Docket No.`.
  - FASB: `Accounting Standards Update\s*No\.\s*20\d\d-\d\d`, "Topic \d{3}".
  - BCBS: "Basel Committee on Banking Supervision", "bis.org", month-year, "Consultative Document" vs "Standards".
  - Academic: "arXiv:", "Abstract", "Keywords", "JEL", DOI.
  - Jurisdiction: country/regulator names (SEBI, RBI, CBUAE, SAMA, QCB, SEC, OCC, FDIC, FRB).
- Suggested LLM schema: `{doc_type ∈ [annual_report, regulatory_standard, federal_register_rule, accounting_standard, academic_paper, other], issuer, entity, title, period_or_fiscal_year, publication_date, version_or_number, jurisdiction, language, confidence}`; give the heuristic candidates as hints; temperature 0; JSON mode. Expected accuracy is high for doc_type/issuer/title (these are on page 1), lower for "version" and "period" in annual reports with multiple years mentioned — validate with regex against the LLM output.
- Cost estimate: 35 docs × ~3k tokens in + 200 out ≈ 110k in / 7k out. At typical cheap-model prices ($0.1-0.5 per M input), < $0.10. Latency: run with 8-16 concurrent requests → < 1 min. Make it non-blocking: index first, fill metadata asynchronously, and fall back to heuristics if the API fails/timeouts.
- Use metadata at query time for filtering/boosting (e.g. query mentions "FY2023" or an issuer name → boost matching docs) — high value for multi-annual-report corpora where the same boilerplate repeats across years/companies.

### Gaps
- No published accuracy numbers found for LLM-based metadata extraction on this document mix; must be validated on the 35-doc corpus (it is small enough to hand-check).
- Exact price of the "cheap LLM" available to the team not known.

## Q6. Embedding throughput on CPU for ~100k paragraphs and options to stay under ~20-30 min

### Takeaway
Static embeddings (Model2Vec potion) embed 100k paragraphs in seconds on CPU but lose substantial retrieval quality (MTEB retrieval ~35 vs ~43 for MiniLM); a small transformer (bge-small / e5-small / gte-small, 33M params, ONNX int8) is likely ~5-20 min for 100k ~80-word paragraphs on an 8-core CPU; bge-m3 (568M) on CPU is likely 1-3 h and should be avoided unless a GPU exists. Combine with BM25 (seconds) for strong hybrid recall.

### Cited Findings
- Model2Vec claims "up to 500 times faster on CPU than the original model"; license MIT — [MinishLab/model2vec](https://github.com/MinishLab/model2vec)
- MTEB averages: potion-base-8M 51.08 (retrieval 31.11); potion-base-32M 52.13 (retrieval 32.67); potion-retrieval-32M retrieval 35.06; all-MiniLM-L6-v2 55.93 (retrieval 42.92); GloVe 300d 45.82 (retrieval 21.80) — [Model2Vec results](https://github.com/MinishLab/model2vec/blob/main/results/README.md)
- bge-small-en-v1.5 is FastEmbed's default model (ONNX Runtime, no PyTorch, 384-d) — [LangChain FastEmbed docs](https://docs.langchain.com/oss/python/integrations/embeddings/fastembed)
- Published CPU throughput numbers for bge-small via FastEmbed are inconsistent and mostly lack hardware details (26 to 96 sentences/s in small-batch tests); FastEmbed maintainers do not publish official latency figures — [ggmbed on PyPI](https://pypi.org/project/ggmbed/); [Markaicode FastEmbed vs HF](https://markaicode.com/vs/fastembed-vs-hugging-face/); [Markaicode benchmark guide](https://markaicode.com/benchmarks/fastembed-production-benchmark-latency/)

### Inferences
- Rough FLOP-based estimate (not sourced; verify with a 2k-paragraph timing run): 100k paragraphs × ~100 tokens = 10M tokens. bge-small (33M params, 12 layers × 384-d) ≈ 2 × 33M × 10M ≈ 7e14 FLOPs; a laptop CPU sustaining 0.5-2 TFLOP/s effective in ONNX/oneDNN fp32 → ~6-25 min; int8 quantized ONNX roughly 1.5-2.5x faster. bge-base/e5-base/gte-base (110M) ≈ 3.3x slower (20-80 min). bge-m3 / multilingual-e5-large (≈560M) ≈ 17x slower (1.5-7 h) → GPU only. nomic-embed-text-v1.5 (137M) ≈ base-sized cost.
- Corpus sizing check: 8,600 pages × ~400-500 words/page ÷ ~70 words/unit ≈ 50-60k units, not 100k, unless tables are row-linearized; so the bge-small estimate is more like 3-15 min.
- Speed levers: sort units by length and batch (cuts padding waste ~30-50%); cap max_seq_len at 256 tokens; ONNX int8 (fastembed or optimum); run embedding in a separate process concurrently with parsing (parsing is I/O+CPU light with PyMuPDF); use all physical cores (`OMP_NUM_THREADS`); cache embeddings by content hash so re-index of the same zip is instant.
- Tiered plan for the timeout: (a) BM25 index + Model2Vec potion-retrieval-32M vectors available within ~1-3 min of upload (index "ready"); (b) bge-small/e5-small dense vectors computed in the background and swapped in when done; (c) if a GPU is detected, use bge-m3 or bge-base instead. Report ingestion time honestly from (a) and (b).
- Reranking (e.g. bge-reranker-base / a cross-encoder) at query time over top-50 BM25+dense results recovers much of the quality lost by a small embedder, and costs nothing at ingestion.

### Gaps
- No authoritative, hardware-specified CPU throughput benchmark for bge-small / e5-small / gte-small / nomic / bge-m3 was found in this session; the MTEB leaderboard does not report speed. The estimates above must be measured on the target machine.
- Model2Vec raw embedding speed (sentences/s) not stated with hardware in the results README (speed numbers there are for classifier training).

## Recommended ingestion pipeline for our system (with time estimate for 8,600 pages)

### Takeaway
PyMuPDF-first, per-page routed pipeline: fast text-layer extraction for everything, Tesseract OCR only on pages with no usable text layer, cheap table handling only on numeric-heavy pages, structure from TOC + fonts + numbered-paragraph regex, one cheap-LLM metadata call per document, BM25 + small ONNX embedder (static embeddings as instant fallback). Expected wall-clock on an 8-core CPU laptop: ~8-20 minutes end-to-end, with a searchable BM25 index in ~2-4 minutes.

### Cited Findings
- PyMuPDF ≈ 0.1 s average per benchmark file with 94% text quality, vs ML pipelines at 0.6-16 s/page on CPU — [py-pdf/benchmarks](https://github.com/py-pdf/benchmarks); [Docling Technical Report v2](https://arxiv.org/html/2501.17887); [Docling Technical Report v1](https://arxiv.org/html/2408.09869)
- Tesseract ≈ 0.16-0.5 s per small page image on CPU — [Codesota](https://codesota.com/ocr/best-for-python); [onnxtr PyPI](https://pypi.org/project/onnxtr/0.3.2); [WebPII](https://arxiv.org/pdf/2603.17357)

### Inferences
Pipeline (each step generic, nothing corpus-specific):
1. **Unzip & enumerate** (seconds). Hash each file; skip if cached.
2. **Per-doc open with PyMuPDF**; collect `doc.metadata`, `doc.get_toc()`, page count. Parallelize over documents AND split very large docs (1,982 pages) into page ranges across a `ProcessPoolExecutor` (N = physical cores) so one giant file doesn't dominate.
3. **Per-page triage**: text chars, image coverage, numeric-token ratio, garbage ratio → route = {text, ocr, table-heavy}.
4. **Text pages**: `get_text("dict", sort=True, flags=dehyphenate)` → blocks with fonts/bboxes. Header/footer removal; heading detection (TOC match > font size > bold-short); numbered-paragraph regex. ~2-5 ms/page (est.).
5. **OCR pages**: `page.get_textpage_ocr(dpi=300, full=True)` (Tesseract) → same block format. 92 pages ≈ 1-3 min single core, <1 min parallel.
6. **Table-heavy pages**: `page.find_tables()` → markdown table unit + row-linearized units; keep raw text blocks too. Budget-capped (e.g. skip if >2 s/page).
7. **Unitize** to ~40-150 words (median ~65), with `doc_id, filename, pages[], section_path, para_id, unit_type ∈ {para, heading, table_md, table_row, list}`.
8. **Metadata**: heuristics immediately; one async cheap-LLM call per doc (first 2-3 pages of extracted text + TOC + PDF metadata), JSON schema; merge into every unit of that doc.
9. **Index**: BM25 (e.g. bm25s / tantivy) immediately; Model2Vec vectors immediately; bge-small/e5-small ONNX int8 vectors in background, length-sorted batches; store in FAISS/hnswlib or LanceDB with metadata filters.
10. **Report** ingestion stats (pages, OCR pages, units, timings) from /index.

Time estimate for 8,600 pages (8-core CPU, no GPU; estimates to be validated with a timing run):
| Stage | Estimate |
|---|---|
| Unzip + open + TOC/metadata | < 30 s |
| Text extraction + structure (≈8,500 born-digital pages, 8 procs) | 1-3 min |
| OCR 92 scanned pages (Tesseract, parallel) | 0.5-2 min |
| Table extraction on numeric-heavy pages (say 1,500 pages × ~0.1-0.3 s / 8 procs) | 0.5-1.5 min |
| LLM metadata (35 calls, concurrent, async) | < 1 min (overlaps) |
| BM25 + Model2Vec on ~50-80k units | < 1 min |
| bge-small/e5-small ONNX int8 dense vectors on ~50-80k units | 4-15 min (background) |
| **Total to "searchable" (BM25 + static)** | **~3-6 min** |
| **Total to full hybrid index** | **~8-20 min** |
- If a GPU is present: swap in bge-m3 (dense+sparse) or bge-base; optionally route the scanned PDF and table-heavy pages through Docling/MinerU on GPU (0.2-0.5 s/page) — still within budget.
- What NOT to do on CPU: Docling/MinerU/Marker/Unstructured hi_res over all pages (≈1.5-40 h), EasyOCR (30 s/page), bge-m3 on CPU over all units, LlamaParse in the timed path (network + third-party data exposure, ~$32-108 for this corpus).
- State to judges: "Expected ingestion ≈ 1.5-2.5 min per 1,000 pages on an 8-core CPU (plus ~1 s per scanned page); index is queryable (lexical + static vectors) after the first phase, dense vectors finish in background."

### Gaps
- All CPU time estimates for PyMuPDF block extraction, find_tables, and small-embedder throughput are engineering estimates, not cited benchmarks; run a 200-page + 5k-unit timing probe on the actual machine before quoting numbers in /index.
- Gold-unit page numbering convention (physical index vs printed label) should be confirmed against the provided gold file.
