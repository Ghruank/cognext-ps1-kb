# Version, Year, Jurisdiction and Entity Disambiguation in Retrieval

Scope: methods for telling near-duplicate documents apart (version, period, jurisdiction, entity) in a RAG knowledge base built from an unlabeled folder dump. Context: CognitionNext '26 PS1. Distractors in our corpus include Basel Dec-2017 vs 2020 revision vs the consolidated Basel Framework; US 2023 vs 2026 capital proposals; FY2025 vs FY2026 annual reports; SAMA vs India vs US rules. Hybrid+rerank, HyDE, multi-query, CRAG, vanilla GraphRAG and ReAct are banned as the core method; layers on top of a vector DB are allowed. Research date: 2026-10-23.

## Q1. Temporal IR and time-aware RAG: how do systems extract and use dates and validity periods?

### Takeaway
The approaches that work best treat time as a constraint you pull out of the query (explicitly), not as text to embed. Examples: MRAG splits a query into content + temporal constraint, and VersionRAG and the legal benchmarks filter by version. The clearest result: on a French tax-law benchmark, a static current-version index got 0% of date-applicable versions right, while a multi-version index reached 98.3%. Adding dates to the embedding text helps only a little (TempRetriever, +4 to +7 pts Top-1), and putting dates in as plain tags can make results worse.

### Cited Findings
- **TimeR4 (EMNLP 2024)**: a retrieve, rewrite, retrieve, rerank pipeline for temporal KG QA. It uses TKG facts to rewrite the question with explicit time constraints, then reranks retrieved facts by those constraints. The retriever is fine-tuned with a contrastive time-aware objective, with reported relative gains of 47.8% and 22.5% on two datasets. Code: github.com/qianxinying/TimeR4. Note that BM25 and other semantic retrievers return facts with the wrong timestamps. — [ACL Anthology 2024.emnlp-main.394](https://aclanthology.org/2024.emnlp-main.394)
- **TempRALM (Gade & Jetcheva, 2024; IEEE CAI 2025)**: the retriever scores documents on semantic relevance plus temporal relevance to the query time. It needs no pre-training and no rebuild or replacement of the index. Reported gains: up to 74% over ATLAS and 32% over a commercial LLM with retrieval. — [arXiv 2401.13222](https://arxiv.org/abs/2401.13222); [SJSU ScholarWorks](https://scholarworks.sjsu.edu/faculty_rsca/6190)
- **TempRetriever (2025)**: each passage takes its source document's publication date. The question date is either explicit, or predicted by a BERT date predictor (MAE 3.51 years, 20% exact accuracy). Fusion options are sum, difference, product, and concatenation ("feature stacking", 1,536-d), with stacking best. Test Top-1: ArchivalQA 62.98 → 69.84 (+6.86); ChroniclingAmericaQA 45.70 → 50.10 (+4.40). Top-20: 90.97 → 94.28 and 72.20 → 77.19. Simple date injection is weaker: DateAsTag 58.37 (below the DPR baseline of 62.98) and DateAsToken 63.44. Hard date filtering was not evaluated. — [arXiv 2502.21024](https://arxiv.org/html/2502.21024v2)
- **MRAG (modular retrieval, training-free)**: an LLM splits the question into main content (MC) and temporal constraint (TC). Contriever retrieves on MC alone, "keeping retrievers from being distracted by dates". Sentences then get semantic score × temporal score. The temporal score comes from dates extracted from the sentence, passed through pre-defined spline functions per constraint type (first/last, before/after/between). Benchmark: TempRAGEval, 1,000 temporally perturbed questions from TimeQA and SituatedQA. Under perturbation, baseline top-1 evidence recall on SituatedQA fell from 45.0% to 20.3%. MRAG gains +7.7 (TimeQA) and +13.9 (SituatedQA) points in top-5 evidence recall, about 4.5% EM/F1 end to end, and Llama3.1-8B TimeQA EM goes 44.0 → 49.2. — [arXiv 2412.15540](https://arxiv.org/html/2412.15540v1)
- **VersionRAG (Oct 2025)**: an LLM extracts title, version and document type from the first pages. Versions of the same document are clustered by title similarity. Explicit changes come from changelogs; implicit changes come from line-level diffs (DeepDiff) between consecutive versions, which an LLM then describes. The hierarchical graph runs Category → Document → Version (temporal sequence edges) → Content (vector chunks) → Change. An LLM router sorts queries into content / version-listing / change-retrieval and extracts doc and version parameters; routing accuracy is 92% for DeepSeek-R1-70B, 91% for GPT-4o-mini and 76% for Llama-3-8B. Content queries use vector search with a version filter. — [arXiv 2510.08109](https://arxiv.org/html/2510.08109v1)
  - Results on VersionQA (100 questions, 34 versioned technical docs: Node.js, Spark, Bootstrap), DeepSeek-R1-70B: overall Naive RAG 58% / GraphRAG 64% / VersionRAG 90%. Content retrieval 76.6 / 70 / 93.3. Version listing 35 / 80 / 100. Change retrieval 25 / 30 / 70. On implicit changes, VersionRAG gets 60% vs 0–10% for the baselines. With GPT-4o-mini overall is 79%, and with Llama-3-8B 55%. Indexing uses 97% fewer tokens than GraphRAG (about 186K tokens for 34 docs). — [arXiv 2510.08109](https://arxiv.org/html/2510.08109v1)
  - Stated limitations: it needs version info on the first pages; date-based or code-name schemes may fail; it handles discrete versions, not continuous calendar time; and errors propagate into the graph. — [arXiv 2510.08109](https://arxiv.org/html/2510.08109v1)
- Related work: DeFuzzRAG (AAAI) handles vague time expressions without retraining (details not read). — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/40276/44237)

### Inferences
- For our corpus, MRAG's main content / temporal-constraint split generalizes directly to a main content / {year, version, jurisdiction, entity} split. Embed only the content part and resolve the constraints against a catalogue. This is a "layer on top of a vector DB" and is not one of the banned techniques.
- TempRetriever's small gains and DateAsTag's regression suggest that "prepend the year to the chunk" is not enough on its own for version separation. Hard or soft constraint matching is needed.
- VersionRAG is the closest published design to our problem. Its weak spot (date-based versions) is exactly our case (2017 vs 2020, FY2025 vs FY2026), so we need date/period extraction as well as version-string extraction.

### Gaps
- I did not read the full TempRALM paper, so the exact form of the temporal score (decay function) is unverified.
- No paper found evaluates mixed constraints (year + jurisdiction + entity together).

## Q2. Legal/regulatory IR: amendments, supersession, consolidated vs original, citation graphs

### Takeaway
Legal QA results consistently show that static RAG over the "current" text cites real but inapplicable versions. Building a multi-version index with explicit validity intervals and filtering by an "as-of" date fixes most of this, with effects that dwarf any embedding improvement. Standard practice models a "work" (the act) separately from its dated "expressions" (versions, consolidations); ELI/FRBR is the reference model.

### Cited Findings
- **FiscalQA Pro (French tax code, arXiv 2608.09393, 2026)**: 32,436 article-versions covering 1938–2031; 209 scored expert-reviewed temporal questions over 33 articles; 69,208 version-aware jurisprudence citation links. They define "temporal misgrounding" as retrieving and citing the currently in-force version when an earlier or future version applies. Mean strict accuracy: closed-book across 11 LLMs 3.0%; static RAG over the current corpus 2.7%, with 0% retrieval of the date-applicable version (it confidently cites a real but inapplicable version). A multi-version retriever with no oracle reaches 98.3%; the oracle-article ablation reaches 99.1%, so remaining error is first-stage recall, not version selection. Scoring uses deterministic regex/numeric nuggets, not an LLM judge, which they argue would share the temporal bias. — [arXiv 2608.09393](https://arxiv.org/abs/2608.09393)
- **German statutory QA (Prior, Schultz, Grabmair, ICAIL 2026, arXiv 2605.23497)**: 312 expert-validated QA pairs (115 post-cutoff amendments, 113 single-provision pre-amendment, 84 multi-provision pre-amendment), including 39 deliberate near-duplicates. Validity periods come from buzer.de in-force dates. At inference, a regex pulls an "as-of" date from the query and filters retrieval to paragraph versions valid then; an LLM separately picks the applicable statute first. On post-cutoff questions, version-filtered RAG scores 0.78–0.88; vanilla Claude Sonnet 3.7 drops to 0.08–0.11, and its outcome correctness goes 24.35% → 86.96% with RAG-kNN. Web search sits at 0.25–0.67 and shows significant recency bias (+0.20–0.24 on recent questions). Removing the temporal filter caused "a substantial drop", with no number given in the text. — [arXiv 2605.23497](https://arxiv.org/html/2605.23497)
- **Chinese statute agents (LegalSearch-R1, 2026)**: training data indexed by amendment period; provision validity windows annotated across 13 amendment versions of Chinese criminal and procedure law (2000–2025); reported gains in temporal consistency. — [arXiv 2605.25920](https://arxiv.org/pdf/2605.25920); [papers.cool](https://papers.cool/arxiv/2605.25920)
- **COLIEE**: Task 3 is statute retrieval for Japanese bar-exam questions; 8 teams submitted 22 methods in 2025. I found no COLIEE work that handles amendments or versioning; the task is version-agnostic. — [ANLP 2026](https://www.anlp.jp/proceedings/annual_meeting/2026/pdf_dir/TS3-11.pdf); [arXiv 2410.12154](https://arxiv.org/html/2410.12154v1)
- **ELI/FRBR model**: eli:LegalResource = work (the statute conceptually); eli:LegalExpression = expression (a language version); eli:Format = manifestation (file format). ELI ontology v1.3 (July 2020) added generic eli:Work/Expression/Manifestation. ELI can identify "a whole act, a specific and provisional version of an act, or a given version in a particular file format." — [EUR-Lex ELI register](https://eur-lex.europa.eu/eli-register/news_item_27.html); [W3C schema.org thread](https://lists.w3.org/Archives/Public/public-schemaorg/2016Feb/0023.html)
- **Detecting amendment provisions**: Turin/Bologna (CIRSFID) work detects modificatory provisions with parsing plus regular expressions. Coverage started with substitution, integration and repeal and was extended to twelve more modification types. — [Unibo ADELE PDF](https://site.unibo.it/adele/en/publications/publications2/automated-extraction.pdf/@@download/file/automated%20extraction.pdf)
- **Citation networks**: a 2026 paper on Greek legislation builds a directed cross-reference network where an edge means one law explicitly cites another, and counts amendment keywords. Highly cited statutes act as hubs. — [arXiv 2606.00030](https://arxiv.org/pdf/2606.00030). A New Zealand study treats legislation as a complex network (centrality, community detection). — [Auckland PDF](https://www.cs.auckland.ac.nz/~mcw/Research/Outputs/SWZ2016.pdf)

### Inferences
- The "consolidated Basel Framework" distractor is a consolidation (a compiled expression of many works), not an original. A cheap, general signal: the document cites or incorporates many other documents in the same family, is very long (1,982 pages), and contains words like "consolidated", "framework", "compiled", "as amended". The legal literature suggests treating consolidations as a separate `doc_role` and, when the query names a specific instrument or date, preferring the original instrument. This is our inference; I found no paper that tests it.
- Cheap supersession edges can come from regex over the text: "amends", "replaces", "supersedes", "revises", "repeals", "finalises", "proposed rule", "final rule", plus references to other doc identifiers or dates. This follows the CIRSFID regex+parse approach.
- An as-of date or year pulled from the query with a regex (as in the German paper) is a proven, cheap, LLM-free first step.

### Gaps
- I did not find how EUR-Lex internally models dated point-in-time consolidations (ELI register pages not read in full).
- No published numbers on retrieval with a consolidated-text distractor alongside the originals.

## Q3. Automatic metadata extraction at ingestion (issuer, doc type, period, FY, entity, jurisdiction, version)

### Takeaway
LLM extraction from the first pages is the de facto approach (VersionRAG, MetaRAG-on-FinanceBench). Published accuracy for generic business fields is scarce. Scholarly-header benchmarks put title extraction at about 0.8–0.9 F1 for classic tools, and the vendor and secondary claims for LLMs (0.91–0.97 F1, "95%+") are unverified. Embedding metadata into the chunk text ("contextual chunks") is the single biggest measured win on FinanceBench, and Anthropic's contextual retrieval cut top-20 retrieval failures by 35–49%.

### Cited Findings
- **Metadata-driven RAG on FinanceBench (arXiv 2510.24402)**: PDFs go to Markdown via Docling. Gemini 2.5 Flash writes document-level metadata (one-liner, summary, 5–20 thematic clusters) and chunk-level metadata (entities, answered questions, retrieval nuggets). The stored payload includes file name, year and quarter. Query-side, gpt-4.1-mini sees the query plus every document's one-liner and returns relevant filenames, which become a strict filter. Results (RAGChecker F1, top-7): naive dense 32.9; hybrid 30.4 standard vs 37.6 with contextual chunks; hybrid+Cohere rerank 38.9 vs 44.1; with filter+rewrite added, 37.3 vs 44.4. Contextual chunks gave "the most significant gains". Adding pre-retrieval filtering dropped claim recall from 50.7 to 47.7, possibly because filtering is "too aggressive". — [arXiv 2510.24402](https://arxiv.org/html/2510.24402v1)
- **FinanceBench**: GPT-4-Turbo with retrieval answered wrongly or refused on 81% of questions. The full set has 10,231 QA-evidence triplets over 40 companies and 361 filings; a 150-case sample is public. — [Patronus docs](https://docs.patronus.ai/docs/research_and_differentiators/financebench). Reported failure modes include the right company but wrong section, confusing fiscal years, and BM25 being thrown by repeated finance terminology. — [NanoFinanceBench docs](https://hakari-bench-leaderboard.hf.space/docs/benchmark-tasks/NanoRTEB/NanoFinanceBench) (a benchmark doc page, not peer-reviewed). A practitioner guide recommends attaching company, form type, period end, fiscal year and amendment status to every chunk and filtering *before* similarity search. — [qveris guide](https://qveris.ai/guides/financial-rag/)
- **Anthropic Contextual Retrieval (Sep 2024)**: an LLM writes 50–100 tokens of document context for each chunk, which is prepended before embedding and BM25 indexing. Contextual embeddings alone cut the top-20 retrieval failure rate by 35%; contextual embeddings + contextual BM25 by 49% (5.7% → 2.9%); with reranking, 67%. A secondary source estimates about $1.02 per million document tokens with prompt caching. — [Anthropic](https://anthropic.com/news/contextual-retrieval); [gradientflow](https://gradientflow.com/contextual-retrieval-rag/)
- **VersionRAG** extracts title/version/type with an LLM from the first pages (at least the first 10 pages are checked to classify changelog vs docs). — [arXiv 2510.08109](https://arxiv.org/html/2510.08109v1)
- **Classic, non-LLM tools (scholarly PDFs)**: in a 2023 multi-tool benchmark, GROBID scored best, with title F1 0.91, abstract 0.82, authors 0.52. — [arXiv 2303.09957](https://arxiv.org/pdf/2303.09957). On German publications, GROBID overall F1 was 0.618 (title 0.812) vs 0.846 for a multimodal model. — [arXiv 2111.05736](https://arxiv.org/pdf/2111.05736). The National Library of Finland fine-tuned small open LLMs on first and last PDF pages to predict title, creator and date (no numbers in the slides). — [SWIB24](https://swib.org/swib24/slides/SWIB24-automating-metadata-extraction-and-cataloguing-experiences-from-the-national-libraries-of-norway-and-finland.pdf)
- **LLM extraction accuracy (weak evidence)**: arXiv 2510.19334 (leases/CUAD) reports JSON tool calling raised Claude 3.5 Sonnet F1 from 0.39 to 0.48 on a proprietary lease set. These are hard contract fields, not simple header fields, and the figure comes via a secondary review. — [arXiv 2510.19334](https://arxiv.org/html/2510.19334v1); [Moonlight review](https://www.themoonlight.io/review/metadata-extraction-leveraging-large-language-models). A vendor blog cites GPT-4o F1 of 0.91–0.97 on scientific-paper metadata ("MOLE" study, not located) — [fast.io](https://fast.io/resources/llm-powered-metadata-extraction-documents/). A vendor case study claims GPT-4o-mini reached 95%+ on legal doc-type/organization classification — [Droptica](https://www.droptica.com/blog/prompt-engineering-data-extraction-how-achieve-95-accuracy-legal-documents/). Treat both as unverified.

### Inferences
- A cheap, general cascade that should work on an unknown corpus:
  1. Filename and path tokens: years, "FY", "v2", "final", "proposal", country codes.
  2. First 1–2 pages: regex for years, dates, fiscal-period phrases ("for the year ended 31 March 2026", "FY2025-26"), version strings, and Federal Register citations ("88 FR", docket numbers, "RIN").
  3. A gazetteer of regulators and jurisdictions (BCBS/BIS, SAMA, RBI, SEBI, Federal Reserve, OCC, FDIC, Federal Register): about 50 strings that generalize across regulatory corpora, mapped to jurisdiction.
  4. PDF metadata fields (CreationDate, Title, Producer).
  5. One small LLM call on page 1 plus the TOC, with a JSON schema, only to fill blanks or resolve conflicts, and storing a confidence and source-span for each field.
  This mirrors the LlamaExtract "citation per field" approach. — [LlamaIndex extract w/ citations](https://developers.llamaindex.ai/llamaparse/extract/v1/examples/extract_data_with_citations/index.md)
- Store the extracted metadata in two places: as structured fields for filtering, and as a short header prepended to every chunk (contextual chunk). The FinanceBench paper shows the header helps even when filters are wrong, which is a hedge against extraction errors.
- Pitfall: an "effective year" differs from the "publication year". For example, a 2017 Basel document is effective 2023+, and a FY2026 report is published in 2026 covering 2025-26. Extract both `issued_date` and `period/effective` where possible, and match query years against both with different weights.

### Gaps
- No credible published per-field accuracy for LLM extraction of fiscal year, issuer or jurisdiction from corporate or regulatory PDFs. Measure it on our own corpus with about 20 hand labels.

## Q4. Entity linking, entity-centric indexes, and resolving query constraints against a document catalogue

### Takeaway
Dense retrievers are weak at entity specificity: DPR averaged 49.7% vs 72.0% for BM25 on EntityQuestions. That is the same failure as "SAMA vs RBI", so resolve entities and constraints symbolically. Self-query and auto-retrieval (LLM → metadata filter + residual query) are standard in LangChain and LlamaIndex, but I found no published accuracy numbers for them. Hard pre-filters can hurt recall when the extraction is wrong, so soft or fallback strategies matter.

### Cited Findings
- **EntityQuestions (Sciavolino et al., EMNLP 2021)**: DPR trained on NQ averages 49.7% vs BM25 72.0%. Dense retrievers generalize only to common entities, and data augmentation does not fix it. — [arXiv 2109.08535](https://ar5iv.arxiv.org/html/2109.08535)
- **Knowledgeable Passage Retriever (EMNLP Findings 2025)**: +12.6% on EntityQuestions over the model without KPR extensions. — via search summary of [weblab U-Tokyo](https://weblab.t.u-tokyo.ac.jp/en/news/20050821/) (not read in full)
- **IR with Entity Linking (arXiv 2404.08678)**: expands queries and documents with linked entities, either as explicit names or hashed entity tokens, using a zero-shot dense entity linker; evaluated with recall@1000 on MS MARCO (numbers not read). — [arXiv 2404.08678](https://arxiv.org/abs/2404.08678)
- **LangChain SelfQueryRetriever**: an LLM produces a structured query (filter + query string) from a declared attribute schema (e.g. year, director, rating), and a Translator converts it to the vector store's filter syntax. — [LangChain docs](https://js.langchain.com/docs/how_to/self_query). **LlamaIndex auto-retrieval**: an LLM infers metadata filters plus a query string, either of which can be blank. In their example, "revenue in 2022" becomes a shorter query plus a year filter. — [LlamaIndex Vectara auto-retriever](https://developers.llamaindex.ai/python/examples/retrievers/vectara_auto_retriever); [Elasticsearch auto-retriever](https://developers.llamaindex.ai/python/examples/vector_stores/elasticsearch_auto_retriever)
- **Catalogue routing in practice**: in the FinanceBench metadata-RAG paper, an LLM selects filenames from a catalogue of one-line summaries, and this hard filter slightly lowered claim recall (50.7 → 47.7). — [arXiv 2510.24402](https://arxiv.org/html/2510.24402v1). In VersionRAG, the LLM extracts doc and version parameters with 76–92% routing accuracy depending on model size. — [arXiv 2510.08109](https://arxiv.org/html/2510.08109v1)
- **Filter mechanics**: post-filtering after ANN can silently return too few results under selective filters. Qdrant reports filtered-search recall of 90.8% for a broad-value filter and 39.7% for an AND of two broad values in naive setups, with other filter shapes above 97%. Qdrant and Weaviate integrate filtering into the HNSW traversal; pgvector post-filters by default. — [Qdrant blog](https://qdrant.tech/blog/pre-filtering-vs-post-filtering/); [Turso blog](https://turso.tech/blog/filtering-in-vector-search-with-metadata-and-rag-pipelines)

### Inferences
- With a small corpus (tens to hundreds of docs), you can skip ANN filter pitfalls entirely. Resolve constraints against an in-memory catalogue (a pandas or SQLite table, one row per doc) to get a candidate doc-id set, then run exact vector search restricted to those doc IDs. This is effectively text-to-SQL over metadata, but a deterministic slot-filler (regex for years and FY, a gazetteer for jurisdiction and issuer, fuzzy matching of entity names against the catalogue's `entity` column) covers most queries without an LLM.
- To avoid the recall loss seen with hard filters, use soft scoring: final = cos_sim + w_year·match(year) + w_juris·match(jurisdiction) + w_entity·match(entity) − w_role·is_consolidation. Fall back to unfiltered search when no catalogue row satisfies the constraints.

### Gaps
- No published accuracy benchmark for LangChain or LlamaIndex self-query on real corpora.
- KPR and 2404.08678 numbers not verified at source.

## Q5. Near-duplicate detection used constructively: version families and diff-aware indexing

### Takeaway
Classic MinHash/LSH (Broder 1997) and SimHash (Charikar; Manku et al. 2007: 64-bit, k=3 on 8B pages) are cheap and mature ways to group documents into near-duplicate "families". VersionRAG shows the constructive pattern (cluster versions, order them, index diffs as change nodes) improves version listing from 35% to 100% and change questions from 25% to 70%. Containment (not just Jaccard resemblance) is the right metric for spotting a consolidated document that subsumes originals.

### Cited Findings
- **Broder 1997, resemblance and containment**: resemblance = |S(A)∩S(B)| / |S(A)∪S(B)| over shingle sets (Jaccard); containment = |S(A)∩S(B)| / |S(A)|. Min-wise independent permutations estimate these at scale; resemblance close to 1 captures "near-duplicate". Applied to the web in "Syntactic Clustering of the Web" (Broder, Glassman, Manasse, Zweig, WWW 1997). — [Princeton COS435 notes](https://www.cs.princeton.edu/courses/archive/spring14/cos435/notes/duplicateDocs_topost.pdf); [Mitzenmacher/Broder JCSS 2000](https://www.eecs.harvard.edu/~michaelm/postscripts/jcss2000.pdf); [Papers We Love](https://paperswelove.org/papers/on-the-resemblance-and-containment-of-documents-973f823a/)
- **SimHash (Manku, Jain, Das Sarma, WWW 2007)**: fingerprints of near-duplicates differ in few bit positions. On 8 billion pages, 64-bit fingerprints with Hamming threshold k=3 were reasonable; the paper also gives a fast method for the Hamming-distance query. — [WWW2007 paper](https://www.www2007.cpsc.ucalgary.ca/papers/paper215.pdf); [Wikipedia SimHash](https://en.wikipedia.org/wiki/SimHash)
- **datasketch MinHashLSH**: threshold and num_perm are fixed at construction (e.g. `MinHashLSH(threshold=0.5, num_perm=128)`); more permutations means better accuracy but higher query cost; band and row split come from the threshold (e.g. 25×5 at 0.5, 9×13 at 0.8 per a third-party tutorial). — [datasketch GitHub (fork)](https://github.com/fpug/datasketch); [mojoauth tutorial](https://mojoauth.com/security-guides/lsh-in-python)
- **VersionRAG**: clusters versions by title similarity, orders them, and makes change nodes from DeepDiff line diffs between consecutive versions. Version listing goes 35% → 100% and change retrieval 25% → 70% over naive RAG. — [arXiv 2510.08109](https://arxiv.org/html/2510.08109v1)

### Inferences
- Families in our corpus are not always textual near-duplicates. The US 2023 vs 2026 proposals, or the Basel 2017 vs 2020 revision, may share only 20–50% of shingles; FY2025 vs FY2026 annual reports share boilerplate. So use a combined family key: MinHash Jaccard on 5-word shingles at a low threshold (about 0.3), or title/first-page embedding similarity above about 0.85, or the same issuer and normalized title stem with the year stripped. Union-find over those edges gives families. Thresholds are our guesses and should be tuned on the corpus.
- Containment of chunk sets flags consolidations: if more than X% of the 2017 document's shingles appear inside the 1,982-page Basel Framework but not the reverse, the larger document is a compilation. Tag it `role=consolidated` and down-weight it when a specific instrument is asked for.
- Diff-aware indexing: within a family, MinHash or exact-hash each chunk. Chunks shared across versions are "common" and get stored once with a `versions=[...]` list. Chunks unique to one version get a `delta=True` flag and higher weight for version-specific queries. When a query targets version V, prefer V's delta chunks; this directly separates near-identical versions.

### Gaps
- No published study measures MinHash family clustering accuracy on regulatory or financial versions specifically.

## Q6. Contrastive / discriminative retrieval: hard negatives and distractor benchmarks

### Takeaway
Dense retrievers are measurably vulnerable to near-duplicate distractors, and hard-negative training helps (TimeR4 contrastive time-aware training: +47.8% / +22.5% relative). In a hackathon with no labels and hours of time, fine-tuning is the least attractive lever compared with symbolic constraints. Its main value is if version-family siblings can serve as free hard negatives.

### Cited Findings
- **Near-duplicate distractors (EADPR, arXiv 2304.03031)**: an LLM generated near-duplicates of evidence passages that omit the key evidence. Dense retriever top-k accuracy fell substantially as more were added; the evidentiality-aware retriever was more robust. — [arXiv 2304.03031](https://arxiv.org/pdf/2304.03031)
- **The Power of Noise (2024)**: on NQ, answer accuracy drops progressively as distracting documents are added, even with a single one. — [arXiv 2401.14887](https://arxiv.org/pdf/2401.14887)
- **The Distracting Effect (2025)**: methods for mining hard distracting passages; fine-tuning LLMs with them gives up to +7.5% answer accuracy. — [arXiv 2505.06914](https://arxiv.org/html/2505.06914v1)
- **BrowseComp-Plus**: corpus built from hard negatives mined via web search to stress retrieval. — [arXiv 2508.06600](https://arxiv.org/pdf/2508.06600)
- **TimeR4**: contrastive time-aware retriever fine-tuning gives 47.8% / 22.5% relative gains on TKGQA. — [ACL Anthology](https://aclanthology.org/2024.emnlp-main.394)
- Guidance (benchmark doc, not peer-reviewed): build hard negatives with the same entity and relation but a different interval; dense retrieval confuses nearby temporal states. — [NanoTempReason docs](https://hakari-bench-leaderboard.hf.space/docs/benchmark-tasks/NanoRARb/NanoTempReasonL2Pure)
- MRAG shows retrievers are "distracted by dates", and stripping the temporal constraint before embedding helps (Q1). — [arXiv 2412.15540](https://arxiv.org/html/2412.15540v1)

### Inferences
- A cheap, label-free alternative to fine-tuning is to do the discrimination at query time, not at the model level. Within a version family, compute each sibling's distinctive terms (TF-IDF of family-unique n-grams: dates, docket numbers, "FY2026", amounts). Score a query against those *discriminative* terms only, which picks the sibling. This is a contrastive layer over the vector DB, not hybrid BM25 over the whole corpus.
- If time permits: auto-generate (query, gold-version chunk, sibling-version chunk) triples from family deltas and fine-tune a small bi-encoder (e.g. sentence-transformers MultipleNegativesRankingLoss) for about 20 minutes. This is the TimeR4/NanoTempReason idea, but it is unproven on our data.

### Gaps
- I found no paper reporting how much hard-negative fine-tuning specifically reduces *version* confusion between same-family documents.

---

### Recommended recipe for our system (implementable in Python in hours)
(This synthesizes the cited findings above; the thresholds are starting guesses.)

1. **Ingest and catalogue (one row per doc)**: parse PDFs (PyMuPDF/Docling), then extract fields in a cascade.
   - Filename/path tokens.
   - Regex over pages 1–2 and the TOC: years, full dates, "year ended …", FY patterns, version and "final/proposed/consultative" markers, FR citations and docket/RIN.
   - A regulator/jurisdiction gazetteer (BCBS→intl, SAMA→SA, RBI/SEBI→IN, Fed/OCC/FDIC/FR→US).
   - PDF metadata.
   - One LLM JSON call on page 1 plus the TOC to fill gaps, with an evidence span per field.
   - Fields: `issuer, entity, jurisdiction, doc_type, status(proposal/final/consolidated), issued_date, period/FY, version, page_count`.
2. **Version families**: MinHash (datasketch, 5-word shingles, 128 perms) plus title/first-page embedding similarity plus same-issuer and year-stripped title stem, joined with union-find. Within each family:
   - Order by date or version.
   - Compute asymmetric *containment* to flag `consolidated` docs (contains most of the siblings, very long).
   - Mark chunks as shared vs `delta` (VersionRAG-style diffs).
3. **Contextual chunks**: prepend a header `[issuer | jurisdiction | doc_type | status | date/FY | version]` to every chunk before embedding (Anthropic −35% failures; largest FinanceBench gain).
4. **Query constraint parsing (MRAG-style split)**: regex and gazetteer pull year/FY/date, jurisdiction, entity and version/status. The residual "main content" is what gets embedded. Use an optional LLM self-query only when the regex finds nothing.
5. **Catalogue resolution, then constrained vector search**:
   - Match constraints against the catalogue to get candidate doc IDs (soft scores, not a hard drop).
   - Run exact vector search over chunks of the candidates.
   - Score: final = cos + Σ w_i·constraint_match_i + w_delta·is_delta(target version) − w_c·is_consolidated (unless the query asks for the framework).
   - If no doc matches the constraints, fall back to unconstrained search plus a "constraint unmet" flag, to avoid the recall loss of hard filters seen on FinanceBench.
6. **Sibling discrimination**: when the top candidates come from one family, choose the version using family-unique distinctive terms and the closeness of its date to the query date (TempRALM-style temporal score). Return the chosen version plus its citation, optionally noting which siblings exist ("2020 revision supersedes 2017 on X").
7. **Evaluate**: build about 30 synthetic queries per family from the delta chunks, with the gold version known by construction. Measure version-accuracy@1, and use these to tune the weights. No labels needed, so this generalizes to the hidden corpus.
