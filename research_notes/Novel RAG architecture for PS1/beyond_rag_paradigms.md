# Beyond-standard-RAG retrieval paradigms for long financial and regulatory documents (2023-2026)

Context: CognitionNext '26 PS1. The corpus is about 35 PDFs (about 8,600 pages) of annual reports, Basel/BCBS rules, US Federal Register capital rules, CECL and quant papers. Answers and contexts are judged by an LLM, and the target is at least 0.95 on answer precision, answer recall, context precision and context recall. Notes compiled 2026-10-23.

Classification key:
- **DISTINCT**: distinct from the banned list.
- **VARIANT**: a variant of a banned technique (hybrid+rerank, self-reflection+rerank, HyDE, multi-query/RAG-Fusion, parent-document/sentence-window, CRAG, vanilla GraphRAG, agentic/ReAct loops, plain top-k).

## 1. Structure-aware and hierarchical retrieval (PageIndex, RAPTOR, HippoRAG 2, LongRAG, MemoRAG, LightRAG): results on financial and long-document benchmarks

### Takeaway
The highest financial-QA number in circulation is the vendor-reported 98.7% on FinanceBench from PageIndex (Mafin 2.5), which navigates a table-of-contents tree with LLM reasoning and uses no vectors. It has no independent verification. Academic hierarchical methods (RAPTOR, HippoRAG 2) report strong gains on general long-document and multi-hop QA, but this session found no financial-benchmark numbers for them.

### Cited Findings
**PageIndex / Mafin 2.5 (VectifyAI, 2025; DISTINCT)**
- How it works: PageIndex is "vectorless, reasoning-based RAG." It builds a hierarchical table-of-contents tree index of each document, and an LLM reasons over that tree to choose sections. It uses no chunking and no vector similarity. — [PyPI: PageIndex](https://pypi.org/project/PageIndex/)
- FinanceBench results reported by the vendor:

  | System | Accuracy | Coverage of FinanceBench |
  |---|---|---|
  | Mafin 2.5 (PageIndex) | 98.7% | 100% |
  | Fintool | 98% | 66.7% |
  | Quantly | 94% | 100% (results not public) |
  | Perplexity | 45% | 66.7% |
  | ChatGPT-4o + Search | 31% | 66.7% |
  | Mafin 1 | 38.0% | not stated |

  The same 98.7% is reported with both GPT-4o and DeepSeek-v3 as the base model. — [VectifyAI/Mafin2.5-FinanceBench](https://github.com/VectifyAI/Mafin2.5-FinanceBench)
- Evaluation setup: the FinanceBench public question set, with all documents in a single database, so this is a corpus-wide setting. Ambiguous, multi-answer or invalid questions were resolved with "expert human annotations" (a `human_evaluations` folder). — [VectifyAI/Mafin2.5-FinanceBench](https://github.com/VectifyAI/Mafin2.5-FinanceBench)
- Caveats:
  - The README gives no judging method and no date.
  - Competitor numbers are third-party and unverified.
  - The vendor itself says FinanceBench mostly tests single-document retrieval.
  - Re-adjudicating ground truth with human annotators can inflate scores.

  — [VectifyAI/Mafin2.5-FinanceBench](https://github.com/VectifyAI/Mafin2.5-FinanceBench)

**FinanceBench baselines (Patronus AI, Nov 2023)**
- GPT-4-Turbo with a retrieval system answered incorrectly or refused on 81% of questions. — [FinanceBench paper (alphaxiv 2311.11944)](https://www.alphaxiv.org/abs/2311.11944v1); [Patronus PDF](https://patronus.ai/financebench.pdf)
- Breakdown by configuration, from a secondary summary not checked against the paper's tables:
  - Shared vector store: 19% correct, 68% refused.
  - Per-document vector store: 50% correct.
  - Oracle evidence page: 85% correct.
  - Long-context feeding of the relevant filing: 79% correct.

  — [beancount.io summary](https://beancount.io/bean-labs/research-logs/2026/05/12/financebench-open-book-financial-qa-benchmark)

**RAPTOR (ICLR 2024; DISTINCT)**
- How it works: it recursively embeds, clusters (GMM) and summarizes chunks into a tree. Retrieval uses either tree traversal or a "collapsed tree," where all levels are searched together. — [arXiv 2401.18059](https://www.arxiv.org/abs/2401.18059)
- Results:
  - RAPTOR with GPT-4 improved the best QuALITY result by 20 points of absolute accuracy, reaching 82.6% on QuALITY according to Papers with Code. — [arXiv 2401.18059](https://www.arxiv.org/abs/2401.18059); [PwC](https://www.paperswithcode.com/paper/raptor-recursive-abstractive-processing-for)
  - No financial benchmark was reported.

**HippoRAG / HippoRAG 2 (NeurIPS 2024 / ICML 2025; partly VARIANT)**
- How it works: an OpenIE knowledge graph plus Personalized PageRank over passages. Because it is graph-based, it sits close to the banned "vanilla GraphRAG," but its PPR retrieval mechanism differs.
- HippoRAG v1 single-step Recall@5, compared with ColBERTv2:

  | Dataset | HippoRAG (Contriever) | ColBERTv2 |
  |---|---|---|
  | MuSiQue | 52.1 | 49.2 |
  | 2WikiMultiHopQA | 89.5 | 68.2 |
  | HotpotQA | 76.2 | 79.3 |

  — [arXiv 2405.14831](https://arxiv.org/pdf/2405.14831)
- HippoRAG 2 (arXiv 2502.14802, Feb 2025, ICML 2025) beats standard RAG on factual, sense-making and associative tasks, with about 7% over a state-of-the-art embedding model on associative memory. — [arXiv 2502.14802](https://arxiv.org/abs/2502.14802)
- A secondary source puts HippoRAG 2 at an average 59.8 F1, against 57.0 for NV-Embed-v2 with a Llama-3.3-70B reader. — [emergentmind](https://www.emergentmind.com/topics/hipporag)

**Element and structure-aware chunking for financial reports (Feb 2024; DISTINCT as an indexing layer)**
- Chunking by document element (title, paragraph, table, as detected by Unstructured) "largely improves" RAG on FinanceBench compared with fixed 128, 256 or 512-token chunks. — [arXiv 2402.05131](https://arxiv.org/abs/2402.05131v3)

**DocFinQA (ACL 2024)**
- The benchmark extends FinQA to full SEC filings, raising average context from under 700 words to about 123k words.
- A fine-tuned ColBERT was the best retriever across almost all LLMs, and GPT-3.5 reached about 42.6% accuracy. Models struggle most on the longest documents. — [arXiv 2401.06915](https://arxiv.org/html/2401.06915v1); [Liner summary](https://liner.com/review/docfinqa-longcontext-financial-reasoning-dataset)

### Inferences
**Hackathon feasibility**
- **PageIndex is the most promising.** It is open-source Python, it builds the TOC tree once per PDF with an LLM, and at query time an LLM picks nodes. The cost is one or more LLM calls per query over a compact TOC.
  - Indian and GCC annual reports and BCBS / Federal Register rules have strong heading and numbering structure (Part / Section / §217.x), so tree navigation suits this corpus. Section-level retrieval naturally gives high context recall.
  - Risk: LLM tree navigation adds latency of one or two calls per hop.
  - Feasibility: HIGH. The repo exists, and 35 documents is a small indexing job.
- **RAPTOR**: feasibility MEDIUM. It needs summarization of every cluster, which for 8,600 pages is roughly thousands of cheap LLM calls. It is better for thematic and sense-making questions than for exact numeric facts.
- **HippoRAG 2**: feasibility MEDIUM-LOW in 12 hours. OpenIE over 8,600 pages is costly. Its benefit is on multi-hop entity questions, which are likely a minority here.

**Why these methods help.** The 85% oracle versus 19% shared-store gap in FinanceBench shows that the bottleneck is *locating the right document and section*, not reading. This is exactly what structure-aware navigation and metadata routing target.

### Gaps
- No independent replication of PageIndex's 98.7% was found, and the judging protocol was not visible.
- No numbers were found for RAPTOR or HippoRAG on FinanceBench, DocFinQA or Loong.
- The following were not retrieved and verified in this session, so their reported numbers need checking against the papers:
  - LongRAG (arXiv 2406.15319): about 4K-token retrieval units with a long-context reader.
  - MemoRAG (arXiv 2409.05591): a memory model generates clues that guide retrieval.
  - LightRAG (arXiv 2410.05779): dual-level graph and vector retrieval, close to GraphRAG and so likely a VARIANT.
- No exact FinanceBench accuracy table for element-based chunking (arXiv 2402.05131) was extracted.

## 2. Late interaction (ColBERTv2, PLAID, ColPali) and learned sparse retrieval (SPLADE): precision gains and cost

### Takeaway
Late interaction gives a consistent but modest gain over single-vector dense retrieval. On DocFinQA, a fine-tuned ColBERT was the best retriever. ColPali's headline gain on visually rich pages is large on average but small on financial-report pages, where OCR-based BM25 stays competitive. Neither approach on its own gets anywhere near 0.95.

### Cited Findings
- **DocFinQA:** a fine-tuned ColBERT gave the best downstream QA across almost all LLMs tested. — [arXiv 2401.06915](https://arxiv.org/html/2401.06915v1)
- **HippoRAG baselines:** ColBERTv2 Recall@5 was 49.2 on MuSiQue, 68.2 on 2Wiki and 79.3 on HotpotQA. It is the strongest non-graph baseline there and beats HippoRAG on HotpotQA. — [arXiv 2405.14831](https://arxiv.org/pdf/2405.14831)
- **ColPali (arXiv 2407.01449, Jul 2024), headline result:** on ViDoRe v1, average nDCG@5 is about 81.3, against about 66-67 for the best OCR plus text-retrieval pipeline. Secondary sources disagree on the exact baseline figure. — [Landing AI write-up](https://landing.ai/developers/visual-document-retrieval-colpali)
- **ColPali on financial reports (TAT-DQA subset)**, from a 2026 paper:

  | Method | Score |
  |---|---|
  | ColPali | 65.8 |
  | ColPali + reranker | 70.4 |
  | OCR + BM25 | 62.7 |
  | OCR + BGE-M3 | 50.5 |

  On financial reports, the gain over text BM25 is about 3 points, not 15. — [arXiv 2609.13268](https://arxiv.org/pdf/2609.13268)
- **ViDoRe v2 (May 2025)** has harder ESG, economics and insurance subsets. colpali-v1.3 scores about 0.5 nDCG@5 on ESG reports and colqwen2.5 about 0.6 on average. — [arXiv 2505.17166](https://arxiv.org/pdf/2505.17166)

### Inferences
**Classification.** ColBERT, PLAID and SPLADE are *retriever scoring functions*, not architectures. Used as the first stage in "retrieve top-k then rerank," they are a VARIANT of plain top-k or hybrid retrieval. Used as a component inside a novel layer, for example token-level MaxSim to score the nodes of a structure tree, they are acceptable.

**Hackathon feasibility**
- ColBERTv2 via RAGatouille or PyLate is HIGH feasibility on CPU or GPU for about 8,600 pages.
- ColPali needs a GPU and page-image embedding: about 8,600 pages × about 1,030 patch vectors each. That gives MEDIUM feasibility and a large index.
- ColPali's main value is for charts and scanned tables in annual reports.

### Gaps
- Verified ColBERTv2 vs dense vs SPLADE numbers on BEIR-FiQA were not retrieved in this session. Gains in published BEIR tables are typically a few nDCG points, but this is unverified here.
- PLAID latency and storage figures were not retrieved. The papers to check are PLAID (arXiv 2205.09707) and ColBERTv2 (arXiv 2112.01488).
- SPLADE-v3 numbers on financial data were not found.

## 3. Long-context LLMs vs RAG (Self-Route, OP-RAG, LaRA): when whole documents or sections win, and at what cost

### Takeaway
With a capable model, feeding the long context beats chunk RAG on average. A *router* (Self-Route) keeps nearly the same accuracy at 38-61% of the token cost. Order-preserving retrieval of a moderate number of chunks (OP-RAG) can beat full long context outright. LaRA (ICML 2025) concludes that no single strategy wins across tasks.

### Cited Findings
**Li et al., "RAG or Long-Context LLMs?" (Google DeepMind / UMich, Jul 2024, EMNLP Industry 2024)**
- Average scores (Table 1, Contriever retriever):

  | Model | Long context | RAG | Self-Route |
  |---|---|---|---|
  | GPT-4o | 48.67 | 32.60 | 48.89 |
  | Gemini-1.5-Pro | 49.70 | 37.33 | 46.41 |
  | GPT-3.5-Turbo | 32.07 | 30.33 | 35.32 |

  — [arXiv 2407.16833](https://arxiv.org/html/2407.16833v1)
- Self-Route uses 38.39% of long-context tokens with Gemini-1.5-Pro and 61.40% with GPT-4o, cost reductions of 65% and 39%. — [arXiv 2407.16833](https://arxiv.org/html/2407.16833v1)
- RAG and long context produce identical predictions on 63% of queries, which is why routing works. — [arXiv 2407.16833](https://arxiv.org/html/2407.16833v1)
- RAG failure causes: multi-step reasoning, general queries, long or complex queries, and implicit queries. — [arXiv 2407.16833](https://arxiv.org/html/2407.16833v1)
- The paper has internal inconsistencies between text and tables. For example, the text states gaps of 13.1% and 7.6%, while the table implies 16.1 and 12.4. — [arXiv 2407.16833](https://arxiv.org/html/2407.16833v1)

**OP-RAG, "In Defense of RAG in the Era of Long-Context LMs" (NVIDIA, Sep 2024)**
- How it works: retrieved chunks are kept in *original document order*, not relevance order. Quality follows an inverted U as k grows. — [arXiv 2409.01666](https://arxiv.org/html/2409.01666v1)
- En.QA (∞Bench) F1 with Llama-3.1-70B:

  | Setup | Tokens | F1 |
  |---|---|---|
  | OP-RAG | 16K | 44.43 |
  | OP-RAG | 48K | 47.25 |
  | No RAG, full context | about 117-128K | about 34.3 |

  — [The Decoder](https://the-decoder.com/?p=17461); [SuperAnnotate](https://www.superannotate.com/blog/rag-vs-long-context-llms)
- On the same task, GPT-4o long context scored 32.36 F1 and Gemini-1.5-Pro scored 43.08. These figures are secondary-sourced. — [The Decoder](https://the-decoder.com/?p=17461)

**LaRA (ICML 2025)**
- 2,326 test cases, 4 task types, 3 context types and 11 LLMs.
- The best choice between RAG and long context depends on model strength, context length, task type and retrieval characteristics. There is "no silver bullet." Code is at github.com/Alibaba-NLP/LaRA. — [PMLR v267](https://proceedings.mlr.press/v267/li25dv.html); [arXiv 2502.09977](https://arxiv.org/html/2502.09977v1)

**FinanceBench**
- Long-context feeding gave GPT-4-Turbo 79%, close to the 85% oracle and far above shared-store RAG at 19%. These figures are secondary-sourced. — [beancount.io](https://beancount.io/bean-labs/research-logs/2026/05/12/financebench-open-book-financial-qa-benchmark)

### Inferences
**Classification**
- Self-Route is a router, not a retrieval loop: DISTINCT, though it uses a self-reflection signal. Pairing it with reranking would drift toward the banned "self-reflection + reranking."
- OP-RAG is a small change to top-k (ordering): mostly a VARIANT on its own.
- Section-level long-context reading after structural navigation is DISTINCT.

**Design implication.** Whole documents of 200-400 pages exceed what is cheap per query. Navigating to the correct *section* (10-40 pages) and then reading it in document order captures most of the long-context advantage at a fraction of the cost. This combines PageIndex-style navigation with OP-RAG-style ordered context.

**Feasibility:** HIGH. Self-Route is one extra LLM call, and OP-RAG is a sort.

### Gaps
- LaRA's per-task numbers were not retrieved.
- No study combining structure-navigation with section-level long context on FinanceBench was found.

## 4. Retrieval as classification or set selection (Adaptive-k, adaptive retrieval, whole-section retrieval)

### Takeaway
Choosing *how many* passages to keep from the score distribution (Adaptive-k, EMNLP 2025) cuts tokens by up to 10x at equal accuracy. Gap-based cutoffs miss secondary evidence on multi-hop questions (TAA-k, 2026). Context precision near 0.95 requires a set-selection or cutoff step rather than a fixed k.

### Cited Findings
- **Adaptive-k** (Taguchi, Maekawa, Bhutani; EMNLP 2025; arXiv 2506.08479):
  - How it works: a single pass picks k at the largest gap in the sorted query-passage similarity distribution. It needs no tuning and no extra LLM calls.
  - Results: it matches or beats fixed-k on factoid and aggregation QA with up to 10x fewer tokens than full context, while still retrieving about 70% of relevant passages.

  — [arXiv 2506.08479](https://arxiv.org/pdf/2506.08479); [ACL Anthology](https://preview.aclanthology.org/setup/2025.emnlp-main.1017)
- **Tail-Aware Adaptive-k (TAA-k)** (arXiv 2606.11907, Jun 2026): argues gap heuristics are "overly aggressive in multi-hop tasks, failing to capture secondary evidence," and reports better recall. — [arXiv 2606.11907](https://arxiv.org/pdf/2606.11907)
- **Cluster-based Adaptive Retrieval (CAR)** (arXiv 2511.14769, Nov 2025): sets the cutoff at cluster boundaries in the ordered distance list. — [arXiv 2511.14769](https://arxiv.org/pdf/2511.14769)
- **"From Ranking to Selection"** (arXiv 2508.09497, Aug 2025) reframes retrieval as selecting a set rather than ranking. — [arXiv 2508.09497](https://arxiv.org/html/2508.09497v1)

### Inferences
**Classification.** Adaptive cutoff and set selection are DISTINCT: a post-scoring layer that changes the output from top-k to a set. An LLM per-passage relevance classifier is close to reranking and could be judged a VARIANT unless it is framed as a binary set filter over structurally selected candidates.

**Fit with the scoring metrics.** The 70% passage recall of Adaptive-k is far below 0.95. Use it for *precision* trimming only after a high-recall candidate stage, such as section navigation.

**Feasibility:** HIGH. It is about 20 lines of numpy.

### Gaps
- Adaptive-RAG (Jeong et al., NAACL 2024, arXiv 2403.14403) routes queries by complexity. Its numbers were not retrieved in this session.
- No financial-domain evaluation of adaptive-k was found.
- Exact TAA-k recall numbers were not extracted.

## 5. Metadata and attribute filtering, self-querying and structured query planning before similarity search

### Takeaway
On FinanceBench, metadata in front of the search (file filtering plus query rewriting) and metadata attached to the chunk ("contextual chunks") give the largest gains in a 2025 study, though F1 stays in the 40s with a reranked hybrid backbone. Anthropic's contextual retrieval cuts top-20 retrieval failure by 35-67%.

### Cited Findings
- **Metadata-driven RAG for financial QA** (Dadopoulos et al., arXiv 2510.24402, Oct 2025), FinanceBench F1:

  | Configuration | F1 |
  |---|---|
  | Naive RAG | 32.9 |
  | Hybrid + Cohere rerank | 38.9 |
  | Hybrid + Cohere rerank, with contextual chunks | 44.1 |
  | Filtering + rewriting + hybrid + rerank, with contextual chunks | 44.4 (best) |
  | Proposed custom metadata reranker | 43.2 |

  Metadata-driven chunk expansion *hurt*: hallucination rose from 14.7% to 22.2%. — [alphaxiv 2510.24402](https://alphaxiv.org/abs/2510.24402v1)
- **Anthropic Contextual Retrieval (Sep 2024)**, top-20 retrieval failure rate:

  | Setup | Failure rate | Reduction |
  |---|---|---|
  | Baseline | 5.7% | n/a |
  | Contextual embeddings | 3.7% | 35% |
  | Contextual embeddings + contextual BM25 | 2.9% | 49% |
  | Plus reranking | 1.9% | 67% |

  — [Maginative](https://www.maginative.com/article/anthropics-contextual-retrieval-technique-enhances-rag-accuracy-by-67); [Gradient Flow](https://gradientflow.com/contextual-retrieval-rag/)
- Gains vary by domain, from 84% to 41% depending on content type. — [Almond AI critique](https://medium.com/almond-ai/responding-to-anthropics-contextual-retrieval-for-rag-apps-why-context-is-not-all-you-need-af503985aa55)
- Contextualization costs about $1.02 per million document tokens with prompt caching. — [getcoai](https://getcoai.com/news/anthropic-introduces-contextual-retrieval-to-boost-accuracy-of-rag-systems)

### Inferences
**Classification**
- Metadata pre-filtering and self-querying (LLM produces a filter for entity, fiscal year, document type or regulation section, and then search runs) is DISTINCT. It is a new planning layer that narrows the search space.
- Contextual chunking is an indexing enrichment. Combined with hybrid + rerank as the core it is a VARIANT, but as an indexing layer under a different core it is acceptable.

**Corpus fit.** With 35 documents, a document-level router (which bank, which year, which regulation) is almost trivially accurate. That removes the biggest failure seen in FinanceBench, the 19% shared-store vs 50% per-document gap.

**Feasibility:** HIGH. Extract metadata per PDF once, then have the LLM emit a JSON filter.

### Gaps
- No public study pairs metadata routing with structural tree navigation on financial documents.
- Per-configuration context precision and recall from 2510.24402 were not available.

## 6. Highest published numbers on financial-document QA, and what made them work

### Takeaway
The top reported FinanceBench numbers, 94-98.7% (Mafin 2.5/PageIndex, Fintool, Quantly), all come from vendors. Their common ingredients are document-level scoping, structure- or section-level navigation instead of chunk similarity, and strong LLM readers over whole sections. The best academic chunk-RAG pipelines on FinanceBench stay around 40-50%. The single-page oracle ceiling was 85% in 2023.

### Cited Findings
- Vendor-reported FinanceBench scores:

  | System | Accuracy | Coverage |
  |---|---|---|
  | Mafin 2.5 (PageIndex) | 98.7% | full benchmark |
  | Fintool | 98% | 66.7% |
  | Quantly | 94% | full benchmark |

  All are unverified third-party or vendor numbers. — [VectifyAI/Mafin2.5-FinanceBench](https://github.com/VectifyAI/Mafin2.5-FinanceBench)
- Academic and baseline references:
  - Shared-store GPT-4-Turbo: 19% correct. Oracle: 85%. Long context: 79%. — [FinanceBench](https://www.alphaxiv.org/abs/2311.11944v1); [beancount.io](https://beancount.io/bean-labs/research-logs/2026/05/12/financebench-open-book-financial-qa-benchmark)
  - Best hybrid + rerank + metadata pipeline: F1 44.4. — [arXiv 2510.24402](https://alphaxiv.org/abs/2510.24402v1)
  - DocFinQA best: about 42.6% (GPT-3.5 with fine-tuned ColBERT). — [Liner](https://liner.com/review/docfinqa-longcontext-financial-reasoning-dataset)

### Inferences
**A candidate architecture, consistent with the evidence.** It is distinct from the banned list, and every stage is buildable in 12 hours in Python:
1. **Metadata router** (DISTINCT): picks the document(s) by entity, year or regulation.
2. **PageIndex-style TOC-tree navigation** (DISTINCT): the LLM selects section nodes. An optional late-interaction score can act as a prior on the nodes.
3. **Adaptive set selection** within the selected sections (DISTINCT): gives precision.
4. **Ordered section-level context to the reader** (OP-RAG ordering), with a Self-Route-style fallback to wider sections when the reader signals insufficiency.

**Expected strengths and risks**
- Section-level retrieval should maximize context recall.
- Context precision is the main risk, because whole sections contain irrelevant text. Mitigate it with paragraph-level set filtering inside the selected sections.
- Vendor 98.7% numbers were achieved with human-adjudicated ground truth. A 0.95 target on LLM-judged context precision is likely harder than the answer-accuracy figures suggest.

### Gaps
- No peer-reviewed system reports at least 0.95 on FinanceBench, DocFinQA, FinQA or TAT-QA under RAGAS-style context precision and recall. Top TAT-QA and FinQA leaderboard numbers were not retrieved.
- No published Loong or LongBench numbers for PageIndex or RAPTOR were found.
- Cost and latency per query for PageIndex were not published in the sources reviewed.
