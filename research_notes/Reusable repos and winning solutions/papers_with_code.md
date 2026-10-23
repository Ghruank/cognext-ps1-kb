# Papers with released code for complete, precise evidence sets and judge-friendly long-form answers (2023-2026)

Notes compiled 2026-10-23. Repo metadata (last push, license, stars) was read from the GitHub API that day. "Last push" means the repo's `pushed_at` date. Numbers marked "(not re-verified)" come from prior knowledge of the paper, not from a source fetched during this session. Treat them as leads to check.

## Q1. High-recall / exhaustive retrieval for report generation (TREC NeuCLIR RG, ARGUE, nuggets)

### Takeaway
The nugget ecosystem is open-source and maintained: castorini/nuggetizer (AutoNuggetizer), hltcoe/auto-argue (ARGUE), GINGER and Crucible (nugget-first generation). For us, the most reusable idea is Crucible/GINGER-style generation. Extract atomic nuggets from evidence, deduplicate them, then write one cited sentence per nugget. That structure matches a judge that scores atomic points. The evaluators also let us build a local proxy for the hackathon judge.

### Cited Findings
- **AutoNuggetizer** (Pradeep, Lin et al., Waterloo). An LLM creates nuggets and assigns them to answers. It was used for the TREC 2024 RAG Track. Over 21 topics and 45 runs, fully automatic scores correlated strongly with mostly-manual nugget evaluation. Automating only the assignment step, with human-made nuggets, agreed even better with the manual baseline. — [arXiv 2411.09607](https://arxiv.org/abs/2411.09607v1); full paper "The Great Nugget Recall" [arXiv 2504.15068](https://arxiv.org/abs/2504.15068)
  - Code: https://github.com/castorini/nuggetizer. Apache-2.0, last push 2026-04-19, PyPI `nuggetizer` v0.0.6. It supports OpenAI-compatible and Azure endpoints. Adoption: easy, `pip install` and point it at the deepseek endpoint. — [GitHub](https://github.com/castorini/nuggetizer)
  - Companion RAG baseline framework: https://github.com/castorini/ragnarok. Apache-2.0, last push 2026-04-18. — [GitHub](https://github.com/castorini/ragnarok)
- **Auto-ARGUE** (Walden, Yang et al., JHU HLTCOE). An LLM implementation of the ARGUE report-evaluation framework, described as the only public ARGUE implementation. On the 51 runs of the TREC 2024 NeuCLIR RG pilot, with Llama-3.3-70B as judge, it showed good system-ranking correlation with humans on sentence precision and nugget recall. A Streamlit viewer, ARGUE-viz, is also available. — [arXiv 2509.26184](https://arxiv.org/abs/2509.26184v4)
  - Code: https://github.com/hltcoe/auto-argue. MIT, last push 2025-10-16 (GitHub search showed an update on 2026-10-15). Run-format validator: https://github.com/hltcoe/rag-run-validator (Apache-2.0, last push 2026-07-14). — [GitHub](https://github.com/hltcoe/auto-argue)
- **NeuCLIR 2024 RG facts.** There were 51 RG runs from 4 teams. Citation precision is the share of citations pointing to a relevant document. Nugget recall is the share of nugget questions answered by at least one sentence. — [TREC 2024 NeuCLIR overview, arXiv 2509.14355](https://arxiv.org/pdf/2509.14355)
  - In the HLTCOE Chinese table, run `hltcoe-eugene-gpt4o-fixed` scored ARGUE 0.726, citation precision 0.859 and nugget recall 0.327. Run `jhu-orion-aggregated-w-claude` scored citation precision 0.900 but nugget recall 0.177. Precision and recall clearly trade off, and nugget recall is low across the board. — [HLTCOE at TREC 2024 NeuCLIR, arXiv 2510.00143](https://arxiv.org/pdf/2510.00143)
- **GINGER** (UiS-IAI, 2025). The pipeline runs nugget detection, BERTopic clustering by query facet, DuoT5 pairwise ranking of clusters, a one-sentence summary per cluster, and a fluency rewrite that adds no content. It reports state-of-the-art results on TREC RAG'24 under AutoNuggetizer. — [arXiv 2503.18174](https://arxiv.org/pdf/2503.18174); LiveRAG version [arXiv 2506.22210](https://arxiv.org/html/2506.22210v1)
  - Code: https://github.com/iai-group/ginger-response-generation. No license file, last push 2025-08-27, 4 stars. — [GitHub](https://github.com/iai-group/ginger-response-generation)
- **Crucible** (2026). A "Nugget-Augmented Generation" system. It builds a bank of Q&A nuggets from retrieved documents and uses it to guide extraction, selection and report generation, keeping citation provenance throughout. On NeuCLIR 2024 it substantially outperforms GINGER on nugget recall, density and citation grounding. — [arXiv 2601.13222](https://arxiv.org/pdf/2601.13222)
  - Code: none found this session.
- **DoGMaTiQ** (2026) generates Q&A nugget sets in three stages: document-grounded generation, paraphrase clustering, and quality-based subselection. It integrates with AutoArgue. — [arXiv 2605.04458](https://arxiv.org/pdf/2605.04458)
- **CoverageBench** (2026) is a suite of test collections for measuring information coverage of retrieval, with topics, nuggets, relevance labels and baseline rankings released. It argues that high recall is sufficient but not necessary for coverage. — [arXiv 2603.20034](https://arxiv.org/pdf/2603.20034)

### Inferences
- Our judge, with atomic points vs. gold and extra points penalised, is essentially ARGUE- or AutoNuggetizer-style. Nuggetizer or Auto-ARGUE pointed at deepseek-v4.1-flash would give a cheap local proxy for answer precision and recall. Gold nuggets can be built from our own best answers or from a few hand-checked ones.
- Crucible's Q&A-nugget bank, followed by one cited sentence per nugget, fits "extra points count against us" well. Every sentence carries exactly one supported point, and redundancy is removed before writing.
- NeuCLIR results show citation precision near 0.9 but nugget recall around 0.2-0.35. Even the best published report systems are far from 0.9 recall on open collections.

### Gaps
- I did not confirm whether Crucible's code is public.
- I did not extract the full per-language NeuCLIR tables (Tables 13-15 of 2509.14355).
- I found no TREC system that explicitly trains its retriever to optimise nugget recall.

## Q2. LLM-as-retriever and long-context evidence selection (PageIndex, RAG-Challenge-2, Self-Route, OP-RAG, LongRAG, SetR)

### Takeaway
Reading whole sections with an LLM and marking relevant paragraph IDs is well supported by repos (PageIndex, RAG-Challenge-2 LLM reranking, SetR set selection). Reasoning-based set selection (SetR) beats per-passage reranking on coverage. Keeping retrieved chunks in document order (OP-RAG) is a free win. Self-Route's recall collapses as the amount of relevant information grows, which is exactly our case (median 81 gold paragraphs). Skip it.

### Cited Findings
- **PageIndex** (VectifyAI). A "vectorless, reasoning-based RAG" document index: an LLM navigates a tree index (table of contents) of a long document. MIT, last push 2026-10-23, about 39.5k stars. An MCP server exists at https://github.com/VectifyAI/pageindex-mcp. — [GitHub](https://github.com/VectifyAI/PageIndex)
- **RAG-Challenge-2 winner** (IlyaRice). The pipeline uses Docling PDF parsing (on a 4090 GPU), vector search with parent-document (page) retrieval, LLM reranking for context relevance, chain-of-thought structured-output answering, and query routing for multi-company comparisons. The best config used o3-mini. The README claims it won both prize nominations. MIT, about 2.5k stars. — [GitHub](https://github.com/IlyaRice/RAG-Challenge-2)
- **SetR** ("Shifting from Ranking to Set Selection for RAG", ACL 2025, LG AI Research). Chain-of-thought first works out the query's information requirements, then the model selects a passage set that covers them. On multi-hop RAG benchmarks it beats proprietary LLM rerankers and open baselines on answer correctness and retrieval quality. — [arXiv 2507.06838](https://arxiv.org/html/2507.06838v2)
  - Code: https://github.com/LGAI-Research/SetR. BSD-3-Clause, last push 2025-09-11. — [GitHub](https://github.com/LGAI-Research/SetR)
- **Self-Route** (Li et al., Google DeepMind, 2024). The model routes each query to RAG or full long context by self-reflection. This cut cost by 65% for Gemini-1.5-Pro and 39% for GPT-4o. LC and RAG gave identical predictions for over 60% of queries. — [arXiv 2407.16833](https://arxiv.org/html/2407.16833v1)
  - On HoloBench aggregation QA, Self-Route's context recall dropped from 65.79 to 21.54 as the relevant information grew from 5k to 50k tokens. — [Adaptive-k paper, arXiv 2506.08479](https://arxiv.org/html/2506.08479)
  - No official code found.
- **OP-RAG** ("In Defense of RAG", 2024). Retrieved chunks are kept in their original document order. Answer quality follows an inverted U as more chunks are added, so there is a sweet spot that beats full long-context input with far fewer tokens. — [arXiv 2409.01666](https://arxiv.org/html/2409.01666v1)
  - On EN.QA, one secondary source reports Llama3.1-70B at 34.26 F1 using about 117k tokens of full context, versus OP-RAG at 47.25 F1 using 48k tokens. — [MLnotes](https://mlnotes.substack.com/p/whats-order-preserve-rag-op-rag)
  - The Decoder gives different figures: 44.43 F1 at 16k tokens versus 34.32. The two secondary sources conflict. — [The Decoder](https://the-decoder.com/?p=17461)
  - No official code found. The method is a few lines to implement.
- **LongRAG** (TIGER-AI-Lab, 2024). Uses long retrieval units (grouped documents) with a long-context reader. MIT, last push 2024-08-25. — [GitHub](https://github.com/TIGER-AI-Lab/LongRAG)
  - From prior knowledge (not re-verified): it uses about 4K-token units and raised NQ answer recall@1 from about 52% to about 71%.

### Inferences
- Cost estimate for us: about 8,600 pages is roughly 4-5M tokens. Having the LLM read every page of only the 1-3 target documents per question (routed by metadata or ToC, PageIndex style) is about 100-400 pages, or 50-200k input tokens per question. That is cheap on a flash model. We can chunk by section, ask for paragraph IDs with a one-line reason each, and parallelise.
  - This likely gives higher recall than any embedding top-k on "every red flag" questions. Precision depends on the prompt.
  - It is a filter/classifier pattern, not a ReAct loop, so it is not banned.
- Use SetR's prompt structure: first list the information requirements (the facets of "credit-risk red flags"), then select paragraphs per requirement. It maps directly onto the judge's point-by-point recall.
- Always pass selected paragraphs in document order (OP-RAG).

### Gaps
- I found no published cost/recall curve for LLM-reads-every-section paragraph tagging on financial reports.
- I did not verify OP-RAG's exact numbers from the primary PDF.
- I did not confirm whether Self-Route or OP-RAG have official repos. I believe neither does.

## Q3. Learned paragraph-relevance classifiers with few labels; SPLADE-v3, ColBERTv2/RAGatouille, CPU rerankers

### Takeaway
Every component has maintained, permissively licensed code, but these are the banned "hybrid+rerank" ingredients when used as the core. Use them only for candidate generation and recall backstops. SetFit is the cheapest route to a paragraph-relevance classifier trained on a few dozen LLM-labelled paragraphs. Note the SPLADE license caveat.

### Cited Findings
- **SetFit** (Hugging Face). Few-shot classification on Sentence Transformers. Apache-2.0, last push 2026-10-06, about 2.9k stars. It is CPU-trainable. — [GitHub](https://github.com/huggingface/setfit)
- **SPLADE** (NAVER). License "NOASSERTION" on GitHub (custom; the model weights are commonly CC-BY-NC, check before use). Last push 2024-05-03. — [GitHub](https://github.com/naver/splade)
- **ColBERT** (Stanford). MIT, last push 2025-10-14, about 3.9k stars. — [GitHub](https://github.com/stanford-futuredata/ColBERT)
- **RAGatouille**. Apache-2.0, last push 2025-05-17. Pushes have slowed. — [GitHub](https://github.com/AnswerDotAI/RAGatouille)
- **PyLate** (LightOn). Late-interaction training and retrieval, MIT, last push 2026-07-23. It is the more active ColBERT-style library. — [GitHub](https://github.com/lightonai/pylate)
- **FlashRank**. CPU-oriented ultra-light rerankers, Apache-2.0, last update 2026-10-22, about 1k stars. — [GitHub](https://github.com/PrithivirajDamodaran/FlashRank)
- **rerankers** (AnswerDotAI). Unified API over cross-encoders, ColBERT, FlashRank, LLM rerankers and others. Apache-2.0, last push 2025-12-20. — [GitHub](https://github.com/AnswerDotAI/rerankers)
- **mxbai-rerank**. Apache-2.0, last push 2025-09-17. — [GitHub](https://github.com/mixedbread-ai/mxbai-rerank)
- **bge-reranker-v2-m3** lives in FlagEmbedding. MIT, last push 2026-08-24, about 12.3k stars. — [GitHub](https://github.com/FlagOpen/FlagEmbedding)

### Inferences
- Recommended use: label about 300-1,000 paragraphs with the LLM pass from Q2 on a few documents. Train a SetFit or logistic-regression head on bge-m3 embeddings per question type, then use it to pre-filter or backstop paragraphs the LLM never saw. This is cheap on CPU.
- bge-reranker-v2-m3 (568M parameters) is slow on CPU over thousands of paragraphs. FlashRank's MiniLM-class models are fast but weaker. This is inferred from model sizes, not benchmarked here.

### Gaps
- I did not fetch SPLADE-v3's paper or Hugging Face license page, or CPU latency benchmarks for these rerankers.
- I found no paper on few-label paragraph classifiers specifically for long financial reports.

## Q4. Set selection / adaptive-k / conformal retrieval

### Takeaway
Two pieces are directly reusable. Adaptive-k has code (BSD-3) and is a tiny largest-gap cutoff. JHU's conformal filter needs only a calibration set and gives a statistical recall target. On aggregation tasks, Adaptive-k's recall plateaus at about 70%, so it cannot reach 0.9 alone.

### Cited Findings
- **Adaptive-k** (Taguchi, Maekawa, Bhutani, EMNLP 2025). The method sorts query-chunk cosine similarities, cuts at the largest adjacent drop within the top 90% of chunks, then adds a buffer of 5 chunks. It needs no tuning and no extra LLM calls. — [arXiv 2506.08479](https://arxiv.org/html/2506.08479)
  - On HoloBench it keeps context recall at about 70% across information amounts: 75.74, 68.54, 66.16 and 67.43 at 5k, 10k, 25k and 50k tokens. — [arXiv 2506.08479](https://arxiv.org/html/2506.08479)
  - It uses 2-10x fewer tokens than full context on aggregation QA, and is up to 9 points better than Self-Route on high-information tasks. — [arXiv 2506.08479](https://arxiv.org/html/2506.08479)
  - Code: https://github.com/megagonlabs/adaptive-k-retrieval. BSD-3-Clause, last push 2025-09-15. — [GitHub](https://github.com/megagonlabs/adaptive-k-retrieval)
- **Conformal context filtering** ("Principled Context Engineering for RAG", Chakraborty, Yang, Khashabi, Lawrie, Duh, JHU, Nov 2025).
  - Method: split conformal prediction right after retrieval. Snippets are 500-character windows with 100-character overlap. The threshold is the (1-α) quantile of nonconformity scores of relevant snippets in a calibration set. Scorers are 1 - cosine similarity (Qwen3-Embedding-8B) or 1 - an LLM 0-1 relevance rating. — [arXiv 2511.17908](https://arxiv.org/html/2511.17908v1)
  - Coverage: target coverage is met for α from 0.05 to 0.40, and context shrinks 2-3x. — [arXiv 2511.17908](https://arxiv.org/html/2511.17908v1)
  - NeuCLIR answer quality: ARGUE F1 went from 0.69 unfiltered to 0.720 with Conformal-Embedding at α=0.05, which removed 22.2% of snippets. Conformal-LLM removed 46.5% at α=0.05 and scored 0.710. — [arXiv 2511.17908](https://arxiv.org/html/2511.17908v1)
  - Code: none found.
- **Conformal-RAG / Conditional Conformal Factuality** (SIGIR 2025). Conformal filtering of answer sub-claims, which retains about 60% more high-quality sub-claims than applying conformal prediction directly to the LLM. Code: https://github.com/n4feng/ResponseQualityAssessment (no license, last push 2025-07-01, 0 stars). — [arXiv 2506.20978](https://arxiv.org/abs/2506.20978); [GitHub](https://github.com/n4feng/ResponseQualityAssessment)
- **SetR** (see Q2) is the LLM-based set-selection option. — [GitHub](https://github.com/LGAI-Research/SetR)

### Inferences
- The conformal recipe fits our metric directly. If we hand-label about 10-20 calibration questions' paragraphs, we can pick α to hit a target such as 0.9 paragraph recall and know the precision cost in advance. With LLM-rating scores, our LLM tagging pass becomes the scorer.
- Calibrating on our own LLM-labelled silver data is weaker than the paper's setup, which used Llama-70B labels. It is still principled.

### Gaps
- No code is released for 2511.17908.
- I found no conformal method that controls precision and recall jointly at the paragraph level.

## Q5. Version-aware retrieval

### Takeaway
VersionRAG has public code (no license file). It models versions as a hierarchical graph and filters by version validity. The idea is easy to reimplement: attach doc_id, year and version metadata, filter before selection, and add an explicit "what changed" path.

### Cited Findings
- **VersionRAG** (Huwiler, Stockinger, Fürst, ZHAW, Oct 2025). Version histories are modelled as a hierarchical graph so retrieval can filter by version validity. On VersionQA (100 questions, 34 versioned technical docs) it reaches 90% accuracy, versus 58% for naive RAG and 64% for GraphRAG. It scores 60% on implicit change detection, where baselines score 0-10%. Indexing uses 97% fewer tokens than GraphRAG. — [arXiv 2510.08109](https://arxiv.org/pdf/2510.08109)
  - Code: https://github.com/danielhuwiler/versionrag. No license, last push 2025-10-03, 27 stars. — [GitHub](https://github.com/danielhuwiler/versionrag)

### Inferences
- With no license, borrow the design (version graph plus validity filter plus change-diff node), not the code.
- For "Basel vs. the 2026 US proposal" questions, a per-document version tag plus routing to both documents covers most of the need. Pre-computed diffs between near-duplicate consolidated texts handle "what changed".

### Gaps
- I did not examine other temporal-RAG repos.

## Q6. Citation-grounded long-form generation; prompts that maximise RAGChecker / FActScore precision and recall

### Takeaway
ALCE (evaluation), Attribute-First-then-Generate, FRONT and LongCite all have code. The shared winning pattern is to extract or select supporting quotes first, then generate conditioned only on them. RAGChecker and FActScore give claim-level precision and recall scorers that closely mirror our judge.

### Cited Findings
- **ALCE** (Princeton, EMNLP 2023). A benchmark and evaluation for generating text with citations. MIT, last push 2024-10-09. — [GitHub](https://github.com/princeton-nlp/ALCE)
- **Attribute First, then Generate** (ACL 2024). Content is selected and attributed first, then text is generated from it, so every sentence is locally attributable. Apache-2.0, last push 2024-12-19. — [GitHub](https://github.com/lovodkin93/attribute-first-then-generate)
- **FRONT** (ACL 2024 Findings). Trains LLMs to extract fine-grained supporting quotes first, then answer from them. On ALCE, FRONT on LLaMA-2-7B reports an average 14.21% citation-quality gain and beats ChatGPT. — [arXiv 2408.04568](https://arxiv.org/pdf/2408.04568)
  - Code: https://github.com/LuckyyySTA/Fine-grained-Attribution. No license, last push 2024-10-24. — [GitHub](https://github.com/LuckyyySTA/Fine-grained-Attribution)
- **LongCite** (THUDM, 2024). The CoF (coarse-to-fine) pipeline builds sentence-level-citation SFT data (LongCite-45k). LongCite-8B and 9B reach state-of-the-art citation quality on LongBench-Cite, surpassing advanced proprietary models. Apache-2.0, last push 2024-12-31. — [arXiv 2409.02897](https://arxiv.org/abs/2409.02897); [GitHub](https://github.com/THUDM/LongCite)
- **RAGChecker** (Amazon). Claim-level diagnosis of RAG: precision and recall of answer claims vs. ground truth, plus context-utilisation metrics. Apache-2.0, last push 2024-12-13, about 1.1k stars. — [GitHub](https://github.com/amazon-science/RAGChecker)
- **FActScore** (EMNLP 2023). Atomic-fact precision for long-form text. MIT, last push 2025-04-13. — [GitHub](https://github.com/shmsw25/FActScore)

### Inferences
- Prompt pattern to adopt: quote first, then nugget, then sentence.
  1. Extract verbatim spans with paragraph IDs.
  2. Turn them into a deduplicated list of atomic claims.
  3. Drop claims not entailed by a quote.
  4. Write one sentence per claim with a citation and no connective commentary.
- This pattern maximises claim precision, because the judge penalises extra points. Recall comes from the exhaustive evidence set, not from the writer.
- RAGChecker's claim precision and recall is the closest open proxy to the hackathon judge. Run it with deepseek as the extractor and checker on a small dev set.
- LongCite's trained models need a GPU. Only the CoF prompts and the LongBench-Cite eval are reusable on our CPU setup.

### Gaps
- I did not extract the exact prompt templates from the RAGChecker, ALCE or LongCite repos this session.
- I found no study that tunes report prompts specifically against RAGChecker scores.

## Q7. Work reporting at or near 0.9+ on both precision and recall of evidence selection

### Takeaway
I found no paper with code that reports 0.9+ on both precision and recall of evidence or paragraph selection over long, unlabeled document collections. The closest results each hit 0.9 on one axis only. The pattern that reaches high numbers is tight scoping (right document or version first), then calibrated or LLM-judged filtering.

### Cited Findings
- Conformal filtering meets user-set coverage targets (for example 95% at α=0.05) while removing 22-70% of snippets. That gives high recall by construction, but precision is not reported at 0.9. — [arXiv 2511.17908](https://arxiv.org/html/2511.17908v1)
- NeuCLIR systems reach citation precision of 0.86-0.90, but nugget recall is only 0.18-0.33. — [arXiv 2510.00143](https://arxiv.org/pdf/2510.00143)
- Adaptive-k plateaus at about 70% context recall on aggregation tasks. — [arXiv 2506.08479](https://arxiv.org/html/2506.08479)
- VersionRAG's 90% is answer accuracy, not evidence precision or recall. — [arXiv 2510.08109](https://arxiv.org/pdf/2510.08109)

### Inferences
- Scoring 0.9/0.9 on our context metrics most plausibly needs three things:
  - (a) Document- and version-level routing, so the candidate pool is one or a few documents.
  - (b) Exhaustive LLM reading of candidate sections with requirement-guided (SetR-style) paragraph tagging, for recall.
  - (c) A second calibrated precision filter (conformal threshold on LLM relevance), plus document-order presentation.
- Gold sets cluster in about 5 page ranges per document. Expanding selected paragraphs to their contiguous neighbours within a section (span smoothing) probably boosts recall cheaply. This is an inference, not sourced.

### Gaps
- I found no benchmark with gold sets of 80-350 paragraphs per query like ours. Published recall figures come from much smaller relevant sets.

## Ranked adopt / adapt / skip list for our system

All entries are based on the findings above.

**ADOPT (use the code directly)**
1. **castorini/nuggetizer** (Apache-2.0). Local judge proxy for answer precision and recall. Easy.
2. **megagonlabs/adaptive-k-retrieval** (BSD-3). Largest-gap cutoff as a cheap set-size heuristic and fallback. Trivial.
3. **amazon-science/RAGChecker** (Apache-2.0). Claim-level precision, recall and context-utilisation diagnostics on a dev set. Easy-moderate.
4. **huggingface/setfit** (Apache-2.0). Few-label paragraph-relevance classifier trained on LLM-tagged paragraphs. Easy, runs on CPU.
5. **VectifyAI/PageIndex** (MIT). ToC/tree navigation to scope sections before LLM paragraph tagging. Moderate.

**ADAPT (borrow the method or prompts)**
6. **Crucible / GINGER** nugget-first generation: Q&A nugget bank, dedupe, one cited sentence per nugget. No Crucible code was found. GINGER's repo has no license.
7. **SetR** (BSD-3) requirement-first set selection prompt, run with deepseek over whole sections.
8. **Conformal filtering** (JHU 2511.17908): calibrate the LLM-relevance threshold on a few labelled questions to hit a recall target. About 30 lines of code to write ourselves.
9. **OP-RAG**: keep document order in the context. Trivial.
10. **Attribute-First / FRONT / ALCE**: quote-first, then generate, and citation-check prompts.
11. **VersionRAG**: version metadata graph plus validity filter plus change detection. There is no license, so reimplement it.
12. **RAG-Challenge-2** (MIT): parent-page retrieval, LLM page reranking and company routing code. Treat it as a reference implementation of the LLM-as-reranker step.
13. **hltcoe/auto-argue** (MIT): second judge proxy, sentence precision plus nugget recall.

**SKIP**
- **Self-Route**: recall collapses on high-information aggregation tasks, and no code is released.
- **LongCite / FRONT model training**: needs a GPU for SFT. Use the ideas only.
- **SPLADE-v3** as core: the license is unclear, it falls under banned hybrid retrieval, and it brings little recall gain over LLM reading of scoped sections. Optional as a candidate generator only.
- **ColBERT / RAGatouille / heavy CPU cross-encoders** (bge-reranker-v2-m3) as core: banned-adjacent and slow on CPU. FlashRank or PyLate are acceptable only as a pre-filter.
- **LongRAG**: the long-unit idea is already implied by section-level LLM reading, and the repo is stale (2024).
- **Conformal-RAG claim filter** (n4feng): no license and 0 stars. Reimplement the idea if we want claim-level filtering.
