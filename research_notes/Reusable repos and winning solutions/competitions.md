# Reusable repos and winning solutions from RAG competitions (for CognitionNext '26 PS1)

Research date: 2026-10-23. Star counts, push dates and licenses come from the GitHub API (`gh api repos/...`) on that date unless noted.

## Q1. Enterprise RAG Challenge (TimeToAct / Rinat Abdullin), rounds 1, 2 and 3: who won, how, scores, code

### Takeaway
Ilya Rice ("Ilia Ris") won ERC Round 2 (Feb 2025, QA over 100 annual reports). His MIT-licensed repo `IlyaRice/RAG-Challenge-2` (about 2.5k stars) is the most directly reusable codebase for us. It covers Docling PDF parsing, a separate FAISS index per company with regex company routing, parent-page retrieval, batched LLM reranking with a 0.7 LLM / 0.3 vector score blend, and Pydantic structured-output answer schemas with `step_by_step_analysis`, `reasoning_summary`, `relevant_pages` and `final_answer`. Round 1 was won by Daniel Weller (84 points) with "bulk checklists with structured outputs".

### Cited Findings
**Round 2 results** ([ranking.csv](https://github.com/trustbit/enterprise-rag-challenge/blob/main/round2/ranking.csv)):
- 1st place: "Ilia Ris v.4", R=82.8, G=80.8, Score=122.2.
- The best non-Ilya entry was rank 3, "Submission_20250227_153643", with R=85.3, G=77.5, Score=120.1. It had higher retrieval but lower generation.
- Rank 9: Sergey Nikonov main v2, R=84.1, G=72.9.
- Rank 16: "multi-query, gpt-4o", 109.4.
- Rank 17: "non-cloud, r1-distill-llama-70b", 109.2.
- Rank 22: "gpt-4o-mini+pgvector", 106.9.
- Rank 24: SWISSCOM_RAG, 106.3.

**Round 2 resources:**
- The repo holds the ground-truth answers (`round2/answers.json`), the questions, all submissions and a leaderboard image — [README](https://github.com/trustbit/enterprise-rag-challenge/blob/main/round2/README.md).
- The README links to the [final leaderboard](https://www.timetoact-group.at/insights/enterprise-rag-challenge/leaderboards/team-leaderboard#teamleaderboard), the [winner overview](https://www.timetoact-group.at/en/insights/the-winner-of-the-enterprise-rag-challenge), Ilya's [blog post](https://www.timetoact-group.at/en/techblog/techblog/how-i-won-the-enterprise-rag-challenge) and an [interactive leaderboard](https://abdullin.com/erc#r2).
- Repo `trustbit/enterprise-rag-challenge`: 320 stars, last push 2025-04-01, license NOASSERTION — [GitHub](https://github.com/trustbit/enterprise-rag-challenge).

**Round 1 results** ([scores.csv](https://github.com/trustbit/enterprise-rag-challenge/blob/main/round1/scores.csv)):
- Daniel Weller: 84, GPT-4o, "Bulk checklists with structured outputs".
- Ilya Rice: 76, "Langchain RAG with GPT-4o".
- Artem Multi Stage Checklist: 73, Gemini Flash.
- Aleksandr Bobrov: 71, "Knowledge Graph with Neo4j & Vectors".
- Anonymous 1256: 69, a qwen72b ReAct + RAG system.

**Round 2 setup:** 100 annual reports of up to 1,000 pages each, a 2.5-hour parse/index window, and 100 template questions (Yes/No, name(s), numeric). Every answer had to carry page references — [Ilya Rice blog (abdullin.com)](https://abdullin.com/ilya/how-to-build-best-rag/).

**Winning architecture** ([blog](https://abdullin.com/ilya/how-to-build-best-rag/)):
- Docling was the best of about two dozen parsers he tested. He customized it and ran it on a rented 4090, parsing 100 reports in about 40 minutes.
- Chunks are about 300 tokens with 50-token overlap.
- One FAISS `IndexFlatIP` index per company, using text-embedding-3-large.
- Regex company-name routing picks the index. Comparative questions are split into per-company sub-questions.
- Parent-page retrieval: the top 30 chunks are mapped to pages and deduplicated.
- LLM rerank scores 3 pages per request. The final score is 0.7 × LLM + 0.3 × vector, and the top 10 pages are kept.
- GPT-4o-mini handles reranking and answering, at under 1 cent per question for reranking. o3-mini was the best answerer.

**What did not work:**
- Minimal hybrid BM25+vector "often reduced retrieval quality."
- LLM table serialization "slightly reduced effectiveness" and was dropped from the final run.
- Llama 3.3 70B finished a couple of points behind o3-mini. Even Llama 8B beat about 80% of participants.
- Source: [blog](https://abdullin.com/ilya/how-to-build-best-rag/).

**Repo `IlyaRice/RAG-Challenge-2`:** 2,461 stars, last push 2025-05-12, MIT license. The README lists Docling parsing, parent document retrieval, LLM reranking, structured CoT outputs and multi-company query routing. Configs are `max_nst_o3m` (best), `ibm_llama70b` and `gemini_thinking` (full-context, "not RAG, actually") — [GitHub](https://github.com/IlyaRice/RAG-Challenge-2).

**Files in `src/` (sizes in bytes)** ([tree](https://github.com/IlyaRice/RAG-Challenge-2/tree/main/src)):
- `pdf_parsing.py` (21.7k)
- `tables_serialization.py` (14.7k)
- `parsed_reports_merging.py` (16k)
- `text_splitter.py` (4.4k)
- `ingestion.py` (4k)
- `retrieval.py` (10.9k)
- `reranking.py` (6.7k)
- `questions_processing.py` (23.4k)
- `prompts.py` (28.4k)
- `api_requests.py` (24.6k)
- `api_request_parallel_processor.py` (18.9k)
- `pipeline.py` (19.9k)

**Concrete pieces worth copying** (read from the source via the GitHub API):
- **`src/prompts.py`:**
  - `AnswerWithRAGContextNamePrompt`, `...NumberPrompt`, `...BooleanPrompt`, `...NamesPrompt` and `ComparativeAnswerPrompt` each define a Pydantic `AnswerSchema`.
  - Each schema has `step_by_step_analysis` ("at least 5 steps and at least 150 words… avoid being tricked"), `reasoning_summary` (~50 words), `relevant_pages: List[int]` and `final_answer` (which allows `"N/A"`). Each comes with a worked one-shot example.
  - The file also has `RerankingPrompt` with a 0 to 1 relevance rubric (e.g. "0.5 = Moderately Relevant"), single-block and multi-block variants, the `RetrievalRankingSingleBlock` schema, `RephrasedQuestionsPrompt` (comparative question decomposition) and `AnswerSchemaFixPrompt` (a reparser for invalid JSON).
  - Source: [prompts.py](https://github.com/IlyaRice/RAG-Challenge-2/blob/main/src/prompts.py).
- **`src/reranking.py`:** `LLMReranker.rerank_documents(query, documents, documents_batch_size=4, llm_weight=0.7)` blends `llm_weight*relevance_score + (1-llm_weight)*distance` and pads missing scores with 0.0. There is also a `JinaReranker` class — [reranking.py](https://github.com/IlyaRice/RAG-Challenge-2/blob/main/src/reranking.py).
- **`src/retrieval.py`:** `BM25Retriever` and `VectorRetriever` both expose `retrieve_by_company_name(company_name, query, top_n, return_parent_pages)`. They filter on `metainfo["company_name"]`, keep one FAISS index per document sha1, and expand chunks to parent pages — [retrieval.py](https://github.com/IlyaRice/RAG-Challenge-2/blob/main/src/retrieval.py).
- **`src/pdf_parsing.py`:** sets up Docling with `DoclingParseV2DocumentBackend`, `do_ocr=True` with `EasyOcrOptions(lang=['en'], force_full_page_ocr=False)`, `do_table_structure=True`, `do_cell_matching=True` and `TableFormerMode.ACCURATE` — [pdf_parsing.py](https://github.com/IlyaRice/RAG-Challenge-2/blob/main/src/pdf_parsing.py).
- **`src/text_splitter.py`:** `RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=50)` per page.
- **`src/pipeline.py`:** `RunConfig` with `parent_document_retrieval`, `llm_reranking`, `llm_reranking_sample_size=30`, `top_n_retrieval=10` and `use_serialized_tables`.
- **`src/tables_serialization.py`:** `TableSerializer` takes the table plus the surrounding text (`_get_table_context`) and has the LLM serialize it into context-rich blocks.

**Round 3 (late 2025):** ERC3 was an AI-agents round. Ilya's repo `IlyaRice/Enterprise-RAG-Challenge-3-AI-Agents` has 71 stars, last push 2025-12-28, MIT. The organizers' sample agents are `trustbit/erc3-agents` (47 stars) — [GitHub](https://github.com/IlyaRice/Enterprise-RAG-Challenge-3-AI-Agents), [GitHub](https://github.com/trustbit/erc3-agents).

**DeepSeek-based ERC port:** `liangdabiao/enterprise-rag-challenge-release` (7 stars, last push 2026-08-31, license NOASSERTION) is a "human-like reading" visual RAG: check the table of contents, look at the page images, dig deeper, then answer with page citations. It runs on `deepseek-v4-flash-vision-exp` and has `01_skill/scripts/router.py`, `ingest.py`, `agentic_tools.py` and `ds_client.py`, plus round 2 to 5 self-evaluation reports (in Chinese) — [GitHub](https://github.com/liangdabiao/enterprise-rag-challenge-release).

**Other ERC participant repos** (all small and unverified for rank): `dennismeissel/RAGathon` (4 stars), `f-krause/enterprise_rag_challenge_fk` (1 star) and `mrvladd-d/RAG-Challenge` — [GitHub search](https://github.com/search?q=enterprise+rag+challenge&type=repositories).

### Inferences
- The ERC R2 Score appears to be R/2 + G: 82.8/2 + 80.8 = 122.2, and the same holds for other rows. So generation was weighted about twice as much as retrieval.
- The ERC answers are short factoids, while ours are long analyst reports. Copy the parsing, company routing, parent-page and LLM-rerank layers. Do not copy the per-type answer schemas as-is. Adapt them to a report schema in which each claim carries `relevant_pages`.
- Our banned list includes "plain hybrid+rerank". Ilya's core is a per-company route, then parent-page expansion, then LLM listwise relevance scoring. That is arguably a layered design rather than plain hybrid+rerank, and his blog says he did not rely on BM25 hybrid.
- His LLM-rerank pattern (scoring a few pages per call with a 0 to 1 score) maps directly onto our context-precision metric. Pages or paragraphs scored below a threshold can be dropped before we return contexts.
- Context recall needs about 81 gold paragraphs per question, so his top-10-pages cutoff is far too small for us. We would need to rerank and return many more paragraphs. This is an inference about our metric, not something ERC tested.

### Gaps
- I did not read the full text of the TimeToAct "how I won" techblog. The abdullin.com mirror was used instead.
- Identities and code for rank 3 and rank 5 (anonymous submission names) were not found.
- I did not open `questions_processing.py` in detail.

## Q2. Meta CRAG KDD Cup 2024 and CRAG-MM (KDD Cup 2025)

### Takeaway
Team db3 (Peking University) won all three CRAG 2024 tasks and the CRAG-MM 2025 Grand Prize. Their write-ups are papers only. I found no official db3 code repo. Their approach is fine-tuned LLMs with refusal training, which does not suit a CPU server with a fixed API model, so CRAG offers little reusable code for us.

### Cited Findings
- **CRAG 2024 Task 1:** AIcrowd announced Team db3 1st, ahead of md_dh and ElectricSheep — [AIcrowd winners](https://discourse.aicrowd.com/t/meta-crag-challenge-2024-winners-announcement/10786).
- **CRAG 2024 overall:** db3 (Jiazun Chen and Yikuan Xia, PKU) "took the top spot across all three tasks" — [AIcrowd db3 spotlight](https://discourse.aicrowd.com/t/behind-the-winning-strategy-of-team-db3-meta-crag-2024/17032).
- **db3 2024 method:** a web/open-data retrieval-and-answering framework with a tuned LLM to reduce hallucination — [arXiv 2410.00005](https://arxiv.org/pdf/2410.00005). A second solution paper is [arXiv 2409.15337](https://arxiv.org/pdf/2409.15337).
- **CRAG-MM 2025:** db3 won the Grand Prize for Egocentric Image QA ($5,000) — [AIcrowd 2025 winners](https://discourse.aicrowd.com/t/meta-crag-challenge-2024-winners-announcement/17308).
  - db3's own paper reports 2nd in Task 1, 2nd in Task 2 and 1st in Task 3, using domain-specific retrieval and refusal training with SFT, DPO and RL — [arXiv 2509.09681](https://arxiv.org/pdf/2509.09681).
  - BlackPearl (Meituan) reports 3rd, 3rd and 1st on the automatic evaluation, and 2nd on Task 3 after human evaluation — [arXiv 2507.21520](https://arxiv.org/html/2507.21520v1).
- **Code that exists:**
  - `BlackPearl-Lab/KddCup-2025-CRAG-MM-Solution`: 14 stars, no license. Files include `0_recall.py`, `1_rerank.py`, `2_syth_data.py`, and DeepSpeed LoRA training — [GitHub](https://github.com/BlackPearl-Lab/KddCup-2025-CRAG-MM-Solution).
  - `cruiseresearchgroup/KDD-Cup-2025-Meta-CRAG-MM`: 3rd-place Task 1, 10 stars — [GitHub](https://github.com/cruiseresearchgroup/KDD-Cup-2025-Meta-CRAG-MM).
  - `BUAADreamer/ConfRAG`: top-7 — [GitHub](https://github.com/BUAADreamer/ConfRAG).
  - Official baselines: `facebookresearch/CRAG` (310 stars, last push 2025-06-14) — [GitHub](https://github.com/facebookresearch/CRAG).

### Inferences
- CRAG winners rely on training, either fine-tuning or refusal DPO, which we cannot do with a fixed deepseek API. CRAG also scores short answers with a hallucination penalty. The only transferable idea is "prefer abstaining to hallucinating" for each claim.
- Note that "CRAG (the method)", Corrective RAG, is banned for us. That is a different thing from the CRAG benchmark.

### Gaps
- I could not confirm whether db3 released code. None was linked in the search results.

## Q3. TREC RAG 2024/2025, TREC NeuCLIR 2024 report generation, TREC 2025 RAGTIME (long cited reports)

### Takeaway
These tracks are the closest match to our "long cited report" task. Their evaluation is nugget recall (similar to our atomic-point answer recall) plus citation or sentence support (similar to our precision). The reusable code is mostly evaluation tooling from Waterloo (Ragnarök, nuggetizer) and the JHU HLTCOE group (auto-argue), not winning generators. GINGER (nugget detect, cluster, rank, summarize) is the most relevant published generation design, but I found no code for it.

### Cited Findings
- **TREC RAG 2024 participation:** 93 runs from 20 groups for the RAG task and 53 runs from 11 groups for the Augmented Generation (AG) task — [arXiv 2504.15205](https://arxiv.org/html/2504.15205v1).
- **TREC RAG 2024 evaluation:**
  - Nugget evaluation uses AutoNuggetizer — [arXiv 2411.09607](https://arxiv.org/html/2411.09607v1), [arXiv 2504.15068](https://arxiv.org/pdf/2504.15068).
  - Support evaluation uses a GPT-4o judge against humans, with weighted precision penalizing overcitation and weighted recall penalizing undercitation — [arXiv 2504.15205](https://arxiv.org/html/2504.15205v1).
- **Ragnarök (Waterloo):** a reusable RAG framework and baselines for TREC RAG 2024, also used for NeuCLIR report generation — [arXiv 2406.16828](https://arxiv.org/pdf/2406.16828). Repo `castorini/ragnarok`: 66 stars, last push 2026-04-18, Apache-2.0 — [GitHub](https://github.com/castorini/ragnarok).
- **nuggetizer:** `castorini/nuggetizer`, 29 stars, last push 2026-04-19, Apache-2.0. It has `scripts/create_nuggets.py`, `assign_nuggets.py` and `calculate_metrics.py`, plus a prompt-templates reference at `.claude/skills/nuggetizer-quickstart/references/prompt-templates.md` — [GitHub](https://github.com/castorini/nuggetizer).
- **GINGER (Łajewska and Balog, Univ. of Stavanger, SIGIR 2025):**
  - Pipeline: nugget detection, clustering, ranking, top-cluster summarization and fluency enhancement. The authors claim state of the art on TREC RAG'24 AG (301 queries) under AutoNuggetizer — [arXiv 2503.18174](https://arxiv.org/pdf/2503.18174).
  - TREC runs: `ginger_top_5` and `ginger-fluency_top_{5,10,20}` — [TREC proceedings](https://pages.nist.gov/trec-browser/trec33/rag/proceedings), [uis-iai paper](https://trec.nist.gov/pubs/trec33/papers/uis-iai.rag.pdf).
  - They later applied it to LiveRAG — [arXiv 2506.22210](https://arxiv.org/html/2506.22210v1).
  - No GitHub repo was found (`iai-group/ginger` returns 404).
- **TREC 2024 NeuCLIR report generation pilot:**
  - 51 runs from 4 teams. Top runs score about 0.9 ARGUE, but top citation precision is only about 0.3 and nugget recall is below 0.5 — [Overview arXiv 2509.14355](https://arxiv.org/pdf/2509.14355).
  - HLTCOE's paper lists `fas-hltcoe-eugene-gpt4o` at ARGUE 0.872, citation precision 0.918 and nugget recall 0.303 — [arXiv 2510.00143](https://arxiv.org/pdf/2510.00143).
  - The two papers' citation precision figures conflict (about 0.3 vs 0.9). They probably use different definitions or aggregation; this is unresolved.
- **Auto-ARGUE:** an LLM judge for report generation — [repo hltcoe/auto-argue](https://github.com/hltcoe/auto-argue) (2 stars, MIT, last push 2025-10-16).
- **TREC 2025 RAGTIME** (multilingual report generation, 2,000- and 10,000-character limits; [arXiv 2602.10024](https://arxiv.org/html/2602.10024)):
  - Metrics are sentence support (every citation must independently support the sentence), nugget coverage, and F1.
  - Top long-report runs by Auto-ARGUE F1:
    - hltcoe-rerank `hltime-gpt5.searcher`: F1 0.694 (coverage 0.638, support 0.805).
    - GenAIus `genaius-question`: F1 0.678.
    - hltcoe-multiagt `lg_nt_4q12r3l_mt_c`: F1 0.647, with support 0.918.
  - Top short-report run: the coordinators' `extractive_rag`, F1 0.562.
  - Top multilingual IR run: fusion of PLAID-X, learned sparse and Qwen3 embeddings, reranked with Qwen3 and Jina.
  - Track site: [trec-ragtime.github.io](https://trec-ragtime.github.io/). GenAIus TREC paper: [GenAIus.rag.pdf](https://trec.nist.gov/pubs/trec34/papers/GenAIus.rag.pdf).
- **Related report-generation work:** "Incorporating Q&A Nuggets into RAG" — [arXiv 2601.13222](https://arxiv.org/pdf/2601.13222). "Insider Knowledge: How Much Can RAG Systems Gain from Evaluation Secrets?" — [arXiv 2601.13227](https://arxiv.org/pdf/2601.13227).

### Inferences
- **Extractive generation:** the coordinators' `extractive_rag` baseline won short-report F1. Extractive or "cite-verbatim-then-compose" generation seems to score well on support and precision, which suits our answer-precision metric.
- **Nugget pipeline:** GINGER-style processing (extract atomic nuggets from evidence, cluster, rank, then write one sentence per cluster with citations) is structurally aligned with our atomic-point recall/precision judge. It is also not on the banned list. We would have to build it ourselves from the paper.
- **Local evaluation:** nuggetizer's `create_nuggets.py`/`assign_nuggets.py` prompts could serve as our local judge to approximate the hackathon's atomic-point scoring.

### Gaps
- No full ranked list of TREC RAG 2024 AG top runs was found. The official overview's per-run tables were not retrieved.
- Code for GenAIus and HLTCOE's RAGTIME generators was not found.

## Q4. NeurIPS 2025 MMU-RAG and SIGIR 2025 LiveRAG

### Takeaway
RMIT-ADM+S won LiveRAG 2025 (by the organizers' manual Borda ranking) with G-RAG, built on an LLM yes/no-odds reranker. They then won MMU-RAG's open-source dynamic evaluation with R2RAG, which routes on query complexity and evidence sufficiency. G-RAG code is public (MIT). RAGentA (BSD-3) is a reusable multi-agent attributed-QA design with in-text citations.

### Cited Findings
- **LiveRAG 2025 final ranking:** 1 RMIT-ADMS, 2 RAGtifier (L3S), 3 UDInfo (Delaware), 4 Magikarp. The challenge had 70 teams, 500 questions and a 2-hour window — [LiveRAG report arXiv 2507.04942](https://arxiv.org/html/2507.04942v1), [ADM+S](https://www.admscentre.org.au/arc-centre-of-excellence-team-wins-global-liverag-challenge-at-sigir-2025/).
  - L3S's own account places Magikarp first — [L3S](https://www.l3s.de/?p=40623). The report's Borda table is the official order.
- **LiveRAG scores by session** (correctness / faithfulness):
  - RMIT-ADMS: 1.199 / 0.477 (S1).
  - RAGtifier: 1.134 / 0.552 (S2).
  - UDInfo: 1.201 / 0.623 (S2).
  - Magikarp: 1.232 / 0.656 (S2).
  - Sessions are not directly comparable — [arXiv 2507.04942](https://arxiv.org/html/2507.04942v1).
- **LiveRAG architectures** ([arXiv 2507.04942](https://arxiv.org/html/2507.04942v1)):
  - RMIT: an LLM reranker that judges whether each doc contains needed information, ranked by "Yes" token odds — [arXiv 2506.14516](https://arxiv.org/abs/2506.14516).
  - RAGtifier: passages in reverse score order — [arXiv 2506.14412](https://arxiv.org/abs/2506.14412).
  - UDInfo: separate rule-based rewriters for sparse and dense retrieval — [arXiv 2506.17493](https://arxiv.org/abs/2506.17493).
  - Magikarp: question expansion from top results, used for reranking — [arXiv 2506.20476](https://arxiv.org/abs/2506.20476).
  - Ped100X: topic-taxonomy document classification that prunes results by the question's topic — [arXiv 2506.12571](https://arxiv.org/abs/2506.12571).
- **LiveRAG code:**
  - `rmit-ir/G-RAG-LiveRAG`: 12 stars, last push 2026-07-02, MIT. Has `scripts/run.py`, `src/services/answer_utils.py`, `src/services/llms/general_openai_client.py` (OpenAI-compatible) and `src/evaluators/context_recall/evaluator.py` — [GitHub](https://github.com/rmit-ir/G-RAG-LiveRAG).
  - `faerber-lab/RAGentA`: 27 stars, last push 2025-07-10, BSD-3-Clause. Has `RAGentA.py`, `hybrid_retriever.py`, `api_agent.py` and `run_RAGentA.py` — [GitHub](https://github.com/faerber-lab/RAGentA).
  - `muktac5/CIIR-LiveRAG` — [GitHub](https://github.com/muktac5/CIIR-LiveRAG).
  - `kimdanny/Starlight-LiveRAG` (learning to rank retrievers) — [GitHub](https://github.com/kimdanny/Starlight-LiveRAG).
- **MMU-RAG (NeurIPS 2025, CMU LTI + Amazon):** 81 teams registered and only 8 submitted working systems — [ADM+S](https://www.admscentre.org.au/mmu-rag-challenge-win/), [NeurIPS page](https://neurips.cc/virtual/2025/competition/127730).
  - RMIT's R2RAG won Best Dynamic Evaluation (Open Source, Text-to-Text). It routes retrieval strategy on inferred query complexity and evidence sufficiency, uses small LLMs on a single consumer GPU, and builds on G-RAG. No code link was found — [arXiv 2602.20735](https://arxiv.org/abs/2602.20735).
  - Tohoku, Studio Ousia and Dentsu Soken won Best Static Evaluation (Open Source) with "An Open and Reproducible Deep Research Agent for Long-Form QA" — [Tohoku NLP](https://www.nlp.ecei.tohoku.ac.jp/?p=14325).

### Inferences
- **Evidence filtering:** RMIT's yes/no-odds relevance filter is a cheap way to decide which paragraphs to return, directly targeting our context-precision metric. With deepseek over an API, logprobs may not be available, so a structured 0 to 1 score (as in Ilya's prompt) is the practical substitute.
- **Topic pruning:** Ped100X's taxonomy-based document pruning resembles our need to route between corpus types (annual reports vs Basel/Fed rules vs CECL vs quant papers).
- **Banned methods:** UDInfo and Magikarp rely on query rewriting or expansion, which is close to the banned HyDE and multi-query methods. Avoid copying them as the core.

### Gaps
- I did not inspect RAGentA's prompt text or the G-RAG answer prompts line by line.
- The MMU-RAG Tohoku agent's code URL was not found.

## Q5. FinanceBench / FinQA / DocFinQA / SEC-QA leaderboards with open code

### Takeaway
PageIndex (VectifyAI, MIT, about 39.6k stars, active today) and its Mafin 2.5 FinanceBench release (claimed 98.7%) are the strongest open finance-doc results. PageIndex builds an LLM-generated table-of-contents tree per PDF and navigates it by reasoning. It is "vectorless", which makes it a candidate for our non-banned core or routing layer. Other FinanceBench repos claim 82 to 92% but are small, single-author projects.

### Cited Findings
- **Mafin 2.5** (`VectifyAI/Mafin2.5-FinanceBench`, 98 stars, last push 2025-10-20, no license stated):
  - Claims 98.7% accuracy with 100% coverage on the FinanceBench public set, with both GPT-4o and DeepSeek-V3. All docs sit in a single database, and ambiguous questions were resolved by expert annotation.
  - The release includes `eval.py`, `result_gpt4o.json`, `result_deepseekv3.json` and `human_evaluations/`.
  - Comparison claims: Quantly 94%, Fintool 98% (at 66.7% coverage), Perplexity 45%.
  - Source: [GitHub](https://github.com/VectifyAI/Mafin2.5-FinanceBench).
- **PageIndex** (`VectifyAI/PageIndex`, 39,552 stars, last push 2026-10-23, MIT):
  - The package has `pageindex/page_index_classic.py` (tree generation, 56k), `page_index_md.py`, `tree_optimize.py` (50k), `agent_tools.py` (78k), `local_chat.py`, `utils.py`, and a heuristic layout parser `pageindex/flash/` (heading detection, TOC/boilerplate classification, column splitting, pdfium char-level parsing).
  - `config.yaml` sets `toc_check_page_num: 20`, `max_page_num_each_node: 10`, `max_token_num_each_node: 20000` and `if_add_node_summary: "yes"`, and takes LiteLLM model names.
  - Cookbooks include `pageindex-citation.ipynb`, `pageindex-metadata.ipynb` and `pageindex-vs-vector-10k/`.
  - Example tree outputs include `examples/results/2023-annual-report_structure.json` and `Regulation Best Interest_*_structure.json`.
  - Source: [GitHub](https://github.com/VectifyAI/PageIndex).
- **FinanceBench dataset:** `patronus-ai/financebench`, 382 stars, last push 2024-12-03, no license — [GitHub](https://github.com/patronus-ai/financebench).
- **Smaller FinanceBench repos (self-reported, unverified):**
  - `Zhonghui-li/sec-filing-agent`: "92% coverage with zero hallucinations", XBRL exact numbers, LangGraph + pgvector — [GitHub](https://github.com/Zhonghui-li/sec-filing-agent).
  - `julienlucas/agentic-rag-with-tools`: 88%, 19 stars — [GitHub](https://github.com/julienlucas/agentic-rag-with-tools).
  - `AsyncBuilds/FinRag`: 82%, "LLM navigates section headings instead of searching embeddings", 13 stars — [GitHub](https://github.com/AsyncBuilds/FinRag).
  - `avnlp/rag-pipelines`: Milvus hybrid + BAML structured generation, 16 stars — [GitHub](https://github.com/avnlp/rag-pipelines).
- **FinQA:** `czyssrs/FinQA`, 406 stars, MIT, last push 2022-06-06. A table+text numerical-reasoning dataset and retriever–generator baseline — [GitHub](https://github.com/czyssrs/FinQA).

### Inferences
- **Fit with our corpus:** PageIndex's tree-of-sections approach suits our 35 long, well-structured PDFs (annual reports and regulations have strong ToCs). It is not on the banned list, and its MIT license allows copying `page_index_classic.py` wholesale.
- **Cost risk:** tree building uses many LLM calls per document. Indexing 8,600 pages with deepseek-v4.1-flash needs a cost and time check.
- **Context recall risk:** PageIndex returns sections or nodes, not paragraphs. To score on context recall (about 81 gold paragraphs) we would expand selected nodes into paragraphs and then filter.
- **Model support:** LiteLLM model naming means we can probably point PageIndex at an OpenAI-compatible deepseek endpoint. This is unverified.
- **Comparability:** FinanceBench scores (short-answer accuracy) are not comparable to our long-report atomic-point scoring.

### Gaps
- No open leaderboard with code was found for DocFinQA or SEC-QA. The `kensho-technologies/docfinqa` repo does not exist (404).
- The 98.7% Mafin claim is self-reported with human-resolved ambiguous items, so it is not independently verified.

## Q6. Kaggle document-retrieval competitions with public top solutions

### Takeaway
The only relevant Kaggle example I checked is the LLM Science Exam (2023). Its top teams all retrieved Wikipedia context (TF-IDF plus small embedders). It is multiple choice over Wikipedia and of low relevance to us.

### Cited Findings
- **Competition design:** questions were GPT-3.5-generated from Wikipedia science snippets — [Kaggle](https://kaggle.com/competitions/kaggle-llm-science-exam).
- **Top-team retrieval:** a Japanese recap says all top teams retrieved Wikipedia context — [zenn.dev](https://zenn.dev/yume_neko/articles/7347ba6b081e93).
- **10th-place code:** three retrievers mixing TF-IDF with gte-small embeddings — [GitHub Lizhecheng02/Kaggle-LLM_Science_Exam](https://github.com/Lizhecheng02/Kaggle-LLM_Science_Exam).

### Inferences
- Kaggle offers little for long cited financial reports. ERC, TREC RAGTIME and NeuCLIR, and PageIndex are far closer matches.

### Gaps
- The 1st-place writeup URL was not retrieved.
- No Kaggle competition on financial-PDF RAG with a public top solution was found.

## Q7. Directly reusable component repos (parsing, tables, routing, evidence selection, citation prompts)

### Takeaway
For parsing, Docling (MIT, about 69k stars) is what the ERC winner used and is the safest choice. MinerU and Marker are alternatives, but check their licenses. For routing, evidence selection and answer schemas, copy from RAG-Challenge-2 (MIT) and PageIndex (MIT). For citation-grounded long answers and local evaluation, use LongCite and ALCE prompts plus nuggetizer.

### Cited Findings
**PDF parsing** (stars, last push, license from `gh api` on 2026-10-23):

| Repo | Stars | Last push | License | Note |
|---|---|---|---|---|
| [docling-project/docling](https://github.com/docling-project/docling) | 69,450 | 2026-10-23 | MIT | Used by the ERC2 winner |
| [opendatalab/MinerU](https://github.com/opendatalab/MinerU) | 82,373 | 2026-10-23 | NOASSERTION | Check custom license |
| [datalab-to/marker](https://github.com/datalab-to/marker) | 40,608 | 2026-10-22 | Apache-2.0 (code) | Model weights may carry separate terms (unverified) |
| [microsoft/markitdown](https://github.com/microsoft/markitdown) | 191,797 | 2026-10-23 | MIT | Lightweight; weak on tables |
| [pymupdf/pymupdf4llm](https://github.com/pymupdf/pymupdf4llm) | 2,281 | 2026-10-23 | AGPL-3.0 | Copyleft risk |
| [Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured) | 15,602 | 2026-10-23 | Apache-2.0 | |

The ERC winner's own Docling config is in [src/pdf_parsing.py](https://github.com/IlyaRice/RAG-Challenge-2/blob/main/src/pdf_parsing.py), and the Docling-JSON-to-page-text merge is in `src/parsed_reports_merging.py`.

**Table handling:**

| Repo | Stars | Last push | License |
|---|---|---|---|
| [camelot-dev/camelot](https://github.com/camelot-dev/camelot) | 3,842 | 2026-10-14 | MIT |
| [jsvine/pdfplumber](https://github.com/jsvine/pdfplumber) | 10,842 | 2026-10-15 | MIT |

- Docling's TableFormer ACCURATE mode is used in the ERC2 config.
- ERC2's `src/tables_serialization.py` does LLM serialization of tables with surrounding context. The author found it slightly hurt ERC scores — [blog](https://abdullin.com/ilya/how-to-build-best-rag/).

**Routing by company or metadata:**
- ERC2 `retrieval.py` `retrieve_by_company_name` keeps a per-document index keyed by `metainfo.company_name`. `prompts.py` `RephrasedQuestionsPrompt` splits multi-company questions — [GitHub](https://github.com/IlyaRice/RAG-Challenge-2).
- PageIndex's `cookbook/pageindex-metadata.ipynb` covers metadata routing — [GitHub](https://github.com/VectifyAI/PageIndex).

**LLM evidence selection:**
- ERC2 `reranking.py` `LLMReranker` plus `RerankingPrompt` and `RetrievalRankingSingleBlock` in `prompts.py` — [GitHub](https://github.com/IlyaRice/RAG-Challenge-2).
- RMIT G-RAG uses a yes-odds reranker — [GitHub](https://github.com/rmit-ir/G-RAG-LiveRAG).
- RAGentA uses multi-agent filtering — [GitHub](https://github.com/faerber-lab/RAGentA).

**Citation-grounded generation:**
- [THUDM/LongCite](https://github.com/THUDM/LongCite): 524 stars, last push 2024-12-31, Apache-2.0. Fine-grained sentence citations in long-context QA.
- [princeton-nlp/ALCE](https://github.com/princeton-nlp/ALCE): 527 stars, last push 2024-10-09, MIT. Citation prompts plus citation precision and recall evaluation.
- [stanford-oval/storm](https://github.com/stanford-oval/storm): 31,684 stars, last push 2025-09-30, MIT. Outline-then-write cited long reports.
- [assafelovic/gpt-researcher](https://github.com/assafelovic/gpt-researcher): 30,262 stars, last push 2026-10-10, Apache-2.0. HLTCOE used a GPT-Researcher pipeline at LiveRAG — [arXiv 2507.04942](https://arxiv.org/html/2507.04942v1).
- [castorini/ragnarok](https://github.com/castorini/ragnarok) (Apache-2.0) produces TREC-format cited answers.

**Full-stack references:**
- [infiniflow/ragflow](https://github.com/infiniflow/ragflow): 92,748 stars, Apache-2.0. Its "deepdoc" parser is aimed at complex documents (claim from general knowledge; not verified this session).
- [gomate-community/TrustRAG](https://github.com/gomate-community/TrustRAG): 1,281 stars, no license.

**Local eval of atomic-point recall:**
- [castorini/nuggetizer](https://github.com/castorini/nuggetizer) (Apache-2.0).
- [hltcoe/auto-argue](https://github.com/hltcoe/auto-argue) (MIT).

### Inferences
- **Fastest path:** fork RAG-Challenge-2's `pdf_parsing.py`, `parsed_reports_merging.py`, `retrieval.py`, `reranking.py` and `prompts.py`. Swap OpenAI for a deepseek base_url in `api_requests.py`. Replace the per-type answer schemas with a report schema, for example `sections[] -> claims[] {text, citations:[{file,page}]}` plus `relevant_pages`.
- **Optional layers:** add a PageIndex tree as a structure-aware routing layer and nuggetizer as a local judge.
- **CPU constraint:** Docling with EasyOCR and TableFormer ACCURATE on CPU for 8,600 pages may be slow. The ERC winner needed a 4090 to parse 100 reports in about 40 minutes. We should pre-parse offline if the rules allow, or turn OCR off for born-digital PDFs.
- **Licenses:** pymupdf4llm (AGPL) and MinerU (custom) need license review before we copy any code.

### Gaps
- I did not verify RAGFlow deepdoc, LongCite or ALCE prompt file paths this session.
- Docling's CPU throughput on our corpus is unmeasured.
- Whether deepseek-v4.1-flash supports strict JSON-schema structured outputs, which ERC2 relies on, was not checked. ERC2's `AnswerSchemaFixPrompt` reparser is the fallback.
