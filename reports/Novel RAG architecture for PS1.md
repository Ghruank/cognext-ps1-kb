# Select calibrated evidence sets, not top-k chunks

Build the PS1 system as a **calibrated evidence-set selector layered on a modified vector index**. Its job is to estimate, for every paragraph-sized unit in the routed documents, the probability that the judge will call it relevant. It then returns the set that maximizes expected context precision plus capped context recall, and it feeds that whole set to one cheap writer call organised by facet. Six named layers make this work. **Typed Evidence Units** fix the chunk-size dilemma by matching gold granularity: about 63 words, typed as prose, table, formula or visual. A **Version-Family Lattice** fixes near-duplicates and versions with MinHash families, containment-based consolidation tags, delta units and soft constraint matching. **Run-Aware Calibrated Relevance** is a LightGBM model plus Platt scaling over corpus-agnostic features, including document-order neighbour scores, because gold arrives in about 5 runs per document. The **Expected-Utility Set Selector**, with a conformal recall floor, sizes the set per question. It is the piece that reconciles a median gold count of 81 (max 352) with the instinct to feed the LLM only 8 to 60 chunks. The **Facet-Partitioned Writer** and a **Deterministic Claim Gate** handle generation. The key insight is that **one latent quantity, the expected gold count Ĝ, drives everything downstream**: how many units to return, how many tokens the writer sees, and how long the answer should be. Feeding the full selected set costs about **$0.002 of input at the median and $0.009 at the maximum** on deepseek-v4.1-flash, so honesty about which chunks were used is cheap. The 0.95 target is realistic for context precision and plausible for context recall. It is far beyond any published result for long-form answer recall: the best RAGChecker system reached F1 52.7, and GPT-4o covers about 58% of key points on Long²RAG. The first job of the build is therefore an oracle run that measures the true ceiling under this judge.

## The judge rewards paragraph-sized sets sized per question

The official metrics set the design constraints, so start there. **Context recall is judged-relevant returned units divided by the count of gold units, and it is not capped at 1.** Context precision is judged-relevant divided by returned. Both come from one deepseek-v4.1-flash call at temperature 0 that sees every returned chunk at once. Three consequences follow, and together they dissolve the chunk-size/completeness dilemma.

First, **granularity is set by the count denominator, not by retrieval convenience.** A question with 81 gold paragraphs cannot reach recall 0.95 with 10 large chunks, however complete they are. Whole-page retrieval also fails, because gold covers only about 40% of a touched page's words, so pages drag precision down. The unit must therefore be paragraph-sized, about 63 words like gold. Completeness then comes from *how many* units are selected, not from how big each one is. This is why the architecture moves the "completeness" problem out of chunking and into set selection.

Second, **the judge scores relevance, not ID overlap.** Our unit boundaries never need to reproduce the gold parser's boundaries; they only need comparable size and self-contained meaning. A genuinely relevant paragraph that the gold set happened to omit still counts as relevant for both precision and recall. That opens an obvious exploit: shred text into tiny fragments to inflate the count above |gold|. The team should refuse it, because it is metric gaming that a reviewer will spot. It also likely collapses on the hidden set, where the judge sees fragments with no meaning. The selector below therefore optimizes **capped** recall, E[min(1, R)], so the objective itself never pays for overshooting gold.

Third, **the gold parser was a typed, vision-language document parser.** Gold section names split into 4,838 Section, 1,056 table, 277 math and 68 visual elements. Tables are about 17% of gold units, so table extraction is first-order, not polish. Visual elements (about 1%, for example a Moody's rating shown only as an image) are a long tail worth only a stretch-goal VLM pass.

The generation side has a parallel structure. The answer judge is essentially RAGChecker's overall precision and recall against gold claims. RAGChecker found that **context utilization correlates most strongly with overall F1**, and that raising k from 5 to 20 lifted claim recall from 61.5 to 77.6 ([RAGChecker](https://arxiv.org/html/2408.08067v2)). Gold answers were written *from the gold units*, with a median of 754 words, headings, `[file.pdf pX]` citations and evidence-scoped hedges. The winning answer therefore uses every relevant returned unit, states each fact atomically in the evidence's own words, and adds nothing the evidence does not support. Any true-but-ungrounded point counts as "extra" against precision.

## Six layers turn a vector index into an evidence-set selector

The vector index stays at the core, which is allowed. What changes is what goes into it (typed units with family and role metadata) and what comes out (a calibrated, per-question-sized set rather than top-k). None of the six layers is a banned technique used as the core. Hybrid scores and a cross-encoder appear only as *features* of a calibrated classifier, and no step loops or reflects.

| Layer | Failure mode it solves | Core math / ML | Key evidence |
|---|---|---|---|
| 1. Typed Evidence Units (TEU) | Chunk-size dilemma; tables and scans lost | Structure-aware unitization to 40-150 words (median ≈65), typed {prose, table, table-row, formula, visual}; OCR on text-less pages | Element-based chunking beats fixed 128-512 token chunks on FinanceBench ([arXiv 2402.05131](https://arxiv.org/abs/2402.05131v3)) |
| 2. Version-Family Lattice (VFL) | Near-duplicate documents and versions; consolidated distractors | MinHash Jaccard + containment, union-find families, delta/shared units, MRAG-style constraint split, soft catalogue scoring | VersionRAG 58%→90% overall ([arXiv 2510.08109](https://arxiv.org/html/2510.08109v1)); multi-version index 98.3% vs 0% applicable-version retrieval for static RAG ([arXiv 2608.09393](https://arxiv.org/abs/2608.09393)) |
| 3. Run-Aware Calibrated Relevance (RACR) | Uncalibrated scores; scattered, contiguous gold runs | LightGBM on query-relative features incl. neighbour scores; Platt scaling; Σp count calibration | DTA beats EUM for rare classes and domain shift ([Ye et al. 2012](https://arxiv.org/abs/1206.4625)); Platt safer on small data ([Niculescu-Mizil & Caruana](https://mlanthology.org/icml/2005/niculescumizil2005icml-predicting)) |
| 4. Expected-Utility Set Selector (EUSS) | Fixed k too small for 352-gold questions and too large for factoid ones | Top-k prefix optimality (PRP for F), prefix-sum expected P and R, conformal risk control floor, submodular facet coverage | Ye et al. ([arXiv 1206.4625](https://arxiv.org/abs/1206.4625)); CRC ([arXiv 2208.02814](https://arxiv.org/abs/2208.02814)); Lin & Bilmes ([ACL P11-1052](https://aclanthology.org/P11-1052/)) |
| 5. Facet-Partitioned Writer (FPW) | Lost-in-the-middle and key-point loss over 7-30k tokens | Units grouped by facet → document → run in document order; each group written in parallel, each unit seen once; ID-plan then answer | Long²RAG KPR falls with length ([arXiv 2410.23000](https://arxiv.org/html/2410.23000v3)); OP-RAG ordering ([arXiv 2409.01666](https://arxiv.org/html/2409.01666v1)); Attribute-First ([ACL 2024](https://aclanthology.org/2024.acl-long.182)) |
| 6. Deterministic Claim Gate (DCG) | Hallucinated numbers and entities; false answers on unanswerable questions | Regex number and entity containment against cited units; optional MiniCheck NLI; two-signal abstention | MiniCheck reaches GPT-4-level checking at 400x lower cost ([arXiv 2404.10774](https://arxiv.org/abs/2404.10774)) |

### Typed Evidence Units make the unit the gold's size and shape

Ingest with PyMuPDF, not ML layout parsers. **Docling takes 3.1 s per page and MinerU 3.3 s per page on an 8-vCPU CPU**, roughly 7-8 hours for 8,600 pages ([Docling report](https://arxiv.org/html/2501.17887)). PyMuPDF averages 0.1 s per benchmark file at 94% text quality ([py-pdf/benchmarks](https://github.com/py-pdf/benchmarks)).

The ingestion steps are:

1. Triage each page: low character count plus a page-covering image marks it as scanned, which catches the 92-page SAMA scan.
2. Send scanned pages through PyMuPDF's built-in Tesseract path.
3. Strip repeating headers and footers.
4. Detect headings from `get_toc()` first and font size second ([pymupdf4llm API](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html)).
5. Start a new unit at every numbered-paragraph regex match (Basel `52.`, FASB `326-20-30-1`, `§217.x`), storing the number as `para_id`.
6. Merge or split blocks to 40-150 words.

On numeric-heavy pages, `find_tables()` emits one markdown table unit, which is what the judge sees, plus row-linearized shadow units that are used for retrieval only. Pipeline table detectors are weak; even MinerU-Pipeline scores only 81.9 TEDS on OmniDocBench ([OmniDocBench](https://github.com/opendatalab/OmniDocBench)). Never let a shadow row be returned as a chunk: map it back to its parent table unit, or to the row-group unit if the table is long, so returned units stay paragraph-sized. Every unit carries `doc_id, pages, section_path, para_id, type, family_id, role, is_delta`. Embedding text gets a contextual header (`issuer | doc_type | status | period | section_path`). That header produced FinanceBench's largest gain (hybrid+rerank 38.9 → 44.1 F1) ([arXiv 2510.24402](https://arxiv.org/html/2510.24402v1)), and Anthropic's version cut top-20 retrieval failures by 35-49% ([Anthropic](https://anthropic.com/news/contextual-retrieval)). Display text stays clean.

### The Version-Family Lattice treats near-duplicates as structure, not noise

Near-duplicate failure has two faces in this corpus, and they need different fixes. The first is **wrong-version retrieval**: a 2023 US proposal paragraph answering a 2026 question, or an FY2025 report answering an FY2026 question. The second is **distractor consolidation**: the 1,982-page Basel Framework is never cited in gold yet restates most Basel text.

The literature is unambiguous that time and version must be handled symbolically. Static RAG over a current-version index retrieved the date-applicable version **0%** of the time on a tax-law benchmark, against **98.3%** for a multi-version retriever ([FiscalQA Pro](https://arxiv.org/abs/2608.09393)). Prepending dates as tags made dense retrieval *worse* than baseline (58.37 vs 62.98 Top-1) ([TempRetriever](https://arxiv.org/html/2502.21024v2)). Dense retrievers are also weak at entity specificity: DPR averages 49.7% against BM25's 72.0% on EntityQuestions ([Sciavolino et al.](https://ar5iv.arxiv.org/html/2109.08535)).

The VFL is built in five steps.

**Step 1: catalogue.** Build one catalogue row per document with a cheap cascade: filename tokens, regex over pages 1-2 (fiscal-year phrases, "Federal Register / Vol.", RIN and docket numbers, "Consultative Document"), a regulator gazetteer, PDF metadata, and one JSON-mode LLM call to fill blanks. By our estimate, about 110k input tokens in total, the LLM call costs well under $0.10 for 35 documents.

**Step 2: families.** Form families by union-find over three kinds of edge: MinHash Jaccard on 5-word shingles at a low threshold (about 0.3), high title or first-page embedding similarity, or the same issuer with the same year-stripped title stem. MinHash and SimHash are mature for exactly this ([Broder](https://www.cs.princeton.edu/courses/archive/spring14/cos435/notes/duplicateDocs_topost.pdf); [Manku et al.](https://www.www2007.cpsc.ucalgary.ca/papers/paper215.pdf)).

**Step 3: consolidation tags.** Within a family, compute the asymmetric **containment** C(A,B) = |S(A)∩S(B)| / |S(A)|. A document that contains most of its siblings' shingles, but not the reverse, is tagged `role=consolidated`. That is the general signal that the Basel Framework is a compilation.

**Step 4: delta units.** Hash every unit. A unit whose text occurs in more than one sibling is `shared`, and is stored once with a version list. Otherwise it is `delta`, which is VersionRAG's change-node idea at paragraph grain. VersionRAG lifted change-retrieval accuracy from 25% to 70% ([arXiv 2510.08109](https://arxiv.org/html/2510.08109v1)).

**Step 5: query-time constraint split.** Following MRAG, split the query into main content plus constraints (year/FY, jurisdiction, entity, status, comparison intent). Embed only the content; MRAG showed retrievers are "distracted by dates" and gained +7.7 to +13.9 points of top-5 evidence recall ([arXiv 2412.15540](https://arxiv.org/html/2412.15540v1)). Then resolve the constraints against the catalogue *softly*, as features rather than filters. On FinanceBench, a hard LLM filename filter *lowered* claim recall from 50.7 to 47.7 ([arXiv 2510.24402](https://arxiv.org/html/2510.24402v1)).

When the parser detects comparison intent ("vs", "compared with", "changes from"), family siblings are *unioned* rather than discriminated, and delta units get a positive feature weight. This matters for the B family, where questions span 3-9 documents. Consolidation is a feature with a learned weight, not a rule, because on the hidden corpus a consolidated text may be the right source.

### Run-Aware Calibrated Relevance learns what the judge will accept

Every downstream decision needs **calibrated probabilities** p_i = P(judge marks unit i relevant | q), not rank scores. Most rerankers are rank-correct but uncalibrated, and verbalized LLM confidences are overconfident ([arXiv 2606.03535](https://arxiv.org/pdf/2606.03535)).

**Candidate pool.** First route documents by document-level aggregates of unit scores plus catalogue matches. Then score *every* unit in the routed documents rather than a fixed top-k. That is about 1-5k units per query, and it gives recall headroom for the 352-unit questions.

**Model.** A heavily regularized LightGBM binary classifier: 7-15 leaves, min_data_in_leaf of at least 50, monotone constraints on similarity features. Training uses GroupKFold by question, stratified by gold-size bucket. Features must be corpus-agnostic so they transfer to the hidden corpus:

- raw dense, BM25 and (optionally) MiniLM cross-encoder scores
- their within-query z-scores, ranks and gaps from the top score
- document-level max and mean
- VFL constraint matches and role flags
- unit type
- predicted answer cardinality
- **the scores of the previous and next units in document order**

LightGBM's lambdarank default focuses on the top 30, which conflicts with 81-352-unit gold sets, so use a pointwise objective ([LightGBM parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)).

**Why neighbour scores matter here.** Gold touches a median of 9 pages per (question, document) in about 5 non-contiguous runs of about 1.7 pages each. Relevance is therefore strongly autocorrelated along the document. Neighbour features let a weak paragraph inside a strong run be promoted, *without* returning the whole parent section, which is banned and would cost precision. If time allows, a principled upgrade is a two-state HMM over units in document order. The sticky self-transition would be set from the empirical mean gold run length, emissions would be the classifier's likelihood ratios, and forward-backward would give smoothed posteriors in O(n) per document. This is our design and has not been tested on this data, so ship the feature version first.

**Calibration.** Fit Platt scaling, p = 1/(1+exp(A·s+B)), on pooled out-of-fold scores ([Platt scaling](https://en.wikipedia.org/wiki/Platt_scaling)). Its two parameters are safe with roughly 100 effective queries, whereas isotonic regression overfits limited calibration data ([HKUST IJCAI'13](https://cse.hkust.edu.hk/~jamesk/papers/ijcai13a.pdf)). Then add **count calibration**: regress log(G+1) on log(Σp+1) and query-type features to get Ĝ. Recall's denominator is a count, so the sum of probabilities must track it.

**Label source.** Gold membership is only a proxy for the target. The real target is the judge's verdict, so a sample of high-scoring non-gold candidates should be relabelled with the judge's own prompt. This judge-aligned distillation tells us whether "non-gold but judge-relevant" is common. If it is, training on gold alone would systematically underestimate p.

### The Expected-Utility Set Selector sizes each answer's evidence

This layer is the novel centre of the design, and it rests on a clean theorem. **If labels are independent given p, the expected-F-optimal set is always a top-k prefix by probability** (Lewis's PRP for F-measure, restated as Theorem 9 in [Ye et al. 2012](https://arxiv.org/abs/1206.4625)). That shrinks the search from 2^n subsets to n+1 cutoffs.

Sort by p and take prefix sums S_k = Σ_{i≤k} p_i. Then the expected context precision is E[CP_k] = S_k/k, which is exact, and the expected capped recall is E[CR_k] ≈ min(1, S_k/Ĝ). Pick k* = argmax_k U(E[CP_k], E[CR_k]) in O(n) time. If the leaderboard averages precision and recall, adding unit k+1 helps exactly when **p_{k+1} > P_k·Ĝ/(Ĝ+k+1)**. That threshold falls toward P_k/2 as k approaches Ĝ, matching Lipton et al.'s result that the F1-optimal threshold is half the optimal F1 ([arXiv 1402.1892](https://arxiv.org/abs/1402.1892)). If the official aggregate is per-question F1, use Ye et al.'s exact O(n²) algorithm instead. At n ≤ 5,000 it costs milliseconds.

**Behaviour on PS1's question families.**

- For the A-family questions with up to 352 gold units, Ĝ is large and k* rises toward a few hundred.
- For short factual lookups, Ĝ is small and k* drops to 1-3.
- For B32, which has zero gold units, Ĝ < 0.5 and the selector returns nothing, which triggers abstention.

That two-way adaptivity is exactly what fixed top-k and gap heuristics lack. Adaptive-k recalls only about 70% of relevant passages, and only 27% on NQ ([arXiv 2506.08479](https://arxiv.org/html/2506.08479v1)). A sharp score gap often appears right after a cluster of near-duplicates, so gap rules would truncate the large questions badly.

**Two guards around the optimizer.**

*Recall floor.* Conformal risk control picks λ̂ = inf{λ : (n/(n+1))·R̂_n(λ) + 1/(n+1) ≤ α}, with per-question loss 1 − recall_q. This bounds the expected miss rate at α, about 0.1-0.2 with 100 questions ([Angelopoulos et al.](https://arxiv.org/abs/2208.02814)). The final set is top-k* ∪ {p_i ≥ λ̂}, but only where cross-validation shows the union helps. In practice, conformal filters removed 22-58% of context at α = 0.05-0.20 with flat-to-better downstream F1 ([arXiv 2511.17908](https://arxiv.org/html/2511.17908v2)). The guarantee assumes exchangeability, which a new corpus breaks, so on the hidden set it is a heuristic.

*Facet guarantee.* The first few slots greedily maximize probabilistic facet coverage, Σ_f w_f(1 − Π_{i∈S}(1 − p_{i,f})). This is monotone submodular, so greedy achieves (1 − 1/e) of the optimum ([Lin & Bilmes](https://aclanthology.org/P11-1052/)). Do *not* add an MMR redundancy penalty. Recall counts every relevant unit, so penalizing near-duplicates would hurt the score.

## Feeding all 81 paragraphs costs a fifth of a cent

Two research notes disagree with the training data. The cost note budgets 30-60 paragraphs for the generator, and the generation note suggests 8-15 chunks. Gold, however, has a median of 81 units and a maximum of 352. Following either note literally creates an honesty problem: the system returns 81 units for the context metric while the writer actually saw 15. **Resolve it by making the returned set and the generator's input the same set, and by changing how the set is presented rather than how big it is.**

The arithmetic makes this affordable. A 63-word unit with its header is about 85-90 tokens. The median set is therefore **about 7k input tokens, or $0.0021** at deepseek-v4.1-flash's $0.30 per million input tokens. The 352-unit maximum is **about 30k tokens, or $0.009**. A long answer of about 1k output tokens adds $0.0012 at $1.20 per million ([OpenCode Zen](https://opencode.ai/docs/zen/)). That puts the median query near $0.0035 and the worst case near $0.011. This is close to the cost note's own naive scenario of $0.0126 with thinking on, so **disable DeepSeek's default high-effort thinking** with `{"thinking":{"type":"disabled"}}` ([DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode)), which removes 1-3k hidden output tokens.

Size is not the problem. Attention is. Long²RAG shows key-point recall *decreasing* as packed evidence grows, and shows that replacing documents with snippets or summaries is worse still ([arXiv 2410.23000](https://arxiv.org/html/2410.23000v3)). Models also neglect mid-context evidence ([Lost in the Middle](https://arxiv.org/abs/2307.03172)). The Facet-Partitioned Writer addresses this in five ways:

1. **Order and grouping.** Units are grouped facet → document → run and kept in original document order inside each run. Order-preserving RAG at 16-48k tokens beat full-context reading on En.QA, 44.4-47.3 F1 against about 34 ([OP-RAG](https://arxiv.org/html/2409.01666v1)).
2. **Shared-text collapsing.** Shared VFL units that appear in several family members are shown once with all their citations. That cuts tokens without hiding any unit.
3. **Partitioning.** When k* exceeds about 60, each facet group goes to its own parallel call, with the static system prompt and few-shot block placed first so that prefix caches at $0.006 per million ([DeepSeek KV cache](https://api-docs.deepseek.com/guides/kv_cache)). Each unit appears in exactly one call, so total input tokens equal those of the single-call version, decode time drops about 3-4x, and no call sees more than about 25 units.
4. **Output format.** Each call emits an ID-only plan (heading → unit IDs) and then the answer, in one completion. This is the prompt-only form of attribute-first generation ([Slobodkin et al.](https://aclanthology.org/2024.acl-long.182)) and of Anthropic's quote-first guidance ([Anthropic](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations)). Unit IDs replace verbatim quotes because output tokens cost 4x input tokens.
5. **Stitching.** A deterministic step merges the sections under gold-style headings. It is not an LLM call.

Ĝ also sets answer length. A small Ĝ selects the 1-3 sentence factual template, because gold has 1-3 points there and any elaboration becomes "extra". A large Ĝ selects the analyst template, with headings, one atomic fact per bullet, `[file.pdf pX]` citations, and "the excerpts do not specify…" hedges for facets the evidence leaves unaddressed. Gold uses those hedges and the judge counts them as points. Whether Ĝ predicts gold answer length is a hypothesis to check in experiment E3.

The **Deterministic Claim Gate** then runs with zero LLM tokens:

- Every number, after normalizing commas, %, crore/lakh and years, and every capitalized entity in a bullet must appear in its cited units or the set. Otherwise the bullet is dropped or re-cited.
- Optionally, MiniCheck-Flan-T5-L (MIT, 75.0 on LLM-AggreFact) deletes a bullet only when p < 0.3 against both the cited unit and the full set ([LLM-AggreFact](https://llm-aggrefact.github.io/)).
- Abstention fires only when Ĝ < 0.5 *and* the maximum p is below a threshold tuned to reject B32 without rejecting any answerable question. The output is then the exact gold string, "No information was retrieved for this question.", because any paraphrase would be atomized into non-matching points.

## Ranked alternatives and why each lost

| Rank | Alternative | Verdict | Reason |
|---|---|---|---|
| 1 | PageIndex-style TOC-tree navigation as the candidate generator | Strong backup or hybrid | Vendor-reported 98.7% on FinanceBench, unverified and with human-adjudicated ground truth ([VectifyAI](https://github.com/VectifyAI/Mafin2.5-FinanceBench)). It adds 1-2 LLM calls per hop. Section-level output hurts precision because gold covers about 40% of a page. Best use: an extra document/section-routing feature for RACR |
| 2 | Late interaction (ColBERTv2) as a scoring feature | Add if GPU or time allows | Best retriever on DocFinQA ([arXiv 2401.06915](https://arxiv.org/html/2401.06915v1)). It is a scoring function, not an architecture, so it fits as one more RACR feature |
| 3 | Hard-negative fine-tuning of a small bi-encoder on VFL sibling pairs | Stretch goal | Distractors measurably hurt dense retrievers ([arXiv 2304.03031](https://arxiv.org/pdf/2304.03031)), and version siblings give free hard negatives. No evidence quantifies the gain on version confusion |
| 4 | Self-Route / long-context per document | Rejected | Long context beat RAG for GPT-4o, 48.67 vs 32.60 ([arXiv 2407.16833](https://arxiv.org/html/2407.16833v1)), but 200-1,982-page documents make it costly. It also does not produce returned chunks for the context metrics |
| 5 | RAPTOR cluster-summary tree | Rejected | Thousands of summarization calls, suited to thematic questions rather than exact figures, and no financial results ([arXiv 2401.18059](https://www.arxiv.org/abs/2401.18059)) |
| 6 | HippoRAG 2 (OpenIE + PPR) | Rejected | Close to banned GraphRAG, and OpenIE over 8,600 pages is costly ([arXiv 2502.14802](https://arxiv.org/abs/2502.14802)) |
| 7 | Gap-based adaptive-k as the cutoff | Fallback only | About 70% recall; unsuitable for summarization-style queries ([arXiv 2506.08479](https://arxiv.org/html/2506.08479v1)). Keep the TAR knee rule as a label-free sanity bound ([arXiv 2106.09871](https://arxiv.org/pdf/2106.09871)) |
| 8 | ColPali page-image retrieval | Rejected on CPU | Only about 3 points over OCR+BM25 on financial reports ([arXiv 2609.13268](https://arxiv.org/pdf/2609.13268)) and needs a GPU |
| 9 | LLMLingua / RECOMP evidence compression | Rejected | Compression lowers long-form key-point recall ([arXiv 2410.23000](https://arxiv.org/html/2410.23000v3)) and breaks the "returned = used" honesty |

## Risks: what 0.95 realistically means

The biggest risk is **chasing a target the published record says nobody has hit**. No peer-reviewed system reports 0.95 under LLM-judged context or answer metrics on financial QA. The best academic FinanceBench pipeline sits at F1 44.4 ([arXiv 2510.24402](https://arxiv.org/html/2510.24402v1)), and the best RAGChecker configuration at 52.7 ([RAGChecker](https://arxiv.org/html/2408.08067v2)).

Our situation differs in one way that matters. Gold answers were generated by an LLM *from* the gold units, so a writer that sees nearly the gold set and imitates its style has a far higher ceiling than open-ended benchmarks suggest. How high is unknown until the oracle run.

Expected outcomes by metric:

- **Context precision** near 0.9 is plausible, because the judge is lenient toward on-topic paragraphs; FActScore found LLM estimators mostly err toward "Supported" ([arXiv 2305.14251](https://arxiv.org/pdf/2305.14251)).
- **Context recall** near 0.9 is plausible because of the count-based definition.
- **Answer recall** is the hardest metric. It is capped by how many of 40-80 gold points one writer can surface.

Other risks, in rough order of severity:

1. **Calibration does not transfer to the hidden corpus**, and that set has a different question mix. The mitigation is three-fold: rank-normalized features, only two Platt parameters, and a fallback to the knee rule when the mean p drifts from training.
2. **The judge disagrees with gold.** Relabelling a sample with the judge prompt measures the gap.
3. **The judge degrades on long inputs.** A 352-unit set is a 30k-token judge prompt, so the judge may miss relevant units in the middle. Facet-grouped ordering of the *returned* list helps the judge as well.
4. **The banned-list boundary.** A cross-encoder used as a feature could be read as "rerank". Run an ablation without it so the classifier stands on its own.
5. **Cost-accounting ambiguity.** It is unclear whether cost is billed or counted at list price. Separately, the free `deepseek-v4.1-flash-free` tier may use data for training ([OpenCode Zen](https://opencode.ai/docs/zen/)).
6. **Zen behaviour is undocumented.** Rate limits and cache behaviour need empirical checks before relying on parallel calls.
7. **Filename aliases** (fact-sheet-SUMMARY.pdf, 91fr14952.pdf, SAMA_EN_3575_VER1.pdf) only affect our own dev scripts, which need an alias map. The official judge reads chunk text.
8. **Zero-gold handling.** Confirm how the notebook handles division by zero for B32 before tuning abstention.

## A 12-hour plan for two people

Person A owns retrieval (layers 1-4). Person B owns generation, evaluation and judge alignment (layers 5-6). The hand-offs are a unit schema at hour 2, a working candidate scorer at hour 5, and calibrated p with Ĝ at hour 8.

| Hours | Person A: retrieval | Person B: generation and evaluation |
|---|---|---|
| 0-1 | Both: run the **official evaluator unmodified** on E0 (naive baseline) and E1 (oracle) | Wrap the notebook for batch runs; cache judge responses by input hash |
| 1-3 | TEU ingestion: PyMuPDF, OCR triage, header stripping, numbered paragraphs, tables; compare unit word-count histogram with gold | E1 oracle: feed gold units to the writer and tune the gold-style prompt until answer metrics plateau. This is the ceiling |
| 3-5 | bge-small embeddings + BM25; catalogue cascade; MinHash families, containment, delta flags; constraint parser | Writer templates by Ĝ bucket; ID-plan-then-answer format; Langfuse cost logging with manual usage details |
| 5-8 | Feature table; LightGBM with GroupKFold; Platt; Ĝ count calibration; EUSS selector | Judge-aligned relabelling of about 1-2k non-gold candidates (feeds A's labels); claim gate; abstention |
| 8-10 | Conformal floor, facet slots, neighbour/run features; no-cross-encoder ablation | Facet-partitioned parallel writing; check cache hits; end-to-end runs on held-out folds |
| 10-12 | Freeze; ingestion timing; fallback switches | Final official-evaluator run; cost and latency table; write-up |

Run the experiments in this order, because each one decides a design question before code is sunk into it.

**E1, the oracle ceiling.** Return exactly the gold units and generate from them. Context precision on this set shows how often the judge rejects gold, and the answer metrics show the writer ceiling. Every later number is read against these two.

**E2, judge versus gold agreement.** Have the judge relabel high-scoring non-gold candidates. This decides whether RACR trains on gold or on judge labels.

**E3, Ĝ predictability.** Check how well Σp and query features predict both |gold| and gold answer length across 100 questions. If they do not, the selector needs a dedicated count regressor.

**E4, granularity.** Compare units of about 40, 65 and 120 words under the official judge. This confirms the paragraph-size claim on our data rather than by argument.

**E5, selection rule.** Compare fixed k = 15/60/120 against EUSS on held-out folds. This is the core claim of the architecture.

**E6, generator input.** Compare the full selected set against a 15-unit subset, reporting answer recall and cost. This settles the reconciliation empirically.

**E7, versions.** Ablate the VFL on the 32 B-family questions and on synthetic version queries built from delta units, measuring version accuracy at rank 1.

## Conclusion

The banned list removes every technique that tries to fix retrieval by searching harder. What it leaves open turns out to be the real lever for this metric: **deciding how much to return as a calibrated statistical estimate rather than a fixed constant**. Once relevance is a probability and the gold count is an estimate, chunk size stops being a trade-off between completeness and focus. Units can be as small as the gold's and as numerous as the question needs, and one estimate, Ĝ, sets retrieval depth, generator budget and answer length together. Version confusion splits the same way: it is a symbolic problem that a catalogue and shingle lattice resolve better than any embedding, and the evidence says so plainly.

The most valuable artifact of the 12 hours may be the oracle measurement rather than the system. It shows whether the 0.95 target is reachable at all under this judge, and how much of the gap belongs to retrieval and how much to writing. A team that knows its ceiling can make honest claims to the judges, and a set selector built on calibrated probabilities is the component most likely to carry over to a hidden corpus it has never seen.
