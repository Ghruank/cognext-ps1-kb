# Faithful, Complete Long-Form Grounded Generation (scoring high under an atomic-point precision/recall LLM judge)

Context: CognitionNext '26 PS1. The judge (deepseek-v4.1-flash, temperature 0, JSON output) atomizes both our answer and the gold answer. Precision = (our points - extra) / our points. Recall = matched gold / gold points. The gold answers are LLM-written, analyst-style (median 754 words, max about 1,700) with `[file.pdf pNN]` citations and hedges. Short questions get gold answers of 1-3 sentences. The unanswerable gold answer is "No information was retrieved for this question." Self-reflection loops, ReAct and CRAG are banned as core.

Research date: 2026-10-23. About 20 tool calls. Wherever a number below comes only from a secondary source, it is marked as such.

---

## 1. Atomic-claim evaluation literature: how judges decompose and match, and what answer properties score well

### Takeaway
Every major framework (FActScore, SAFE, RAGAS faithfulness, RAGChecker, Long²RAG KPR, ALCE) does the same three things. It splits text into atomic claims, checks each claim for entailment against a reference (evidence or gold claims), and reports the supported fraction. Our judge is RAGChecker's "overall precision/recall" run against gold claims. The winning answer is therefore made of many short, self-contained, evidence-paraphrasing statements that cover the gold claim set, with no padding that the judge could count as an unsupported point.

### Cited Findings
- **FActScore** (Min et al., EMNLP 2023) breaks a generation into atomic facts and computes the percentage supported by a knowledge source. Human-evaluated FActScores were 42% for InstructGPT, 58% for ChatGPT and 71% for PerplexityAI. ChatGPT fell from 80% to 16% for rare entities. The automated estimator (retrieval plus a strong LM) approximates FActScore with an error rate below 2% in its best configuration, but plain Retrieve→LM can overestimate by up to 17%. The most common estimator error is labelling unsupported facts as Supported. — [FActScore arXiv 2305.14251](https://arxiv.org/pdf/2305.14251)
- **SAFE / LongFact** (Wei et al., Google DeepMind, NeurIPS 2024) uses an LLM to split a long response into individual facts, then checks each one with multi-step Google Search. It agrees with 72% of human annotations and is more than 20x cheaper ($0.19 vs $4.00 per response). Its metric, F1@K, combines precision (supported fraction) with recall against a target fact count K (K=64 is the median fact count, per a secondary source). — [SAFE arXiv 2403.18802](https://ar5iv.labs.arxiv.org/html/2403.18802); K=64 detail from [alphaxiv LongFact](https://www.alphaxiv.org/benchmarks/google-deepmind/longfact) (secondary)
- **RAGAS faithfulness** = (number of response claims supported by the retrieved context) / (total response claims). The response is decomposed into statements and each is checked for inferability from the context. `FaithfulnesswithHHEM` replaces the LLM check with Vectara HHEM-2.1-Open. — [RAGAS docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
- **RAGChecker** (Amazon, NeurIPS 2024 D&B) is the closest analogue to our judge. It decomposes both the response and the ground-truth answer into claims and checks entailment, using Llama3-70B-Instruct through RefChecker. Overall precision is the fraction of response claims entailed by the ground truth. Overall recall is the fraction of ground-truth claims entailed by the response. — [RAGChecker arXiv 2408.08067v2](https://arxiv.org/html/2408.08067v2)
  - Diagnostic generator metrics: faithfulness (response claims entailed by retrieved chunks), relevant and irrelevant noise sensitivity (incorrect claims drawn from chunks), hallucination (incorrect and unsupported by any chunk), self-knowledge (correct but not in any chunk), context utilization (retrieved ground-truth claims that appear in the response). — same source
  - Meta-evaluation on 280 human-labelled pairs (Pearson): RAGChecker reached 49.66 on correctness, 60.67 on completeness and 61.93 overall. The other metrics scored 48.31 overall (RAGAS answer similarity), 35.15 (TruLens) and 17.81 (ARES). Inter-human agreement was 70.09. — same source
  - Findings: **context utilization correlated most strongly with overall F1**. Higher retriever recall raises generator noise sensitivity. Relevant noise is harder to resist than irrelevant noise. Open models trust context more blindly than GPT-4. The best system (E5-Mistral + GPT-4) reached only F1 52.7. — same source
  - Tuning: raising k from 5 to 20 chunks took claim recall from 61.5 to 77.6, faithfulness from 88.1 to 92.2 and F1 from 51.7 to 53.4, while noise sensitivity rose from 34.0 to 35.4. Raising chunk size from 150 to 300 tokens took claim recall from 70.3 to 77.6 and F1 from 52.6 to 53.4. **Explicit prompt requirements** (instructions to be faithful) took faithfulness from 92.2 to 93.6 and context utilization from 59.2 to 63.7, but noise sensitivity from 35.4 to 38.1. — same source
- **Long²RAG KPR** (EMNLP 2024 Findings): KPR = mean over questions of (key points entailed by the response / key points). Key points are extracted from the retrieved documents by GPT-4-class models, filtered, then human-verified (2,055 points across 280 questions, κ=0.56). GPT-4o judges entailment with 87% agreement with annotators. — [Long²RAG arXiv 2410.23000v3](https://arxiv.org/html/2410.23000v3)
- **ALCE** (Gao et al., EMNLP 2023) is the citation benchmark. A statement's citation recall is 1 if it has at least one citation and the NLI model judges the cited passages to entail it. Citation precision penalizes irrelevant citations. On ELI5, "even the best models lack complete citation support 50% of the time." — [ALCE arXiv 2305.14627](https://arxiv.org/abs/2305.14627); recall definition via [AGREE/ALiiCE search snippets](https://arxiv.org/pdf/2311.09533)

### Inferences (for our judge)
- **Granularity.** The judge atomizes whatever we write, so one compound sentence with five numbers becomes five points. Each correct, evidence-derived point raises the matched count. One wrong number makes that point "extra". Explicit numbers with units, dates, entity names and percentages exactly as the evidence states them are the highest-yield points, because gold answers were written from the same paragraphs and will contain them.
- **"Extra" ≠ "wrong".** A point is extra whenever no gold point supports it, even when it is true. Content from retrieved-but-non-gold chunks is therefore as damaging to precision as hallucination. This is RAGChecker's "relevant noise sensitivity". Context *precision* (dropping off-topic chunks before generation) matters as much as generator faithfulness.
- **Recall dominates for long questions.** With about 750-word golds (roughly 40-80 atomic points, assuming about 10-15 words per point), dropping a whole evidence paragraph costs many recall points. One extra sentence costs one precision point out of many. Covering everything relevant is the bigger lever, but only with evidence content.
- **Hedges are points.** Gold hedges such as "the excerpts do not explicitly label X" will be atomized into gold points. If we make the same evidence-scoped hedge when the evidence is silent, it matches. Generic disclaimers ("consult a professional", "this may vary") become extra points, so ban them.
- **Structure.** Markdown headings and bullets are not facts, so they cost nothing. Bullets make the judge's decomposition cleaner (one fact per bullet), which likely gives more reliable matching. Citations like `[file.pdf p52]` are probably ignored or folded into the point. Keep them, since the gold uses them, but never invent page numbers.
- **Length matching.** Short factual questions: answer in 1-3 sentences. Extra elaboration only adds points that are likely extra, because a short gold has 1-3 points.
- **Judge leniency asymmetry.** LLM judges tend to over-credit support (FActScore: ChatGPT-based estimators mostly mislabel unsupported facts as Supported). Close paraphrase of the evidence wording therefore maximizes the chance a point is matched rather than flagged.

### Gaps
- No published data on how deepseek-v4.1-flash in particular atomizes text, e.g. whether it counts citations or headings as points. This should be calibrated empirically on a few dev questions.
- No paper found that directly measures how bullet vs prose format affects atomic-judge precision/recall.

---

## 2. Attributed and grounded generation: quote-first, attribute-first, FRONT, AGREE, constrained citation

### Takeaway
"Select evidence first, then generate conditioned only on the selection" is the consistently best pattern. Training-based versions (FRONT, AGREE) report double-digit citation-quality gains. The prompting-only version (Anthropic's "extract word-for-word quotes first, then answer only from the quotes") is free to adopt and fits our no-loop constraint.

### Cited Findings
- **Anthropic guidance** has three parts. (1) Allow "I don't know": "This technique can drastically reduce false information." (2) For long documents (>20k tokens), "extract word-for-word quotes first before performing its task", with the example prompt "Extract exact quotes ... If you can't find relevant quotes, state 'No relevant quotes found.' ... Use the quotes to analyze ... referencing the quotes by number. Only base your analysis on the extracted quotes." (3) Verify with citations: "For each claim, find a direct quote from the documents that supports it. If you can't find a supporting quote for a claim, remove that claim." It also recommends **external knowledge restriction**: "Explicitly instruct Claude to only use information from provided documents and not its general knowledge." — [Anthropic: Reduce hallucinations](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations)
- **Attribute First, then Generate** (Slobodkin, Hirsch, Cattan, Schuster, Dagan, ACL 2024) splits generation into content selection → sentence planning → sentence-by-sentence generation conditioned on the selected source segments, which then double as fine-grained attributions. On multi-document summarization and LFQA it gives more concise citations while maintaining or improving generation quality and attribution accuracy, and it significantly reduces human fact-verification time. — [ACL Anthology 2024.acl-long.182](https://aclanthology.org/2024.acl-long.182); [Google Research page](https://research.google/pubs/attribute-first-then-generate-locally-attributable-grounded-text-generation/)
- **FRONT** (Huang et al., Findings ACL 2024) is two-stage training: first select supporting quotes, then generate guided by them, with preference optimization to keep grounding and generation consistent. On ALCE with LLaMA-2-7B it reports an **average 14.21% improvement in citation quality across all datasets, surpassing ChatGPT**. It was tested only at 7B and 13B. — [FRONT arXiv 2408.04568](https://arxiv.org/pdf/2408.04568)
- **AGREE** (Ye et al., NAACL 2024) tunes LLMs to self-ground claims and cite retrieved documents, and adds test-time adaptation that retrieves more for unsupported claims. It "substantially outperforms all the baselines" on 5 datasets, with a large citation-precision gap over in-context citing. TTA raised Llama-2 answer correctness by +3.1 (NQ) and +3.7 (ASQA). — [AGREE arXiv 2311.09533](https://arxiv.org/pdf/2311.09533)
- **ALCE baselines**: in-context citing leaves about 50% of ELI5 statements without full citation support even for the best models. — [ALCE](https://arxiv.org/abs/2305.14627)
- **RAGChecker** shows that adding explicit faithfulness requirements to the prompt raises faithfulness by 1.4 points and context utilization by 4.5 points (see §1). — [RAGChecker](https://arxiv.org/html/2408.08067v2)

### Inferences
- We cannot fine-tune in a hackathon. FRONT and AGREE matter mainly as evidence that **quote/segment selection before writing** is the mechanism that drives the gains. The prompt-only equivalent is Anthropic's quote-first pattern in one call: emit `<quotes>` (numbered, verbatim, with file and page), then `<answer>` that may only restate quoted content. This is a single forward pass, not a reflection loop.
- Making the selection step **exhaustive over relevant chunks** (one quote block per relevant chunk) also serves completeness (§4).
- Constrained decoding (forcing a citation token per sentence) is unnecessary with strong API models. Instead, enforce the format "every bullet ends with `[file p#]`" and validate it with a regex after generation.

### Gaps
- Exact AGREE table numbers (citation recall/precision per dataset) were not retrieved; see Table 2 of arXiv 2311.09533.
- "Learning to cite" papers beyond FRONT and AGREE were not separately reviewed.
- No head-to-head numbers were found for prompt-only quote-first versus direct answering on long-form RAG with a modern frontier model.

---

## 3. Cheap post-hoc claim verification: NLI and fact-checker models, accuracy and latency

### Takeaway
On LLM-AggreFact, small dedicated checkers match or beat frontier LLMs. Bespoke-MiniCheck-7B scores 77.4 (vs Claude-3.5 Sonnet 77.2, GPT-4o 75.9), but its license is non-commercial. MiniCheck-Flan-T5-L (0.8B, MIT) scores 75.0 and FactCG-DeBERTa-L (0.4B) scores 75.6. HHEM-2.1-Open (about 0.1B, Apache-2.0) runs on CPU. For a hackathon, MiniCheck-Flan-T5-L or Bespoke-MiniCheck-7B (if non-commercial use is acceptable) is the best claim-vs-evidence filter. Combine it with exact number and string matching.

### Cited Findings
- **LLM-AggreFact leaderboard** (11 datasets: AggreFact-CNN/XSum, TofuEval-MediaS/MeetB, WiCE, REVEAL, ClaimVerify, FactCheck-GPT, ExpertQA, LFQA, RAGTruth). It shows 42 models; the top 14 by average are: — [LLM-AggreFact leaderboard](https://llm-aggrefact.github.io/)
  - nimble-v4-max (27B) 79.5; Jev 1.1.3.0 78.7; nimble-v4 (9B) 77.8; **Bespoke-MiniCheck-7B 77.4**; Claude-3.5 Sonnet 77.2; **Granite Guardian 3.3 (8B) 76.5**; Mistral-Large 2 (123B) 76.5; GPT-4o-2024-05-13 75.9; **FactCG-DeBERTa-L (0.4B) 75.6**; Qwen2.5-72B 75.6; **MiniCheck-Flan-T5-L (0.8B) 75.0**; Llama-3.3-70B 74.5; Llama-3.1-405B 74.4; QwQ-32B 71.8.
  - The RAGTruth column (closest to our use) reads: nimble-v4-max 90.5, nimble-v4 88.9, Jev 87.0, Claude-3.5 Sonnet 86.1, Mistral-Large 2 85.9, GPT-4o 84.3, Bespoke-MiniCheck-7B 84.0, FactCG-DeBERTa-L 82.3, Granite Guardian 3.3 82.2, MiniCheck-Flan-T5-L 78.0. The ExpertQA column stays around 59-61 for every model, which suggests a label-noise ceiling.
  - The page does not label the metric (the MiniCheck paper uses balanced accuracy) and shows no update date.
- **MiniCheck paper** (Tang, Laban, Durrett, EMNLP 2024): MiniCheck-FT5 (770M) "outperforms all systems of comparable size and reaches GPT-4 accuracy" at "400x lower cost". It was trained on synthetic GPT-4-generated hard examples. — [MiniCheck arXiv 2404.10774](https://arxiv.org/abs/2404.10774); [ACL Anthology](https://aclanthology.org/2024.emnlp-main.499)
- **Bespoke-MiniCheck-7B** (HF `bespokelabs/Bespoke-MiniCheck-7B`): fine-tuned from internlm2.5-7b-chat on 35K examples (21K ANLI plus 14K synthetic from Llama-3.1-405B). **License CC BY-NC 4.0 (non-commercial)**. Input is a document plus a single-sentence claim; output is {0,1} with a probability. Documents are chunked only above 32K tokens. Throughput exceeds 500 docs/min on one A6000 with vLLM, and automatic prefix caching is supported. Multi-sentence claims must be split into sentences. — [HF card](https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B)
- **MiniCheck-Flan-T5-Large** (HF `lytang/MiniCheck-Flan-T5-Large`): **MIT license**. "Best fact-checking model with size < 1B", on par with GPT-4 at 400x lower cost, and 4-10% absolute better than similar-scale checkers. Sentence-level claims. About 500 docs/min depending on hardware. Siblings MiniCheck-RoBERTa-Large and MiniCheck-DeBERTa-v3-Large are 0.4B each. Install with `pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"`. — [HF card](https://huggingface.co/lytang/MiniCheck-Flan-T5-Large)
- **FactCG-DeBERTa-L** (0.4B; FactCG, NAACL 2025) is trained on graph-based multi-hop synthetic data and then fine-tuned on MiniCheck D2C. It scores 75.6 average and the best ClaimVerify score (78.5) on the leaderboard. — [FactCG arXiv 2501.17144](https://arxiv.org/pdf/2501.17144); [leaderboard](https://llm-aggrefact.github.io/). License not verified.
- **Vectara HHEM-2.1-Open** (HF `vectara/hallucination_evaluation_model`): Apache-2.0, flan-t5-base (about 0.1B), unlimited context. **About 1.5 s for a 2k-token input on a modern x86 CPU, under 600 MB RAM.** Balanced accuracy: AggreFact-SOTA 76.55% (GPT-4 73.78%), RAGTruth-Summ 64.42% (GPT-4 62.62%), RAGTruth-QA 74.28% (GPT-4 74.11%). — [HF card](https://huggingface.co/vectara/hallucination_evaluation_model)
- **AlignScore** (RoBERTa-large backbone): NVIDIA NeMo Guardrails measured about 45 ms per check in-memory and about 220 ms as an HTTP service. — [NeMo Guardrails AlignScore docs](https://docs.nvidia.com/nemo/guardrails/latest/configure-guardrails/guardrail-catalog/third-party/alignscore)
- **LettuceDetect** (ModernBERT token-level hallucination span detector, MIT per a secondary source): example-level F1 79.22% on the RAGTruth test set (large model), 14.8% better than Luna. 30-60 examples/s on a single GPU. Context 4k-8k tokens (sources disagree). — [LettuceDetect arXiv 2502.17125](https://huggingface.co/papers/2502.17125); [PyPI](https://pypi.org/project/lettucedetect)

### Inferences
- **Best small open model for our use:** Bespoke-MiniCheck-7B (77.4 avg, 84.0 RAGTruth) if non-commercial use is acceptable for a hackathon. It needs a GPU (about 16 GB at bf16). Otherwise use **MiniCheck-Flan-T5-L** (MIT, 75.0, CPU-feasible but slow) or **FactCG-DeBERTa-L** (0.4B, 75.6, fastest). Granite Guardian 3.3 8B (76.5) is an IBM Apache-licensed option, but its license was not verified here.
- These checkers are about 75-77% balanced accuracy, so roughly one judgment in four is wrong. **Use them as a filter on high-risk sentences, not as ground truth.** Recommended policy: delete a sentence only when (a) the checker's probability is below about 0.3 against its cited chunk **and** against the full evidence pack, or (b) it contains a number or proper noun absent from the evidence (deterministic check). This protects recall, which is costlier to lose than precision.
- **Deterministic number and entity check** is the cheapest and most precise verifier: regex-extract every number (normalize 1,234 / 1.2k / %, years) and capitalized entity from each output sentence, and require each to appear in the cited chunk or the evidence pack. Numbers are the most common hallucinated detail in financial-report RAG and the easiest point for the judge to mark as extra.
- Latency budget per answer (inference): about 40 bullets × about 2 chunks. On FactCG or AlignScore that is about 80 × 45 ms ≈ 4 s CPU, or under 1 s GPU. On MiniCheck-7B with vLLM at about 500 docs/min it is about 10 s; with prefix caching on a shared evidence document it is faster.

### Gaps
- Licenses for FactCG-DeBERTa-L, AlignScore and Granite Guardian 3.3 were not verified in this session.
- No per-claim GPU latency numbers were published for MiniCheck-Flan-T5-L.
- The newer leaderboard leaders (nimble-v4, Jev) have no model card reviewed here; their availability and licenses are unknown.

---

## 4. Completeness: outline/facet-first, coverage-driven generation, long-evidence degradation

### Takeaway
Long-form RAG loses key points as context grows (Long²RAG: KPR falls with document length; even GPT-4o covers only about 58% of key points). Models also under-use mid-context evidence (lost-in-the-middle). Compressing evidence (truncation, snippets, summaries) makes recall worse. The best mitigations are to keep the full relevant evidence, force explicit per-chunk coverage (quote/plan step listing every relevant chunk), and place evidence well.

### Cited Findings
- **Long²RAG**: average KPR (fraction of document key points entailed) was GPT-4o 0.579, Claude-3-Sonnet 0.477, GPT-4-Turbo 0.469, Qwen2-72B 0.449, Phi-3-mini-128K (3.8B) 0.434, Qwen2-7B 0.416, Mixtral-8×22B 0.383, Mistral-7B 0.373, Qwen2-1.5B 0.262. **KPR generally decreases as packed retrieved-document length increases**, with a small uptick around 16-25K tokens. **Truncating documents lowers KPR, and replacing documents with snippets or GPT-4o summaries lowers it further**, so the authors conclude full long context beats compression. The paper does not analyze key-point position. — [Long²RAG arXiv 2410.23000v3](https://arxiv.org/html/2410.23000v3)
- **Lost in the Middle** (Liu et al., TACL 2024): on multi-document QA and key-value retrieval, performance is highest when relevant information is at the beginning or end, "and significantly degrades" when it is in the middle, even for long-context models. — [arXiv 2307.03172](https://arxiv.org/abs/2307.03172). A secondary source cites a 20-30 point drop with 20 documents ([inblog glossary](https://inblog.ai/de/glossary/lost-in-the-middle), unverified).
- **RAGChecker**: context utilization is the generator metric most correlated with overall F1. More, larger chunks raise claim recall (61.5 → 77.6) and F1 but also noise sensitivity. — [RAGChecker](https://arxiv.org/html/2408.08067v2)
- **Attribute-First** sentence planning (content selection → plan → generate) is the published form of outline-first grounded generation. — [ACL 2024](https://aclanthology.org/2024.acl-long.182)

### Inferences
- Gold answers were written *from the gold evidence paragraphs*, so the gold point set is roughly the union of facts in those paragraphs that relate to the question. **Coverage-driven generation means every retrieved-and-relevant paragraph must contribute its facts.** Concretely: (1) a relevance filter at chunk level; (2) a plan step listing, for each kept chunk, the facts it contributes, as an outline of headings → bullets with chunk ids; (3) a cheap deterministic check that every kept chunk id is cited at least once in the final answer. If a chunk is uncited, run one targeted "add bullets for chunk X" call. That is a coverage repair keyed on a missing id, not open-ended self-reflection, but confirm with organisers.
- Against lost-in-the-middle: order chunks by relevance with the strongest at the start *and* end ("sandwich" ordering), give each chunk a short header (`[doc p#] section title`), and keep total evidence modest. For a gold of about 750 words, the gold evidence is probably 3-10 paragraphs, so 8-15 well-filtered chunks is sufficient. The quote-extraction step itself counteracts positional neglect, because the model must walk every chunk.
- Do **not** summarize or compress chunks before generation (Long²RAG shows recall loss). Filter whole chunks in or out instead.
- Expected recall ceiling: even GPT-4o reached only about 58% KPR naively. That leaves large headroom for an explicit per-chunk coverage step. This is an inference, not a measured number.

### Gaps
- No paper found that directly measures outline-first vs direct generation on KPR or RAGChecker recall with numbers.
- The exact lost-in-the-middle magnitudes were not verified from the primary paper in this session.
- STORM-style or perspective-guided outline papers were not reviewed.

---

## 5. Abstention and answerability detection in RAG

### Takeaway
Unanswerable detection in RAG is mostly a *retrieval* failure signal. No single RAG configuration wins on both answerable and unanswerable queries (UAEval4RAG). For us, with one known unanswerable gold, the rule is: if the best chunk is below a calibrated reranker threshold, *or* the quote step returns "No relevant quotes found", emit exactly "No information was retrieved for this question."

### Cited Findings
- **UAEval4RAG** (Salesforce, ACL 2025) defines six categories of unanswerable RAG queries and auto-generates them for any knowledge base. It scores an unanswered ratio and an acceptable ratio. Across 4 datasets **no single RAG configuration was best on both answerable and unanswerable queries**. The paper argues that RAG rejections often stem from retrieval failure rather than recognition of unanswerability. — [arXiv 2412.12300](https://arxiv.org/html/2412.12300v3)
- **Anthropic**: explicitly permitting "I don't have enough information" "can drastically reduce false information". The quote-first prompt includes the sentinel "If you can't find relevant quotes, state 'No relevant quotes found.'" — [Anthropic docs](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations)

### Inferences
- Two-signal abstention: (a) reranker max-score threshold, calibrated on the dev set by picking a threshold that rejects the known unanswerable and keeps all answerable ones; (b) the LLM quote step returning zero quotes. Abstain if (b) holds or if (a) holds clearly. Output the exact gold string, since any other wording would be atomized into non-matching points.
- **Partial answerability is much more common than full unanswerability.** When evidence covers some facets but not others, follow the gold style: answer what is supported, then state an evidence-scoped hedge ("The provided excerpts do not specify X"). The judge treats that hedge as a point and the gold often has it. Never fill the gap with parametric knowledge.
- Over-abstention is costly: abstaining on an answerable question gives recall 0. Keep the threshold conservative, i.e. abstain only on very low scores.

### Gaps
- RGB's (Chen et al. 2024) negative-rejection numbers were not retrieved.
- No calibrated numeric thresholds were found for cross-encoder reranker scores; these are dataset-specific.

---

## 6. Prompting patterns that keep the model on the provided context (no parametric knowledge)

### Takeaway
Combine an explicit knowledge restriction, quote-first grounding, per-claim citation requirements, permission to say "not stated", and a ban on generic filler. RAGChecker shows that explicit instructions measurably raise faithfulness and utilization. Anthropic recommends exactly these levers.

### Cited Findings
- "Explicitly instruct Claude to only use information from provided documents and not its general knowledge." Quote-first, then "Only base your analysis on the extracted quotes." Post-draft claim checking: "If you can't find a supporting quote for a claim, remove that claim." — [Anthropic docs](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations)
- RAGChecker defines "self-knowledge" as correct claims not in any chunk; these count as non-faithful. Explicit prompt requirements improved faithfulness from 92.2 to 93.6 and utilization from 59.2 to 63.7. — [RAGChecker](https://arxiv.org/html/2408.08067v2)
- RAGChecker also found that open models trust context more blindly than GPT-4, so they adhere better but are more noise-sensitive. — same source

### Inferences
Prompt rules to include, each targeting a judge failure mode:
1. "Use ONLY facts stated in the EVIDENCE. Do not add background, definitions, implications, recommendations or general knowledge, even if true." This targets extra points from self-knowledge.
2. "Each bullet states one fact and ends with its citation `[file.pdf pNN]` copied from the chunk header." This gives atomic granularity and makes verification possible.
3. "Copy numbers, units, dates and names exactly as written, and do not compute new figures unless the question asks for a calculation." Show inputs if a calculation is asked. Derived numbers are a frequent extra point.
4. "If the evidence does not state something the question asks, write one sentence: 'The provided excerpts do not specify …'." This matches gold hedges.
5. "Ignore evidence chunks unrelated to the question." This reduces relevant-noise extras.
6. "No introduction, no conclusion that repeats points, no generic caveats." Repeated points can be double-counted, and caveats count as extra.
7. Length guidance tied to question type: "factual lookup → 1-3 sentences; analytical → headed sections with bullets covering every relevant excerpt."

### Gaps
- No quantified study isolating each instruction's effect on an atomic judge was found. Rules 3-6 are inferences from the metric definitions.

---

## Recommended generation recipe for our system

(This section is an inference/synthesis built on the cited findings above.)

**Step 0: Route and abstain (no LLM, or one tiny call).**
- Classify the question as *factual-short* or *analytical-long*. Use a heuristic (wh-question about a single value or entity) or a tiny LLM classifier at about 150 input tokens and 5 output tokens.
- Abstain if the top reranker score is below the calibrated threshold. Output exactly `No information was retrieved for this question.`

**Step 1: Evidence pack (no LLM).**
- Keep the top 8-15 reranked chunks above a relevance floor (cross-encoder), each about 200-400 tokens. RAGChecker found larger chunks and more k raise claim recall. Drop chunks below the floor, since off-topic chunks cause extras.
- Merge adjacent chunks from the same page. Give each chunk a header `[C3 | file.pdf p52 | section]`.
- Order chunks as a sandwich: strongest first and last. Do not summarize them.
- Size: about 3-6k tokens of evidence.

**Step 2: Single grounded generation call (quote → plan → write in one pass).**
The output has three sections in one completion, which keeps it a single forward pass and not a loop:
1. `<quotes>`: for every chunk that is relevant, numbered verbatim quotes containing each fact, number or hedge-worthy gap, tagged with chunk id. Emit "No relevant quotes found." if none, which triggers abstention.
2. `<outline>`: headings (for long questions) mapped to quote numbers. Every relevant chunk must appear at least once (coverage).
3. `<answer>`: markdown. Each bullet holds one atomic fact paraphrased tightly from a quote, with `[file.pdf pNN]`. Include an evidence-scoped hedge sentence for any asked-for facet with no quote. Target about 600-900 words for analytical questions (gold median 754) and 1-3 sentences for factual ones.
Use temperature 0 and the system rules from §6. Strip `<quotes>` and `<outline>` before submission.
- Tokens: input about 1.2k (instructions plus question) + 3-6k (evidence) = **about 4-7k**. Output: quotes about 0.8-1.5k + outline about 0.2k + answer about 1.0-1.3k (long) or about 0.1k (short) = **about 2-3k long, about 0.5k short**.

**Step 3: Deterministic and NLI filter (no LLM tokens).**
- Split the answer into sentences or bullets.
- **Number and entity check**: every number (normalized) and capitalized entity must appear in the cited chunk or the evidence pack. Otherwise drop the sentence, or fix the citation if the fact appears in another chunk.
- **Citation check**: the cited page must be among the chunk headers. Otherwise replace it with the best-matching chunk by NLI.
- **NLI check**: run MiniCheck-Flan-T5-L (MIT), or Bespoke-MiniCheck-7B (CC BY-NC) on GPU, for (cited chunk, sentence). If p < 0.5, recheck against the concatenated evidence pack. Delete the sentence only if both checks give p < 0.3. Preserve the hedge sentences, which match the pattern "do not specify/state". Expect about 1-10 s per answer.
- Do **not** regenerate: this is a pure filter, not a reflection loop.

**Step 4: Coverage patch (optional, only if a relevant chunk was not cited).**
- One small call: "Write bullets for the facts in chunk C7 relevant to the question, same rules." Input about 1.5k tokens, output about 200 tokens. Append the bullets under the matching heading and run them through the Step 3 filter. Skip it if organisers consider it a loop.

**Step 5: Format.**
- Keep headings, bullets and citations. Remove any intro or outro sentence that has no citation, since an uncited sentence is likely extra.

**Cost per question (LLM tokens):**
- Long analytical: about 5-7k input + about 2.5-3k output ≈ **8-10k tokens** (+ about 1.7k with coverage patch).
- Short factual: about 3-4k input + about 0.5k output ≈ **3.5-4.5k tokens**.
- Abstained: about 0-0.2k.
- Verification: 0 LLM tokens (local small model).
- Prompt-cache the static system prompt (about 1k tokens) across questions.
- If output tokens are expensive, compress `<quotes>` to quote ids plus short verbatim spans. This cuts about 40% of the output but loses some grounding benefit.

**Calibration to do on dev questions (cheap, high value):**
1. Run the actual judge prompt on 5-10 dev answers to see how it atomizes citations, headings and hedges.
2. Tune the reranker abstention threshold.
3. Tune the NLI deletion threshold by comparing precision gain against recall loss.
4. Check the answer-length target against gold length.
