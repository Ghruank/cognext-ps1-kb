# Cost and latency for long-answer grounded QA (PS1)

Research date: 2026-10-23. Scope: per-query LLM token cost and end-to-end latency for 700-word analyst answers built from 100-300 evidence paragraphs (median 81 paragraphs of about 63 words, roughly 7k tokens; maximum 352 paragraphs, roughly 30k tokens), using the OpenCode gateway, with Langfuse tracing.

## 1. OpenCode inference gateway (Zen / Console): models, pricing, caching, JSON, streaming, rate limits

### Takeaway
OpenCode's pay-as-you-go gateway lists `deepseek-v4.1-flash` on the OpenAI chat-completions endpoint at $0.30 per 1M input tokens, $1.20 per 1M output tokens and $0.006 per 1M cached-read tokens. That is 50x cheaper for cached input, so prefix caching matters a lot. The judge uses the same model. OpenCode's docs do not cover rate limits, JSON mode or streaming. DeepSeek's own API docs do cover JSON output and streaming.

### Cited Findings
- Endpoints: OAI chat completions at `https://opencode.ai/inference/openai/v1/chat/completions` (used by DeepSeek, GLM, Kimi, MiniMax, Qwen3.8 Max and the free models), plus `/openai/v1/responses` (GPT), `/anthropic/v1/messages` (Claude, most Qwen) and `/google/v1beta/models/<id>` (Gemini). In OpenCode itself, models are referenced as `opencode/<model-id>`. — [OpenCode Zen docs](https://opencode.ai/docs/zen/)
- Prices on the chat-completions endpoint, in USD per 1M tokens (input / output / cached read). Page fetched 2026-10-23; the page shows no publication date. — [OpenCode Zen docs](https://opencode.ai/docs/zen/)
  - `deepseek-v4.1-flash`: $0.30 / $1.20 / $0.006
  - `deepseek-v4-pro`: $1.74 / $3.48 / $0.145
  - `glm-5.3-flash`: $0.15 / $0.50 / $0.03
  - `minimax-m3`: $0.30 / $1.20 / $0.06
  - `kimi-k2.6`: $0.95 / $4.00 / $0.16
  - `mistral-large-4` (50% off): $0.68 / $2.09 / $0.07
  - `qwen3.8-max`: $2.00 / $6.00 / $0.25 (cache write $2.50)
- Cheap models on other endpoints, same units:
  - `qwen3.8-flash` (Anthropic endpoint): $0.15 / $0.47 / $0.016
  - `gpt-6-luna`: $0.10 / $0.50 / $0.01
  - `claude-haiku-5-5` at 100K tokens or less: $0.10 / $0.50 / $0.01
  - `gpt-5-nano`: $0.05 / $0.40 / $0.005
  - `gemini-3.5-flash-lite`: $0.30 / $2.50 / $0.03
  — [OpenCode Zen docs](https://opencode.ai/docs/zen/)
- Free models (some only "for a limited time") include `deepseek-v4.1-flash-free`, `glm-5.3-flash-free`, `big-pickle`, `nemotron-3.5-lightning-free` and `mimo-v2.6-flash-free`. For some of them, "During its free period, collected data may be used to improve the model." — [OpenCode Zen docs](https://opencode.ai/docs/zen/)
- Billing:
  - "Console uses pay-as-you-go pricing… charged per request."
  - Each credit purchase carries a processing fee of 4.6% + $0.31.
  - Auto-recharge and per-workspace and per-member monthly budgets are available.
  - Providers are zero-retention, except the OpenAI and Anthropic APIs, which retain requests for 30 days.
  — [OpenCode Zen docs](https://opencode.ai/docs/zen/)
- Rate limits, JSON mode and streaming are not mentioned on the Zen docs page. — [OpenCode Zen docs](https://opencode.ai/docs/zen/)
- DeepSeek first-party API, for reference. `deepseek-flash` is DeepSeek-V4.1-Flash-1017.
  - Context is 1M tokens and maximum output is 384K. JSON output is supported. The concurrency limit is 2500.
  - Peak prices: cache-hit input $0.006, cache-miss input $0.30, output $1.20. Off-peak prices are half of these ($0.003 / $0.15 / $0.60).
  - Peak hours are 01:00-04:00 and 06:00-10:00 UTC, Monday to Friday.
  - The legacy name `deepseek-v4-flash` is now served by V4.1-Flash.
  — [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing)
- DeepSeek thinking mode:
  - "Thinking mode is enabled by default, with the default effort being `high`."
  - Disable it with `{"thinking": {"type": "disabled"}}`, sent in `extra_body` from the OpenAI SDK, or use `reasoning_effort` low/high/max.
  - When streaming, reasoning arrives in `delta.reasoning_content`, separate from `delta.content`.
  — [DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode)
- Conflicting third-party prices for older V4 Flash:
  - OpenCode Data lists `deepseek-v4-flash-0423` (released 2026-04-23) at $0.19 input, $0.51 output and $0.03 cached input. — [OpenCode Data](https://opencode.ai/data/deepseek/deepseek-v4-flash-0423.md)
  - A search snippet and aggregators (Bifrost) report $0.14 / $0.28 for V4 Flash and $0.15 / $0.60 for `deepseek-v4.1-flash` (Zen). — [Bifrost](https://www.getmaxim.ai/bifrost/model-library/compare/opencode-zen/deepseek-v4.1-flash.md)
  - Treat the Zen docs table as authoritative.

### Inferences
- Zen's `deepseek-v4.1-flash` price ($0.30 / $1.20 / $0.006) exactly equals DeepSeek's first-party peak price, so Zen probably passes DeepSeek pricing through without the off-peak discount. This is unverified.
- Output tokens cost 4x input tokens. A 700-word answer (about 950 tokens) costs about as much as about 3.8k tokens of uncached input. Hidden reasoning tokens add to that unless thinking is disabled.
- `deepseek-v4.1-flash-free` would make tokens cost $0. Whether the hackathon scores "cost" from token counts multiplied by list price, or from actual billing, decides whether the free model helps. Scoring is probably by token counts. Its data-use terms and capacity are also unknown.
- `qwen3.8-flash`, `gpt-6-luna`, `claude-haiku-5-5` and `gpt-5-nano` are 2-6x cheaper on input than `deepseek-v4.1-flash`. They are candidates for auxiliary calls such as query rewriting or claim checks, but most use non-OAI-CC endpoints, which needs separate client code.

### Gaps
- No OpenCode documentation found on rate limits, RPM/TPM, JSON mode, `response_format`, or `stream_options.include_usage` passthrough. Test these empirically.
- Whether Zen passes DeepSeek's `prompt_cache_hit_tokens` through in `usage`, and whether it bills reasoning tokens as output: not documented.
- No published throughput (tokens/s) or time-to-first-token (TTFT) for V4.1-Flash via Zen.

## 2. Evidence compression: LLMLingua family, RECOMP, extractive selection, dedup

### Takeaway
Query-aware compression reliably cuts evidence by 2-6x, and up to about 20x for extractive or summary methods, with small accuracy loss. LongLLMLingua even gains accuracy on multi-document QA by removing distractors. Recall risk grows with compression, and is worst for multi-hop questions and for long answers that need many distinct facts. For a 700-word synthesis answer, moderate compression (2-3x) is safer than 20x.

### Cited Findings
- LLMLingua claims "up to 20x compression with minimal performance loss". The repo example compresses 2,365 tokens to 211 (11.2x). — [LLMLingua repo](https://github.com/microsoft/LLMLingua)
- LongLLMLingua is the question-aware version for RAG. On NaturalQuestions with GPT-3.5-Turbo, performance rises by up to 21.4% with about 4x fewer tokens. It cuts cost by 94.0% on LooGLE, and improves end-to-end latency 1.4x-2.6x at 2x-6x compression on prompts of about 10k tokens. — [LongLLMLingua arXiv 2310.06839](https://arxiv.org/abs/2310.06839)
- LLMLingua-2 is a token-classification compressor built on XLM-RoBERTa-large or mBERT, distilled from GPT-4. At 2x-5x compression it is 3x-6x faster than existing prompt-compression methods and speeds up end-to-end latency 1.6x-2.9x. It was evaluated out-of-domain on LongBench, ZeroSCROLLS, GSM8K and BBH. — [LLMLingua-2 arXiv 2403.12968](https://arxiv.org/abs/2403.12968)
- Repo usage:
  - `PromptCompressor(model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank", use_llmlingua2=True).compress_prompt(prompt, rate=0.33, force_tokens=['\n','?'])`
  - LongLLMLingua: `rate=0.55, condition_in_question="after_condition", reorder_context="sort", dynamic_context_compression_ratio=0.3, rank_method="longllmlingua"`
  — [LLMLingua repo](https://github.com/microsoft/LLMLingua)
- RECOMP, with Flan-UL2 20B as reader and top-5 documents as evidence:
  - NQ: 660 tokens, 39.39 EM. Extractive compressor (Contriever-based, 110M): 37 tokens, 36.57 EM. Abstractive compressor (T5-large, 775M): 36 tokens, 37.04 EM.
  - TriviaQA: 677 tokens, 62.37 EM. Extractive: 38 tokens, 58.99 EM.
  - HotpotQA: 684 tokens, 32.80 EM. Extractive: 75 tokens, 30.40 EM. Abstractive: 64 tokens, 28.20 EM.
  - So a compression rate of about 5-11% loses about 2-4 EM points.
  - "Selective augmentation": the compressor can output an empty string when the documents are irrelevant.
  — [RECOMP arXiv 2310.04408](https://arxiv.org/html/2310.04408)
- MiniCheck (770M, FT5) reaches GPT-4-level grounding fact-checking on LLM-AggreFact "for 400x lower cost". It is also usable as a sentence-level support filter. — [MiniCheck arXiv 2404.10774](https://arxiv.org/abs/2404.10774)

### Inferences
- The tested settings are short-answer QA (EM). A 700-word analyst answer depends on recall of many facts across paragraphs. RECOMP-style 5% compression would likely hurt answer recall and the judge's context-recall score. For our case, prefer:
  1. Paragraph-level cross-encoder selection plus near-duplicate removal. This works in whole paragraphs, keeps citations intact and helps context precision.
  2. Optionally, LLMLingua-2 at rate about 0.5 on the kept paragraphs.
- Token-level compression such as LLMLingua produces ungrammatical text. If the judge scores "context precision/recall" on the passages we return, return the original paragraphs to the judge and feed only the compressed text to the generator.
- Dedup: with 81-352 paragraphs per question, near-duplicate removal with MinHash, or embedding cosine above about 0.92-0.95, is nearly free on CPU. No published number was found for its typical token savings; it depends on the corpus.

### Gaps
- No published benchmark of compression's effect on long-form answer recall (as opposed to EM). Validate on our own dev set.
- No source found for LLMLingua-2 CPU latency per 10k tokens. GPU numbers dominate the papers.

## 3. Prompt caching and semantic answer caching

### Takeaway
DeepSeek caching is automatic and best-effort, and Zen bills cached reads at $0.006/M versus $0.30/M, a 98% discount. It pays off only for repeated prefixes. Within one query, that means multi-call pipelines that reuse the same system prompt and evidence block. Semantic answer caching has meaningful false-hit rates and little value for a one-shot evaluation set.

### Cited Findings
- DeepSeek context caching:
  - "enabled by default for all users… without needing to modify their code."
  - A hit requires a full match on a "cache prefix unit". Units are created at request boundaries, at detected common prefixes and at fixed token intervals.
  - Cache construction "takes seconds". The cache is cleared "within a few hours to a few days". It is best-effort, with no 100% hit guarantee.
  - Usage reports `prompt_cache_hit_tokens` and `prompt_cache_miss_tokens`.
  — [DeepSeek KV cache guide](https://api-docs.deepseek.com/guides/kv_cache)
- Zen lists a cached-read price for most paid models, including `deepseek-v4.1-flash` at $0.006/M, so it supports DeepSeek cache-hit pricing. — [OpenCode Zen docs](https://opencode.ai/docs/zen/)
- Semantic caching evidence:
  - MeanCache reports precision of 0.72 versus 0.52 for GPTCache on 1,000 queries, with 89 versus 233 false hits. — [MeanCache arXiv 2403.02694](https://arxiv.org/pdf/2403.02694)
  - vCache argues for learned per-prompt thresholds instead of one static threshold. — [vCache arXiv 2502.03771](https://arxiv.org/pdf/2502.03771v4)
  - Vendor claims of 25-60% hit rates at cosine 0.90-0.95 are unverified. — [PremAI blog](https://blog.premai.io/semantic-caching-for-llms-how-to-cut-api-bills-by-60-without-hurting-quality)

### Inferences
- Put all static content first: system prompt, answer template and few-shot example. Put the evidence next, and the question last; or put the question after the evidence. Then any second call (outline, then section writing, then self-check) on the same evidence pays $0.006/M for the shared prefix. Because "cache construction takes seconds", a second call issued immediately may miss. Parallel calls fired at the same moment may also miss. This must be measured.
- A semantic answer cache is useful only if the evaluation repeats or paraphrases questions. Use a strict threshold (at least 0.95) and key it on the retrieved evidence-ID set as well, to avoid false hits.

### Gaps
- Not confirmed whether Zen returns `prompt_cache_hit_tokens` in its usage object, or which minimum prefix length caches.

## 4. Small local models that replace LLM calls in the query path

### Takeaway
Run reranking, dedup, query parsing and claim checks on CPU with small encoders, and keep the LLM for the single final generation. MiniLM-L6 cross-encoders score 100 pairs in roughly 0.1-0.25 s on CPU. Larger multilingual rerankers such as bge-reranker-v2-m3 or jina-v2 (278-568M) are about 20x slower on CPU, about 40 ms per pair, or 4 s per 100 pairs.

### Cited Findings
- Sentence-Transformers MS MARCO cross-encoders (NDCG@10 on TREC DL19 / MRR@10 / docs per second; hardware not stated, reportedly a V100 GPU):
  - TinyBERT-L2: 69.84 / 32.56 / 9000
  - MiniLM-L2: 71.01 / 34.85 / 4100
  - MiniLM-L4: 73.04 / 37.70 / 2500
  - MiniLM-L6: 74.30 / 39.01 / 1800
  - MiniLM-L12: 74.31 / 39.02 / 960
  — [SBERT pretrained cross-encoders](https://sbert.net/docs/cross_encoder/pretrained_models.html); V100 attribution per [PromptLayer](https://www.promptlayer.com/models/ms-marco-minilm-l-6-v2)
- One CPU measurement: a MiniLM-L-6 (22M) cross-encoder ran at 478 pairs/s on CPU (2.1 ms per pair). Jina Reranker v2 multilingual (278M) ran at 25 pairs/s on CPU (40 ms per pair) versus 748 pairs/s on GPU. — [EnsembleLink arXiv 2601.21138](https://arxiv.org/pdf/2601.21138)
- bge-reranker-v2-m3 (about 568M) is described as the highest-accuracy option and better suited to GPU. No CPU number was found. — [MariaDB reranker docs](https://mariadb.com/docs/tools/mariadb-ai-rag/api-reference/reranker.md)
- MiniCheck-FT5 (770M) matches GPT-4 at grounding checks for 400x lower cost. The paper's DeBERTa-v3-large variant maps neutral and contradiction to "unsupported", with threshold 0.5. — [MiniCheck](https://arxiv.org/abs/2404.10774)

### Inferences
- About 2 ms per pair means 81 paragraphs take about 0.17 s and 352 paragraphs about 0.75 s with MiniLM-L6 on CPU. Our 63-word paragraphs are short, so this is plausible. bge-reranker-v2-m3 on CPU for 352 pairs would take about 14 s, which is too slow. Use it only on a GPU or on a pre-filtered top-30.
- Query parsing (entities, time ranges, intent) can use GLiNER or spaCy and a small classifier, in well under 50 ms. No benchmark was cited for this; it is an engineering estimate.
- A claim check after generation, with MiniCheck or DeBERTa NLI over about 30 sentences by their cited paragraphs, costs CPU seconds rather than LLM tokens. Run it only if it raises the judge's precision.

### Gaps
- No authoritative CPU benchmark exists for bge-reranker-v2-m3, nli-deberta-v3 or MiniCheck. Benchmark these locally. The machine has 12 cores, and faiss and sentence-transformers are not installed yet.

## 5. Latency: streaming, parallelism, two-stage generation, ANN index, budgets

### Takeaway
Decoding the roughly 1k-token answer dominates end-to-end time. Retrieval and reranking over about 100k vectors take well under 1 s. To cut wall-clock time: disable thinking, stream, and split the answer into 3-4 sections generated in parallel over a shared, cached evidence prefix.

### Cited Findings
- FAISS: "The only index that can guarantee exact results is IndexFlatL2 or IndexFlatIP". HNSW "is the best option" when RAM is plentiful or the dataset is small. HNSW memory is (d·4 + M·2·4) bytes per vector, with M from 4 to 64. For only 1k-10k searches, "direct computation is the most efficient option". — [FAISS wiki: Guidelines to choose an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- An unsourced course table gives brute force about 1,200 ms, HNSW about 1 ms at 98% recall, and IVF about 5 ms. No hardware or dimensions are given, so do not rely on it. — [TDS course page](https://tds.s-anand.net/2026-02/docs/week-3/similarity-search/)
- LongLLMLingua and LLMLingua-2 report 1.4x-2.9x end-to-end speedups from input compression on prompts of about 10k tokens. — [2310.06839](https://arxiv.org/abs/2310.06839), [2403.12968](https://arxiv.org/abs/2403.12968)
- DeepSeek thinking is on by default at high effort, which adds reasoning tokens before the answer. — [DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode)

### Inferences
- At 100k vectors × 1024 dimensions in float32 (about 400 MB), an exact flat inner-product search takes on the order of tens of milliseconds with BLAS on a modern CPU. This is an estimate, not cited. Flat is exact, so recall cannot be lost, and that makes it the safe default. HNSW (M=32, efSearch=64-128) is optional if latency is scored strictly. It adds about 25 MB of memory at M=32 by the formula above.
- Rough budget, assuming 50-100 output tokens/s for V4.1-Flash (no published figure found):

  | Stage | Time |
  |---|---|
  | Query parse | under 50 ms |
  | Embed and search | under 100 ms |
  | Cross-encoder rerank | 0.2-0.8 s |
  | LLM time to first token | about 0.5-2 s |
  | Decoding 1k tokens | 10-20 s single-stream, or 3-6 s with 4 parallel sections |

- Thinking mode could add several seconds and 1-3k output tokens.

### Gaps
- No measured V4.1-Flash time to first token or tokens/s via Zen, and no Zen rate limits, which matter for parallel calls. Measure both.

## 6. Langfuse (Python SDK v3): cost and latency per trace, custom model prices, export

### Takeaway
Use `from langfuse.openai import OpenAI` with `base_url` set to OpenCode and `stream_options={"include_usage": True}`. Define a custom model `deepseek-v4.1-flash` with input, output and cached-input prices in Project Settings > Models, or send `cost_details` explicitly. Ingested values override inferred ones.

### Cited Findings
- Drop-in import: `from langfuse.openai import openai`, or `OpenAI` / `AsyncOpenAI`. It captures prompts, completions (including streamed ones), latencies, errors, usage and USD cost.
  - Streams need `stream_options={"include_usage": True}`. The final chunk has empty `choices`, so guard with `if chunk.choices:`.
  - Set session and tags via `metadata={"langfuse_session_id":…, "langfuse_tags":[…]}` or `propagate_attributes(...)`.
  - Call `get_client().flush()` in short-lived scripts.
  — [Langfuse OpenAI integration](https://langfuse.com/integrations/model-providers/openai-py)
- How cost is computed:
  - "ingested values take priority over inferred ones."
  - Otherwise, the model name is matched by a regex `match_pattern` against model definitions (user-defined definitions win), and price is multiplied by usage.
  - Custom models are added in Project Settings > Models or via `POST /api/public/models`. They apply only to new generations.
  - Usage buckets must be exclusive, so cached tokens must be subtracted from input. Usage keys must exactly match the price keys.
  - Pricing tiers can apply conditionally, for example when input exceeds 200k.
  — [Langfuse token and cost tracking](https://langfuse.com/docs/observability/features/token-and-cost-tracking)
- OpenAI usage schema recognition is strict: "the usage object must contain only the OpenAI usage fields". Extra keys from a gateway, such as `cost`, are stored flat with no mapping or subtraction. — [Langfuse token and cost tracking](https://langfuse.com/docs/observability/features/token-and-cost-tracking)
- Manual v3 API: `langfuse.start_as_current_observation(as_type="generation", …)` then `generation.update(usage_details={...}, cost_details={...})`, or `langfuse.update_current_generation(...)` inside an `@observe` function. — [same](https://langfuse.com/docs/observability/features/token-and-cost-tracking)
- Export options: UI batch export to CSV or JSON (respects filters), scheduled blob-storage export, SDK/public API access, and a Metrics API for aggregated cost, tokens and scores. — [Langfuse export](https://langfuse.com/docs/api-and-data-platform/features/export-from-ui)

### Inferences
- DeepSeek's usage includes non-OpenAI keys (`prompt_cache_hit_tokens`, `prompt_cache_miss_tokens`), which may break Langfuse's strict schema mapping. The safest approach is manual:

  ```python
  usage_details = {"input": miss, "input_cached": hit, "output": completion}
  cost_details = {"input": miss * 0.30e-6, "input_cached": hit * 0.006e-6, "output": completion * 1.20e-6}
  ```

- Trace one span per pipeline stage (retrieve, rerank, compress, generate, verify) so per-stage latency appears in the trace tree.

### Gaps
- Not confirmed: the exact API endpoint and SDK method names for fetching traces, such as `GET /api/public/traces` or `langfuse.api.trace.list`, and whether time-to-first-token is captured automatically for streams. See https://api.reference.langfuse.com.

## Recommended cost/latency plan for our system

### Takeaway
Use one generation call on `deepseek-v4.1-flash` with thinking disabled, over about 3-4.5k tokens of reranked, deduplicated whole paragraphs. That is about 5k input and 1k output tokens per query, roughly $0.0027 per query, with a p50 of about 8-15 s end to end. It falls to about 5-8 s if sections are generated in parallel.

### Cited Findings
- Price basis: $0.30/M input, $1.20/M output, $0.006/M cached, for `deepseek-v4.1-flash` on Zen. — [OpenCode Zen docs](https://opencode.ai/docs/zen/)
- Cross-encoder CPU cost is about 2 ms per pair (MiniLM-L6). — [arXiv 2601.21138](https://arxiv.org/pdf/2601.21138)
- Thinking can be disabled via `extra_body={"thinking":{"type":"disabled"}}`. — [DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode)

### Inferences
**Pipeline:**
1. Parse the query locally with no LLM call: rules or GLiNER plus a small intent classifier.
2. Retrieve with an exact FAISS flat index (or HNSW M=32) plus BM25 hybrid.
3. Run all candidate paragraphs (81-352) through a MiniLM-L6 or L12 cross-encoder on CPU, taking 0.2-0.8 s.
4. Drop near-duplicate paragraphs (cosine above 0.93).
5. Keep paragraphs by score threshold and an adaptive cap of about 30-60 paragraphs (about 2.5-5k tokens). Keep the paragraph IDs for citations and for the judge's context scoring.
6. Optionally apply LLMLingua-2 at rate 0.5 only when evidence exceeds about 6k tokens. Check recall first.
7. Make one streamed generation call with thinking disabled, a static system prompt and template first, then the evidence, then the question, and `max_tokens` about 1,200.
8. Optionally verify claims locally with MiniCheck or NLI. Do not add an LLM verify call unless dev-set gains justify about $0.001-0.002 extra.

**Per-query token and cost budget (Zen list prices):**

| Scenario | Input tokens | Output tokens | Input cost | Output cost | Total per query |
|---|---|---|---|---|---|
| Typical (recommended plan) | 4.5k (0.7k static + 3.5k evidence + 0.3k question and format) | 1.0k | $0.00135 | $0.0012 | **about $0.0026** |
| Worst case (352 paragraphs, capped at 60) | about 5.5k | 1.0k | — | — | about $0.003 |
| Naive (all 30k evidence tokens, thinking on with about 2k reasoning tokens) | 30k | 3k | $0.009 | $0.0036 | about $0.0126, about 5x more |
| Parallel 4-section variant | 4 × 4.5k, three of them cache hits if the cache warms | 4 × 280 | about $0.0014 | about $0.0013 | about $0.003 |

The parallel variant cuts decode wall-clock time about 3-4x. If cache hits do not happen, it costs about $0.0067.

**Other notes:**
- Context precision benefits from the reranked, thresholded set. Recall is protected by not using token-level compression aggressively and by keeping an adaptive top-k floor of at least 20 paragraphs.
- Sending all LLM calls in off-peak UTC windows does not help, because Zen appears to charge DeepSeek's flat peak price.
- `deepseek-v4.1-flash-free` is $0, but data may be used for training and availability is limited. Use it for development, not for the scored run, unless the rules allow it.
- Log everything in Langfuse with manual `usage_details` and `cost_details`, plus a custom model definition.

### Gaps
- Real V4.1-Flash throughput, Zen rate limits, Zen cache-hit behaviour and how scoring computes "cost" (list price or billed) all need measuring or confirming with the organizers.
