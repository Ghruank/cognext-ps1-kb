# ALPHA: NEO + page-coherence refinement

ALPHA is NEO, unchanged up to the unit verifier, plus one new stage (4b) that re-decides every
candidate unit after verification. `neo/` is untouched. ALPHA lives here, and its storage
symlinks NEO's indexes.

```
1 parse    NeoRAG mode + hard document/version filter + facets          (LLM, 1 call)
2 pages    tree page ranking inside the filtered documents, top 40 pages
3 expand   cross-version change counterparts for change questions
4 verify   every unit on those pages judged with the context judge's rule (LLM, batched)
4b refine  coherence classifier over the verdict pattern                  (no LLM, ~10 ms)
5 write    grounded answer over the kept units
```

## Why NEO's precision was low

Measured on the 93 train questions with saved Claude verdicts (`python -m alpha.exp_coherence`):

| Selection over NEO's candidate pool | P | R | F1 |
| --- | ---: | ---: | ---: |
| keep every candidate | 0.100 | 2.03 | 0.19 |
| NEO verifier as is | 0.350 | 0.681 | 0.462 |
| ALPHA, threshold 0.25 | 0.427 | 0.819 | 0.561 |
| **ALPHA, verifier + coherence refinement (th 0.2, shipped)** | **0.406** | **0.955** | **0.569** |
| ALPHA + cross-encoder features (A/B, th 0.25) | 0.425 | 0.768 | 0.547 |
| oracle selection (pool ceiling) | 0.946 | 2.04 | 1.29 |

The candidate pool is not the bottleneck. It already contains enough gold for P 0.95 and
R 2.0 (recall is uncapped: relevant retrieved / gold chunk count). The loss is entirely in
**selection**. The verifier judges each unit alone, so it makes two kinds of error:

* **Isolated look-alikes** (precision loss). A unit on a page where nothing else was verified,
  with low query similarity, is usually a false positive. Gold comes in runs of blocks on the
  same pages.
* **Short lines inside gold runs** (recall loss). Table rows, headers and list items are
  rejected on their own, but their neighbours are verified.

Stage 4b is a gradient-boosted classifier over features of the verdict *pattern*: the
fraction of the unit's page that was verified, verified neighbours (±1, ±3), the document's
verified share, model2vec query similarity and its rank, lexical overlap with the question
and with each facet, and unit length/type/digit density. We also tried Self-Healing-RAG's two-stage cross-encoder rerank
(`alpha/ce_scores.py`, MS-MARCO MiniLM) as a feature. It **did not help** (see the A/B row): the model is
out of domain for regulatory text and adds nothing the other features lack. It is off by default
(`ALPHA_CE=1` to retry). The classifier is trained
on proxy labels with 5-fold question-grouped CV, so every number above is out-of-fold.

## Caveats (say these out loud)

* All numbers are the **text-overlap proxy** (unit carries gold text). The official context
  precision is LLM-judged and more lenient, so official P will be higher than proxy P.
* The classifier was trained on **Claude Haiku** verdicts. At query time ALPHA verifies with
  Groq (`alpha/llm.py`), whose verdict pattern differs somewhat, so expect some drift.
* The threshold trades P for R: 0.15 gives P 0.38 / R 1.14, 0.30 gives P 0.44 / R 0.70. We ship 0.2
  (best F1, recall above 0.9).
* Groq free tier (8k TPM) makes live verification slow: one question's verify step takes minutes
  and can stall on rate limits. The refine stage itself needs no LLM and takes under 1 s.

## API (the submission endpoints)

```
pip install -r alpha/requirements.txt
# .env at the repo root: GROQ_API_KEY, GROQ_API_KEY_2, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
uvicorn alpha.api:app --host 0.0.0.0 --port 8000
```

* `POST /index` takes multipart `collection_name` + `file` (a .zip of PDFs). It parses layout blocks,
  embeds pages, writes one catalogue card per document (Groq), and builds the tree and graph under
  `alpha/storage/<collection>_*`. Collections are independent.
* `POST /query` takes JSON `{"collection_name", "query"}` and returns `answer`, `chunks` (text, doc,
  page, citation), `trace` (one `{step_name, description, input, output}` per stage:
  `neorag_query_parse`, `tree_page_ranking`, `change_expansion`, `unit_verification`,
  `coherence_refinement`, `llm_answer_generation`) and `usage` (tokens, latency). Every query is
  also traced to Langfuse as `alpha-query` when the keys are set.

Tested end to end on 2026-10-24: indexing one 9-page PDF took 5 s. A query took 112 s, almost all
of it waiting on Groq's free-tier limit. Ingestion is CPU-bound (PyMuPDF), plus one LLM card per document.

## Experiments

```
.venv/bin/python -m alpha.exp_coherence          # experiment table + retrain alpha/storage/coherence.pkl
.venv/bin/python -m alpha.run outputs/alpha.jsonl A21,B05
```
