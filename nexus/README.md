# NEXUS

NEXUS is a new document-native retrieval architecture for PS1. It does not extend
STAR. Its main unit of work is a query-time **evidence contract**: scope constraints
and distinct facts that must be covered.

## Design

- Ingestion is deterministic and LLM-free. The source map stores parser-native PDF
  blocks, tables, sections, page numbers, and document metadata in SQLite FTS5.
- Query scope, intent, years, and evidence facets are derived deterministically from the
  question and document catalog; planning does not call an LLM.
- Retrieval searches each facet independently and unions its page routes. Every
  parser-native block on a routed page remains eligible, so one lexical hit does not
  discard other evidence on that page.
- A flat page-vector sidecar uses the cached static Model2Vec encoder for query and page
  vectors. It scans page vectors directly, uses no vector database or ANN service,
  and does not replace the source-scope and facet-coverage plan.
- A deterministic version-family layer uses source names, dates, and text containment.
  Compare/trend queries may expand an explicitly selected source to its family. The
  benchmark collection also imports NEO's precomputed paired change links; fresh uploads
  receive family links without LLM calls, but no learned change-pair edges.
- A mathematical facet term scorer filters routed page blocks independently per evidence
  obligation. It uses no LLM; its current overlap-proxy results show this rule still needs
  improvement before it can be considered benchmark-ready.
- At most one LLM call is made after retrieval/filtering to frame the final answer. If it is
  disabled or unavailable, NEXUS returns a cited extractive answer.

This is an initial implementation, not a claim that the PS1 target is already met.
NEXUS keeps the organizer's evaluator untouched. Build and run locally from `kb/`:

```bash
python -m nexus.run --build --ids A01,A02 --offline --no-answer
```

`--offline` disables the final answer-framing LLM call; deterministic planning, filtering,
and extractive answers still run. The output is resumable JSONL with page-cited chunks
and the four-field PS trace steps. Run the
gold-overlap diagnostic with `python -m nexus.score_proxy outputs/nexus.jsonl`;
for official judge scores use `python -m eval.run_eval outputs/nexus.jsonl`.

Default indexing remains LLM-free. For the saved-metadata ablation only, rebuild with
`--llm-metadata` to include NEO's existing LLM-generated document catalog and available
page cards; these cached artifacts are reused without new LLM calls. This ablation has
not reached NEO's proxy scores.

The minimal two-route API is `uvicorn nexus.api:app --host 0.0.0.0 --port 8000`.
Set `NEXUS_DATA_DIR` to persistent storage before deployment. Langfuse uses the
standard `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL` settings.
