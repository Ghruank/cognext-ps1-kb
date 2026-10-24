# NEXUS design review and benchmark findings

## What the available results establish

- STAR's reported 0.20 precision / 0.68 recall / 0.35 F1 is weak on precision. NEXUS does not reuse STAR's selector.
- NEO's saved Claude-assisted proxy run scored 93 questions with available verdicts at P=0.350, R=0.681, F1=0.462; seven verdicts were missing. It used LLM query planning, dense page-vector ranking, version expansion, and Claude unit verification. This is a text-overlap proxy, not the organizer evaluator.
- A scorer-path bug made earlier NEXUS proxy results invalid: NEXUS indexes `neo/storage/train_docs.pkl`, while `score_proxy` had defaulted to the incompatible STAR `storage/train_docs.pkl`. The default is now corrected. Re-scoring both systems on the same 93 questions and NEO unit IDs gives:

  | Run | Precision | Recall | F1 |
  | --- | ---: | ---: | ---: |
  | NEO + Claude verifier | 0.350 | 0.681 | 0.462 |
  | NEXUS lexical + deterministic clauses/TOC, metadata-free | 0.098 | 0.099 | 0.098 |
  | NEXUS + saved LLM catalog/page cards | 0.089 | 0.124 | 0.104 |
  | NEXUS + LLM catalog/page cards, dense pages, 60 blocks/facet, threshold 0.42 | 0.080 | 0.281 | 0.125 |

- The saved LLM catalog covers all 35 documents, but saved page cards cover only 586 of 6,958 pages and just four documents. This is a partial ingestion-time LLM ablation, not full page enrichment. On the same 93 questions, adding the saved LLM artifacts raises recall from 0.099 to 0.124 without dense routing, and to 0.281 with dense routing and a 60-block budget; precision stays low (0.080), so F1 reaches only 0.125. This is a modest retrieval gain, not a NEO-beating result. Full LLM page coverage could improve page routing, but these results do not support expecting it to close the gap by itself.
- NEXUS's offline 100-question dense+metadata run scored P=0.081, R=0.268, F1=0.124 on the corrected proxy. These are retrieval diagnostics; extractive fallback answers were used. The official judge was attempted on A01 but could not resolve `opencode.ai`; its zeroes are failure placeholders. No official answer/context scores are available.

## RAG_Techniques review under PS1 constraints

The [RAG_Techniques repository](https://github.com/NirDiamant/rag_techniques) is a catalogue of recipes, not evidence that a technique will meet PS1's 0.95 targets. The most compatible pieces are multi-faceted filtering and hierarchical indices over metadata, TOC/section, page, and parser-native blocks. NEXUS now splits compound questions by grammar, matches TOC headings, routes through a flat static page-vector sidecar, then expands routed pages to parser-native blocks. Version-aware routing remains separate.

The ingestion-time LLM ablation has now been run with the cached NEO catalog and page cards. It gave a modest recall/F1 improvement but did not approach NEO, and its page cards cover only 8.4% of pages. A full-coverage page-card run is the next controlled experiment if ingestion-time LLM use is allowed. It should emit compact, source-grounded page descriptions and retain exact page provenance; query-time planning and filtering can remain deterministic. The current weak precision also points to the evidence selector: the NEXUS selector is lexical overlap plus a fixed per-facet budget, while NEO's comparable run includes a Claude unit verifier. Do not attribute the gap solely to indexing. Avoid vanilla question generation or reranking as the core architecture. Dense-plus-sparse fusion and reranking alone are specifically insufficient under PS1 section 4.

## Research implications

- VersionRAG's core transferable idea is explicit version-aware routing and change paths. Its reported 90% is on VersionQA (100 questions over 34 versioned technical documents), so it does not imply a 0.90 score on PS1's financial corpus or its separate context metrics. See [VersionRAG](https://arxiv.org/abs/2510.08109).
- GNN-Ret reports gains by propagating evidence through passage relatedness edges, but those experiments use its own QA data and trained graph retrieval setup. NEXUS currently has no relevance training labels suitable for held-out training and must avoid leakage from benchmark gold. A trained GNN is therefore an experiment to validate later, not a safe one-hour shortcut. See [GNN-Ret, NAACL 2025](https://aclanthology.org/2025.naacl-long.337/).
- RAGChecker motivates separating retrieval and answer diagnostics; aggregate F1 alone cannot tell whether candidate ranking, evidence selection, or answer writing is failing. PS1's own evaluator remains the deciding measurement. See [RAGChecker, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract.html).
- LSTM retrieval research supports sequence/context modeling when there is a useful order signal, but neural ranking models are trained against relevance data and are data-hungry. NEXUS has ordered document blocks, yet no sufficiently large, leakage-safe query-to-block training set. A randomly initialized LSTM or one trained on the 100 benchmark questions would be more likely to overfit than to generalize. The sensible future experiment is a small sequence-aware block selector over candidate neighborhoods, evaluated against identical per-query precision/recall and latency on a held-out split; it should not be added to the one-hour benchmark path. See [Neural ranking models for document retrieval](https://link.springer.com/article/10.1007/s10791-021-09398-0) and [Microsoft Research's neural IR tutorial](https://www.microsoft.com/en-us/research/publication/introduction-neural-information-retrieval/).

## NEXUS architecture decisions

1. Keep the LLM-free parser-native source map: original page/block text, table/prose type, section, and citation provenance.
2. Derive query scope and evidence obligations deterministically. Route each obligation independently so one strong lexical match cannot hide a missing facet.
3. Use SQLite FTS and optional flat page vectors as address mechanisms; keep document, year, and version constraints ahead of evidence selection. The vector store is not the architecture.
4. Keep filtering deterministic and facet-aware. The current overlap scorer is a prototype; its low held-out proxy recall means its pruning rule needs redesign before any score claim.
5. Use NEO's version-family/change signal for compare/trend intent. Avoid unbounded cross-version expansion; expand only the family of an explicitly routed source and retain paired change units in that family.
6. Do not claim the 0.90 target is met until the untouched organizer evaluator has scored generated answers and contexts. No architecture choice guarantees that target without evidence from the actual benchmark.
7. Use at most one final LLM call for answer framing. Query planning, retrieval, and evidence selection remain deterministic. Keep LSTM as a held-out ablation, not an untrained component; the matching signal required to train it is absent today.

## One-hour evaluation plan and current blocker

- The local 100-question no-LLM diagnostic completed in under a minute in the prior run. It is suitable for fast routing regressions, but not as the reported score.
- The full path needs query-planning/verifier/answer calls and then official judge calls. The current execution environment cannot resolve external API hosts (Groq and Langfuse DNS failed in the prior run), so a fresh API key alone will not make an official run possible here.
- Once outbound network access to the configured LLM and judge endpoints exists, run a small end-to-end smoke subset first, then the full 100 questions under a 55-minute wall-clock cap, stopping early if provider throttling makes that cap impossible. Save raw per-question outputs and official evaluator output separately.
