"""NEXUS inference path: compile evidence obligations, cover them, verify, then answer."""

import sqlite3
import time
from pathlib import Path

from nexus.index import document_catalog
from nexus.planner import compile_plan
from nexus.retrieve import eliminate, retrieve

from nexus.answer import ABSTAIN, write_answer, write_extractive_answer


def _trace(name: str, description: str, inp, out) -> dict:
    # The PS trace schema requires exactly these four keys.
    return {"step_name": name, "description": description, "input": inp, "output": out}


def _query(question: str, index_path: str | Path, *, use_llm: bool = True,
           per_facet_limit: int | None = 40, verify_relevance: bool = True, generate_answer: bool = True,
           use_dense: bool = True, max_per_facet: int = 24,
           vector_threshold: float = 0.42) -> dict:
    t0 = time.perf_counter()
    trace = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    db = sqlite3.connect(index_path)
    db.row_factory = None
    catalog = document_catalog(db)
    # Planning is deterministic. The optional LLM is reserved for the final answer frame.
    plan = compile_plan(question, catalog)
    usage["prompt_tokens"] += plan.usage.get("prompt_tokens", 0)
    usage["completion_tokens"] += plan.usage.get("completion_tokens", 0)
    usage["calls"] += bool(plan.usage)
    trace.append(_trace("evidence_contract", "Compile query scope and distinct evidence obligations",
                        question, {"intent": plan.intent, "entities": plan.entities, "periods": plan.periods,
                                   "documents": plan.documents, "facets": [f.__dict__ for f in plan.facets],
                                   "reasoning": plan.reasoning}))

    candidates, route = retrieve(db, plan, question=question, per_facet_limit=per_facet_limit,
                                 use_dense=use_dense, vector_threshold=vector_threshold)
    trace.append(_trace("facet_coverage_search",
                        "Search each evidence obligation independently and union page-level routes",
                        {"facets": [f.name for f in plan.facets], "document_scope": plan.documents}, route))
    db.close()

    if verify_relevance:
        chosen, filt = eliminate(plan, candidates, max_per_facet=max_per_facet)
        restored = []
        covered = {fi for e in chosen for fi in e.facets}
        for fi, facet in enumerate(plan.facets):
            if fi in covered:
                continue
            eligible = [e for e in candidates if fi in e.facets]
            if eligible:
                keep = max(eligible, key=lambda e: (e.score, len(e.text)))
                if keep not in chosen:
                    chosen.append(keep)
                    restored.append({"facet": facet.name, "id": keep.id, "doc": keep.doc, "page": keep.page})
                    covered.add(fi)
        filt["coverage_restored"] = restored
        filt["kept"] = len(chosen)
        usage["prompt_tokens"] += filt.get("prompt_tokens", 0)
        usage["completion_tokens"] += filt.get("completion_tokens", 0)
        usage["calls"] += filt.get("calls", 0)
        trace.append(_trace("conservative_evidence_filter",
                            "Remove only clear non-evidence; preserve uncertain passages and fail open",
                            {"candidate_blocks": len(candidates)}, filt))
    else:
        chosen, filt = candidates, {"calls": 0, "kept": len(candidates), "failed_batches": 0}

    evidence = [e.as_dict() for e in chosen]
    answer = ABSTAIN
    answer_mode = "not_generated"
    answer_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    if generate_answer:
        if use_llm:
            try:
                answer, answer_usage = write_answer(question, evidence)
                answer_mode = "single_final_llm_call"
            except (RuntimeError, ValueError, TypeError, KeyError):
                answer, answer_usage = write_extractive_answer(question, evidence,
                                                                [f.__dict__ for f in plan.facets])
                answer_mode = "extractive_fallback"
        else:
            answer, answer_usage = write_extractive_answer(question, evidence,
                                                            [f.__dict__ for f in plan.facets])
            answer_mode = "extractive"
        usage["prompt_tokens"] += answer_usage.get("prompt_tokens", 0)
        usage["completion_tokens"] += answer_usage.get("completion_tokens", 0)
        usage["calls"] += answer_usage.get("calls", 0)
    trace.append(_trace("evidence_answer", "Produce a cited answer from the verified evidence set",
                            {"question": question, "evidence_count": len(evidence)},
                            {"answer": answer, "mode": answer_mode,
                             "llm_calls": answer_usage.get("calls", 0)}))
    return {"answer": answer, "chunks": [{"text": e["text"], "doc": e["doc"], "pages": e["pages"],
                                             "id": e["id"]} for e in evidence],
            "trace": trace, "usage": usage, "latency_s": time.perf_counter() - t0,
            "plan": {"intent": plan.intent, "facets": [f.__dict__ for f in plan.facets]},
            "retrieval": route | {"filter": filt}}


def query(question: str, index_path: str | Path, *, use_llm: bool = True,
          per_facet_limit: int | None = 40, verify_relevance: bool = True,
          generate_answer: bool = True, observe: bool = True, use_dense: bool = True,
          max_per_facet: int = 24, vector_threshold: float = 0.42) -> dict:
    if not observe:
        return _query(question, index_path, use_llm=use_llm, per_facet_limit=per_facet_limit,
                      verify_relevance=verify_relevance, generate_answer=generate_answer, use_dense=use_dense,
                      max_per_facet=max_per_facet, vector_threshold=vector_threshold)
    from nexus.observe import finish_observation, query_observation

    with query_observation(question) as observation:
        result = _query(question, index_path, use_llm=use_llm, per_facet_limit=per_facet_limit,
                        verify_relevance=verify_relevance, generate_answer=generate_answer, use_dense=use_dense,
                        max_per_facet=max_per_facet, vector_threshold=vector_threshold)
        finish_observation(observation, result)
        return result
