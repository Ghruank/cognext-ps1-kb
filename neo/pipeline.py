"""NEO: NeoRAG graph routing over an LLM-free retrieval tree. Index and query entry points.

Index (no LLM except one catalogue card per document):
        PDFs -> layout-block units (neo.ingest) -> tree: document/section/page/unit with
        unit BM25 + vectors, table descriptors, page vectors (neo.tree, neo.embed_pages)
        -> NeoRAG graph: categories, documents, versions, change edges (neo.graph)
Query:  1 parse     NeoRAG retrieval mode + hard document/version filter + facets
        2 pages     tree scoring inside the filtered documents, top pages kept
        3 expand    cross-version change counterparts for change questions
        4 verify    every unit on those pages judged with the context judge's criterion
        5 write     grounded answer over verified units only (neo.writer)
"""

import json
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from engine.index import embed_query
from neo.graph import NeoGraph, build_graph
from neo.parser import parse
from neo.tree import Tree
from neo.verify import verify
from neo.writer import write

STORE = Path(__file__).resolve().parent / "storage"
N_PAGES = 60
PAGE_MODEL = "BAAI/bge-small-en-v1.5"


def unit_index(tree: Tree) -> SimpleNamespace:
    """The tree's units in the shape engine.versions.Lattice expects of an Index."""
    return SimpleNamespace(units=tree.units, vecs=tree.unit_vecs, pos={u["id"]: i for i, u in enumerate(tree.units)})


def build(collection: str, catalog: dict) -> tuple[NeoGraph, Tree]:
    docs = pickle.load(open(STORE / f"{collection}_docs.pkl", "rb"))
    tree_path = STORE / f"{collection}_tree.pkl"
    tree = pickle.load(open(tree_path, "rb")) if tree_path.exists() else Tree.build(docs)
    pv_path = STORE / f"{collection}_pagevecs_{PAGE_MODEL.split('/')[-1]}.pkl"
    if pv_path.exists():
        pv = pickle.load(open(pv_path, "rb"))
        pos = {k: i for i, k in enumerate(pv["keys"])}
        tree.page_vecs = np.stack([pv["vecs"][pos[k]] if k in pos else np.zeros(pv["vecs"].shape[1], np.float32)
                                   for k in tree.pages])
    pickle.dump(tree, open(tree_path, "wb"))
    graph = build_graph(docs, catalog, unit_index(tree), {})
    pickle.dump(graph, open(STORE / f"{collection}_graph.pkl", "wb"))
    return graph, tree


def load(collection: str) -> tuple[NeoGraph, Tree]:
    return (pickle.load(open(STORE / f"{collection}_graph.pkl", "rb")),
            pickle.load(open(STORE / f"{collection}_tree.pkl", "rb")))


_enc = None


def _page_query_vec(text: str) -> np.ndarray:
    global _enc
    if _enc is None:
        from fastembed import TextEmbedding
        _enc = TextEmbedding(PAGE_MODEL, threads=8)
    return np.asarray(list(_enc.query_embed([text]))[0], dtype=np.float32)


def _step(trace: list, name: str, description: str, inp, out) -> None:
    trace.append({"step_name": name, "description": description, "input": inp, "output": out})


def query(question: str, graph: NeoGraph, tree: Tree, *, n_pages: int = N_PAGES, write_answer: bool = True) -> dict:
    t0 = time.perf_counter()
    trace: list[dict] = []
    tokens = {"prompt": 0, "completion": 0}

    def count(u: dict) -> None:
        tokens["prompt"] += u.get("prompt_tokens", 0)
        tokens["completion"] += u.get("completion_tokens", 0)

    plan = parse(question, graph)
    count(plan.usage)
    _step(trace, "neorag_query_parse",
          "Classify retrieval mode and fix the in-scope documents/versions from the category->document->version graph",
          question, {"retrieval": plan.retrieval, "documents": plan.documents, "facets": plan.facets, "reason": plan.reason})

    if not plan.documents:
        _step(trace, "abstain", "No document in the collection covers the question", plan.reason, [])
        return {"answer": "No information was retrieved for this question.", "chunks": [], "trace": trace,
                "tokens": tokens, "latency_s": time.perf_counter() - t0}

    text = question + (" " + " ; ".join(plan.facets) if plan.facets else "")
    scores = tree.page_scores(text, embed_query(text), plan.documents, contiguity=0, section_weight=0,
                              page_vec_weight=1.0 if tree.page_vecs is not None else 0.0,
                              qvec_page=_page_query_vec(text) if tree.page_vecs is not None else None)
    top = sorted(scores, key=lambda p: -scores[p])[:n_pages]
    _step(trace, "tree_page_ranking",
          "Score every page of the in-scope documents (unit dense+BM25 max, page BM25, page vector) and keep the top pages",
          {"documents": plan.documents, "pages_scored": len(scores), "kept": n_pages},
          [{"doc": tree.pages[p][0], "page": tree.pages[p][1], "score": round(scores[p], 3)} for p in top])

    positions = sorted({u for p in top for u in tree.page_units[p]})
    if plan.retrieval == "ChangeRetrieval" and graph.change_links:
        pos = {u["id"]: i for i, u in enumerate(tree.units)}
        have = set(positions)
        extra = {pos[b] for a, b in graph.change_links if pos.get(a) in have and b in pos} | \
                {pos[a] for a, b in graph.change_links if pos.get(b) in have and a in pos}
        positions = sorted(have | extra)
        _step(trace, "change_expansion", "Add the counterpart paragraph in the other version for every change edge",
              {"units": len(have)}, {"added": len(extra - have)})

    chosen, ver_usage = verify(question, plan.facets, tree.units, positions)
    count(ver_usage)
    units = [tree.units[k] for k in chosen]
    _step(trace, "unit_verification",
          "Judge each candidate unit for relevance to the query (same criterion as the context judge)",
          {"candidate_units": len(positions), "llm_calls": ver_usage.get("calls", 0)},
          [{"id": u["id"], "doc": u["doc"], "pages": u["pages"]} for u in units])

    answer = ""
    if write_answer:
        answer, w_usage = write(question, units)
        count(w_usage or {})
        _step(trace, "llm_answer_generation", "Write the grounded, cited answer from verified units only",
              {"units": len(units)}, answer)

    return {"answer": answer,
            "chunks": [{"text": u["text"], "doc": u["doc"], "pages": u["pages"], "id": u["id"]} for u in units],
            "trace": trace, "tokens": tokens, "latency_s": time.perf_counter() - t0}


def to_json(result: dict) -> str:
    return json.dumps(result, default=str)
