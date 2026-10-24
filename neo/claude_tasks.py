"""Run NEO's two LLM steps through Claude Code subagents (file in, file out), then score.

    python -m neo.claude_tasks parse-input          # -> claude/parse_input.md (library + all questions)
    python -m neo.claude_tasks verify-input         # needs claude/parse_output.json -> claude/verify/<id>.md
    python -m neo.claude_tasks score                # needs claude/verdict/<id>.json -> NEO vs STAR proxy

Everything else is the NEO pipeline unchanged: LLM-free ingestion (neo.ingest), the NeoRAG
graph (neo.graph), tree page ranking (neo.tree), cross-version change expansion.
"""

import json
import pickle
import sys

import numpy as np

from engine.index import embed_query
from eval import bench
from neo.parser import PROMPT as PARSE_PROMPT, _resolve
from neo.pipeline import STORE, _page_query_vec, load

DIR = STORE / "claude"
N_PAGES = 40
UNIT_WORDS = 100

VERIFY_RULE = ("For each chunk, decide if it is genuinely relevant and useful for answering the query: it states a fact, "
               "figure, rule, condition, event or explanation that belongs in a complete answer. Chunks about a different "
               "entity, year, version or topic than the query asks about are not relevant, even if they look similar.")


def parse_input() -> None:
    graph, _ = load("train")
    DIR.mkdir(parents=True, exist_ok=True)
    head, rules = PARSE_PROMPT.split("Question: {question}")
    qs = "\n".join(f"{q['id']}: {q['question']}" for q in bench.load())
    (DIR / "parse_input.md").write_text(
        head.format(library=graph.describe())
        + "\nAnswer the following for EACH question below." + rules.split("Return JSON")[0].replace("{{", "{").replace("}}", "}")
        + f"\nQuestions:\n{qs}\n\nWrite one JSON object mapping every question id to "
        '{"retrieval": str, "documents": [filenames exactly as listed], "facets": [str]}.\n')
    print(f"wrote {DIR / 'parse_input.md'}")


def _plans(graph) -> dict[str, dict]:
    raw = json.load(open(DIR / "parse_output.json"))
    docs = list(graph.catalog)
    plans = {}
    for qid, r in raw.items():
        chosen = []
        for name in r.get("documents", []):
            d = _resolve(str(name), docs)
            if d and d not in chosen:
                chosen.append(d)
        if r.get("retrieval") == "ChangeRetrieval":
            for d in list(chosen):
                chosen += [v for v in graph.versions(d) if v not in chosen and v not in graph.consolidated]
        plans[qid] = {"retrieval": r.get("retrieval", "ContentRetrieval"), "documents": chosen,
                      "facets": [str(f) for f in r.get("facets", [])][:8]}
    return plans


def candidates(question: str, plan: dict, graph, tree) -> list[int]:
    if not plan["documents"]:
        return []
    text = question + (" " + " ; ".join(plan["facets"]) if plan["facets"] else "")
    s = tree.page_scores(text, embed_query(text), plan["documents"], contiguity=0, section_weight=0,
                         page_vec_weight=1.0, qvec_page=_page_query_vec(text))
    top = sorted(s, key=lambda p: -s[p])[:N_PAGES]
    positions = {u for p in top for u in tree.page_units[p]}
    if plan["retrieval"] == "ChangeRetrieval":
        pos = {u["id"]: i for i, u in enumerate(tree.units)}
        positions |= {pos[b] for a, b in graph.change_links if pos.get(a) in positions and b in pos}
        positions |= {pos[a] for a, b in graph.change_links if pos.get(b) in positions and a in pos}
    return sorted(positions)


def verify_input() -> None:
    graph, tree = load("train")
    plans = _plans(graph)
    (DIR / "verify").mkdir(parents=True, exist_ok=True)
    index = {}
    for q in bench.load():
        plan = plans.get(q["id"], {"retrieval": "ContentRetrieval", "documents": [], "facets": []})
        pos = candidates(q["question"], plan, graph, tree)
        index[q["id"]] = {"plan": plan, "positions": pos}
        lines = [f"Query: {q['question']}", "Parts of the query: " + "; ".join(plan["facets"]), "", VERIFY_RULE, "",
                 f"Chunks ({len(pos)}):"]
        for k, p in enumerate(pos):
            u = tree.units[p]
            lines.append(f"[{k}] ({u['doc']} p{u['pages'][0]}) " + " ".join(u["text"].split()[:UNIT_WORDS]))
        (DIR / "verify" / f"{q['id']}.md").write_text("\n".join(lines) + "\n")
    json.dump(index, open(DIR / "verify_index.json", "w"))
    sizes = {i: len((DIR / "verify" / f"{i}.md").read_text().split()) for i in index}
    json.dump(sizes, open(DIR / "verify_sizes.json", "w"))
    print(f"wrote {len(index)} verify files; words per file: median {int(np.median(list(sizes.values())))}, "
          f"total {sum(sizes.values())}")


def score() -> None:
    from eval.gold_map import label_units
    from scripts.exp_retrieval import proxy, summarize
    _, tree = load("train")
    index = json.load(open(DIR / "verify_index.json"))
    labels = label_units(pickle.load(open(STORE / "train_docs.pkl", "rb")))
    qs = [q for q in bench.load() if (DIR / "verdict" / f"{q['id']}.json").exists() or not index[q["id"]]["positions"]]
    neo_rows = []
    for q in qs:
        pos = index[q["id"]]["positions"]
        path = DIR / "verdict" / f"{q['id']}.json"
        ids = json.load(open(path)).get("relevant_ids", []) if path.exists() else []
        sel = [tree.units[pos[k]]["id"] for k in {int(k) for k in ids if 0 <= int(k) < len(pos)}]
        neo_rows.append(proxy(np.array([u in labels[q["id"]] for u in sel], dtype=float), len(q["gold"])))
    missing = [q["id"] for q in bench.load() if q not in qs]
    print(f"{len(qs)} questions scored" + (f"; missing verdicts: {missing}" if missing else ""))
    summarize(f"NEO (Claude Haiku 5.5), n={len(qs)}", neo_rows)


if __name__ == "__main__":
    {"parse-input": parse_input, "verify-input": verify_input, "score": score}[sys.argv[1]]()
