"""Facet-wise routing and deterministic, auditable evidence selection."""

import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from nexus.index import fts_query
from nexus.schema import Evidence, QueryPlan
from nexus.vectors import load_page_vectors, query_vectors


def _search_facet(db: sqlite3.Connection, queries: list[str], docs: list[str], kind: str,
                  limit: int | None) -> list[tuple]:
    sql = """SELECT u.id,u.doc,u.page,u.section,u.kind,u.text,bm25(units_fts,1.0,0.7,0.0,0.0,0.0)
             FROM units_fts JOIN units u ON u.rowid=units_fts.rowid
             WHERE units_fts MATCH ?"""
    found = {}
    for query in queries:
        q = fts_query(query)
        if not q:
            continue
        args: list = [q]
        query_sql = sql
        if docs:
            query_sql += " AND u.doc IN (" + ",".join("?" for _ in docs) + ")"
            args.extend(docs)
        if kind in ("table", "prose"):
            query_sql += " AND u.kind=?"
            args.append(kind)
        query_sql += " ORDER BY bm25(units_fts,1.0,0.7,0.0,0.0,0.0)"
        if limit is not None:
            query_sql += " LIMIT ?"
            args.append(limit)
        for row in db.execute(query_sql, args):
            prev = found.get(row[0])
            if prev is None or float(row[6]) < float(prev[6]):
                found[row[0]] = row
    ranked = sorted(found.values(), key=lambda row: float(row[6]))
    return ranked[:limit] if limit is not None else ranked


def _search_page_cards(db: sqlite3.Connection, queries: list[str], docs: list[str],
                       limit: int | None) -> list[tuple[str, int, float]]:
    """Find pages through saved ingestion-time descriptions/questions, if indexed."""
    sql = """SELECT c.doc,c.page,bm25(page_cards_fts,1.0,0.8)
             FROM page_cards_fts JOIN page_cards c
               ON c.doc=page_cards_fts.doc AND c.page=page_cards_fts.page
             WHERE page_cards_fts MATCH ? AND c.kind!='filler'"""
    found: dict[tuple[str, int], float] = {}
    for query in queries:
        q = fts_query(query)
        if not q:
            continue
        args: list = [q]
        query_sql = sql
        if docs:
            query_sql += " AND c.doc IN (" + ",".join("?" for _ in docs) + ")"
            args.extend(docs)
        query_sql += " ORDER BY bm25(page_cards_fts,1.0,0.8)"
        if limit is not None:
            query_sql += " LIMIT ?"
            args.append(limit)
        for doc, page, rank in db.execute(query_sql, args):
            key = (str(doc), int(page))
            found[key] = min(found.get(key, float("inf")), float(rank))
    ranked = sorted(((d, p, score) for (d, p), score in found.items()), key=lambda row: row[2])
    return ranked[:limit] if limit is not None else ranked


def retrieve(db: sqlite3.Connection, plan: QueryPlan, *, question: str = "", per_facet_limit: int | None = None,
             vector_threshold: float = 0.42, use_dense: bool = True) -> tuple[list[Evidence], dict]:
    """Union facet-specific page hits; do not collapse the whole query to one top-k list.

    Each facet contributes its own route. Matching pages are expanded to all native blocks
    on that page, preserving table and paragraph context. The LLM relevance gate is applied
    after lexical, metadata, TOC, and optional page-vector routes are unioned.
    """
    docs = list(plan.documents)
    compare_mode = plan.intent.lower() in {"compare", "trend"} or bool(
        __import__("re").search(r"\b(compare|compared|comparison|change|changed|difference|trend|versus|\bvs\b|across years|over time)\b", question, __import__("re").I))
    # Version-family routing is query-time only. For cross-version questions, expand an
    # explicitly selected source to its family before lexical/dense page search.
    family_expanded = 0
    if compare_mode and docs:
        try:
            placeholders = ",".join("?" for _ in docs)
            family_rows = db.execute(
                f"SELECT DISTINCT v2.doc FROM version_nodes v1 JOIN version_nodes v2 ON v1.family=v2.family "
                f"WHERE v1.doc IN ({placeholders})", docs).fetchall()
            expanded = list(dict.fromkeys([*docs, *(r[0] for r in family_rows)]))
            family_expanded = len(expanded) - len(docs)
            docs = expanded
        except sqlite3.OperationalError:
            pass  # legacy collection without graph tables
    routes: dict[str, set[int]] = defaultdict(set)
    candidates: dict[str, Evidence] = {}
    route_counts = []
    card_route_counts = []
    dense_counts = []
    facet_pages: dict[int, set[tuple[str, int]]] = defaultdict(set)
    page_facets: dict[tuple[str, int], set[int]] = defaultdict(set)
    page_route_scores: dict[tuple[str, int], float] = defaultdict(float)
    db_path = Path(db.execute("PRAGMA database_list").fetchone()[2])
    vector_path = db_path.with_suffix(".vectors.pkl")
    vector_keys, page_matrix = load_page_vectors(vector_path) if use_dense and vector_path.exists() else ([], None)
    scope = set(docs)
    dense_pages = [(i, key) for i, key in enumerate(vector_keys) if not scope or key[0] in scope]
    dense_unavailable = False
    for fi, facet in enumerate(plan.facets):
        rows = _search_facet(db, [*facet.terms, *facet.section_hints], docs,
                             facet.evidence_type, per_facet_limit)
        try:
            card_rows = _search_page_cards(db, [*facet.terms, *facet.section_hints], docs,
                                           per_facet_limit)
        except sqlite3.OperationalError:
            card_rows = []
        pages = {(r[1], int(r[2])) for r in rows}
        dense_hits = 0
        used_threshold = vector_threshold
        semantic_status = "ok"
        if page_matrix is not None and dense_pages:
            qtext = " ".join([question, facet.name, *facet.terms[:8]])
            qvecs = query_vectors([qtext])
            if qvecs is None:
                dense_unavailable = True
                semantic_status = "query encoder unavailable"
            else:
                qvec = qvecs[0]
                idx = np.fromiter((i for i, _ in dense_pages), dtype=np.int64)
                scores = page_matrix[idx] @ qvec
                used_threshold = max(vector_threshold, float(scores.max()) - 0.18) if len(scores) else vector_threshold
                for j in np.flatnonzero(scores >= used_threshold):
                    doc, page = dense_pages[int(j)][1]
                    routes[doc].add(int(page))
                    facet_pages[fi].add((doc, int(page)))
                    page_facets[(doc, int(page))].add(fi)
                    page_route_scores[(doc, int(page))] = max(page_route_scores[(doc, int(page))], float(scores[j]))
                    dense_hits += 1
        route_counts.append({"facet": facet.name, "matched_units": len(rows), "matched_pages": len(pages)})
        for doc, page in pages:
            routes[doc].add(page)
            facet_pages[fi].add((doc, page))
            page_facets[(doc, page)].add(fi)
        dense_counts.append({"facet": facet.name, "semantic_pages": dense_hits,
                             "semantic_threshold": round(used_threshold, 4) if semantic_status == "ok" else None,
                             "status": semantic_status})
        for row in rows:
            uid, doc, page, section, kind, text, rank = row
            ev = candidates.get(uid)
            if ev is None:
                ev = Evidence(str(uid), str(doc), int(page), str(section), str(kind), str(text))
                candidates[uid] = ev
            ev.facets.append(fi)
            # FTS5 BM25 is negative for better matches; normalize into an ordering score.
            lexical_score = 1.0 / (1.0 + abs(float(rank)))
            ev.score += lexical_score
            page_route_scores[(str(doc), int(page))] = max(page_route_scores[(str(doc), int(page))], lexical_score)

        for doc, page, rank in card_rows:
            routes[doc].add(page)
            facet_pages[fi].add((doc, page))
            page_facets[(doc, page)].add(fi)
            card_score = 1.1 / (1.0 + abs(rank))
            page_route_scores[(doc, page)] = max(page_route_scores[(doc, page)], card_score)
        card_route_counts.append({"facet": facet.name, "matched_pages": len(card_rows)})

    # Version graph: for compare/trend questions, pull the explicitly paired delta unit
    # from another version when one endpoint was found lexically. No general graph walk.
    change_expanded = 0
    if compare_mode and candidates:
        try:
            hits = list(candidates)
            linked = []
            for start in range(0, len(hits), 500):
                batch = hits[start:start + 500]
                linked.extend(db.execute(
                    "SELECT u.id,u.doc,u.page,u.section,u.kind,u.text,l.src "
                    "FROM change_links l JOIN units u ON u.id=l.dst WHERE l.src IN (" +
                    ",".join("?" for _ in batch) + ")", batch).fetchall())
            scope = set(docs)
            for uid, doc, page, section, kind, text, src in linked:
                if scope and doc not in scope:
                    continue
                if uid in candidates:
                    continue
                source = candidates[src]
                candidates[uid] = Evidence(str(uid), str(doc), int(page), str(section), str(kind),
                                           str(text), facets=list(source.facets), score=source.score * 0.8)
                routes[str(doc)].add(int(page))
                page_facets[(str(doc), int(page))].update(source.facets)
                page_route_scores[(str(doc), int(page))] = source.score * 0.8
                change_expanded += 1
        except sqlite3.OperationalError:
            pass  # legacy collection without graph tables

    # Expand matched pages to parser-native blocks for coverage. The deterministic
    # facet scorer selects evidence after page reconstruction.
    if routes:
        clauses, params = [], []
        for doc, pages in routes.items():
            if pages:
                clauses.append("(doc=? AND page IN (" + ",".join("?" for _ in pages) + "))")
                params.extend([doc, *sorted(pages)])
        if clauses:
            sql = "SELECT id,doc,page,section,kind,text FROM units WHERE " + " OR ".join(clauses)
            for uid, doc, page, section, kind, text in db.execute(sql, params):
                ev = candidates.get(uid)
                if ev is None:
                    ev = Evidence(str(uid), str(doc), int(page), str(section), str(kind), str(text))
                    candidates[uid] = ev
                ev.facets = sorted(set(ev.facets) | page_facets[(doc, int(page))])
                ev.score = max(ev.score, page_route_scores[(str(doc), int(page))])
    evs = sorted(candidates.values(), key=lambda x: (x.doc, x.page, x.id))
    return evs, {"facets": route_counts, "page_card_routes": card_route_counts,
                 "semantic_routes": dense_counts, "semantic_unavailable": dense_unavailable,
                 "routed_pages": sum(map(len, routes.values())),
                 "candidate_blocks": len(evs), "scoped_documents": len(docs) if docs else None,
                 "version_graph": {"compare_mode": compare_mode, "family_documents_added": family_expanded,
                                   "change_units_added": change_expanded}}


def eliminate(plan: QueryPlan, candidates: list[Evidence], *, max_per_facet: int = 24) -> tuple[list[Evidence], dict]:
    """Select candidates with deterministic facet-term overlap scoring."""
    if not candidates:
        return [], {"calls": 0, "kept": 0, "failed_batches": 0}
    from nexus.index import tokens

    selected: dict[str, Evidence] = {}
    facet_stats = []
    for fi, facet in enumerate(plan.facets):
        terms = set(tokens(" ".join([*facet.terms, *facet.section_hints])))
        ranked = []
        for ev in candidates:
            text_terms = set(tokens(ev.section + " " + ev.text))
            overlap = len(terms & text_terms)
            if overlap:
                score = overlap / max(1, len(terms)) + ev.score
                ranked.append((score, overlap, ev))
        ranked.sort(key=lambda x: (x[0], x[1], len(x[2].text)), reverse=True)
        keep_n = min(max(1, max_per_facet), len(ranked))
        for score, _, ev in ranked[:keep_n]:
            ev.facets = sorted(set(ev.facets) | {fi})
            ev.score = max(ev.score, score)
            selected[ev.id] = ev
        facet_stats.append({"facet": facet.name, "matching_candidates": len(ranked), "kept": keep_n})
    if not selected:
        selected[candidates[0].id] = candidates[0]
    kept = sorted(selected.values(), key=lambda e: (-e.score, e.doc, e.page, e.id))
    return kept, {"calls": 0, "kept": len(kept), "failed_batches": 0,
                  "method": "facet-term-overlap", "facets": facet_stats}
