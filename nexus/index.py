"""Lossless, LLM-free source map backed by SQLite FTS5.

The index stores parser-native blocks and page/section metadata. It does not create
summaries, embeddings, or LLM-generated chunks. FTS is one address mechanism; query
scope and facet coverage determine what gets inspected.
"""

import json
import re
import sqlite3
from types import SimpleNamespace
from pathlib import Path

from neo.ingest import parse_folder, parse_pdf, repair_ligatures

TOKEN = re.compile(r"[\w][\w./%-]*", re.UNICODE)
STOP = {"a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "give",
        "how", "in", "is", "it", "of", "on", "or", "the", "to", "was", "were", "what", "when",
        "which", "who", "with", "would", "your"}


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    return db


def build_from_docs(docs: list[dict], path: str | Path, *, page_vectors: tuple | None = None,
                    version_graph=None, llm_catalog: dict | None = None,
                    page_cards: dict | None = None) -> dict:
    """Build/replace a collection index from already parsed layout blocks."""
    path = Path(path)
    if version_graph is None:
        version_graph = _deterministic_version_families(docs)
    if path.exists():
        path.unlink()
    vector_path = path.with_suffix(".vectors.pkl")
    if page_vectors is None:
        vector_path.unlink(missing_ok=True)
    db = _connect(path)
    db.executescript("""
        CREATE TABLE documents (
            doc TEXT PRIMARY KEY,
            pages INTEGER NOT NULL,
            title TEXT,
            first_pages TEXT,
            toc_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE units (
            id TEXT PRIMARY KEY,
            doc TEXT NOT NULL,
            page INTEGER NOT NULL,
            block INTEGER NOT NULL,
            kind TEXT NOT NULL,
            section TEXT NOT NULL,
            text TEXT NOT NULL,
            FOREIGN KEY(doc) REFERENCES documents(doc)
        );
        CREATE INDEX units_doc_page ON units(doc, page, block);
        CREATE TABLE page_cards (
            doc TEXT NOT NULL, page INTEGER NOT NULL, kind TEXT NOT NULL,
            card TEXT NOT NULL, questions TEXT NOT NULL,
            PRIMARY KEY(doc,page)
        );
        CREATE VIRTUAL TABLE page_cards_fts USING fts5(
            card, questions, doc UNINDEXED, page UNINDEXED,
            tokenize='unicode61 remove_diacritics 2'
        );
        CREATE TABLE version_nodes (
            doc TEXT PRIMARY KEY, family TEXT NOT NULL, years TEXT NOT NULL,
            consolidated INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX version_nodes_family ON version_nodes(family);
        CREATE TABLE change_links (src TEXT NOT NULL, dst TEXT NOT NULL, PRIMARY KEY(src,dst));
        CREATE VIRTUAL TABLE units_fts USING fts5(
            text, section, doc UNINDEXED, page UNINDEXED, kind UNINDEXED,
            content='units', content_rowid='rowid', tokenize='unicode61 remove_diacritics 2'
        );
        CREATE TRIGGER units_ai AFTER INSERT ON units BEGIN
            INSERT INTO units_fts(rowid,text,section,doc,page,kind)
            VALUES (new.rowid,new.text,new.section,new.doc,new.page,new.kind);
        END;
    """)
    n_units = 0
    for d in docs:
        meta = d.get("pdf_meta") or {}
        toc = d.get("toc") or []
        catalog = (llm_catalog or {}).get(d["doc"], {})
        db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?)", (
            d["doc"], int(d.get("n_pages", 0)), meta.get("title"),
            d.get("first_pages", ""), json.dumps(toc, ensure_ascii=False),
            json.dumps(catalog, ensure_ascii=False),
        ))
        for i, u in enumerate(d.get("units", [])):
            page = int((u.get("pages") or [1])[0])
            uid = str(u.get("id") or f'{d["doc"]}#{i}')
            db.execute("INSERT INTO units(id,doc,page,block,kind,section,text) VALUES(?,?,?,?,?,?,?)", (
                uid, d["doc"], page, int(u.get("block", i)), str(u.get("type", "prose")),
                str(u.get("section", "")), str(u.get("text", "")),
            ))
            n_units += 1
    for (doc, page), card in (page_cards or {}).items():
        kind = str(card.get("kind", "content"))
        text = str(card.get("card", ""))
        questions = " ".join(str(q) for q in card.get("questions", []))
        db.execute("INSERT INTO page_cards VALUES (?,?,?,?,?)", (doc, int(page), kind, text, questions))
        db.execute("INSERT INTO page_cards_fts(card,questions,doc,page) VALUES(?,?,?,?)",
                   (text, questions, doc, int(page)))
    if version_graph is not None:
        for doc in version_graph.family:
            years = version_graph.doc_years.get(doc, [])
            db.execute("INSERT INTO version_nodes VALUES (?,?,?,?)", (
                doc, version_graph.family[doc], json.dumps(sorted(years)),
                int(doc in version_graph.consolidated),
            ))
        valid_units = {row[0] for row in db.execute("SELECT id FROM units")}
        links = {(a, b) for a, b in version_graph.change_links if a in valid_units and b in valid_units}
        links |= {(b, a) for a, b in links}
        db.executemany("INSERT OR IGNORE INTO change_links VALUES (?,?)", sorted(links))
    db.commit()
    db.execute("INSERT INTO units_fts(units_fts) VALUES('optimize')")
    db.commit()
    pages = db.execute("SELECT COUNT(DISTINCT doc || ':' || page) FROM units").fetchone()[0]
    db.close()
    if page_vectors is not None:
        from nexus.vectors import save_page_vectors

        save_page_vectors(vector_path, *page_vectors)
    return {"documents": len(docs), "pages": pages, "units": n_units,
            "llm_catalog_documents": len(llm_catalog or {}), "llm_page_cards": len(page_cards or {}),
            "page_vectors": page_vectors is not None,
            "version_families": len({version_graph.family[d] for d in version_graph.family}) if version_graph else 0,
            "change_links": len(links) if version_graph else 0,
            "index": str(path)}


def build_folder(folder: str | Path, path: str | Path, workers: int = 8, *, use_dense: bool = True) -> dict:
    """Parse source PDFs deterministically, then index their layout blocks."""
    from nexus.vectors import create_page_vectors

    if workers <= 1:
        paths = sorted(str(p) for p in Path(folder).rglob("*") if p.suffix.lower() == ".pdf")
        docs = [parse_pdf(p) for p in paths]
        repair_ligatures(docs)
    else:
        docs = parse_folder(str(folder), workers=workers)
    vectors = create_page_vectors(docs) if use_dense else None
    graph = _deterministic_version_families(docs)
    return build_from_docs(docs, path, page_vectors=vectors, version_graph=graph)


def _deterministic_version_families(docs: list[dict]):
    """Build NEO's Lattice family layer from source metadata and text, with no LLM calls.

    Fresh API collections have no NEO card artifact. Filename/title metadata provides a
    conservative annual-report issuer hint; Lattice's document-shingle containment handles
    version/consolidation relationships. Change-pair edges are only imported from the saved
    NEO graph when indexing that corpus, because generic uploads have no trained unit vectors.
    """
    if not docs:
        return None
    from engine.versions import Lattice

    catalog = {}
    for d in docs:
        name = str(d.get("doc", ""))
        meta = d.get("pdf_meta") or {}
        title = str(meta.get("title") or name)
        text = f"{name} {title}"
        words = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        tokens_ = [x.lower() for x in re.findall(r"[a-z0-9]+", words)]
        generic = {"annual", "report", "integrated", "financial", "statement", "statements", "limited",
                   "ltd", "inc", "company", "corporation", "the", "and", "of", "for", "fy", "year"}
        issuer = " ".join(x for x in tokens_ if not re.fullmatch(r"(?:19|20)\d{2}|\d{2}", x) and x not in generic)
        report = "annual report" in text.lower() or "annualreport" in text.lower()
        years = sorted({int(x) for x in re.findall(r"\b(?:19|20)\d{2}\b", text)})
        catalog[name] = {
            "title": title, "entities": [issuer] if issuer else [],
            "doc_type": "annual_report" if report else "other",
            "period": " ".join(map(str, years)), "date": "", "version": "",
        }
    lattice = Lattice(docs, catalog)
    return SimpleNamespace(family=lattice.family, doc_years=lattice.doc_years,
                           consolidated=lattice.consolidated, change_links=[])


def build_from_pickle(pickle_path: str | Path, path: str | Path, *, use_dense: bool = True,
                      use_llm_metadata: bool = False) -> dict:
    import pickle

    with open(pickle_path, "rb") as f:
        docs = pickle.load(f)
    p = Path(pickle_path)
    vectors = None
    if use_dense:
        from nexus.vectors import create_page_vectors

        cached = p.parent / "train_pagevecs_potion-retrieval-32M.pkl"
        vectors = create_page_vectors(docs, cached)
    graph_path = p.parent / "train_graph.pkl"
    graph = None
    llm_catalog = {}
    if graph_path.exists():
        with graph_path.open("rb") as f:
            graph = pickle.load(f)
        if use_llm_metadata:
            llm_catalog = graph.catalog
            hints_path = Path(__file__).resolve().parent / "storage" / "llm_doc_queries.json"
            if hints_path.exists():
                with hints_path.open() as f:
                    query_hints = json.load(f)
                for doc, queries in query_hints.items():
                    llm_catalog.setdefault(doc, {})["query_hints"] = queries
    card_path = p.parent / "train_pagecards.pkl"
    page_cards = {}
    if use_llm_metadata and card_path.exists():
        with card_path.open("rb") as f:
            page_cards = pickle.load(f)
    return build_from_docs(docs, path, page_vectors=vectors, version_graph=graph,
                           llm_catalog=llm_catalog, page_cards=page_cards)


def document_catalog(db: sqlite3.Connection) -> list[dict]:
    try:
        version = {r[0]: {"family": r[1], "years": json.loads(r[2]), "consolidated": bool(r[3])}
                   for r in db.execute("SELECT doc,family,years,consolidated FROM version_nodes")}
    except sqlite3.OperationalError:
        version = {}
    columns = {r[1] for r in db.execute("PRAGMA table_info(documents)")}
    has_metadata = "metadata_json" in columns
    select = "doc,pages,title,first_pages,toc_json" + (",metadata_json" if has_metadata else "")
    out = []
    for row in db.execute(f"SELECT {select} FROM documents ORDER BY doc"):
        metadata = json.loads(row[5]) if has_metadata and row[5] else {}
        out.append({**metadata, "doc": row[0], "pages": row[1], "title": row[2],
                    "first_pages": row[3], "toc": json.loads(row[4]), **version.get(row[0], {})})
    return out


def tokens(text: str) -> list[str]:
    # TOKEN retains punctuation useful inside values (e.g. 2025/26, 12.5%,
    # credit-risk); strip only sentence/quote delimiters at the edges so names
    # such as "India." still match catalog metadata "India".
    cleaned = (t.strip(".,;:!?()[]{}\"'") for t in TOKEN.findall(text))
    return [t.lower() for t in cleaned if len(t) > 1 and t.lower() not in STOP]


def fts_query(text: str) -> str:
    """A quoted phrase query, safe for punctuation-heavy finance terms and FTS syntax."""
    terms = list(dict.fromkeys(tokens(text)))
    return '"' + " ".join(t.replace('"', '""') for t in terms) + '"' if terms else ""
