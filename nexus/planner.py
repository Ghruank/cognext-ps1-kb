"""Compile a question into a deterministic evidence contract."""

import re
import math

from nexus.index import STOP, TOKEN
from nexus.schema import Facet, QueryPlan

NAME_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _terms(text: str, years: list[str]) -> list[str]:
    words = [w.strip(".,;:!?()[]{}\"'").lower() for w in TOKEN.findall(text)]
    words = [w for w in words if len(w) > 1 and w not in STOP]
    phrases = [" ".join(words[i:i + 2]) for i in range(len(words) - 1)]
    phrases += [w for w in words if w in years or len(w) >= 7]
    if len(words) == 1:
        phrases.append(words[0])
    return list(dict.fromkeys(phrases))[:24]


def _clauses(question: str) -> list[str]:
    """Split compound asks by ordinary punctuation/conjunction structure."""
    sentences = re.split(r"(?<=[.!?])\s+", question)
    parts = [part for sentence in sentences for part in
             re.split(r"[,;\n]+|\b(?:and|as well as|also)\b", sentence, flags=re.I)]
    return [part.strip(" .:-") for part in parts if len(_terms(part, [])) >= 1]


def _metadata_scope(question: str, catalog: list[dict]) -> list[str]:
    """Rank documents against LLM-built catalogue fields without query-time LLM calls."""
    q = {w for w in _terms(question, []) for w in re.findall(r"[a-z0-9]+", w.lower())
         if len(w) > 2 and w not in STOP}
    if not q:
        return []
    profiles = {}
    for d in catalog:
        fields = [
            (" ".join(str(x) for x in d.get("entities", []) if x), 3.0),
            (str(d.get("issuer", "")) + " " + str(d.get("title", "")), 3.0),
            (" ".join(str(x) for x in d.get("topics", []) if x), 2.0),
            (" ".join(str(x) for x in d.get("query_hints", []) if x), 2.5),
            (str(d.get("summary", "")), 1.0),
            (" ".join(str(d.get(k, "")) for k in ("doc_type", "jurisdiction", "period", "date", "version")), 1.0),
        ]
        weighted = {}
        for text, weight in fields:
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                if len(token) > 2 and token not in STOP:
                    weighted[token] = max(weighted.get(token, 0.0), weight)
        profiles[d["doc"]] = weighted
    df = {term: sum(term in profile for profile in profiles.values()) for term in q}
    idf = {term: math.log1p((len(profiles) + 1) / (df[term] + 1)) for term in q}
    denom = sum(idf.values()) or 1.0
    scores = {
        doc: sum(idf[t] * weights[t] for t in q if t in weights) / denom
        for doc, weights in profiles.items()
    }
    best = max(scores.values(), default=0.0)
    if best <= 0:
        return []
    # Relative cut preserves tied document families while discarding weak metadata matches.
    return [d["doc"] for d in catalog if scores.get(d["doc"], 0.0) >= best * 0.6]


def _fallback(question: str, catalog: list[dict]) -> QueryPlan:
    years = re.findall(r"\b(?:19|20)\d{2}(?:[-/]\d{2,4})?\b", question)
    words = [w.strip(".,;:!?()[]{}\"'").lower() for w in TOKEN.findall(question)]
    words = [w for w in words if len(w) > 1 and w not in STOP]
    # When the question names an issuer present in document filenames or PDF titles,
    # use that exact identity as a hard scope hint. Otherwise leave scope open.
    qwords = set(words)
    name_hits = []
    name_tokens: dict[str, set[str]] = {}
    name_sequences: dict[str, list[str]] = {}
    generic = {"integrated", "annual", "report", "limited", "ltd", "financial", "statement", "policy"}
    for d in catalog:
        name = f'{d["doc"]} {d.get("title") or ""}'.lower()
        sequence = [x.lower() for x in NAME_TOKEN.findall(name)]
        named = set(sequence)
        named = {x for x in named if len(x) > 3 and x not in generic}
        name_tokens[d["doc"]] = named
        name_sequences[d["doc"]] = sequence
    token_frequency: dict[str, int] = {}
    for named in name_tokens.values():
        for token in named:
            token_frequency[token] = token_frequency.get(token, 0) + 1
    surface_words = [w.lower() for w in NAME_TOKEN.findall(question)]
    surface_words = [w for w in surface_words if len(w) > 1]
    query_bigrams = set(zip(surface_words, surface_words[1:]))
    capitalized = {w.lower() for w in NAME_TOKEN.findall(question)
                   if w[:1].isupper() and len(w) >= 4}
    for d in catalog:
        named = name_tokens[d["doc"]]
        overlap = qwords & named
        named_bigrams = set(zip(name_sequences[d["doc"]], name_sequences[d["doc"]][1:]))
        phrase_match = bool(query_bigrams & named_bigrams)
        # A lone title word scopes a query only when it uniquely identifies one
        # catalog item. Multiple unrelated title-word overlaps are not enough.
        distinctive = len(overlap) == 1 and any(
            token in capitalized and token_frequency.get(token, 0) == 1 for token in overlap)
        if phrase_match or distinctive:
            name_hits.append(d["doc"])
    # A stated period narrows the document catalogue when matching metadata exists.
    period_docs = []
    if years:
        year_values = {y for period in years for y in re.findall(r"(?:19|20)\d{2}", period)}
        period_docs = [d["doc"] for d in catalog if year_values.intersection(
            {str(y) for y in d.get("years", [])})]
        if period_docs and name_hits:
            name_hits = [doc for doc in name_hits if doc in period_docs]
    if not name_hits:
        metadata_candidates = [d for d in catalog if d["doc"] in period_docs] if period_docs else catalog
        name_hits = _metadata_scope(question, metadata_candidates)

    clauses = _clauses(question)
    facets = []
    for i, clause in enumerate(clauses):
        facet_terms = _terms(clause, years)
        if not facet_terms:
            continue
        qterms = set(_terms(clause, years))
        heading_hits = []
        for d in catalog:
            if name_hits and d["doc"] not in name_hits:
                continue
            for item in d.get("toc", []):
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                title = str(item[1])
                title_terms = set(_terms(title, years))
                overlap = qterms & title_terms
                if overlap:
                    heading_hits.append((len(overlap) / max(1, len(qterms)), title))
        heading_hits.sort(key=lambda row: (-row[0], row[1]))
        hints = list(dict.fromkeys(title for _, title in heading_hits[:3]))
        facets.append(Facet(name=f"clause_{i + 1}", terms=facet_terms, section_hints=hints))
    if not facets:
        facets = [Facet(name="question", terms=_terms(question, years) or words[:4])]

    # Exact title phrase or a unique catalog token creates a conservative source scope.
    return QueryPlan(periods=years, documents=name_hits,
                     facets=facets, intent="lookup",
                     reasoning="Deterministic clause, catalogue, period, and TOC routing.")


def compile_plan(question: str, catalog: list[dict]) -> QueryPlan:
    """Compile a deterministic evidence contract without any model calls."""
    return _fallback(question, catalog)
