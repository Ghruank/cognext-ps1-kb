"""Typed query plans and evidence records used by NEXUS."""

from dataclasses import dataclass, field


@dataclass
class Facet:
    """One independently answerable evidence obligation."""

    name: str
    terms: list[str] = field(default_factory=list)
    evidence_type: str = "any"
    section_hints: list[str] = field(default_factory=list)


@dataclass
class QueryPlan:
    """A declarative scope and coverage contract compiled from a question."""

    entities: list[str] = field(default_factory=list)
    periods: list[str] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)
    facets: list[Facet] = field(default_factory=list)
    intent: str = "lookup"
    reasoning: str = ""
    usage: dict = field(default_factory=dict)


@dataclass
class Evidence:
    id: str
    doc: str
    page: int
    section: str
    kind: str
    text: str
    facets: list[int] = field(default_factory=list)
    score: float = 0.0

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "doc": self.doc,
            "pages": [self.page],
            "section": self.section,
            "type": self.kind,
            "text": self.text,
            "facets": self.facets,
            "score": round(self.score, 5),
        }
