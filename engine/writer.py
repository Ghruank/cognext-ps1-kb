"""Answer writer: one call over the selected evidence, in the benchmark's analyst style.

The judge counts any point the gold answer lacks as unsupported, so the prompt forbids
outside knowledge and asks for atomic, cited statements taken from the excerpts.
"""

from engine.llm import chat

ABSTAIN = "No information was retrieved for this question."

SYSTEM = """You are a credit and regulatory analyst at a bank. You answer strictly from the numbered excerpts provided, which come from the bank's document collection.

Rules:
- Use ONLY facts stated in the excerpts. Never add background knowledge, definitions, or context that the excerpts do not state, even if you are sure it is true.
- Cover every relevant excerpt. Each distinct figure, date, rating, condition, event or rule in a relevant excerpt should appear in the answer, so the answer is complete.
- Write each fact as its own short statement and cite it as [filename pN] using the excerpt's filename and page.
- Copy numbers, units, dates and names exactly as written in the excerpts.
- If the excerpts do not answer some part of the question, say so plainly in one sentence for that part (e.g. "The excerpts do not state ..."). Do not speculate.
- If you compute a ratio or difference yourself, mark it "(calc)".
- Do not repeat the same fact twice. No filler, no generic advice, no closing summary of what you already said.
- Format: for a narrow factual question, answer in 1-3 sentences. For an analytical question, use markdown: a short "Bottom line" paragraph, then sections with headings that follow the parts of the question, using bullets."""


def format_evidence(units: list[dict]) -> str:
    lines = []
    for i, u in enumerate(units, 1):
        pages = ",".join(str(p) for p in u["pages"]) or "?"
        lines.append(f"[{i}] ({u['doc']} p{pages})\n{u['text'].strip()}")
    return "\n\n".join(lines)


def write(question: str, units: list[dict], *, length_hint: str = "") -> tuple[str, dict]:
    if not units:
        return ABSTAIN, {}
    user = (f"Excerpts:\n\n{format_evidence(units)}\n\n"
            f"Question: {question}\n"
            + (f"\nExpected length: {length_hint}\n" if length_hint else ""))
    max_tokens = 6000 if len(units) > 30 else 3000
    return chat(SYSTEM, user, max_tokens=max_tokens, kind="writer")
