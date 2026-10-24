"""Query time: map the question's concepts to the vocabulary documents actually use.

Analysts ask about "leverage" or "debt-servicing capacity"; annual reports file the
evidence under "Borrowings", "Finance costs", "Net debt". One small LLM call lists the
line items, metric names and section titles where the answer would be written; those
terms are scored against unit descriptors (tables reduced to captions and row labels).
"""

import json

from neo.llm import chat

SYSTEM = "You know how financial, regulatory and technical documents are organised. Respond only with JSON."

PROMPT = """Question: {question}

List the exact wording a document would use where the evidence for this question is written: financial statement line items, note titles, metric and ratio names, section headings, defined terms, paragraph topics. Include both the narrative terms and the table row labels. 8-25 short terms, no explanations.

Return JSON: {{"terms": [str]}}"""


def expand(question: str) -> tuple[list[str], dict]:
    try:
        content, usage = chat(SYSTEM, PROMPT.format(question=question), max_tokens=600, json_mode=True, kind="neo-expand")
        return [str(t) for t in json.loads(content).get("terms", [])][:25], usage
    except (RuntimeError, ValueError):
        return [], {}
