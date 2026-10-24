"""Langfuse query-level tracing for NEXUS."""

import os
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def query_observation(question: str):
    """Yield a Langfuse generation when configured; keep offline runs self-contained."""
    load_dotenv(ROOT / ".env")
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        yield None
        return
    from langfuse import get_client

    client = get_client()
    with client.start_as_current_observation(
        as_type="generation", name="nexus-query", model="groq/mixed",
        input={"question": question}, metadata={"architecture": "NEXUS", "version": "0.1.0"},
    ) as generation:
        yield generation
    # Inference is currently configured for Groq's free tier; actual usage is still
    # attached so model pricing can be configured in Langfuse later.
    client.flush()


def finish_observation(generation, result: dict) -> None:
    if generation is None:
        return
    usage = result.get("usage", {})
    generation.update(
        output={"answer": result.get("answer"), "retrieved_chunks": len(result.get("chunks", []))},
        usage_details={"input": int(usage.get("prompt_tokens", 0)),
                       "output": int(usage.get("completion_tokens", 0)),
                       "total": int(usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))},
        cost_details={"input": 0.0, "output": 0.0, "total": 0.0},
        metadata={"latency_s": result.get("latency_s"), "llm_calls": usage.get("calls", 0),
                  "candidate_blocks": result.get("retrieval", {}).get("candidate_blocks", 0),
                  "kept_blocks": result.get("retrieval", {}).get("filter", {}).get("kept", 0)},
    )
