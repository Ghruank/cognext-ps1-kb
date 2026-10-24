"""Langfuse query-level tracing for ALPHA."""

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
        as_type="generation", name="alpha-query", model="groq/mixed",
        input={"question": question}, metadata={"architecture": "ALPHA", "version": "1.0.0"},
    ) as generation:
        yield generation
    # Inference is currently configured for Groq's free tier; actual usage is still
    # attached so model pricing can be configured in Langfuse later.
    client.flush()


def finish_observation(generation, result: dict) -> None:
    if generation is None:
        return
    tok = result.get("tokens", {})
    generation.update(
        output={"answer": result.get("answer"), "retrieved_chunks": len(result.get("chunks", []))},
        usage_details={"input": int(tok.get("prompt", 0)), "output": int(tok.get("completion", 0)),
                       "total": int(tok.get("prompt", 0) + tok.get("completion", 0))},
        cost_details={"input": 0.0, "output": 0.0, "total": 0.0},
        metadata={"latency_s": result.get("latency_s"), "steps": [s["step_name"] for s in result.get("trace", [])]},
    )
