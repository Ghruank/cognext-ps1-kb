"""OpenCode chat client for the system's own calls (planner, writer).

Thinking is disabled: it adds hidden output tokens and latency we pay for.
Every call's usage goes to the shared ledger so dev spend is visible next to judge spend.
"""

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from eval.official import record_usage

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

URL = "https://opencode.ai/inference/openai/v1/chat/completions"
MODEL = "deepseek-v4.1-flash"


def chat(system: str, user: str, *, max_tokens: int = 4000, json_mode: bool = False,
         kind: str = "gen", retries: int = 3) -> tuple[str, dict]:
    """Returns (content, usage). Raises after `retries` failed attempts."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {os.environ['OPENCODE_API_KEY']}"}
    err = None
    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            resp = requests.post(URL, headers=headers, json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage") or {}
            usage["latency_s"] = time.perf_counter() - t0
            usage["usd"] = record_usage(kind, usage)
            content = data["choices"][0]["message"]["content"] or ""
            if json_mode:
                json.loads(content)  # validate before returning
            return content, usage
        except (requests.RequestException, KeyError, ValueError) as e:
            err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"OpenCode call failed: {err}")
