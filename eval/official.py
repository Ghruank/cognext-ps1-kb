"""Loads the organisers' evaluator from Evaluation.ipynb without modifying it.

The RAGMetricsEvaluator class and evaluate_dataset function are executed verbatim
from the notebook cells. Only the transport is wrapped: CachedClient subclasses
OpenCodeClient so identical judge calls are served from disk and every paid call
is recorded in a token ledger for budget tracking.
"""

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "data" / "Evaluation" / "Evaluation.ipynb"
CACHE_DIR = ROOT / "eval" / ".cache" / "judge"
LEDGER = ROOT / "eval" / ".cache" / "ledger.jsonl"

# deepseek-v4.1-flash list price on OpenCode, USD per token
PRICE_IN, PRICE_CACHED, PRICE_OUT = 0.30e-6, 0.006e-6, 1.20e-6

_REQUIRED = ("class OpenCodeClient", "class RAGMetricsEvaluator", "def evaluate_dataset")


def _load_notebook_namespace() -> dict:
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    ns: dict = {}
    # same values as the notebook's configuration cell (used as default arguments)
    exec("import json\nimport time\nimport requests\n"
         'OPENCODE_BASE_URL = "https://opencode.ai/inference/openai/v1/chat/completions"\n'
         'OPENCODE_MODEL = "deepseek-v4.1-flash"', ns)
    for cell in cells:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and any(marker in src for marker in _REQUIRED):
            exec(src, ns)
    for name in ("OpenCodeClient", "RAGMetricsEvaluator", "evaluate_dataset"):
        assert name in ns, f"{name} not found in notebook"
    return ns


_NS = _load_notebook_namespace()
OpenCodeClient = _NS["OpenCodeClient"]
RAGMetricsEvaluator = _NS["RAGMetricsEvaluator"]
evaluate_dataset = _NS["evaluate_dataset"]
OPENCODE_BASE_URL = "https://opencode.ai/inference/openai/v1/chat/completions"
OPENCODE_MODEL = "deepseek-v4.1-flash"

_ledger_lock = threading.Lock()


def record_usage(kind: str, usage: dict) -> float:
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    prompt = usage.get("prompt_tokens", 0)
    out = usage.get("completion_tokens", 0)
    cost = (prompt - cached) * PRICE_IN + cached * PRICE_CACHED + out * PRICE_OUT
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with _ledger_lock, LEDGER.open("a") as f:
        f.write(json.dumps({"t": time.time(), "kind": kind, "prompt": prompt,
                            "cached": cached, "out": out, "usd": cost}) + "\n")
    return cost


def spent_usd() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(line)["usd"] for line in LEDGER.open())


class CachedClient(OpenCodeClient):
    """Same request payload as the notebook's client; adds a disk cache and usage ledger.

    mode="official" sends exactly the notebook's payload. mode="dev" sets reasoning
    effort to low, which is cheaper and less likely to run out of tokens on long answers.
    Dev scores are for comparing variants; checkpoint numbers use mode="official".
    """

    def __init__(self, *args, mode: str = "official", **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode

    def call_structured(self, system_prompt: str, user_prompt: str, max_tokens: int = 10000,
                        temperature: float = 0, max_retries: int = 3) -> dict:
        key = hashlib.sha256(json.dumps(
            [self.model, system_prompt, user_prompt, max_tokens, temperature]
            + ([self.mode] if self.mode != "official" else [])).encode()).hexdigest()
        path = CACHE_DIR / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.mode == "dev":
            # thinking fully disabled makes the judge contradict itself (marks matched points
            # as extra); low effort keeps it consistent at lower token use
            payload["reasoning_effort"] = "low"
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(self.base_url, headers=headers, json=payload, timeout=180)
                resp.raise_for_status()
                data = resp.json()
                record_usage(f"judge-{self.mode}", data.get("usage") or {})
                content = data["choices"][0]["message"]["content"]
                if not content:
                    raise ValueError(f"Empty content, finish_reason={data['choices'][0].get('finish_reason')}")
                result = self._safe_json_parse(content)
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result))
                return result
            except (requests.RequestException, KeyError, ValueError) as e:
                last_error = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"OpenCode call failed after {max_retries} attempts: {last_error}")


def make_evaluator(mode: str = "official") -> "RAGMetricsEvaluator":
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    client = CachedClient(api_key=os.environ["OPENCODE_API_KEY"], model=OPENCODE_MODEL,
                          base_url=OPENCODE_BASE_URL, mode=mode)
    return RAGMetricsEvaluator(client)
