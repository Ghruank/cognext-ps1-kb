"""LLM client for NEO's own calls: Groq by default, OpenCode (engine.llm) as a switch.

    NEO_LLM=groq       (default) free tier; rotates over models, each with its own limits
    NEO_LLM=opencode   engine.llm.chat, paid credits

Groq's free tier allows 8k tokens/minute and 1k requests/day per model. A request is
admitted only when the model's sliding one-minute window has room for its estimated
size (prompt + completion budget), so long batch jobs pace themselves instead of
hammering 429s; a 429 still backs off for the time Groq asks for.
"""

import json
import os
import threading
import time
from collections import deque
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

URL = "https://api.groq.com/openai/v1/chat/completions"
# strongest first; a call goes to the first model whose window has room
MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
TPM = 8000
LEDGER = ROOT / "alpha" / "storage" / "groq_ledger.jsonl"

# GROQ_API_KEY first; GROQ_API_KEY_2 for a model only once the first key's daily quota for it is spent
KEYS = [k for k in ("GROQ_API_KEY", "GROQ_API_KEY_2") if os.environ.get(k)]

_lock = threading.Lock()
_windows: dict[tuple, deque] = {(k, m): deque() for k in KEYS for m in MODELS}  # (time, tokens), last minute
_blocked_until: dict[tuple, float] = {b: 0.0 for b in _windows}
_exhausted: set[tuple] = set()


def _estimate(system: str, user: str, max_tokens: int) -> int:
    return (len(system) + len(user)) // 3 + max_tokens


def _acquire(need: int, models: list[str]) -> tuple[str, str]:
    """Returns (key env name, model) of the first bucket with room in its one-minute window."""
    while True:
        with _lock:
            now = time.time()
            live = [(k, m) for m in models for k in KEYS[:1 + sum((kk, m) in _exhausted for kk in KEYS)]
                    if (k, m) not in _exhausted]
            if not live:
                raise RuntimeError("every Groq key has spent its daily quota for these models")
            for b in live:
                w = _windows[b]
                while w and now - w[0][0] > 60:
                    w.popleft()
                if now >= _blocked_until[b] and sum(t for _, t in w) + need <= TPM:
                    w.append((now, need))
                    return b
        time.sleep(0.5)


def _record(kind: str, model: str, usage: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with _lock, LEDGER.open("a") as f:
        f.write(json.dumps({"t": time.time(), "kind": kind, "model": model,
                            "prompt": usage.get("prompt_tokens", 0), "out": usage.get("completion_tokens", 0)}) + "\n")


def _groq(system: str, user: str, *, max_tokens: int, json_mode: bool, kind: str,
          models: list[str] | None, retries: int) -> tuple[str, dict]:
    pool = models or MODELS
    max_tokens = min(max_tokens, 3000)
    need = _estimate(system, user, max_tokens)
    if need > TPM:
        raise ValueError(f"request of ~{need} tokens exceeds the {TPM} tokens/minute limit; split it")
    err = None
    for attempt in range(retries + 2 * len(pool) * len(KEYS)):
        key, model = _acquire(need, pool)
        headers = {"Authorization": f"Bearer {os.environ[key]}"}
        payload = {"model": model, "temperature": 0, "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if "gpt-oss" in model:
            payload |= {"reasoning_effort": "low", "include_reasoning": False}
        elif "qwen" in model:
            payload |= {"reasoning_effort": "none"}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            t0 = time.perf_counter()
            resp = requests.post(URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                wait = float(resp.headers.get("retry-after", 10))
                daily = "per day" in resp.text or wait > 600
                with _lock:
                    if daily:
                        _exhausted.add((key, model))
                    _blocked_until[(key, model)] = time.time() + wait
                err = RuntimeError(f"429 on {model} ({key}){' daily quota spent' if daily else ''}")
                continue
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage") or {}
            usage |= {"latency_s": time.perf_counter() - t0, "usd": 0.0, "model": model}
            _record(kind, model, usage)
            content = data["choices"][0]["message"].get("content") or ""
            if json_mode:
                json.loads(content)
            return content, usage
        except requests.ConnectionError as e:
            # DNS/routing failures are provider-wide; trying every model only adds delay.
            raise RuntimeError(f"Groq connection unavailable: {e}") from e
        except (requests.RequestException, KeyError, ValueError) as e:
            err = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Groq call failed: {err}")


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
# free tier: ~20 requests/day per model, so a model whose daily quota is spent is skipped for the rest of the run
GEMINI_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-3.5-flash-lite",
                 "gemini-3.1-flash-lite", "gemini-flash-lite-latest"]
_gemini_spent: set[str] = set()


def _gemini(system: str, user: str, *, max_tokens: int, json_mode: bool, kind: str, retries: int) -> tuple[str, dict]:
    headers = {"Authorization": f"Bearer {os.environ['GEMINI_API_KEY']}"}
    err = None
    for attempt in range(retries + 2 * len(GEMINI_MODELS)):
        live = [m for m in GEMINI_MODELS if m not in _gemini_spent]
        if not live:
            raise RuntimeError("every Gemini model has spent its daily free-tier quota")
        model = live[0]
        payload = {"model": model, "temperature": 0, "max_tokens": max(max_tokens, 4000),
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            t0 = time.perf_counter()
            resp = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=300)
            if resp.status_code in (429, 503):
                if "per day" in resp.text or "PerDay" in resp.text:
                    _gemini_spent.add(model)
                else:
                    time.sleep(min(60, 5 * (attempt + 1)))
                err = RuntimeError(f"{model} {resp.status_code}: {resp.text[:160]}")
                continue
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage") or {}
            usage |= {"latency_s": time.perf_counter() - t0, "usd": 0.0, "model": model}
            _record(kind, model, usage)
            content = data["choices"][0]["message"].get("content") or ""
            if json_mode:
                content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                json.loads(content)
            return content, usage
        except requests.ConnectionError as e:
            # DNS/routing failures are provider-wide; trying every model only adds delay.
            raise RuntimeError(f"Gemini connection unavailable: {e}") from e
        except (requests.RequestException, KeyError, ValueError) as e:
            err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Gemini call failed: {err}")


def chat(system: str, user: str, *, max_tokens: int = 2000, json_mode: bool = False, kind: str = "neo",
         models: list[str] | None = None, retries: int = 3, provider: str | None = None) -> tuple[str, dict]:
    """Returns (content, usage); same contract as engine.llm.chat."""
    provider = provider or os.environ.get("NEO_LLM_" + kind.upper().replace("-", "_"),
                                         os.environ.get("NEO_LLM", "groq"))
    if provider == "gemini":
        return _gemini(system, user, max_tokens=max_tokens, json_mode=json_mode, kind=kind, retries=retries)
    if provider == "opencode":
        from engine.llm import chat as opencode_chat
        return opencode_chat(system, user, max_tokens=max_tokens, json_mode=json_mode, kind=kind, retries=retries)
    return _groq(system, user, max_tokens=max_tokens, json_mode=json_mode, kind=kind, models=models, retries=retries)
