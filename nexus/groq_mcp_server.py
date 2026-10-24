"""Minimal stdio MCP server exposing one Groq chat-completion tool.

Run with:
    .venv/bin/python nexus/groq_mcp_server.py

The server reads GROQ_API_KEY from the process environment or kb/.env. It never
returns the key to the MCP client and writes protocol messages only to stdout.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

SERVER_NAME = "cognext-groq"
SERVER_VERSION = "0.1.0"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
SUPPORTED_PROTOCOLS = {"2024-11-05", "2025-03-26", "2025-06-18"}

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(SERVER_NAME)

TOOLS = [{
    "name": "groq_chat",
    "title": "Groq chat completion",
    "description": "Generate a response with a Groq-hosted chat model.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "User prompt sent to the model."},
            "system": {"type": "string", "description": "Optional system instruction."},
            "model": {"type": "string", "description": "Groq model ID; defaults to GROQ_MODEL or llama-3.3-70b-versatile."},
            "max_tokens": {"type": "integer", "minimum": 1, "maximum": 8192, "default": 2048},
            "temperature": {"type": "number", "minimum": 0, "maximum": 2, "default": 0.2},
            "json_mode": {"type": "boolean", "default": False,
                          "description": "Ask Groq to return a JSON object."},
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
}]


class RpcError(Exception):
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def _groq_chat(args: dict[str, Any]) -> dict[str, Any]:
    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise RpcError(-32602, "prompt must be a non-empty string")
    system = args.get("system", "You are a helpful assistant.")
    model = args.get("model", DEFAULT_MODEL)
    max_tokens = args.get("max_tokens", 2048)
    temperature = args.get("temperature", 0.2)
    json_mode = args.get("json_mode", False)
    if not isinstance(system, str) or not isinstance(model, str) or not model.strip():
        raise RpcError(-32602, "system and model must be strings")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= 8192:
        raise RpcError(-32602, "max_tokens must be an integer from 1 to 8192")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not 0 <= temperature <= 2:
        raise RpcError(-32602, "temperature must be between 0 and 2")
    if not isinstance(json_mode, bool):
        raise RpcError(-32602, "json_mode must be a boolean")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RpcError(-32001, "GROQ_API_KEY is not set in the environment or kb/.env")

    payload: dict[str, Any] = {
        "model": model,
        "temperature": float(temperature),
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    try:
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage") or {}
    except requests.HTTPError as exc:
        # Expose the provider's status and short error body for diagnosis. Never
        # include request headers, which contain the API key.
        response = exc.response
        detail = ""
        if response is not None:
            try:
                body = response.json()
                err = body.get("error", {}) if isinstance(body, dict) else {}
                detail = err.get("message", "") if isinstance(err, dict) else ""
            except ValueError:
                detail = response.text
            detail = " ".join(str(detail).split())[:400]
        status = response.status_code if response is not None else "unknown"
        log.warning("Groq returned HTTP %s: %s", status, detail or "no response detail")
        suffix = f": {detail}" if detail else ""
        raise RpcError(-32002, f"Groq returned HTTP {status}{suffix}") from exc
    except requests.RequestException as exc:
        # Do not include request headers or environment values in returned errors.
        log.warning("Groq request failed: %s", type(exc).__name__)
        raise RpcError(-32002, f"Groq request failed: {type(exc).__name__}") from exc
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        log.warning("Groq returned an invalid response: %s", type(exc).__name__)
        raise RpcError(-32003, "Groq returned an invalid response") from exc

    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": {"model": model, "usage": usage},
    }


def _handle(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}
    if not isinstance(method, str) or not isinstance(params, dict):
        raise RpcError(-32600, "Invalid Request")

    # Notifications never receive a response.
    if request_id is None:
        if method in {"notifications/initialized", "notifications/cancelled"}:
            return None
        return None

    if method == "initialize":
        requested = params.get("protocolVersion")
        protocol = requested if requested in SUPPORTED_PROTOCOLS else "2024-11-05"
        result = {
            "protocolVersion": protocol,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name != "groq_chat":
            raise RpcError(-32602, f"Unknown tool: {name}")
        if not isinstance(args, dict):
            raise RpcError(-32602, "arguments must be an object")
        try:
            result = _groq_chat(args)
        except RpcError as exc:
            if exc.code == -32602:
                raise
            result = {"content": [{"type": "text", "text": exc.message}], "isError": True}
    else:
        raise RpcError(-32601, f"Method not found: {method}")

    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def main() -> None:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
            if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                raise RpcError(-32600, "Invalid Request")
            response = _handle(message)
        except json.JSONDecodeError:
            response = {"jsonrpc": "2.0", "id": None,
                        "error": {"code": -32700, "message": "Parse error"}}
        except RpcError as exc:
            response = {"jsonrpc": "2.0", "id": message.get("id") if isinstance(message, dict) else None,
                        "error": {"code": exc.code, "message": exc.message}}
        except Exception as exc:  # keep protocol stdout valid, details stay on stderr
            log.exception("Unhandled MCP server error")
            response = {"jsonrpc": "2.0", "id": message.get("id") if isinstance(message, dict) else None,
                        "error": {"code": -32603, "message": type(exc).__name__}}
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
