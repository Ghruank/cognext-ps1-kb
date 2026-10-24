"""Register Langfuse's hosted MCP server in the local Codex user config.

Run from any directory with the project's `kb/.env` present. The API secret is
read from that file, encoded into the required Basic Auth header, and never
printed. The Codex config is written with mode 0600.
"""

from __future__ import annotations

import base64
import os
import re
import stat
from pathlib import Path

from dotenv import dotenv_values


KB_ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path.home() / ".codex" / "config.toml"


def main() -> None:
    env = dotenv_values(KB_ROOT / ".env")
    public = env.get("LANGFUSE_PUBLIC_KEY")
    secret = env.get("LANGFUSE_SECRET_KEY")
    base = (env.get("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com").rstrip("/")
    if not public or not secret:
        raise SystemExit("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set in kb/.env")

    token = base64.b64encode(f"{public}:{secret}".encode()).decode()
    section = ("[mcp_servers.langfuse]\n"
               f'url = "{base}/api/public/mcp"\n'
               f'http_headers = {{ Authorization = "Basic {token}" }}\n')

    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    old = CONFIG.read_text() if CONFIG.exists() else ""
    pattern = re.compile(r"(?ms)^\[mcp_servers\.langfuse\]\s*.*?(?=^\[|\Z)")
    if pattern.search(old):
        updated = pattern.sub(section, old, count=1)
    else:
        updated = old.rstrip() + ("\n\n" if old.strip() else "") + section
    temp = CONFIG.with_suffix(".toml.tmp")
    temp.write_text(updated)
    temp.chmod(0o600)
    temp.replace(CONFIG)
    CONFIG.chmod(stat.S_IRUSR | stat.S_IWUSR)
    print(f"Registered Langfuse MCP at {base}/api/public/mcp in {CONFIG}")
    print("Restart Codex, then confirm the `langfuse` tools are available.")


if __name__ == "__main__":
    main()
