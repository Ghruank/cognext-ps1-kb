# Langfuse MCP setup

Langfuse provides an official hosted MCP endpoint. The helper reads the existing
project-scoped `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL`
from `kb/.env`, then adds the authenticated endpoint to `~/.codex/config.toml` without
printing credentials.

Run this from the project terminal:

```bash
cd /home/ghruank/hackathons/cognext/kb
.venv/bin/python -m nexus.setup_langfuse_mcp
```

Restart Codex after the helper completes. The hosted server provides Langfuse data tools;
it does not change network access for local NEXUS processes, so local Langfuse trace
export still depends on the app environment reaching the configured Langfuse host.

Official setup reference: <https://langfuse.com/docs/api-and-data-platform/features/mcp-server>
