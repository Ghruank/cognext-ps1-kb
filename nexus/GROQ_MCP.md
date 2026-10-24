# Groq MCP tool

`groq_mcp_server.py` exposes a stdio MCP tool named `groq_chat`. It uses the
`GROQ_API_KEY` environment variable or loads it from `kb/.env`. The key is never a
tool argument and is not returned to the client.

Register it with Codex from a network-enabled terminal in this checkout:

```bash
codex mcp add groq -- \
  /home/ghruank/hackathons/cognext/kb/.venv/bin/python \
  /home/ghruank/hackathons/cognext/kb/nexus/groq_mcp_server.py
```

Restart the Codex session, then check `codex mcp list`. The server process must be
able to read `kb/.env` and reach `api.groq.com`. If running from another checkout,
replace the two absolute paths with that checkout's paths.

Tool arguments:

- `prompt` (required): user prompt text
- `system` (optional): system instruction
- `model` (optional): defaults to `GROQ_MODEL` or `llama-3.3-70b-versatile`
- `max_tokens` (optional): 1–8192, default 2048
- `temperature` (optional): 0–2, default 0.2
- `json_mode` (optional): request a JSON object

The current Codex configuration is outside this workspace's writable area, so this
checkout contains the server and registration command but cannot register it in the
active Codex process automatically.
