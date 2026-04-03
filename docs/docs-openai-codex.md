# Connecting gosh.memory to OpenAI Codex CLI

Codex CLI supports Streamable HTTP MCP servers via `config.toml`.

## Prerequisites

Server must be running:
```bash
gosh-memory start --data-dir ./data
# Listening on http://127.0.0.1:8765
```

The server is protected by token authentication. On first start, a token is
auto-generated and saved to `~/.gosh-memory/token` (chmod 600). All requests
(except `GET /health`) must include an `x-server-token` header.

---

## Configuration

Edit `~/.codex/config.toml` (user scope) or `.codex/config.toml` in your project (project scope):

```toml
[mcp_servers.gosh-memory]
url = "http://localhost:8765/mcp"
headers = { x-server-token = "<token from ~/.gosh-memory/token>" }
```

Or via CLI:

```bash
codex mcp add gosh-memory --url http://localhost:8765/mcp
```

Full options:

```toml
[mcp_servers.gosh-memory]
url                  = "http://localhost:8765/mcp"
headers              = { x-server-token = "<token from ~/.gosh-memory/token>" }
startup_timeout_sec  = 10    # default
tool_timeout_sec     = 60    # default
enabled              = true
```

---

## Verify

```bash
codex mcp list
# gosh-memory  http://localhost:8765/mcp  enabled
```

---

## Usage pattern

```
# Recall before answering
memory_recall(key="project", query="what was decided about the pipeline")

# Store after completing a task
memory_store(key="project", content="<session>",
             agent_id="codex", swarm_id="proj",
             session_num=1, session_date="2026-03-16",
             content_type="technical")
```

For the complete agent flow (loading data, building index, query types, scope
isolation), see [README — How agents use memory](../README.md#how-agents-use-memory).

---

## Using via OpenAI Responses API directly

If you're calling the API directly (not CLI), pass gosh.memory as a remote tool:

```python
from openai import OpenAI
client = OpenAI()

resp = client.responses.create(
    model="gpt-5",
    tools=[{
        "type": "mcp",
        "server_label": "gosh-memory",
        "server_url": "http://localhost:8765/mcp",
        "require_approval": "never",
    }],
    input="What do we know about the authentication architecture?",
)
print(resp.output_text)
```

> Note: the Responses API requires a publicly reachable URL. For local development, use `localhost` with Codex CLI instead.
