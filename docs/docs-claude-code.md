# Connecting gosh.memory to Claude Code

gosh.memory uses HTTP transport. Three ways to connect.

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

## Option A — CLI (fastest)

```bash
claude mcp add gosh-memory --transport http http://localhost:8765/mcp
```

Verify:
```bash
claude mcp list
# gosh-memory: http://localhost:8765/mcp
```

---

## Option B — User scope (all projects)

Edit `~/.claude.json`:

> **Important:** MCP servers must be in `~/.claude.json`, not in `~/.claude/settings.json` — that location is silently ignored for MCP.

```json
{
  "mcpServers": {
    "gosh-memory": {
      "type": "http",
      "url": "http://localhost:8765/mcp",
      "headers": {"x-server-token": "<token from ~/.gosh-memory/token>"}
    }
  }
}
```

---

## Option C — Project scope (one project, version-controlled)

Create `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "gosh-memory": {
      "type": "http",
      "url": "http://localhost:8765/mcp",
      "headers": {"x-server-token": "<token from ~/.gosh-memory/token>"}
    }
  }
}
```

This file can be committed to git and shared with the team.

---

## Verify connection

Restart Claude Code, then run `/mcp` inside the session:

```
/mcp
⎿ MCP Server Status
⎿ • gosh-memory: connected
```

---

## How Claude Code uses memory

### Initial setup (once per project)

```bash
# 1. Import your Claude conversation history (CLI)
gosh-memory import \
  --format conversation_json \
  --file ~/Downloads/conversations.json \
  --key my_project --scope agent-private

# 2. Import your codebase (CLI)
gosh-memory import \
  --format git \
  --source-uri https://github.com/you/repo \
  --token ghp_xxx \
  --content-type technical --key my_project
```

Or do everything via MCP tools inside Claude Code:

```
# Import a git repo (same as CLI --format git)
memory_import(key="my_project",
    source_format="git",
    source_uri="https://github.com/you/repo",
    content_type="technical",
    scope="swarm-shared")

# Import a local folder (same as CLI --format directory)
memory_import(key="my_project",
    source_format="directory",
    path="/home/user/my-project",
    content_type="technical",
    scope="swarm-shared")

# Import Claude conversation export
memory_import(key="my_project",
    source_format="conversation_json",
    content=<file contents>,
    scope="agent-private")

# Build the retrieval index (Tier 2/3 + embeddings)
memory_build_index(key="my_project")
```

### Every-turn pattern

```
# BEFORE answering — pull relevant context
result = memory_recall(key="my_project",
    query="authentication architecture decisions")
# result["context"] is a ready-to-use text block for the LLM

# AFTER the turn — store what happened
memory_store(key="my_project",
    content="<full session text>",
    session_num=42,
    session_date="2026-03-17",
    agent_id="claude-code",
    scope="agent-private")
```

`memory_recall` auto-builds the index if needed, so you don't have to call
`memory_build_index` after every store. But calling it explicitly after a
batch of imports is faster — it embeds once instead of per-recall.

### query_type — retrieval strategy

`memory_recall` accepts `query_type` to select the retrieval strategy:

| Type | When to use | Example |
|------|-------------|---------|
| `auto` | Default — auto-detects from question | Any question |
| `lookup` | Simple fact retrieval | "What's Alice's email?" |
| `temporal` | Time-ordered events | "What happened after the deploy?" |
| `aggregate` | Counting / arithmetic | "How many bugs were filed?" |
| `current` | Latest state (newest wins) | "What's the current auth approach?" |
| `synthesize` | Cross-session patterns | "Summarize the project evolution" |
| `procedural` | Step-by-step instructions | "How do I set up the dev env?" |
| `prospective` | Forward-looking | "What will break if we remove X?" |

### Scope isolation

| Scope | Visible to |
|-------|------------|
| `agent-private` | Only the agent that stored it |
| `swarm-shared` | All agents in the same swarm |
| `system-wide` | All agents on the server |

### Agent self-extraction (zero extra LLM calls)

By default `memory_store` calls a separate LLM (the Librarian) to extract
facts. Claude Code can do extraction itself — saving the cost of a second
LLM call. The server still handles format detection, chunking, prompt
selection, post-processing, tier 2/3, embedding, and retrieval.

```
# 1. Server prepares prompts (format detection + chunking + prompt selection)
r = memory_prepare_extraction(key="my_project",
    content="<session text>",
    session_num=1, session_date="2026-03-17",
    speakers="User and Assistant")

# r["prompts"] = [{"system": "You are extracting...", "user": "Conversation..."}]
# r["llm_needed"] = true (false for FACT_LIST/RAW_CHUNKS — already extracted)
# r["extraction_id"] = "abc-123"

# 2. Run each prompt through your own LLM
#    Output must be JSON: {"facts": [...], "temporal_links": [...]}
#    See README for the full JSON schema.

# 3. Send results back — server does everything else
memory_store_extracted(key="my_project",
    extraction_id="abc-123",
    results='[{"facts": [...], "temporal_links": [...]}]',
    scope="agent-private")
```

If `r["llm_needed"]` is `false`, facts are in `r["pre_extracted"]` — call
`memory_store_extracted` with `results="[]"`.

See [README — Agent self-extraction](../README.md#agent-self-extraction-zero-extra-llm-calls)
for the full JSON schema and detailed flow.

### Other useful tools

```
# List facts with filters
memory_list(key="my_project", kind="action_item")

# Check what's in memory
memory_stats(key="my_project")

# Store a credential (never appears in recall)
memory_store_secret(key="my_project", name="db_password", value="xxx")

# Fetch it back
memory_get_secret(key="my_project", name="db_password")

# Force tier rebuild without full index
memory_flush(key="my_project")
# → {"rebuilt": true, "total_consolidated": 89, "total_cross_session": 34}
```

See [README — How agents use memory](../README.md#how-agents-use-memory) for
the complete tool reference with all parameters.
