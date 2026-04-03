# gosh.memory — Documentation

## Getting started

- [Quickstart](../README.md#quick-start) — install, start server, first import
- [How agents use memory](../README.md#how-agents-use-memory) — loading data, building index, retrieving, complete flow
- [CLI reference](../README.md#cli-reference) — all commands and flags
- [MCP Tools reference](../README.md#mcp-tools-reference) — full table of 17 tools
- [Storage](../README.md#storage) — data format and layout

## Authentication

The server is protected by token authentication. On first start, a token is
auto-generated and saved to `~/.gosh-memory/token` (chmod 600). All requests
(except `GET /health`) must include an `x-server-token` header with this token.

You can override the token via the `GOSH_MEMORY_TOKEN` environment variable.

## Integrations

| Client | Guide | Config file | Transport |
|--------|-------|------------|-----------|
| [Claude Code](docs-claude-code.md) | Setup + usage | `~/.claude.json` or `.mcp.json` | HTTP |
| [OpenClaw](docs-openclaw.md) | Setup + usage | `~/.openclaw/mcp_config.json` | HTTP |
| [OpenAI Codex CLI](docs-openai-codex.md) | Setup + usage | `~/.codex/config.toml` | HTTP |
| [Gemini CLI](docs-gemini-cli.md) | Setup + usage | `~/.gemini/settings.json` | HTTP |

All integrations use the same server URL: `http://localhost:8765/mcp`

All integrations require the `x-server-token` header. See individual guides for config examples.
