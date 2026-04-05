<div align="center">

# gosh.memory

**Semantic memory for AI agents**


[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://python.org)
[![MCP](https://img.shields.io/badge/protocol-MCP-purple.svg)](https://modelcontextprotocol.io)

</div>

---

<table>
<tr><td>Self-hosted</td><td>Runs on your machine, your data stays yours</td></tr>
<tr><td>Model agnostic</td><td>Works with any LLM provider — picks the best model for each query</td></tr>
<tr><td>MCP native</td><td>Connects to Claude Code, Gemini, Codex, OpenClaw, or any MCP client</td></tr>
<tr><td>Autoresearch</td><td>Memory Adaptation Loop — self-improvement on your data</td></tr>
</table>

> [!NOTE]
> **Developer Preview** — source-only distribution. Requires Python 3.10+, Rust 1.86+ (for gosh.cli/gosh.agent). Works with OpenAI, Anthropic or Google subscriptions, any Claw or APIs.

> [!WARNING]
> **Work in progress.** APIs, telemetry, and config formats change daily. Pull often. Watch releases and star the repo.

> [!CAUTION]
> **Data migration.** Storage format is not stable yet. Data you load into memory may require migration on future versions. Keep your source files.

---

## Models

gosh.memory uses LLMs in several roles. You choose the models — all are configurable, any provider works.

| Role | What it does | Minimum | Recommended |
|------|-------------|---------|-------------|
| **Librarian** | Extracts atomic facts from raw content | Qwen 30B | Claude Sonnet, Qwen 30B |
| **Inference** | Answers questions from retrieved context | Qwen 30B | Claude Sonnet / Opus |
| **Embeddings** | Vector representations for retrieval | text-embedding-3-large | text-embedding-3-large |

We tested across different models. Qwen 30B performs well as Librarian. Mercury 2 underperforms on extraction. In general — better model, better results. Start with Qwen 30B (free via Groq) and upgrade as needed.

---

## Why external memory

Real-world data is unbounded. Any deployed model is not: it has finite active state, finite precision, finite bandwidth, and finite compute per inference. So it cannot represent arbitrarily growing history without collisions. Once the number of possible histories exceeds the number of internal states, different pasts must collapse to the same representation. Lossless global memory is therefore impossible inside a bounded model.

Long context does not remove this limit. It only moves it. In standard transformers, attention scales quadratically with sequence length and KV cache grows linearly, so larger windows become expensive before they become sufficient.

External persistent memory is therefore not an optimization layer. It is the only architecture for unbounded real-world data.

The same logic explains why semantic recognition should be hybrid. Neural networks are excellent local encoders: they can capture the meaning of a passage, a session, or an episode. But they cannot hold the full corpus and its global invariants in one bounded forward pass. Contradictions, supersession, temporal order, provenance, and episode structure must live in explicit persistent state. Neural for local meaning, explicit memory for global structure: that is not a compromise. It is the only architecture consistent with finite computation.

This is also how an autonomous computer must be built. Not around a model, but around memory. Models are interchangeable: they vary in context length, latency, cost, modality, and capability. Memory does not. There is one underlying history, one persistent data plane, one evolving ground truth that every agent works against. Models are workers. Memory is the substrate. Coherence comes not from one model seeing everything, but from all models converging on the same persistent semantic memory.

---

## Memory Adaptation Loop (MAL)

MAL is the autoresearch system that continuously improves the memory pipeline using production feedback. When a query fails — wrong answer, missing fact, bad retrieval — MAL diagnoses which stage broke, proposes a fix, evaluates it, and applies or rejects.

When MAL identifies a problem that requires new code, it emits a task to `agent_id="coding"` via courier. If a coding agent exists in the swarm, it picks up the task and writes the fix.

MAL is optional and disabled by default. See [MAL documentation](https://github.com/gosh-dot-ai/gosh.docs/blob/main/MEMORY-ADAPTATION-LOOP.md).

---

## Multibench

We run all benchmarks on the same production pipeline, in parallel, through one memory server instance. No special benchmark-only code paths. Cross-contamination between parallel workloads: **< 1.5%**.

[Benchmark results and methodology](https://github.com/gosh-dot-ai/gosh.docs/blob/main/BENCHMARKS.md)

---

## Content types (extending)

Memory should be universal — it should support all types of human content.

| Type | Status | Examples |
|------|:------:|---------|
| Conversations | supported | Chat sessions, Q&A, meetings, interviews |
| Documents | supported | Specs, reports, articles, READMEs, PDFs |
| Code | planned | Source files, diffs, PRs, commit history |
| Agent traces | planned | Tool calls, DOM snapshots, action logs |
| Structured data | planned | Tables, CSVs, JSON records, SQL results |
| Media | planned | Audio files, video files, images |
| Email / messaging | planned | Threads, channels, DMs |
| Knowledge bases | planned | Wikis, FAQs, internal docs |

---

## Quick start

```bash
pip install -e .
gosh-memory start --data-dir ./data
```

Server runs at `http://127.0.0.1:8765`. Token auto-generated at `~/.gosh-memory/token`.

---

## Setup

### gosh.cli

Rust orchestrator for gosh.ai — manages memory and agent services, secrets, and provides CLI commands for all operations. [Full setup guide](https://github.com/gosh-dot-ai/gosh.docs/blob/main/SETUP.md#mode-1-harness--full-stack-via-goshcli)

### Claude Code / Codex CLI / Gemini CLI

Connect gosh.memory as an MCP tool server to your AI assistant. Each has a short config — no code changes needed.

- [Claude Code setup](docs/docs-claude-code.md)
- [Gemini CLI setup](docs/docs-gemini-cli.md)
- [OpenAI Codex CLI setup](docs/docs-openai-codex.md)
- [OpenClaw setup](docs/docs-openclaw.md)

### Standalone MCP server

Run gosh.memory directly without gosh.cli. For custom integrations, testing, or embedding into your own stack. [Standalone setup](https://github.com/gosh-dot-ai/gosh.docs/blob/main/SETUP.md#mode-2-standalone--mcp-server-directly)

---

## MCP

gosh.memory exposes all functionality through [Model Context Protocol](https://modelcontextprotocol.io) over HTTP. Any MCP-compatible client can connect — store data, query memory, manage configuration.

```json
{"mcpServers": {"gosh-memory": {"url": "http://localhost:8765/mcp", "headers": {"x-server-token": "<token>"}}}}
```

The server exposes 20+ tools: `memory_store`, `memory_recall`, `memory_ask`, `memory_list`, `memory_import`, `memory_build_index`, `memory_stats`, and more. See [Memory System](https://github.com/gosh-dot-ai/gosh.docs/blob/dev/MEMORY-SYSTEM.md) for how they work.

---

## Documentation

| Document | What it covers |
|----------|---------------|
| [Setup](https://github.com/gosh-dot-ai/gosh.docs/blob/main/SETUP.md) | Installation, configuration, all deployment modes |
| [Memory System](https://github.com/gosh-dot-ai/gosh.docs/blob/main/MEMORY-SYSTEM.md) | How extraction, retrieval, complexity routing, and inference work |
| [Benchmarks](https://github.com/gosh-dot-ai/gosh.docs/blob/main/BENCHMARKS.md) | Benchmark suite, scores, methodology |
| [MAL](https://github.com/gosh-dot-ai/gosh.docs/blob/main/MEMORY-ADAPTATION-LOOP.md) | Autoresearch adaptation loop |
| [Telemetry](https://github.com/gosh-dot-ai/gosh.docs/blob/main/TELEMETRY-CONTRACT.md) | All telemetry fields for recall, ask, stats, agent |
| [Architecture](https://github.com/gosh-dot-ai/gosh.docs/blob/main/ARCHITECTURE.md) | System components and data flow |

---

## License

MIT. Copyright 2026 (c) Mitja Goroshevsky and GOSH Technology Ltd.
