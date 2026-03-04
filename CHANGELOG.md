# Changelog

All notable changes to Brain MCP will be documented in this file.

## [0.1.8] — 2026-03-04

### Added
- `brain-mcp version` command
- `brain-mcp summarize` command (with guided setup if not configured)
- `brain-mcp dashboard` command (placeholder for v0.2.0)
- Claude Desktop auto-discovery in `brain-mcp init`
- Dashboard-first UX: running `brain-mcp` with no args opens dashboard

### Fixed
- Test assertion for server_name after v0.1.7 rename
- Removed orphaned `config.py` and `architecture.html` from repo

### Changed
- Default behavior: `brain-mcp` (no subcommand) → opens dashboard instead of printing help

## [0.1.7] — 2026-03-04

### Changed
- Renamed MCP server from `brain` to `my-brain` to avoid collisions with user configs

## [0.1.6] — 2026-03-03

### Fixed
- `brain-mcp init --full` crash when embedding not installed
- Config key collision with other MCP servers (now uses `brain-mcp` key)

## [0.1.5] — 2026-03-03

### Changed
- Embedding is now fully optional — `pip install brain-mcp[embed]` for semantic search
- Better UX: clear messages when optional features aren't installed

### Fixed
- Missing `pytz` dependency

## [0.1.4] — 2026-03-02

### Fixed
- Claude Code config path: uses `~/.claude.json` (not `~/.claude/mcp.json`)

## [0.1.3] — 2026-03-02

### Fixed
- Claude Code and Desktop setup paths
- Added `pipx install` as recommended install method

## [0.1.2] — 2026-03-01

### Fixed
- Claude Desktop/Code config path detection
- Safer embedding pipeline (handles missing model gracefully)

## [0.1.1] — 2026-03-01

### Fixed
- Missing `einops` dependency for embedding model

## [0.1.0] — 2026-03-01

### Added
- **25 MCP tools** across 7 categories: search, conversations, synthesis, stats, prosthetic, GitHub, analytics
- **4 conversation ingesters**: Claude Code, ChatGPT, Clawdbot, Generic JSONL
- **Local embedding pipeline** using nomic-embed-text-v1.5 (768-dim vectors)
- **LanceDB vector search** for semantic similarity queries
- **DuckDB SQL** over parquet for fast keyword search and analytics
- **Cognitive prosthetic tools**: tunnel_state, context_recovery, switching_cost, dormant_contexts, open_threads, cognitive_patterns, tunnel_history, trust_dashboard
- **Synthesis tools**: what_do_i_think, alignment_check, thinking_trajectory, what_was_i_thinking
- **Optional LLM summarization** (Anthropic, OpenAI, Ollama) for structured conversation analysis
- **Progressive feature tiers**: works with just conversations, improves with embeddings, and again with summaries
- **CLI** with init, ingest, embed, serve, setup, doctor, status, sync commands
- **Auto-discovery** of Claude Code and ChatGPT conversations
- **npx support** for zero-install usage
- **Configurable principles** for alignment_check (YAML format)
- 100% local — no telemetry, no cloud, no phone-home
