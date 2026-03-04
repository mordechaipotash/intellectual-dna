# Changelog

All notable changes to Brain MCP will be documented in this file.

## [0.1.0] — 2026-03-04

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
