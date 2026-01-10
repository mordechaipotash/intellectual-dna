# Intellectual DNA

A personal knowledge system that turns 3 years of AI conversations (353K messages) into queryable intelligence.

Built by a monotropic polymath who needed a system that works with deep focus, not against it.

## What This Does

```
You: "What do I actually think about agency?"

Brain: Searching 106K embedded messages...

Your position evolved:
- 2023: "AI should do what I say"
- 2024: "AI should preserve my decision sovereignty"
- 2025: "100% human control, 100% machine execution"

Related SEED principle (AGENCY PRESERVATION):
"Maintain human decision-making control while automating everything else"
```

Not a note-taking app. A queryable memory that finds patterns I'd never think to look for.

## The Numbers

| What | Count |
|------|-------|
| Conversation messages | 353,216 |
| Embedded vectors | 106,000 |
| YouTube videos tracked | 32,000 |
| GitHub commits indexed | 1,427 |
| Google searches captured | 52,791 |
| Query time (semantic) | 256ms |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MCP BRAIN SERVER                            │
│           30+ tools exposed to Claude Code/Desktop              │
│  semantic_search · thinking_trajectory · find_contradictions    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LANCEDB VECTORS                            │
│  106K embeddings @ 768 dims · 440MB (was 14GB in DuckDB)        │
│  nomic-embed-text-v1.5 · Apple Silicon MPS · 256ms queries      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FACTS (Immutable)                            │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐       │
│  │ Brain L0-2│ │ YouTube   │ │ GitHub    │ │ Google    │       │
│  │ 353K msgs │ │ 31K vids  │ │ 1.4K      │ │ 52K       │       │
│  │ Parquet   │ │ Parquet   │ │ commits   │ │ searches  │       │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                INTERPRETATIONS (Derived)                        │
│  Versioned, rebuildable, never corrupts source                  │
│  focus/v1 · mvp_velocity/v2 · mood_patterns · weekly_summaries  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AUTO-SYNC LAYER                              │
│  Claude Code Stop Hook → sync.py → embed → briefing             │
│  Every conversation automatically flows into the brain          │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

**Facts vs Interpretations**: Raw data never gets touched. Derived analysis lives in versioned layers. Wrong interpretation? Delete and rebuild. Facts stay clean.

**LanceDB over DuckDB VSS**: Started with DuckDB, discovered duplicate HNSW indexes created 46x overhead (14GB for 300MB of data). Migrated to LanceDB: 440MB, same vectors, native incremental.

**Onion Skin Layers**: L0 (event pointers) → L1 (previews + embeddings) → L2 (full content). Query what you need, not everything.

**Auto-sync via hooks**: Claude Code Stop hook triggers `sync.py`. New conversations flow in automatically. No manual export.

## MCP Tools

The brain exposes 30+ tools via Model Context Protocol:

```python
# Find conceptually similar messages
semantic_search("bottleneck as amplifier", limit=10)

# Track how an idea evolved
thinking_trajectory("agency")

# Time-travel to any month
what_was_i_thinking("2024-08")

# Compare recent vs historical positions
find_contradictions("productivity")

# Cross-source search (conversations + YouTube + GitHub)
unified_search("database optimization")

# 151 weeks of narrative summaries
query_weekly_summaries(month="2024-06")
```

## Daily Briefing Agent

Runs at 6am. Surfaces what I'm circling, contradicting, stalling, forgetting:

```markdown
## Circling Themes
- **brain** (573x) - you keep coming back to this
- **design** (425x) - active focus area

## Contradictions
- **management**: Shifted from "strategic oversight" → "granular data management"

## Stalled Projects
- sparkii (mentioned but no commits in 30 days)

## Summary
[LLM-generated action items via Gemini Flash]
```

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Vector DB | LanceDB | 32x smaller than DuckDB VSS, native incremental |
| Embeddings | nomic-embed-text-v1.5 | 768 dims, local on Apple Silicon |
| Analytics | DuckDB | Fast parquet queries, serverless |
| Storage | Parquet | Columnar, compressed, portable |
| Interface | MCP | Direct integration with Claude Code/Desktop |
| Automation | launchd + hooks | Native macOS, zero external deps |
| LLM (briefings) | Gemini 3 Flash | Fast, cheap synthesis |

## What I Learned

1. **DuckDB VSS has footguns**: Accidentally created duplicate HNSW indexes. 14GB for 300MB of data. LanceDB just works.

2. **Facts vs interpretations prevents rebuild nightmares**: Mixing raw data with derived analysis creates cascading corruption. Keep them separate.

3. **Claude Code hooks are underused**: Auto-sync on session end = zero manual export forever.

4. **Embeddings beat keywords**: "What was I thinking about agency?" finds relevant messages even when I never used that exact word.

5. **Passive search isn't enough**: The brain needs to be proactive. Daily briefings surface what you'd never think to query.

## The 8 SEED Principles

Foundational mental models extracted from 353K messages:

| Principle | Core Idea |
|-----------|-----------|
| **INVERSION** | Reverse the problem - ask what prevents NOT-X |
| **COMPRESSION** | Reduce to essential while preserving decision quality |
| **AGENCY** | 100% human control, 100% machine execution |
| **BOTTLENECK** | Find the constraint, amplify it as leverage |
| **TRANSLATION** | Interface between infinite AI output and finite human comprehension |
| **TEMPORAL** | Human time is the ultimate scarce resource |
| **SEEDS** | Autonomous bounded systems with clear interfaces |
| **COGNITIVE** | Design systems that amplify your brain, not fight it |

## Repository Structure

```
intellectual_dna/
├── data/
│   ├── all_conversations.parquet    # 353K messages
│   ├── youtube_rows.parquet         # 31K videos
│   ├── github_commits.parquet       # 1.4K commits
│   └── facts/brain/                 # L0-L2 onion layers
├── vectors/
│   └── brain.lance/                 # 106K embeddings (440MB)
├── pipelines/
│   ├── embed_messages.py            # Embedding pipeline
│   ├── rebuild.py                   # Unified orchestrator
│   └── migrate_to_lancedb.py        # DuckDB → LanceDB migration
├── live/
│   ├── sync.py                      # Auto-sync from Claude Code
│   └── daily_briefing.py            # 6am briefing agent
└── mordelab/02-monotropic-prosthetic/
    └── mcp_brain_server.py          # MCP server (30+ tools)
```

## The Paradox

**Massive capability, invisible to the world.**

| Dimension | Reality |
|-----------|---------|
| Technical output | Team-level, solo |
| AI fluency | 353K messages of practice |
| Architecture | This system exists |
| Visibility | You're reading this |

The work isn't more building. It's becoming visible.

This README is part of that.

---

## Work With Me

Open to async contract work:

- Context engineering & CLAUDE.md architecture
- MCP server development
- AI orchestration systems

Reach out: [GitHub](https://github.com/mordechaipotash) · [Reddit](https://reddit.com/u/Signal_Usual8630)

---

## Related Projects

- [youtube-transcription-pipeline](https://github.com/mordechaipotash/youtube-transcription-pipeline) - 32K+ videos, 41.8M words transcribed
- [python-data-engineering-portfolio](https://github.com/mordechaipotash/python-data-engineering-portfolio) - 1,059 production scripts
- [seedgarden](https://github.com/mordechaipotash/seedgarden) - The SHELET Protocol for AI-human interfaces

---

*Built by [Mordechai Potash](https://github.com/mordechaipotash)*
