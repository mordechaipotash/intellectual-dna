<p align="center">
  <h1 align="center">🧬 Intellectual DNA</h1>
  <p align="center">
    <strong>Turn 3 years of AI conversations into a queryable second brain</strong>
  </p>
  <p align="center">
    376K messages · 118K embeddings · 31 MCP tools · 256ms queries
  </p>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/MCP-compatible-green?style=flat-square" alt="MCP"></a>
  <a href="#"><img src="https://img.shields.io/badge/vectors-LanceDB-orange?style=flat-square" alt="LanceDB"></a>
  <a href="#"><img src="https://img.shields.io/badge/embeddings-nomic--v1.5-purple?style=flat-square" alt="Embeddings"></a>
  <a href="https://github.com/mordechaipotash/intellectual-dna/stargazers"><img src="https://img.shields.io/github/stars/mordechaipotash/intellectual-dna?style=flat-square" alt="Stars"></a>
</p>

---

```
You: "What do I actually think about agency?"

Brain: Searching 118K embedded messages...

Your position evolved:
  2023: "AI should do what I say"
  2024: "AI should preserve my decision sovereignty"  
  2025: "100% human control, 100% machine execution"

Related SEED principle (AGENCY PRESERVATION):
"Maintain human decision-making control while automating everything else"
```

## What is this?

Every conversation you have with an AI is a thought you externalized. Over 3 years, that's **376,000 thoughts** — but they're scattered across ChatGPT exports, Claude sessions, Gemini chats, and code editor transcripts.

Intellectual DNA turns that scattered history into a **queryable knowledge system**. Not a note-taking app — a second brain that can:

- **Find patterns** you'd never think to search for
- **Track how your thinking evolved** on any topic
- **Surface contradictions** between what you say and what you do
- **Cross-reference** conversations with your GitHub commits, markdown docs, and more

It runs as an **MCP server** — plug it into Claude Code, Claude Desktop, or any MCP-compatible client, and your entire intellectual history becomes context.

## The Numbers

| Metric | Value |
|--------|-------|
| Conversation messages | **376,164** |
| Embedded vectors | **118,533** (768d, nomic-v1.5) |
| GitHub commits indexed | **2,217** across 146 repos |
| Markdown docs harvested | **5,524** |
| MCP tools exposed | **31** |
| Semantic query time | **~256ms** |
| Vector DB size | **493MB** (was 14GB before LanceDB migration) |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MCP BRAIN SERVER                            │
│              31 tools · Claude Code / Desktop                   │
│  semantic_search · thinking_trajectory · alignment_check · ...  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────┐  ┌──────────────────────────────────────┐
│    LANCEDB VECTORS   │  │         DUCKDB + PARQUET             │
│  118K embeddings     │  │  376K messages · keyword search      │
│  768d nomic-v1.5     │  │  columnar · compressed · portable    │
│  493MB on disk       │  │  serverless SQL analytics            │
└──────────────────────┘  └──────────────────────────────────────┘
              │                         │
              └────────────┬────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  DATA SOURCES (Immutable)                       │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐      │
│  │ Claude    │ │ ChatGPT   │ │ Gemini    │ │ Clawdbot  │      │
│  │ Code/     │ │ export    │ │ sessions  │ │ sessions  │      │
│  │ Desktop   │ │           │ │           │ │           │      │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘      │
│  ┌───────────┐ ┌───────────┐ ┌───────────────────────┐        │
│  │ GitHub    │ │ Markdown  │ │ Interpretation layers  │        │
│  │ 2.2K      │ │ 5.5K docs │ │ focus · mood · themes  │        │
│  │ commits   │ │           │ │ spend · velocity · ...  │        │
│  └───────────┘ └───────────┘ └───────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AUTO-SYNC PIPELINE                           │
│  Claude Code hook → sync.py → parquet → embed → ready          │
│  Hourly: clawdbot sessions · Nightly: all sources + vectors    │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### Facts vs Interpretations

Raw data is **immutable**. Derived analysis lives in versioned layers. Wrong interpretation? Delete the version and rebuild. Source data stays clean forever.

```
data/
├── facts/          # NEVER modified — append only
│   ├── brain/      # L0 index → L1 summary → L2 content → L3 raw
│   ├── spend/      # raw → daily → monthly aggregation
│   └── sources/    # original parquets (symlinks)
└── interpretations/ # DERIVED — versioned, rebuildable
    ├── focus/v1/
    ├── mood_patterns/
    └── weekly_summaries/
```

### Onion Skin Layers (L0–L3)

Progressive disclosure — query only what you need:

| Layer | Contents | Use Case |
|-------|----------|----------|
| **L0** index | Event IDs, timestamps, source | Quick lookups, counts |
| **L1** summary | 500-char preview + embedding | Semantic search, browsing |
| **L2** content | Full text, has_code, has_url | Deep reading |
| **L3** deep | Symlinks to original parquets | Source verification |

### LanceDB over DuckDB VSS

Started with DuckDB for vector search. Discovered duplicate HNSW indexes created **46x storage overhead** (14GB for 300MB of data). Migrated to LanceDB: **493MB**, same vectors, native incremental indexing. Same 256ms query time.

## MCP Tools (31)

### 🔍 Search (7)

| Tool | Description |
|------|-------------|
| `semantic_search` | Vector similarity via LanceDB (768d nomic embeddings) |
| `search_conversations` | Keyword search via DuckDB SQL on parquet |
| `unified_search` | Cross-source: conversations + GitHub + markdown |
| `search_ip_docs` | Vector search on curated IP documents |
| `search_markdown` | Keyword search on 5.5K harvested markdown docs |
| `code_to_conversation` | Semantic search across commits + conversations |
| `find_user_questions` | Recent questions asked |

### 🧠 Synthesis (4)

| Tool | Description |
|------|-------------|
| `what_do_i_think` | Synthesize your views on any topic from all evidence |
| `find_precedent` | Find similar situations from the past |
| `alignment_check` | Check if a decision aligns with your principles |
| `thinking_trajectory` | Track how an idea evolved over months/years |

### 💬 Conversation (5)

| Tool | Description |
|------|-------------|
| `get_conversation` | Full conversation by ID |
| `conversations_by_date` | What happened on a specific date |
| `what_was_i_thinking` | Month snapshot: themes, activity, concepts |
| `concept_velocity` | How often a term appears over time |
| `first_mention` | When a concept first appeared in your history |

### 🐙 GitHub (4)

| Tool | Description |
|------|-------------|
| `github_project_timeline` | Repo creation, commits, activity windows |
| `conversation_project_context` | Conversations mentioning a project |
| `validate_date_with_github` | Verify conversation dates via commit timestamps |
| `code_to_conversation` | Bridge code changes to discussion context |

### 📄 Markdown Corpus (4)

| Tool | Description |
|------|-------------|
| `get_breakthrough_docs` | Documents tagged with high breakthrough energy |
| `get_deep_docs` | High depth-score documents |
| `get_project_docs` | All docs for a specific project |
| `get_open_todos` | Documents with open TODO items |

### 📊 Analysis (5)

| Tool | Description |
|------|-------------|
| `query_tool_stacks` | Technology stack patterns |
| `query_problem_resolution` | Debugging and problem-solving patterns |
| `query_spend` | Cost breakdown by source and time period |
| `query_timeline` | Cross-source timeline for any date |
| `query_conversation_summary` | Comprehensive conversation analysis |

### ⚙️ Meta (2)

| Tool | Description |
|------|-------------|
| `brain_stats` | Overview of all data sources and counts |
| `list_principles` / `get_principle` | Your foundational SEED principles |

## Data Flow

```
  Clawdbot Sessions          Claude Code          ChatGPT Export       Gemini
  ~/.clawdbot/agents/     ~/.claude/projects/     conversations.json   sessions
         │                       │                       │                │
         ▼                       ▼                       ▼                ▼
     sync_clawdbot.py        live/sync.py          import pipeline    import pipeline
         │                       │                       │                │
         └───────────────────────┴───────────────┬───────┘────────────────┘
                                                 ▼
                            data/all_conversations.parquet (376K messages)
                                                 │
                                    ┌────────────┴────────────┐
                                    ▼                         ▼
                           embed_new_messages.py      build_*.py (88 pipelines)
                                    │                         │
                                    ▼                         ▼
                           vectors/brain.lance/       data/interpretations/
                           (118K vectors, 493MB)      (focus, mood, themes, ...)
                                    │                         │
                                    └────────────┬────────────┘
                                                 ▼
                                        mcp_brain_server.py
                                         (31 MCP tools)
                                                 │
                                                 ▼
                                    Claude Code · Claude Desktop
                                      Any MCP-compatible client
```

## Quick Start

### Prerequisites

- Python 3.11+
- Apple Silicon Mac recommended (MPS acceleration for embeddings)
- [mcporter](https://github.com/nicobailey/mcporter) or any MCP client

### 1. Clone & Setup

```bash
git clone https://github.com/mordechaipotash/intellectual-dna.git
cd intellectual-dna

# Create virtual environment
python -m venv mcp-env
source mcp-env/bin/activate

# Install dependencies
pip install duckdb lancedb nomic fastmcp pandas pyarrow
```

### 2. Prepare Your Data

The system expects conversation data in parquet format. Export your conversations:

```bash
# Import ChatGPT export
python -m pipelines import_chatgpt /path/to/conversations.json

# Import Claude Code sessions
python -m pipelines import_claude_code

# Or bring your own parquet with columns: 
# [message_id, conversation_id, role, content, created, source]
```

### 3. Embed & Index

```bash
# Generate embeddings (uses nomic-embed-text-v1.5 locally)
python pipelines/embed_new_messages.py

# Check stats
python pipelines/embed_new_messages.py stats
```

### 4. Run the MCP Server

```bash
# Direct
python mordelab/02-monotropic-prosthetic/mcp_brain_server.py

# Or via mcporter config (~/.mcporter/mcporter.json):
{
  "brain": {
    "command": "python",
    "args": ["mordelab/02-monotropic-prosthetic/mcp_brain_server.py"],
    "lifecycle": "keep-alive"
  }
}
```

### 5. Query Your Brain

```python
# Semantic search
semantic_search("what do I think about productivity?", limit=10)

# Track idea evolution
thinking_trajectory("agency")

# Time-travel to any month
what_was_i_thinking("2024-08")

# Cross-source search
unified_search("database optimization")
```

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Vector DB | **LanceDB** | 32x smaller than DuckDB VSS, native incremental, no index footguns |
| Embeddings | **nomic-embed-text-v1.5** | 768d, runs locally on Apple Silicon via MPS |
| Analytics | **DuckDB** | Fast SQL on parquet, serverless, zero config |
| Storage | **Parquet** | Columnar, compressed, portable, ecosystem support |
| Interface | **MCP (FastMCP)** | Direct integration with Claude Code/Desktop |
| Automation | **launchd + hooks** | Native macOS scheduling, zero external deps |
| Pipelines | **88 Python scripts** | Each pipeline is standalone, composable |

## The SEED Principles

Eight foundational mental models extracted from 376K messages:

| Principle | Core Idea |
|-----------|-----------|
| **INVERSION** | Reverse the problem — ask what prevents NOT-X |
| **COMPRESSION** | Reduce to essential while preserving decision quality |
| **AGENCY** | 100% human control, 100% machine execution |
| **BOTTLENECK** | Find the constraint, amplify it as leverage |
| **TRANSLATION** | Interface between infinite AI output and finite human comprehension |
| **TEMPORAL** | Human time is the ultimate scarce resource |
| **SEEDS** | Autonomous bounded systems with clear interfaces |
| **COGNITIVE** | Design systems that amplify your brain, not fight it |

## Repository Structure

```
intellectual-dna/
├── mordelab/02-monotropic-prosthetic/
│   ├── mcp_brain_server.py          # MCP server (31 tools)
│   └── SEED-MORDETROPIC-128KB-MASTER.json  # 8 principles
├── pipelines/                        # 88 data pipelines
│   ├── embed_new_messages.py         # Parquet → LanceDB vectors
│   ├── sync_clawdbot.py             # Clawdbot sessions → parquet
│   ├── sync_github.py               # GitHub repos + commits
│   ├── harvest_markdown.py          # Markdown corpus builder
│   ├── build_*.py                   # 50+ interpretation builders
│   └── rebuild.py                   # Unified orchestrator
├── live/
│   ├── sync.py                      # Auto-sync from Claude Code
│   └── daily_briefing.py            # Morning briefing agent
├── data/                             # (gitignored)
│   ├── facts/                        # Immutable source data
│   │   └── brain/                    # L0-L3 onion layers
│   └── interpretations/              # Derived, versioned analysis
├── vectors/                          # (gitignored)
│   └── brain.lance/                  # 118K vectors (493MB)
├── config.py                         # Central configuration
└── .claude/CLAUDE.md                 # Context engineering for Claude Code
```

## Lessons Learned

1. **DuckDB VSS has footguns** — Accidentally created duplicate HNSW indexes. 14GB for 300MB of data. LanceDB just works.

2. **Facts vs Interpretations prevents rebuild nightmares** — Mixing raw data with derived analysis creates cascading corruption. Keep them separate.

3. **Auto-sync beats manual export** — Claude Code stop hook triggers `sync.py`. New conversations flow in automatically. Zero friction = actually gets used.

4. **Embeddings beat keywords** — "What was I thinking about agency?" finds relevant messages even when you never used that exact word.

5. **88 pipelines > 1 monolith** — Each pipeline is a standalone script. Easy to run, debug, or replace individually.

## Related Projects

- **[brain-canvas](https://github.com/mordechaipotash/brain-canvas)** — Give any LLM its own display (`npx brain-canvas`)
- **[youtube-transcription-pipeline](https://github.com/mordechaipotash/youtube-transcription-pipeline)** — 31K+ videos, transcribed
- **[seedgarden](https://github.com/mordechaipotash/seedgarden)** — The SHELET Protocol for AI-human interfaces

## Work With Me

Open to async contract work in context engineering, MCP server development, and AI orchestration systems.

[GitHub](https://github.com/mordechaipotash) · [Reddit](https://reddit.com/u/Signal_Usual8630)

---

*Built by [Mordechai Potash](https://github.com/mordechaipotash) — a monotropic polymath who needed a system that works with deep focus, not against it.*
