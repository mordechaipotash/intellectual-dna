# Intellectual DNA - Monotropic Prosthetic

## LLM Model Policy
**If using Gemini via OpenRouter, ALWAYS use `google/gemini-3-flash-preview`** - never use older versions like gemini-2.0-flash or gemini-2.5-flash.

## Project Overview
AI-powered knowledge system for querying Mordechai's intellectual patterns from:
- **353K conversation messages** (2023-01 to 2026-01) in DuckDB/Parquet
- **106K embedded messages** with vector search (256ms queries, HNSW-indexed)
- **8 SEED principles** (foundational mental models)

**Quality Score: 82/100** (as of 2026-01-09)
- Embedding coverage: 100% of embeddable user messages
- Timestamp coverage: 100% (using `created` fallback for NULL `msg_timestamp`)
- Data integrity: Zero duplicates, zero NULL content

**Note**: The static KG (973 nodes) was archived 2025-12-16 - it captured aspirational thinking but missed 99.6% of raw intellectual output. Embeddings + raw conversations are more representative.

## Architecture

### Facts vs Interpretations (Implemented 2025-12-25)

The data layer separates immutable truth from derived analysis:

| Layer | Nature | Example | Rebuild? |
|-------|--------|---------|----------|
| **Facts** | Immutable, append-only | conversations, spend, commits | Never modified |
| **Interpretations** | Derived, versioned | focus/v1, cognitive/v2 | Rebuild anytime |
| **Seed** | Foundational principles | 8 SEED mental models | Static reference |

### Onion Skin Brain Layers (L0-L3)

Progressive disclosure from pointers → full content:

| Layer | Purpose | Rows | Contents |
|-------|---------|------|----------|
| **L0 index** | Event pointers | 205K | event_id, type, timestamp, source (excludes untimed events) |
| **L1 summary** | Previews + embeddings | 205K | preview (500 chars), word_count, embedding_768 |
| **L2 content** | Full text | 217K | full_content, has_code, has_url (includes all events) |
| **L3 deep** | Raw sources | - | Symlinks to original parquets |

**Layer differences explained**:
- L2 content > L0/L1: Includes 15K unwatched YouTube videos (no timestamp) + 4K code sessions
- L0 index has 9.7K NULL event_ids: These are IDE `<ide_opened_file>` events (expected)

```
~/intellectual_dna/                           # LOCAL (moved from iCloud 2025-12-17)
├── config.py                                # Central configuration (all paths)
├── pipelines/                               # Data pipelines
│   ├── __main__.py                          # CLI: python -m pipelines <cmd>
│   ├── rebuild.py                           # Unified rebuild: all|facts|brain|temporal|focus
│   ├── build_facts_spend.py                 # Spend → raw/daily/monthly parquets
│   ├── build_brain_layers.py                # Brain L0-L2 builder
│   ├── build_temporal_dim.py                # Date dimension table
│   ├── build_focus_v1.py                    # Focus interpretation (keywords)
│   ├── import_claude_code.py                # Claude Code JSONL → parquet
│   ├── embed_messages.py                    # Embedding pipeline
│   └── utils/                               # Shared utilities
├── mordelab/02-monotropic-prosthetic/       # MCP server
│   ├── mcp_brain_server.py                  # MCP server (30+ tools)
│   ├── embeddings.duckdb                    # Vector embeddings (14GB, HNSW-indexed)
│   ├── SEED-MORDETROPIC-128KB-MASTER.json   # 8 principles (active)
│   └── mcp-env/                             # Python venv for MCP
├── data/
│   ├── facts/                               # IMMUTABLE TRUTH
│   │   ├── brain/                           # Onion layers
│   │   │   ├── index.parquet                # L0: 150K event pointers
│   │   │   ├── summary.parquet              # L1: previews + embeddings
│   │   │   ├── content.parquet              # L2: full text
│   │   │   └── deep/                        # L3: symlinks to sources
│   │   ├── spend/                           # Three-tier spend
│   │   │   ├── raw.parquet                  # 2.5K records
│   │   │   ├── daily.parquet                # ~1K daily aggregates
│   │   │   └── monthly.parquet              # 155 monthly summaries
│   │   ├── sources/                         # Original parquets (symlinks)
│   │   └── temporal_dim.parquet             # Date dimension (2.3K days)
│   ├── interpretations/                     # DERIVED, VERSIONED
│   │   └── focus/v1/                        # Daily focus detection
│   │       ├── config.json                  # Algorithm parameters
│   │       ├── daily.parquet                # 957 days with keywords
│   │       └── README.md                    # Version documentation
│   ├── seed/                                # FOUNDATIONAL PRINCIPLES
│   │   └── principles.json                  # → SEED-MORDETROPIC-128KB-MASTER.json
│   └── backups/                             # Parquet backups
└── .claude/                                 # Claude Code config
```

## Core MCP Brain Tools (Post-KG Architecture)

After archiving the static KG, these are the primary tools:

### 0. `unified_search(query, limit=15)` ⭐ CROSS-SOURCE
**Searches ALL sources** - conversations (semantic), YouTube, GitHub, markdown in one query
```python
unified_search('bottleneck', limit=10)
# Returns integrated results across all data sources
```

### 1. `semantic_search(query, limit=10)` ⭐ PRIMARY
**Most powerful** - Vector similarity search across 106K embedded messages (256ms, HNSW-indexed)
```python
semantic_search('bottleneck as amplifier', limit=5)
# Finds conceptually similar messages even without exact keywords
```

### 2. `thinking_trajectory(topic)` ⭐ DEEP ANALYSIS
**Combines semantic + temporal + patterns** - Track idea evolution
```python
thinking_trajectory('agency')
# Shows when idea emerged, how it evolved, related concepts
```

### 3. `search_conversations(term, limit=15, role='user')`
Full-text keyword search across 349K messages (401ms)
```python
search_conversations('think out the box', role='user', limit=10)
# Find your signature phrases and working patterns
```

### 4. `list_principles()` / `get_principle(name)`
The 8 SEED foundational mental models (still active)
```python
list_principles()  # Overview of all 8
get_principle('compression')  # Deep dive into one
```

**Principle names**: inversion, compression, bottleneck, agency, seeds, translation, temporal, cognitive_architecture

### 5. Temporal Tools
| Tool | Purpose |
|------|---------|
| `what_was_i_thinking(month)` | Time-travel to YYYY-MM |
| `concept_velocity(term)` | Track idea acceleration |
| `first_mention(term)` | Find genesis moment |

### 6. V2 Interpretation Tools (NEW)
| Tool | Purpose |
|------|---------|
| `query_focus_v2(month, limit)` | LLM-generated daily summaries (957 days) |
| `query_mvp_velocity(month, limit)` | Development patterns: oneshot, iterate, asap (36 months) |
| `query_tool_stacks(month, limit)` | Technology stack evolution + adoptions/drops |
| `query_problem_resolution(month, limit)` | Debugging patterns, hardest problems, aha moments |
| `query_focus(month, limit)` | V1 keyword-based daily focus |
| `query_spend(month, source)` | Cost breakdown by API source |
| `query_timeline(date)` | What happened on a specific date |

### 7. GitHub Cross-Reference Tools
| Tool | Purpose |
|------|---------|
| `github_project_timeline(project)` | Project creation date, commit history, activity |
| `conversation_project_context(project)` | Find conversations mentioning a project |
| `code_to_conversation(query)` | Semantic search across commits AND conversations |
| `validate_date_with_github(conv_id)` | Validate conversation dates against GitHub evidence |
| `github_stats()` | Statistics about imported GitHub data |

### 8. YouTube Tools
| Tool | Purpose |
|------|---------|
| `youtube_search(query)` | Search 31K watched videos by keyword |
| `youtube_semantic_search(query)` | Find videos by concept similarity |
| `youtube_stats()` | Viewing patterns and statistics |
| `search_youtube_searches(query)` | Search 1.2K YouTube search queries |

### 9. Google Browsing Tools (NEW)
| Tool | Purpose |
|------|---------|
| `search_google_searches(query)` | Search 52K Google searches |
| `search_google_visits(query)` | Search 58K website visits |
| `google_stats()` | Browsing statistics overview |

### 10. GitHub File Changes (NEW)
| Tool | Purpose |
|------|---------|
| `search_file_changes(query)` | Search 1.1K file changes by filename/repo |

### 11. V1 Interpretation Tools (NEW)
| Tool | Purpose |
|------|---------|
| `query_signature_phrases(category)` | 432 signature phrases (v2, 8 categories) |
| `query_insights(month, category, limit)` | 1.1K extracted insights |
| `query_questions(month, category, limit)` | 1.4K user questions (v2, 12 categories) |
| `query_monthly_themes(limit)` | 36 monthly theme narratives (v2, 100% content) |
| `query_mood(month, limit)` | 897 daily mood/energy patterns |
| `query_weekly_summaries(month, limit)` | 151 weeks with rich narrative summaries 🔥 |

### 12. Hidden Gem Tools (NEW)
| Tool | Purpose |
|------|---------|
| `query_accomplishments(month, limit)` | 796 days with ~20 accomplishments each |
| `query_glossary(term)` | 105 SEED-style term definitions |
| `query_phrase_context(phrase, limit)` | 200 phrases with meaning + style insights |
| `query_project_arcs(project, limit)` | 185 projects with lifecycle tracking |

### 13. Intellectual Evolution Tool (NEW)
| Tool | Purpose |
|------|---------|
| `query_intellectual_evolution(limit)` | 11 quarter comparisons with evolved beliefs, new frameworks, pivotal insights |

### ⚠️ Archived KG Tools
These now return "KG Archived" message with alternatives:
- `list_themes`, `get_theme`, `list_frameworks`
- `get_inversions`, `get_core_insights`, `get_formulas`
- `get_key_quotes`, `find_connections`, `brain_stats`

## Slash Commands

| Command | Purpose |
|---------|---------|
| `/brain-search <term>` | Search embeddings + conversations |
| `/brain-month <YYYY-MM>` | Time-travel to specific month |
| `/brain-velocity <term>` | Track idea acceleration |
| `/brain-align <decision>` | Check against SEED principles |
| `/brain-genesis <term>` | Find first mention |
| `/brain-synth <topic>` | Deep synthesis of position |

## Pipeline Commands

### Rebuild Pipeline (NEW - 2025-12-25)

Unified rebuild for all data layers:

```bash
# Show status of all layers
python pipelines/rebuild.py status

# Rebuild everything (facts → brain → temporal → interpretations)
python pipelines/rebuild.py all

# Rebuild specific layers
python pipelines/rebuild.py facts      # Spend: raw/daily/monthly
python pipelines/rebuild.py brain      # L0-L2: index/summary/content
python pipelines/rebuild.py temporal   # Date dimension table
python pipelines/rebuild.py focus      # focus/v1 interpretation
```

### Data Operations

```bash
# Show brain status (messages, embeddings, coverage)
python -m pipelines status

# Import Claude Code conversations
python -m pipelines import-claude --days 8              # Dry run
python -m pipelines import-claude --days 8 --import     # Actually import
python -m pipelines import-claude --all --import --merge  # Import all + merge

# Run embedding pipeline
python -m pipelines embed --batches 100                 # Embed 100 batches
python -m pipelines embed --all                         # Embed all remaining

# Search embeddings
python -m pipelines search "bottleneck as amplifier"

# Show embedding stats
python -m pipelines stats
```

## Embedding System

### Stack
- **Model**: nomic-embed-text-v1.5 via sentence-transformers (768 dimensions)
- **Hardware**: Apple Silicon MPS acceleration
- **Storage**: DuckDB with VSS extension (persistent HNSW index)
- **Scope**: User messages only, >20 chars
- **Location**: `mordelab/02-monotropic-prosthetic/embeddings.duckdb` (14GB)
- **Performance**: 256ms queries (34x optimized), model pre-warmed at startup

### Running Embeddings
```bash
# Using new pipeline CLI (recommended)
python -m pipelines embed --batches 50      # Embed 2500 msgs
python -m pipelines embed --all             # Embed all remaining
python -m pipelines search "your query"     # Test search

# Or using old scripts (still work)
cd mordelab/02-monotropic-prosthetic
./mcp-env/bin/python embed_messages.py 50
```

### Performance
- ~13-15 messages/second on Apple Silicon (M4)
- Uses sentence-transformers (no Ollama required)
- Automatic NULL message_id handling with synthetic IDs

## YouTube Data

### Overview
- **31,832 videos** in library (exported from YouTube)
- **16,386 videos** actually watched (have `watched_date`)
- **15,446 videos** unwatched (in library but no watch timestamp)
- **15,458 videos** with transcripts
- Metadata: channel, views, likes, duration, upload date

### Schema
Key columns in `youtube_rows.parquet`:
- `title`, `youtube_id`, `url`, `channel_name`
- `full_transcript`, `transcript_word_count`
- `watched_date`, `watch_count`, `first_watched`, `last_watched`
- `content_embedding_1024` (vector for semantic search)
- `view_count`, `like_count`, `engagement_rate`

### Value for Intellectual DNA
YouTube data represents **consumption patterns** - what ideas you were exposed to, complementing:
- **Conversations** = what you said/thought
- **GitHub** = what you built
- **YouTube** = what you consumed

## Development

### Environment
```bash
cd mordelab/02-monotropic-prosthetic
source mcp-env/bin/activate
python mcp_brain_server.py  # Runs on stdio (model pre-warms automatically)
```

### Testing Tools
```bash
./mcp-env/bin/python -c "
from mcp_brain_server import *
print(semantic_search('agency', limit=3))
print(list_principles())
print(conversation_stats())
print(what_was_i_thinking('2024-08'))
"
```

## Data Sources (Actual Coverage)

| Source | Location | Rows | Temporal Coverage |
|--------|----------|------|-------------------|
| **Conversations** | `data/all_conversations.parquet` | 353,216 | 100% (via `created` fallback) |
| **YouTube** | `data/youtube_rows.parquet` | 31,832 | 51.5% watched_date (correct) |
| **Google Searches** | `data/google_searches.parquet` | 52,791 | 100% |
| **Google Visits** | `data/google_visits.parquet` | 58,650 | 100% |
| **GitHub Commits** | `data/github_commits.parquet` | 1,427 | 100% |
| **GitHub Repos** | `data/github_repos.parquet` | 132 | 100% |

### Conversation Sources
| Source | Messages | User Msgs | Notes |
|--------|----------|-----------|-------|
| claude-code | 196,989 | ~65K | 47K have NULL msg_timestamp (use `created`) |
| chatgpt | 129,791 | ~43K | Full timestamps |
| claude_desktop | 25,408 | ~8K | Full timestamps |
| gemini | 1,028 | ~0.3K | Full timestamps |

### Temporal Columns
- `msg_timestamp`: Primary timestamp (86.4% coverage)
- `created`: Conversation creation time (100% coverage, fallback for msg_timestamp)
- `year`, `month`: Derived fields (100% coverage)
- MCP tools use `year`/`month`/`created` - works for all messages

## Health Check

```bash
# Run health check
./mordelab/02-monotropic-prosthetic/mcp-env/bin/python scripts/health_check.py

# Run maintenance
./scripts/maintain.sh
```

## Conventions

### Python
- Use pathlib for all file paths
- Type hints on functions
- DuckDB for analytics
- Absolute paths always

### Paths (LOCAL - no more iCloud!)
```python
# CORRECT - Local path (moved 2025-12-17)
base = Path("/Users/mordechai/intellectual_dna")

# Also works
base = Path.home() / "intellectual_dna"
```

## Roadmap

### Phase 1 ✅ Complete
- MCP server with tools
- Serverless mode (DuckDB)
- Temporal analysis tools

### Phase 2 ✅ Complete
- nomic-embed-text via sentence-transformers
- DuckDB VSS extension for vector search
- 96K user messages embedded

### Phase 3 ✅ Complete
- SEED principles integrated
- **KG archived** (2025-12-16) - was aspirational slice, not representative

### Phase 4 ✅ Complete
- YouTube data integration (31K videos with embeddings)
- GitHub cross-reference tools

### Phase 5 ✅ Complete (2025-12-25)
- **Facts vs Interpretations architecture** - Immutable truth vs derived analysis
- **Onion Skin brain layers (L0-L3)** - Progressive disclosure
- **Three-tier spend tracking** - raw → daily → monthly aggregation
- **Temporal dimension table** - 2.3K days with activity metrics
- **First interpretation: focus/v1** - Daily keyword extraction
- **Unified rebuild pipeline** - `python pipelines/rebuild.py`

### Phase 6 ✅ Complete (2025-12-26)
- **V2 interpretation pipelines** (Gemini 3 Flash / Grok 3):
  - `mvp_velocity/v2`: Development pattern extraction (36 months)
  - `tool_stacks/v2`: Technology stack tracking (36 months)
  - `problem_resolution/v2`: Debugging pattern analysis (36 months)
  - `focus/v2`: Daily focus with LLM summaries
- **HNSW index optimization**: 34x speedup (8.5s → 256ms)
- **Model pre-warming**: Eliminates cold start
- **Brain MRI diagnostic**: Health score 70/100

### Phase 7 - Future
- Auto-clustering to discover themes from embeddings
- Signature phrase extraction (e.g., "think out the box")
- Cross-source synthesis (conversations + YouTube + GitHub)
- Embed assistant responses (222K more messages)

## Tips for Claude

1. **Use `thinking_trajectory(topic)` for deep analysis** - Combines semantic + temporal
2. **Use `semantic_search(query)` as primary entry** - 96K embedded messages, 256ms
3. **SEED principles**: `list_principles()` / `get_principle(name)` still active
4. **KG tools are archived** - Will return message pointing to alternatives
5. **YouTube tools**: Search what Mordechai consumed, not just what he said
6. **GitHub tools**: Cross-reference code projects with conversation context
7. **V2 interpretations**: `query_focus()`, mvp_velocity, tool_stacks, problem_resolution
8. **Restart Claude Code** after MCP server changes to pick up new tools
