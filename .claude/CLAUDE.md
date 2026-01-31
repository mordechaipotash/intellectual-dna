# Intellectual DNA - Monotropic Prosthetic

## LLM Model Policy
**If using Gemini via OpenRouter, ALWAYS use `google/gemini-3-flash-preview`** - never use older versions like gemini-2.0-flash or gemini-2.5-flash.

## Project Overview
AI-powered knowledge system for querying Mordechai's intellectual patterns from:
- **359K conversation messages** (2023-01 to 2026-01) in DuckDB/Parquet
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
│   ├── mcp_brain_server.py                  # MCP server (37 active, 55 deprecated)
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

### 14. Phase 1 55x Mining Tools (NEW 2026-01-11)
*Surfaces 8 hidden interpretation layers that were computed but never exposed*

| Tool | Purpose | Rows |
|------|---------|------|
| `query_conversation_titles(month, limit)` | Meta-level naming of conversations | 957 titles |
| `query_weekly_expertise(month, limit)` | Technical domains focused on each week | 153 weeks |
| `query_tool_preferences(month, limit)` | Which tools/stacks preferred each month | 36 months |
| `query_tool_combos(limit)` | Tool stack combinations used together | 5 records |
| `query_problem_chains(month, limit)` | Weekly debugging patterns and solutions | 155 weeks |
| `query_intellectual_themes(theme)` | Auto-discovered document clusters | 8 themes |
| `query_youtube_links(month, limit)` | How videos connect to your work | 101 links |

### 15. Phase 2 Spend Mining Tools (NEW 2026-01-11)
*Extracts insights from 732K OpenRouter API calls ($898 total spend, 1.47B tokens)*

| Tool | Purpose | Data |
|------|---------|------|
| `query_model_efficiency(top_n, sort_by)` | Model ROI - cost per million tokens | 116 models |
| `query_provider_performance(month)` | Provider comparison (Together, DeepInfra, Novita) | 42 providers |
| `query_spend_temporal(month, view)` | Usage patterns by day/month/dow | 307 days |
| `query_spend_summary()` | Comprehensive spend overview | All data |

**Usage examples**:
```python
query_model_efficiency(top_n=10, sort_by='efficiency')  # Most cost-effective models
query_provider_performance('2025-06')  # June 2025 providers
query_spend_temporal(view='dow')  # Day of week patterns
query_spend_summary()  # Full overview
```

### 16. Phase 3 Conversation Analysis Tools (NEW 2026-01-11)
*Extracts patterns from 353K messages, 15K conversations, 154K threaded messages*

| Tool | Purpose | Data |
|------|---------|------|
| `query_conversation_stats(top_n, sort_by)` | Per-conversation metrics | 15,210 conversations |
| `query_deep_conversations(min_messages, limit)` | Long sustained discussions (100+ msgs) | 581 deep convos |
| `query_question_patterns(month)` | Q&A type analysis (how-to, debug, etc) | 11,989 questions |
| `query_correction_patterns(month)` | When you correct AI ("no no no", etc) | 1,877 corrections |
| `query_conversation_summary()` | Comprehensive conversation overview | All data |

**Usage examples**:
```python
query_conversation_stats(sort_by='questions')  # Most question-heavy convos
query_deep_conversations(min_messages=200)  # Really long conversations
query_question_patterns('2025-06')  # June 2025 question types
query_correction_patterns()  # All-time correction analysis
```

### 17. Phase 4 Behavioral Archaeology Tools (NEW 2026-01-11)
*Mines 112K Google searches/visits for behavioral patterns (2011-2025)*

| Tool | Purpose | Data |
|------|---------|------|
| `query_search_patterns(month, intent)` | Search intent categorization | 52,791 searches |
| `query_browsing_patterns(category)` | Website visit analysis | 58,650 visits |
| `query_research_velocity(month)` | Research intensity tracking | 283 deep dive days |
| `query_curiosity_terms(month, limit)` | Top search topics | 2,173 terms |
| `query_behavioral_summary()` | Comprehensive overview | All data |

**Usage examples**:
```python
query_search_patterns(intent='debugging')  # Show debugging searches
query_browsing_patterns(category='github')  # GitHub visit analysis
query_research_velocity()  # Research intensity overview
query_curiosity_terms(month='2024-06')  # June 2024 search terms
```

### 18. Phase 5 Code Productivity Tools (NEW 2026-01-11)
*Mines 1.4K commits across 132 repos for productivity patterns*

| Tool | Purpose | Data |
|------|---------|------|
| `query_code_velocity(month, view)` | Commit velocity (daily/weekly/monthly) | 186 days tracked |
| `query_repo_stats(active_only, limit)` | Repository statistics | 121 active repos |
| `query_language_stats()` | Programming language distribution | 6 languages |
| `query_commit_patterns(commit_type)` | Commit message patterns | 10 types |
| `query_high_productivity_days(limit)` | Days with 5+ commits | 79 days |
| `query_code_summary()` | Comprehensive code overview | All data |

**Usage examples**:
```python
query_code_velocity(view='monthly')  # Monthly commit patterns
query_repo_stats(limit=10)  # Top 10 repos
query_commit_patterns(commit_type='fix')  # Show fix commits
query_high_productivity_days()  # Most productive days
```

### 19. Phase 6 Markdown Knowledge Tools (NEW 2026-01-11)
*Surfaces 5.5K markdown documents (6.3M words) from knowledge base*

| Tool | Purpose | Data |
|------|---------|------|
| `query_markdown_stats(limit)` | Document-level statistics | 5,524 docs |
| `query_markdown_categories(limit)` | Category analysis | 28 categories |
| `query_markdown_projects(limit)` | Project clustering | 50+ projects |
| `query_curated_docs(doc_type, limit)` | TOP_TIER_IP + DEEP_DOCS | 701 curated docs |
| `query_markdown_summary()` | Comprehensive markdown overview | All layers |

**Usage examples**:
```python
query_markdown_stats(limit=20)  # Top 20 documents by word count
query_markdown_categories()  # All 28 categories
query_markdown_projects(limit=10)  # Top 10 projects
query_curated_docs(doc_type='top_tier')  # 440 TOP_TIER_IP docs
query_curated_docs(doc_type='deep')  # 261 DEEP_DOCS
query_markdown_summary()  # Full overview
```

### 20. Phase 7 Cross-Dimensional Synthesis Tools (NEW 2026-01-11)
*Connects ALL data sources: accomplishments × mood × commits × spend × research*

| Tool | Purpose | Data |
|------|---------|------|
| `query_productivity_matrix(month, view, top_n)` | Daily/weekly/monthly productivity scores | 2,793 days |
| `query_learning_arcs(month, topic, limit)` | Learning velocity: Search→Watch→Discuss→Build | 125 months |
| `query_project_success(category, limit)` | Project outcome prediction | 185 projects |
| `query_unified_timeline(year, limit)` | Cross-source unified timeline | 125 months |
| `query_cross_dimensional_summary()` | Comprehensive synthesis overview | All layers |

**Usage examples**:
```python
query_productivity_matrix(view='peak')  # Peak productivity days
query_productivity_matrix(view='monthly')  # Monthly productivity
query_learning_arcs(topic='python')  # Python learning arc
query_project_success(category='shipped')  # Shipped projects only
query_unified_timeline(year='2025')  # 2025 timeline
query_cross_dimensional_summary()  # Full synthesis overview
```

### 21. Phase 9 Discovery Pipeline Tools (NEW 2026-01-11)
*Automated insight generation: anomalies, trends, recommendations*

| Tool | Purpose | Data |
|------|---------|------|
| `query_anomalies(anomaly_type, limit)` | Spending spikes, productivity outliers | 14 spikes, 187 outliers |
| `query_trends(trend_type, limit)` | Rising/declining concepts | 49 rising, 262 declining |
| `query_recommendations(rec_type, limit)` | Dormant topics, stalled projects | 629 dormant topics |
| `query_weekly_synthesis(weeks)` | Auto-generated weekly reports | 470 weeks |
| `query_discovery_summary()` | Full discovery insights overview | All layers |

**Usage examples**:
```python
query_anomalies(anomaly_type='spending')  # Spending spikes only
query_trends(trend_type='rising')  # Rising concepts
query_recommendations(rec_type='dormant')  # Dormant topics to revisit
query_weekly_synthesis(weeks=5)  # Last 5 weeks
query_discovery_summary()  # Full overview
```

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
8. **Phase 1 55x tools**: `query_intellectual_themes()`, `query_problem_chains()`, `query_weekly_expertise()` - surfaces hidden interpretation layers
9. **Restart Claude Code** after MCP server changes to pick up new tools
