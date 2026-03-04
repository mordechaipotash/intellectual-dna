# brain-mcp

**You've had thousands of AI conversations. You can't search any of them.**

brain-mcp fixes that. It's a local [MCP server](https://modelcontextprotocol.io) that ingests your conversations from Claude Code, ChatGPT, Cursor, and more — then gives your AI assistant the ability to search, synthesize, and reason over everything you've ever discussed.

Your data stays on your machine. Nothing leaves your computer.

---

## Quick Start (3 commands)

```bash
npx brain-mcp init          # discover your conversations
npx brain-mcp init --full   # import + embed (one-time, ~5 min)
npx brain-mcp setup claude  # auto-configure Claude Code
```

That's it. Restart Claude Code, and ask: *"Search my past conversations about authentication"*

### Alternative: pip install

```bash
pip install brain-mcp
brain-mcp init --full
brain-mcp setup claude
```

---

## What Happens

`brain-mcp init` auto-discovers your AI conversations:

```
Discovering AI conversations...

   found  Claude Code       438 sessions    ~/.claude/projects/
   found  ChatGPT            47 exports     ~/Downloads/chatgpt-export/
   --     Cursor             not found
   --     Windsurf           not found

Config saved to ~/.config/brain-mcp/brain.yaml
```

`brain-mcp init --full` imports everything and creates searchable embeddings:

```
Importing conversations...
   Claude Code  ████████████████████ 23,456 messages  (32s)
   ChatGPT      ████████████████████  8,923 messages  (12s)

Creating embeddings...
   Embedding    ████████████████████ 19,847 embedded  (2m 31s)

Your brain is ready!
   Messages:    32,379
   Embedded:    19,847
   Sources:     2 (Claude Code, ChatGPT)
```

---

## Why Brain MCP?

| Feature | Brain MCP | Mem0 | Letta |
|---------|:---------:|:----:|:-----:|
| **Approach** | Conversation archaeology — search & synthesize across your full AI history | Key-value memory store | Tiered memory agents |
| **MCP Tools** | 25 | ~5 | ~5 |
| **Data Location** | 100% local | Cloud | Cloud/Self-hosted |
| **Privacy** | No telemetry, no phone-home | Cloud-dependent | Cloud-dependent |
| **Cost** | Free (MIT) | Freemium | Freemium |
| **Unique Feature** | Cognitive prosthetic — reconstructs where you left off in any domain | Automatic memory extraction | Persistent agent memory |
| **Setup** | 3 commands | API key + cloud setup | Complex multi-service |
| **AI Sources** | Claude Code, ChatGPT, Clawdbot, Generic JSONL | N/A (runtime only) | N/A (runtime only) |

Brain MCP isn't a memory layer — it's **conversation archaeology**. It searches, synthesizes, and reconstructs context from your entire AI conversation history. The prosthetic tools don't just find information — they rebuild the *state* of your thinking.

---

## 🔒 Data Privacy

Brain MCP is radically local:

- **All data stays on your machine** — conversations, embeddings, summaries, everything
- **No telemetry** — we don't collect any usage data, ever
- **No phone-home** — the software never contacts any server
- **No cloud dependency** — works fully offline after initial model download
- **No accounts** — no sign-up, no API keys required (unless you opt into LLM summarization)
- **You own everything** — all data is in standard formats (parquet, LanceDB) you can read with any tool
- **MIT licensed** — fork it, audit it, modify it

The embedding model (~275MB) is downloaded once and runs locally. After that, Brain MCP works entirely offline.

---

## 25 Tools

### Search (4 tools)

| Tool | What it does |
|------|-------------|
| `semantic_search` | Find conceptually similar messages, even without exact keywords |
| `search_conversations` | Full-text keyword search across all messages |
| `unified_search` | Cross-source search (conversations + GitHub + docs) |
| `search_docs` | Search markdown documentation corpus |

### Conversations (3 tools)

| Tool | What it does |
|------|-------------|
| `get_conversation` | Retrieve a full conversation by ID |
| `conversations_by_date` | Browse conversations from a specific date |
| `search_summaries` | Search structured conversation summaries |

### Synthesis (4 tools)

| Tool | What it does |
|------|-------------|
| `what_do_i_think` | Synthesize your views on a topic across conversations |
| `alignment_check` | Check if a decision aligns with your stated principles |
| `thinking_trajectory` | Track how your thinking evolved over time |
| `what_was_i_thinking` | Monthly snapshot: what was on your mind in 2024-08? |

### Stats & Analytics (4 tools)

| Tool | What it does |
|------|-------------|
| `brain_stats` | Overview with 7 views (messages, domains, pulse, etc.) |
| `unfinished_threads` | Threads with unresolved questions |
| `query_analytics` | Custom SQL analytics over your conversation data |
| `github_search` | Search your GitHub repos and commits (if configured) |

### Cognitive Prosthetic (8 tools)

This is what makes brain-mcp different from "just search." These tools don't just find information — they reconstruct the *state* of your thinking.

| Tool | What it does |
|------|-------------|
| `tunnel_state` | "Load game" — where you left off in a domain |
| `context_recovery` | Full re-entry briefing after time away |
| `switching_cost` | Should you switch? Quantified cost before you do |
| `dormant_contexts` | Abandoned topics with unresolved questions |
| `open_threads` | All unfinished business, every domain, one view |
| `cognitive_patterns` | When and how you think best, backed by data |
| `tunnel_history` | Your engagement with a domain over time |
| `trust_dashboard` | Everything preserved. Nothing lost. Proof. |

### Principles (2 tools)

| Tool | What it does |
|------|-------------|
| `list_principles` | List your configured principles |
| `get_principle` | Get details about a specific principle |

---

## Supported Sources

| Source | Auto-detected | Format |
|--------|:---:|--------|
| **Claude Code** | yes | JSONL sessions in `~/.claude/projects/` |
| **ChatGPT** | yes | JSON export from Settings > Data Controls |
| **Clawdbot** | yes | JSONL sessions in `~/.clawdbot/agents/` |
| **Cursor** | coming soon | Workspace storage |
| **Windsurf** | coming soon | Workspace storage |
| **Generic JSONL** | manual | `{role, content, timestamp}` per line |

### ChatGPT Export

1. Go to [chat.openai.com](https://chat.openai.com) > Settings > Data Controls > Export
2. Download the zip, extract to `~/Downloads/`
3. Run `brain-mcp init` — it auto-detects the export

---

## CLI Reference

```bash
brain-mcp init              # Discover sources, create config
brain-mcp init --full       # Discover + ingest + embed (one command)
brain-mcp ingest            # Import conversations to parquet
brain-mcp embed             # Create/update vector embeddings
brain-mcp serve             # Start MCP server (stdio)
brain-mcp setup claude      # Auto-configure Claude Code
brain-mcp setup cursor      # Auto-configure Cursor
brain-mcp setup windsurf    # Auto-configure Windsurf
brain-mcp doctor            # Health check (what's working, what's missing)
brain-mcp status            # One-line status
brain-mcp sync              # Incremental update (new conversations only)
```

### Health Check

```bash
$ brain-mcp doctor

Brain MCP Health Check

   ok Python 3.12.1
   ok Config: ~/.config/brain-mcp/brain.yaml
   ok Parquet: 32,379 messages
   ok Vectors: 19,847 embeddings
   warn  Summaries: not generated (prosthetic tools in basic mode)
      -> Run: brain-mcp summarize (requires ANTHROPIC_API_KEY)
   ok Claude Code: configured
   --  Cursor: not configured
```

---

## How It Works

```
You ──────────────────────────────────────────────────────────

  "Search my conversations about authentication"
  "What do I think about microservices?"
  "Where did I leave off with the database redesign?"

                          │
                          ▼

MCP Server ─────────────────────────────────────────────────

  25 tools registered:
  Search │ Synthesis │ Prosthetic │ Stats │ Principles

                          │
                          ▼

Data Layer ─────────────────────────────────────────────────

  DuckDB (SQL over parquet)    LanceDB (vector search)
  ┌─────────────────────┐      ┌────────────────────┐
  │  all_conversations   │      │  message vectors   │
  │  .parquet            │      │  (768-dim, local)  │
  └─────────────────────┘      └────────────────────┘

                          │
                          ▼

Ingest Pipeline ────────────────────────────────────────────

  Claude Code │ ChatGPT │ Clawdbot │ Generic JSONL
  (auto-discovered from your machine)
```

All data stored at `~/.config/brain-mcp/`. Embedding model runs locally (nomic-embed-text-v1.5, ~275MB download on first run).

---

## Progressive Feature Tiers

brain-mcp works at every scale. You don't need 100K messages to get value.

| What you have | What works |
|---------------|-----------|
| **Just conversations** (after `ingest`) | Keyword search, date browsing, stats, analytics |
| **+ Embeddings** (after `embed`) | Semantic search, synthesis, trajectory tracking |
| **+ Summaries** (after `summarize`) | Full prosthetic tools with structured analysis |

Every tool works at every tier — just with increasing depth. The prosthetic tools (tunnel_state, switching_cost, etc.) give useful results from raw messages, and richer results with summaries.

---

## Configuration

Config lives at `~/.config/brain-mcp/brain.yaml` (created by `brain-mcp init`).

| Key | Description | Default |
|-----|-------------|---------|
| `data_dir` | Parquet file storage | `~/.config/brain-mcp/data` |
| `vectors_dir` | LanceDB vectors | `~/.config/brain-mcp/vectors` |
| `sources` | Conversation sources (auto-detected) | `[]` |
| `embedding.model` | Embedding model | `nomic-ai/nomic-embed-text-v1.5` |
| `embedding.dim` | Vector dimensions | `768` |
| `summarizer.enabled` | Enable summarization | `false` |
| `summarizer.provider` | LLM provider | `anthropic` |
| `summarizer.model` | LLM model | `claude-sonnet-4-20250514` |
| `domains` | Domain categories for prosthetic tools | (auto-generated) |

---

## Requirements

- Python 3.11+ (auto-managed if using npx)
- ~500MB disk (embedding model + data)
- ~2GB RAM for embedding
- macOS, Linux, or WSL

---

## Project Structure

```
brain-mcp/
├── pyproject.toml              # Package config
├── brain_mcp/                  # Python package
│   ├── cli.py                  # CLI (init, ingest, embed, serve, etc.)
│   ├── config.py               # Config loader
│   ├── sync.py                 # Incremental sync
│   ├── server/                 # MCP server + 25 tools
│   │   ├── server.py           # FastMCP server
│   │   ├── db.py               # DuckDB + LanceDB connections
│   │   ├── tools_search.py
│   │   ├── tools_synthesis.py
│   │   ├── tools_prosthetic.py
│   │   ├── tools_stats.py
│   │   ├── tools_conversations.py
│   │   ├── tools_github.py
│   │   └── tools_analytics.py
│   ├── ingest/                 # Conversation ingesters
│   │   ├── claude_code.py
│   │   ├── chatgpt.py
│   │   ├── clawdbot.py
│   │   └── generic.py
│   ├── embed/                  # Local embedding pipeline
│   └── summarize/              # Optional LLM summarization
├── npm/                        # npx wrapper
│   ├── package.json
│   └── bin/brain-mcp.js
├── principles/                 # Template for personal principles
├── tests/
├── CONTRIBUTING.md
├── CHANGELOG.md
└── LICENSE
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up the dev environment, run tests, and submit PRs.

## License

MIT — see [LICENSE](LICENSE).
