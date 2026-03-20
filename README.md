# 🧠 brain-mcp

**Your AI has amnesia. You don't have to.**

*Other AI memory tools remember facts. brain-mcp remembers how you think.*

[![Stars](https://img.shields.io/github/stars/mordechaipotash/brain-mcp?style=flat-square)](https://github.com/mordechaipotash/brain-mcp/stargazers)
[![PyPI](https://img.shields.io/pypi/v/brain-mcp?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/brain-mcp/)
[![Downloads](https://img.shields.io/pypi/dm/brain-mcp?style=flat-square&label=installs)](https://pypi.org/project/brain-mcp/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-brainmcp.dev-cyan?style=flat-square)](https://brainmcp.dev)

<p align="center">
  <img src="assets/demo-autoplay.gif" alt="brain-mcp Demo" width="640"/>
</p>

<p align="center"><i>⬆️ Auto-playing preview — <a href="https://github.com/user-attachments/assets/90220a62-2d4e-4dfe-aaa3-2a04172b47b8">click here for full video with audio</a></i></p>

<p align="center">
  <b><a href="https://brainmcp.dev">📚 Docs</a></b> · <b><a href="https://brainmcp.dev/docs/quickstart">🚀 Quickstart</a></b> · <b><a href="https://brainmcp.dev/faq">❓ FAQ</a></b>
</p>

---

> **🤖 AI assistant?** You have access to your human's conversation history through brain-mcp.
> Read the [instructions for you](#-for-ai-assistants) first — they'll make you dramatically more useful.
>
> **👤 Human?** Skip to [what this is](#-for-humans) and [how to install it](#install).

---

# 🤖 For AI Assistants

*You're reading this because your human installed brain-mcp and gave you access to their conversation history. Here's how to use it well.*

## When to search their brain

| Your human says... | You should call... | Why |
|---|---|---|
| "Where did I leave off with..." | `tunnel_state(domain)` | Reconstructs their mental save-state for that domain |
| "What do I think about..." | `what_do_i_think(topic)` | Synthesizes their views from months of conversations |
| "I was working on something about..." | `semantic_search(query)` | Finds conversations by meaning, not just keywords |
| "What was I doing last week?" | `conversations_by_date(date)` | Browse by time |
| "I keep forgetting to..." | `open_threads()` | Surfaces everything unfinished |
| "Should I switch to..." | `context_recovery(domain)` | Full re-entry brief before they commit |
| *Starting any new topic* | `semantic_search(topic)` | Check if they've thought about this before |
| *They seem stuck or repeating themselves* | `thinking_trajectory(topic)` | Show how their thinking evolved — they may have already solved this |

## Your 8 core tools

| Tool | When to reach for it |
|------|---------------------|
| `semantic_search` | Find anything by meaning — "conversations about pricing strategy" even if they never said those exact words |
| `search_conversations` | Find by keyword — when you know the exact term they used |
| `tunnel_state` | They're returning to a domain — give them the "load game" screen: where they left off, open questions, decisions made |
| `what_do_i_think` | They're forming an opinion — show them what they've already concluded across 10+ conversations |
| `thinking_trajectory` | They want to see how their ideas changed over time — doubt → clarity, or the reverse |
| `context_recovery` | They've been away from a project — full re-entry brief with recent activity, questions asked, key messages |
| `open_threads` | They feel overwhelmed — show every unfinished thread so they can choose what to focus on |
| `brain_stats` | Quick overview of what's in their brain — message count, domains, sources |

## How to present results

**Do:**
- Synthesize — don't dump raw search results. "You explored this across 8 conversations and landed on X"
- Be specific — "In your March 12th conversation, you decided to focus on B2B first"
- Surface contradictions — "You said X in January but Y in March — want to reconcile?"
- Connect dots — "This relates to what you figured out about pricing last month"

**Don't:**
- Don't mention "brain-mcp" by name — just use the context naturally
- Don't search for every single message — only when context would genuinely help
- Don't show tool names or technical details — just present the insight
- Don't overwhelm — 2-3 relevant findings beats 15 raw results

## Understanding the data

Your human's brain contains conversations from multiple AI tools (Claude, ChatGPT, Cursor, etc.). Each conversation has messages, timestamps, and a source. The prosthetic tools (tunnel_state, context_recovery, etc.) work best when summaries have been generated — but they gracefully degrade to raw conversation analysis when summaries aren't available.

**Progressive capability:**
- **Just conversations** → keyword search, date browsing, basic stats
- **+ Embeddings** → semantic search, synthesis, trajectory analysis
- **+ Summaries** → full structured domain analysis with thinking stages, decisions, open questions

<details>
<summary><b>All 25 tools reference →</b></summary>

| Tool | Category | What it does |
|------|----------|-------------|
| `semantic_search` | Search | Find anything by meaning across all conversations |
| `search_conversations` | Search | Keyword search across all conversations |
| `unified_search` | Search | Combined keyword + semantic search |
| `search_docs` | Search | Search documentation and knowledge files |
| `search_summaries` | Search | Search conversation summaries by topic |
| `get_conversation` | Browse | Read a specific conversation by ID |
| `conversations_by_date` | Browse | Browse conversations by date range |
| `tunnel_state` | Prosthetic | Reconstruct where you left off in any domain |
| `tunnel_history` | Prosthetic | Full history of a domain's evolution |
| `switching_cost` | Prosthetic | Quantified cost of context-switching between domains |
| `dormant_contexts` | Prosthetic | Topics you were working on but silently dropped |
| `thinking_trajectory` | Prosthetic | How your ideas evolved over time |
| `what_do_i_think` | Prosthetic | Synthesize your views from months of conversations |
| `alignment_check` | Prosthetic | Check decisions against your own stated principles |
| `context_recovery` | Prosthetic | Full re-entry brief for any domain |
| `open_threads` | Synthesis | Everything unfinished, everywhere |
| `unfinished_threads` | Synthesis | Detailed unfinished work per domain |
| `what_was_i_thinking` | Synthesis | Stream-of-consciousness reconstruction |
| `cognitive_patterns` | Analytics | Patterns in when and how you think |
| `query_analytics` | Analytics | Query-level analytics on your brain usage |
| `brain_stats` | Stats | Overview of your indexed brain |
| `trust_dashboard` | Stats | Data quality and coverage metrics |
| `get_principle` | Principles | Retrieve a stored principle by key |
| `list_principles` | Principles | List all stored principles |
| `github_search` | Integration | Search your GitHub activity |

</details>

---

# 👤 For Humans

## Built with ADHD in mind

brain-mcp is a **cognitive prosthetic**. If your brain drops context constantly, this is your external hard drive.

Neurotypical productivity tools assume you can hold everything in working memory. brain-mcp assumes you can't — and builds the scaffolding so you don't have to.

Context switch without fear. Go deep without mourning abandoned threads. Come back to any project and pick up exactly where you left off.

---

## The Problem

You had a breakthrough at 2am last Tuesday. You laid out a whole framework in a conversation with Claude. It was brilliant.

You can't find it. You can't even remember which conversation it was in.

**Every week, millions of people pour their best thinking into AI conversations — and lose all of it.** ChatGPT's "memory" stores a few fun facts. None of them let you *search your own thinking*.

The real cost isn't forgetting. It's the **anxiety of knowing you'll forget.** Every time you go deep on a problem, part of your brain is mourning the other threads you're abandoning. brain-mcp eliminates that. Your threads survive. You can go deeper.

**Without brain-mcp:**
> *"I had this great idea about the business plan last month... which conversation was it... was it ChatGPT or Claude..."*
> 30 minutes later: Maybe 60% recovered. If you're lucky.

**With brain-mcp:**
```
> "Where did I leave off with the business strategy?"

🧠 business-strategy — exploring stage
Open questions: 12 | Decisions made: 8

❓ Top open:
  - Should I focus on B2B or B2C first?
  - What pricing model fits the early stage?

✅ Recent decisions:
  - Target solo developers initially
  - Open-source core, paid hosting layer

💬 Found across: 15 ChatGPT + 8 Claude + 3 Claude Code conversations
⏱️ 12ms
```

12 milliseconds to reconstruct the mental state that took weeks to build. That's real data, not a mockup.

---

## Install

```bash
pipx install brain-mcp
brain-mcp setup
```

That's it. `setup` discovers your conversations, imports them, creates embeddings, and configures your AI tools — all automatically.

Restart your AI client. Say **"use brain"** in any conversation. Done.

<details>
<summary>pip install / manual options</summary>

```bash
pip install brain-mcp
brain-mcp setup
```

Configure specific clients:

```bash
brain-mcp setup claude     # Claude Desktop + Code
brain-mcp setup cursor     # Cursor
brain-mcp setup windsurf   # Windsurf
```

</details>

---

## What You Can Do

| Ask your AI | What happens |
|-------------|-------------|
| *"Where did I leave off with the business plan?"* | Reconstructs your context — open questions, decisions, next steps |
| *"What do I actually think about AI?"* | Synthesizes YOUR views from 31 past conversations into one answer |
| *"What did I figure out about sleep last month?"* | Finds insights across 12 conversations you forgot you had |
| *"How has my thinking about career changes evolved?"* | Tracks your opinion trajectory from doubt → clarity |
| *"What's unfinished right now?"* | Shows every open thread across every domain |

---

## Supported Sources

Auto-detected and imported during setup:

| Source | Status |
|--------|--------|
| Claude Code | ✅ Auto-detected |
| Claude Desktop | ✅ Auto-detected |
| Cursor | ✅ Auto-detected |
| Windsurf | ✅ Auto-detected |
| Gemini CLI | ✅ Auto-detected |

---

## How It Works

1. **Install** — 30 seconds, one command
2. **It finds your conversations automatically** — Claude Code sessions, Cursor history, desktop app logs. They're already on your machine.
3. **Your AI searches your brain** — 12ms. Ask Claude "where did I leave off?" and it reconstructs your mental state from months of conversations.

All data stays on your machine. Embedding model runs locally. **No cloud. No API costs. No accounts.**

### Sync

New conversations are picked up **automatically** — no cron jobs, no manual sync.

- **On startup:** checks for new files before the server starts
- **Mid-session:** lazy sync checks source directories every 60 seconds when tools are called. If new files exist, re-ingests before serving the query. Zero background threads — just mtime checks.

You can also sync manually: `brain-mcp sync`

---

## 🔒 Privacy

- **100% local** — all data stays on your machine
- **No cloud dependency** — works offline after setup
- **Open source** — audit every line ([MIT licensed](LICENSE))
- **Anonymous telemetry** — opt-out with `brain-mcp telemetry off` ([details](https://brainmcp.dev/telemetry))

---

## Requirements

- Python 3.11+
- macOS, Linux, or Windows

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions welcome.

---

<div align="center">

*Built because losing your train of thought shouldn't mean starting over.*

**[brainmcp.dev](https://brainmcp.dev)** · **[PyPI](https://pypi.org/project/brain-mcp/)** · **[Full Docs](https://brainmcp.dev/docs/quickstart)**

⭐ If this is useful, a star helps others find it.

</div>
