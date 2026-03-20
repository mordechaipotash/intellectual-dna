# 🧠 brain-mcp

**You've had thousands of AI conversations. You can't search any of them.**

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

## The Problem

You had a breakthrough at 2am last Tuesday. You laid out a whole framework in a conversation with Claude. It was brilliant.

You can't find it. You can't even remember which conversation it was in.

**Every week, millions of people pour their best thinking into AI conversations — and lose all of it.** ChatGPT's "memory" stores a few fun facts. None of them let you *search your own thinking*.

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

> *Built with ADHD in mind. If your brain drops context constantly, this is your external hard drive.*

---

## Install

```bash
pipx install brain-mcp          # recommended
brain-mcp init                   # discover your conversations
brain-mcp ingest                 # import them (fast, no GPU)
brain-mcp setup claude           # auto-configure Claude Desktop + Code
```

Restart Claude. **25 tools available.** Keyword search works immediately.

```bash
# Optional: enable semantic search
pipx inject brain-mcp fastembed  # ~107MB, no GPU needed
brain-mcp embed
```

<details>
<summary>pip install / other clients</summary>

```bash
pip install brain-mcp
brain-mcp init && brain-mcp ingest
brain-mcp setup claude           # Claude Desktop + Code
brain-mcp setup cursor           # Cursor
brain-mcp setup windsurf         # Windsurf
```

</details>

After setup, just say **"use brain"** in any conversation. Your AI searches your full thinking history when relevant.

---

## What You Can Do

| Ask your AI | What happens |
|-------------|-------------|
| *"What did I figure out about sleep last month?"* | Finds insights across 12 conversations you forgot you had |
| *"Where did I leave off with the business plan?"* | Reconstructs your context — open questions, decisions, next steps |
| *"How has my thinking about career changes evolved?"* | Tracks your opinion trajectory from doubt → clarity |
| *"What would it cost to switch focus right now?"* | Quantifies what you'd abandon — open threads, unfinished thinking |
| *"What do I actually think about AI?"* | Synthesizes YOUR views from 31 past conversations into one answer |

---

## 25 Tools (What Makes This Different)

Most MCP memory tools are key-value stores. brain-mcp has **8 cognitive prosthetic tools** that no other memory system offers:

| Tool | What it does |
|------|-------------|
| `tunnel_state` | "Load your save game" — reconstructs where you were in any domain |
| `switching_cost` | Quantified cost of switching between domains (built for ADHD) |
| `dormant_contexts` | Topics you were working on but silently dropped |
| `thinking_trajectory` | How your ideas evolved over time, not just the latest version |
| `what_do_i_think` | Synthesizes your actual views from months of conversations |
| `alignment_check` | Checks decisions against your own stated principles |
| `open_threads` | Everything unfinished, everywhere |
| `context_recovery` | Full re-entry brief for any domain |

Plus **17 more**: semantic search, keyword search, conversation browsing, stats, synthesis, analytics, and more. [Full tool reference →](https://brainmcp.dev/docs/tools)

### Progressive Tiers — Every tool works at every level:

| What you have | What works |
|---------------|-----------|
| Just conversations | Keyword search, date browsing, stats |
| + Embeddings | Semantic search, synthesis, trajectory |
| + Summaries | Full prosthetic tools with structured domain analysis |

---

## Supported Sources

| Source | Auto-detected | Status |
|--------|:---:|--------|
| Claude Code | ✅ | Supported |
| Claude Desktop | ✅ | Supported |
| ChatGPT | ✅ | Supported |
| Cursor | ✅ | Supported |
| Clawdbot | ✅ | Supported |
| Gemini CLI | ✅ | Supported |
| Generic JSONL | Manual | Supported |

---

## How It Works

1. **You already have the data.** Claude Code sessions, ChatGPT exports — they're files on your machine.
2. **brain-mcp indexes them.** Keyword search works instantly. Add embeddings for semantic search.
3. **Your AI gets 25 new tools.** Ask Claude "where did I leave off?" and it searches your brain. 12ms.

All data stays on your machine. Embedding model runs locally. **No cloud. No API costs. No accounts.**

---

## 🔒 Privacy

- **100% local** — all data stays on your machine
- **No telemetry** — zero tracking, zero phone-home
- **No cloud dependency** — works offline after setup
- **Open source** — audit every line ([MIT licensed](LICENSE))

---

## Requirements

- Python 3.11+
- macOS (Apple Silicon recommended), Linux, or WSL

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions welcome.

---

<div align="center">

*Built because losing your train of thought shouldn't mean starting over.*

**[brainmcp.dev](https://brainmcp.dev)** · **[PyPI](https://pypi.org/project/brain-mcp/)** · **[Full Docs](https://brainmcp.dev/docs/quickstart)**

⭐ If this is useful, a star helps others find it.

</div>
