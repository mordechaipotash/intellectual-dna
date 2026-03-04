# Getting Started with brain-mcp

This guide walks you through setup, ingestion, and launching the MCP server step by step.

## Prerequisites

- **Python 3.11+**
- **~2GB disk** for the embedding model (downloaded on first run)
- **~4GB RAM** for embedding (the model runs locally)
- Git

### Optional (for summarization)

The prosthetic tools (`tunnel_state`, `context_recovery`, etc.) require conversation summaries. Generating these requires an LLM API key from one of:
- Anthropic (Claude)
- OpenAI
- Ollama (local, free)

## Step 1: Clone & Setup

```bash
git clone https://github.com/your-user/brain-mcp.git
cd brain-mcp
./setup.sh
```

This creates a Python venv, installs dependencies, and copies `brain.yaml.example` to `brain.yaml`.

## Step 2: Configure Sources

Edit `brain.yaml` to point to your conversation data:

```yaml
sources:
  # Claude Code (default path)
  - type: claude-code
    path: ~/.claude/projects/

  # ChatGPT export
  - type: chatgpt
    path: ~/Downloads/chatgpt-export/

  # Any JSONL with {role, content, timestamp} lines
  - type: generic
    path: ~/my-conversations/
    name: my-source
```

See [Adding Sources](adding-sources.md) for details on each format.

## Step 3: Ingest Conversations

```bash
./ingest.sh
```

This reads all configured sources and merges them into `data/all_conversations.parquet`.

You should see output like:
```
📥 [claude-code] claude-code ← /home/user/.claude/projects/
   → 15,230 messages
✅ Merged 15,230 messages → data/all_conversations.parquet
```

## Step 4: Create Embeddings

```bash
./embed.sh
```

First run downloads the embedding model (~275MB). Subsequent runs are incremental — only new messages get embedded.

```
Loading nomic-ai/nomic-embed-text-v1.5...
Model loaded!
Total user messages: 8,102
Filtering noise messages...
Embedding 7,540 messages...
✅ Done! Embedded 7,540 messages.
```

## Step 5: Start the MCP Server

```bash
./start.sh
```

The server runs on stdio (MCP standard). Connect it to your AI assistant:

### Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "brain": {
      "command": "/path/to/brain-mcp/start.sh"
    }
  }
}
```

### Other MCP Clients

Any MCP-compatible client can connect via stdio. Point it at `start.sh` or `python -m server.server`.

## Step 6 (Optional): Generate Summaries

Summaries power the prosthetic tools. To enable:

1. Set up your LLM API key:
   ```bash
   export ANTHROPIC_API_KEY="sk-..."  # or OPENAI_API_KEY
   ```

2. Enable in `brain.yaml`:
   ```yaml
   summarizer:
     enabled: true
     provider: anthropic
     model: claude-sonnet-4-20250514
     api_key_env: ANTHROPIC_API_KEY
   ```

3. Run:
   ```bash
   source venv/bin/activate
   python -m summarize.summarize
   ```

## What's Next

- [Tools Reference](tools-reference.md) — all 23 tools with parameters and examples
- [Adding Sources](adding-sources.md) — how to add Claude Code, ChatGPT, or custom sources
- Set up a cron job to re-run `./ingest.sh && ./embed.sh` periodically
