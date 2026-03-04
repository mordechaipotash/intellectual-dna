# Adding Conversation Sources

brain-mcp supports four source types out of the box. Each is configured in `brain.yaml` under the `sources` list.

## Claude Code

Reads JSONL session files from Claude Code's local project directories.

```yaml
sources:
  - type: claude-code
    path: ~/.claude/projects/
```

**What it reads:** Every `*.jsonl` file under the projects directory. Each file is one conversation session containing `user` and `assistant` messages with timestamps.

**Default location:** `~/.claude/projects/` (created automatically by Claude Code).

**What you get:** Message content, project names (extracted from directory structure), model names, timestamps.

## ChatGPT

Reads the `conversations.json` file from a ChatGPT data export.

```yaml
sources:
  - type: chatgpt
    path: ~/Downloads/chatgpt-export/
```

**How to export:**
1. Go to [chat.openai.com](https://chat.openai.com)
2. Settings → Data Controls → Export Data
3. Wait for the email, download the ZIP
4. Unzip — the `conversations.json` file is what we need
5. Point `path` to the directory containing `conversations.json`

**What you get:** Full conversation history with titles, timestamps, model names, and message threading.

## Clawdbot

Reads Clawdbot agent session JSONL files.

```yaml
sources:
  - type: clawdbot
    path: ~/.clawdbot/agents/
```

**What it reads:** Session files from `~/.clawdbot/agents/*/sessions/*.jsonl`. Each file represents one conversation session.

**What you get:** Messages with roles, models, session metadata, and timestamps.

## Generic JSONL

For any conversation data you have in JSONL format. This is the catch-all ingester.

```yaml
sources:
  - type: generic
    path: ~/my-conversations/
    name: my-custom-source  # appears as source name in parquet
```

### Required Format

Each line must be a JSON object with at least:

```json
{"role": "user", "content": "What is the meaning of life?", "timestamp": "2024-01-15T10:30:00Z"}
{"role": "assistant", "content": "The meaning of life...", "timestamp": "2024-01-15T10:30:05Z"}
```

### Supported Fields

| Field | Required | Description |
|-------|----------|-------------|
| `role` | ✅ | `"user"` or `"assistant"` |
| `content` | ✅ | Message text |
| `timestamp` | ✅ | ISO 8601 or Unix epoch (seconds or ms) |
| `conversation_id` | | Groups messages into conversations |
| `model` | | LLM model name |
| `project` | | Project/workspace name |
| `title` | | Conversation title |
| `message_id` | | Unique message identifier |
| `parent_id` | | Parent message for threading |

### Timestamp Formats

The generic ingester accepts:
- ISO 8601: `2024-01-15T10:30:00Z`, `2024-01-15T10:30:00.000Z`
- Date+time: `2024-01-15 10:30:00`
- Date only: `2024-01-15`
- Unix epoch (seconds): `1705312200`
- Unix epoch (milliseconds): `1705312200000`

## Multiple Sources

You can configure as many sources as you want:

```yaml
sources:
  - type: claude-code
    path: ~/.claude/projects/

  - type: chatgpt
    path: ~/exports/chatgpt-2024/

  - type: chatgpt
    path: ~/exports/chatgpt-2025/

  - type: generic
    path: ~/logs/copilot-chats/
    name: copilot

  - type: generic
    path: ~/logs/custom-bot/
    name: my-bot
```

Each source gets a `source` tag in the parquet, so you can filter by source in searches.

## Writing a Custom Ingester

If you have a format that doesn't fit the generic JSONL pattern, write a custom ingester:

1. Create `ingest/my_source.py`
2. Import and use `make_record` and `finalize_conversation` from `ingest.schema`
3. Return a list of canonical records

```python
from brain_mcp.ingest.schema import make_record, finalize_conversation

def ingest_my_source(path):
    records = []
    for msg in read_your_format(path):
        record = make_record(
            source="my-source",
            conversation_id=msg.conv_id,
            role=msg.role,
            content=msg.text,
            timestamp=msg.time,
            msg_index=msg.index,
        )
        if record:
            records.append(record)
    return finalize_conversation(records)
```

4. Add your type to `ingest.sh` (in the Python dispatch block)
5. Add it to `brain.yaml`

## Re-ingesting

Running `./ingest.sh` rebuilds the entire parquet from scratch each time. It's idempotent — safe to re-run whenever you get new data.

After re-ingesting, run `./embed.sh` to embed any new messages (incremental — only new messages get embedded).
