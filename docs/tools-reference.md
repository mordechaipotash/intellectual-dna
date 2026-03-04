# Tools Reference

brain-mcp exposes 23 MCP tools organized into five tiers. All tools are available after running the ingest + embed pipeline. Prosthetic tools additionally require summaries.

## Tier Overview

| Tier | Tools | Requires |
|------|-------|----------|
| 🔍 Search | 4 | parquet + embeddings |
| 💬 Conversation | 3 | parquet |
| 🧪 Synthesis | 4 | parquet + embeddings + summaries |
| 📊 Stats | 2 | parquet |
| 🧠 Prosthetic | 8 | summaries |
| 📜 Principles | 2 | principles YAML |

---

## 🔍 Search Tools

### `semantic_search`

Search conversations using vector similarity. Finds messages that are *conceptually* similar even without matching keywords.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Search query |
| `limit` | int | 10 | Max results |

**Example:** `semantic_search(query="how to handle burnout")`

---

### `search_summaries`

Hybrid vector + keyword search across conversation summaries. Supports filtering and extraction modes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Search query |
| `extract` | string | `"summary"` | `"summary"`, `"questions"`, `"decisions"`, or `"quotes"` |
| `limit` | int | 10 | Max results |
| `domain` | string | None | Filter by domain (e.g. `"ai-dev"`) |
| `importance` | string | None | `"breakthrough"`, `"significant"`, `"routine"` |
| `thinking_stage` | string | None | `"exploring"`, `"crystallizing"`, `"refining"`, `"executing"` |
| `source` | string | None | Filter by source |
| `mode` | string | `"hybrid"` | `"hybrid"`, `"vector"`, `"fts"` |

**Examples:**
```
search_summaries(query="API design patterns", extract="decisions")
search_summaries(query="machine learning", domain="ai-dev", importance="breakthrough")
```

---

### `unified_search`

Search across all data sources at once: conversations, GitHub repos, and markdown docs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Search query |
| `limit` | int | 15 | Max results |

---

### `search_docs`

Search markdown documentation files.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | `""` | Search query |
| `filter` | string | None | `"ip"`, `"breakthrough"`, `"deep"`, `"project"`, `"todos"` |
| `project` | string | None | Filter by project name |
| `limit` | int | 10 | Max results |

---

## 💬 Conversation Tools

### `search_conversations`

Full-text keyword search across all conversation messages.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `term` | string | `""` | Search keyword |
| `limit` | int | 15 | Max results |
| `role` | string | None | `"user"` or `"assistant"`. With `role="user"` + empty term: shows recent questions |

---

### `get_conversation`

Retrieve a full conversation by ID.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conversation_id` | string | required | Conversation ID |

---

### `conversations_by_date`

Find all conversations from a specific date.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `date` | string | required | Date in `YYYY-MM-DD` format |
| `limit` | int | 30 | Max results |

---

## 🧪 Synthesis Tools

### `what_do_i_think`

Synthesize your views on a topic from multiple conversations, or find past precedents.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `topic` | string | required | Topic or situation |
| `mode` | string | `"synthesize"` | `"synthesize"` or `"precedent"` |

**Examples:**
```
what_do_i_think(topic="microservices vs monolith")
what_do_i_think(topic="switching jobs", mode="precedent")
```

---

### `alignment_check`

Check whether a decision aligns with your configured principles.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `decision` | string | required | The decision to evaluate |

**Example:** `alignment_check(decision="should I take the contract role?")`

---

### `thinking_trajectory`

Track how your thinking on a concept has evolved over time.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `topic` | string | required | Concept to trace |
| `view` | string | `"full"` | `"full"`, `"velocity"`, or `"first"` |

---

### `what_was_i_thinking`

Snapshot of your intellectual activity during a specific month.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `month` | string | required | Month in `YYYY-MM` format |

---

## 📊 Stats Tools

### `brain_stats`

Brain overview and analytics with multiple views.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `view` | string | `"overview"` | View: `"overview"`, `"domains"`, `"pulse"`, `"conversations"`, `"embeddings"`, `"github"`, `"markdown"` |

---

### `unfinished_threads`

Find conversation threads with open/unresolved questions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `domain` | string | None | Filter by domain |
| `importance` | string | `"significant"` | Minimum importance level |

---

## 🧠 Cognitive Prosthetic Tools

These tools require summaries (run the summarize pipeline). They turn a search engine into a cognitive aid.

### `tunnel_state`

Reconstruct your cognitive save-state for a domain. The "load game" button.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `domain` | string | required | Domain to reconstruct (e.g. `"ai-dev"`) |
| `limit` | int | 10 | Max summaries to analyze |

**Returns:** Thinking stage, open questions, recent decisions, concepts, emotional tone.

---

### `dormant_contexts`

Find abandoned or neglected domains with unresolved questions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_importance` | string | `"significant"` | Minimum importance filter |
| `limit` | int | 20 | Max results |

---

### `context_recovery`

Full re-entry briefing for picking up a domain after time away.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `domain` | string | required | Domain to recover |
| `summary_count` | int | 5 | How many recent summaries to include |

---

### `tunnel_history`

Engagement history and meta-view of a domain over time.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `domain` | string | required | Domain to trace |

---

### `switching_cost`

Quantify the cognitive cost of switching between two domains.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `current_domain` | string | required | Domain you're in now |
| `target_domain` | string | required | Domain you want to switch to |

**Returns:** Switching cost score, open questions you'd leave behind, shared concepts between domains.

---

### `cognitive_patterns`

Analyze when and how you think best, backed by data.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `domain` | string | None | Specific domain, or all |

---

### `open_threads`

Global view of unfinished business across all domains.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit_per_domain` | int | 5 | Max threads per domain |
| `max_domains` | int | 20 | Max domains to scan |

---

### `trust_dashboard`

System-wide proof that the safety net works. Shows coverage, freshness, and completeness.

*No parameters.*

---

## 📜 Principles Tools

### `list_principles`

List all configured principles from your principles YAML file.

*No parameters.*

---

### `get_principle`

Get detailed information about a specific principle.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | required | Principle name (partial match) |
