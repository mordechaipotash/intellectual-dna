# Brain MCP v2 — Full Specification

**Date:** 2026-02-13
**Status:** Planning
**Author:** Steve + Mordechai

---

## Overview

Brain MCP v2 upgrades the existing 31-tool MCP server to leverage v6 structured summaries (9,979 conversations with open questions, decisions, quotables, connections, cognitive patterns, and 25 normalized domains).

### What Changed (v5 → v6)

| Field | v5 | v6 |
|-------|----|----|
| Domains | 4,100 raw labels | 25 canonical categories |
| Open questions | Not tracked | 112,057 across all convos |
| Decisions | Not tracked | 36,565 with context |
| Quotable phrases | Not tracked | 33,732 authentic quotes |
| Connections | Not tracked | Cross-conversation links |
| Cognitive pattern | Not tracked | connecting-domains, iterative-refinement, etc. |
| Problem solving | Not tracked | architectural-redesign, first-principles, etc. |
| Emotional tone | Not tracked | excited, frustrated, reflective, etc. |
| Importance | 3 levels (clean) | 3 levels (normalized) |
| Thinking stage | 4 stages (dirty) | 4 stages (normalized) |
| Concepts | 40K raw | 34K deduplicated, normalized |

### Data Sources

| Source | Location | Format | Records |
|--------|----------|--------|---------|
| v6 Summaries | `data/brain_summaries_v6.parquet` | Parquet | 9,979 |
| v6 Vectors | `vectors/brain_summaries.lance/summary` | LanceDB | 9,979 |
| Raw Messages | `data/all_conversations.parquet` | Parquet | 374K+ |
| Raw Vectors | `vectors/brain.lance/message` | LanceDB | 118K+ |
| GitHub | `data/github_repos.parquet` + `github_commits.parquet` | Parquet | varies |
| Markdown | `vectors/brain.lance/markdown` | LanceDB | varies |

---

## Domain Taxonomy (25 categories)

```
frontend-dev (1,294)     data-engineering (989)    devops (974)
backend-dev (721)        database (717)            ai-dev (665)
business-strategy (574)  torah (451)               other (441)
automation (367)         personal (314)            career (312)
cognitive-architecture (310)  python (289)         document-processing (282)
wotc (250)               health (207)             finance (169)
prompt-engineering (164) web-scraping (107)        documentation (102)
ai-image (100)           ai-strategy (70)          mobile-dev (59)
optilab (51)
```

---

## Tool Inventory

### Tier 1: Core Search (6 tools)

#### 1.1 `search_summaries` ⚡ UPGRADE
Search v6 conversation summaries with hybrid vector + keyword search.

**Parameters:**
- `query` (str, required) — search query
- `limit` (int, default 10) — max results
- `domain` (str, optional) — filter by canonical domain (e.g. "torah", "ai-dev")
- `importance` (str, optional) — "routine", "significant", "breakthrough"
- `thinking_stage` (str, optional) — "exploring", "crystallizing", "refining", "executing"
- `source` (str, optional) — "chatgpt", "claude-code", "claude_desktop", "clawdbot"
- `mode` (str, default "hybrid") — "hybrid", "vector", "fts"

**Returns:** Ranked list with summary, domain, importance, thinking_stage, key concepts, conversation_id.

**Implementation:**
- Query `vectors/brain_summaries.lance/summary` table
- Filters applied as SQL WHERE clauses on LanceDB
- Cross-encoder reranking when available
- Uses v6 normalized domains instead of raw messy ones

**Change from v5:** Updated table path (`brain_summaries.lance/summary` not `brain_summaries/brain_summaries`), new filter parameters (domain, thinking_stage), cleaner output format.

---

#### 1.2 `search_open_questions` 🆕
Find unresolved questions across all conversations.

**Parameters:**
- `query` (str, required) — topic to search questions about
- `domain` (str, optional) — filter by domain
- `limit` (int, default 20) — max results

**Returns:** List of open questions with source conversation context.

**Implementation:**
```python
# DuckDB on parquet
SELECT conversation_id, title, source, domain_primary, 
       open_questions, summary, importance
FROM brain_summaries_v6.parquet
WHERE open_questions NOT LIKE '%none identified%'
  AND open_questions != '[]'
  [AND domain_primary = '{domain}']
ORDER BY importance_rank, msg_count DESC
LIMIT {limit}
```
Then filter questions by semantic similarity to query (embed query, compare to each question text).

**Use case:** "What questions do I still have about WOTC?" → returns all open questions from WOTC conversations.

---

#### 1.3 `search_decisions` 🆕
Find past decisions on a topic.

**Parameters:**
- `query` (str, required) — topic to search decisions about
- `domain` (str, optional) — filter by domain
- `limit` (int, default 20) — max results

**Returns:** List of decisions with context (when, what conversation, what was decided).

**Implementation:** Similar to search_open_questions but filtering on `decisions` field.

**Use case:** "What did I decide about Supabase schema?" → returns all schema decisions chronologically.

---

#### 1.4 `semantic_search` (KEEP)
Search raw messages via LanceDB vectors. Unchanged from v5.

---

#### 1.5 `search_conversations` (KEEP)
Keyword search on raw conversations via DuckDB. Unchanged from v5.

---

#### 1.6 `unified_search` (KEEP)
Cross-source search (conversations + GitHub + markdown). Unchanged from v5.

---

### Tier 2: Synthesis (6 tools)

#### 2.1 `what_do_i_think` ⚡ UPGRADE
Synthesize Mordechai's views on a topic using v6 summaries.

**Parameters:**
- `topic` (str, required)

**Returns:** Synthesized view combining breakthrough + significant conversations, key decisions, open questions, and quotable phrases on the topic.

**Change from v5:** Now pulls structured data (decisions, open_questions, quotable) instead of just raw message snippets. Much richer synthesis.

**Implementation:**
1. Vector search on `brain_summaries.lance/summary` for topic
2. Pull top 20 results, prioritize breakthrough > significant
3. Extract and deduplicate: key_insights, decisions, open_questions, quotable
4. Format as synthesis with sections: Summary → Key Decisions → Open Questions → Authentic Quotes

---

#### 2.2 `quote_me` 🆕
Find authentic Mordechai quotes on a topic.

**Parameters:**
- `topic` (str, required) — what to find quotes about
- `domain` (str, optional) — filter by domain
- `limit` (int, default 10) — max quotes

**Returns:** Quotable phrases with source conversation context.

**Implementation:**
1. Vector search summaries for topic
2. Extract `quotable` field from matching conversations
3. Filter quotes by relevance to topic (simple keyword/semantic match)
4. Return with attribution (conversation title, date, domain)

**Use case:** Building a presentation about bottleneck thesis → get authentic quotes to use.

---

#### 2.3 `open_questions_report` 🆕
Report of unresolved questions in a domain.

**Parameters:**
- `domain` (str, required) — one of the 25 canonical domains
- `thinking_stage` (str, optional) — filter by stage
- `limit` (int, default 30)

**Returns:** Grouped, deduplicated list of open questions in that domain.

**Implementation:**
```python
# Filter parquet by domain, extract all open_questions
# Group by sub-topic (semantic clustering)
# Deduplicate similar questions
# Sort by importance (breakthrough > significant > routine)
```

**Use case:** "What questions are open in cognitive-architecture?" → structured report of what's unresolved.

---

#### 2.4 `decision_trail` 🆕
Timeline of decisions on a topic.

**Parameters:**
- `topic` (str, required)
- `limit` (int, default 20)

**Returns:** Chronological list of decisions related to topic with conversation context.

**Implementation:**
1. Search summaries for topic
2. Extract decisions from matching conversations
3. Sort chronologically (by conversation date from raw parquet)
4. Format as timeline

**Use case:** "Decision trail for Brain MCP" → see how architectural decisions evolved.

---

#### 2.5 `find_precedent` ⚡ UPGRADE
Find similar past situations. Now uses v6 structured data.

**Change from v5:** Returns structured summary + decisions + outcome instead of raw message snippets.

---

#### 2.6 `alignment_check` (KEEP)
Check if a decision aligns with SEED principles. Unchanged.

---

### Tier 3: Analytics (7 tools)

#### 3.1 `domain_map` 🆕
Overview of thinking distribution across all 25 domains.

**Parameters:**
- `source` (str, optional) — filter by source
- `importance` (str, optional) — filter by importance

**Returns:** Domain breakdown with counts, percentages, and top concepts per domain.

**Implementation:**
```python
# DuckDB aggregation
SELECT domain_primary, COUNT(*) as count,
       COUNT(CASE WHEN importance='breakthrough' THEN 1 END) as breakthroughs
FROM brain_summaries_v6.parquet
GROUP BY domain_primary
ORDER BY count DESC
```

---

#### 3.2 `thinking_pulse` 🆕
Current state of thinking across domains — what's crystallizing vs exploring.

**Parameters:**
- `domain` (str, optional) — specific domain or all

**Returns:** Matrix of domain × thinking_stage with counts. Highlights what's crystallizing (ready to ship) vs exploring (early stage).

---

#### 3.3 `cognitive_profile` 🆕
Aggregate view of cognitive patterns and problem-solving approaches.

**Parameters:** none

**Returns:** Distribution of cognitive_pattern, problem_solving_approach, prompting_pattern across all conversations.

**Use case:** Self-awareness tool — "How do I typically solve problems?"

---

#### 3.4 `concept_velocity` (KEEP)
Track how often a term appears over time. Unchanged.

---

#### 3.5 `first_mention` (KEEP)
When a concept first appeared. Unchanged.

---

#### 3.6 `thinking_trajectory` ⚡ UPGRADE
Track how an idea evolved. Now includes v6 thinking_stage progression.

**Change from v5:** Shows thinking stage transitions (exploring → crystallizing → executing) alongside raw message evolution.

---

#### 3.7 `brain_stats` ⚡ UPGRADE
Overview stats. Updated with v6 field counts.

**Change from v5:** Adds domain distribution, open question count, decision count, breakthrough count, thinking stage distribution.

---

### Tier 4: Connections (3 tools)

#### 4.1 `find_connections` 🆕
Find conversations connected by shared concepts.

**Parameters:**
- `conversation_id` (str, required) — starting conversation
- `limit` (int, default 10)

**Returns:** Conversations that share concepts or are listed in `connections_to`.

**Implementation:**
1. Get concepts and connections_to from source conversation
2. Search other conversations for matching concepts
3. Rank by concept overlap count
4. Return with shared concepts highlighted

---

#### 4.2 `concept_cluster` 🆕
Group conversations by shared concepts.

**Parameters:**
- `concept` (str, required) — concept to cluster around
- `limit` (int, default 20)

**Returns:** All conversations mentioning this concept, grouped by domain and thinking stage.

---

#### 4.3 `unfinished_threads` 🆕
Conversations in "exploring" stage with significant open questions.

**Parameters:**
- `domain` (str, optional)
- `importance` (str, optional) — default "significant"

**Returns:** List of conversations that have open questions and are still in exploring/crystallizing stage.

**Use case:** "What threads am I leaving unfinished in ai-dev?" → find things worth revisiting.

---

### Tier 5: Existing (kept as-is) (9 tools)

These tools work on raw data and don't need v6 changes:

| Tool | Purpose |
|------|---------|
| `get_conversation` | Full conversation by ID |
| `conversations_by_date` | What happened on a specific date |
| `find_user_questions` | Recent questions asked |
| `what_was_i_thinking` | Month snapshot |
| `list_principles` / `get_principle` | SEED principles |
| `github_project_timeline` | Repo activity |
| `conversation_project_context` | Conversations mentioning a project |
| `validate_date_with_github` | Verify dates via commits |
| `code_to_conversation` | Cross-reference code + conversations |

### Tier 6: Existing (kept as-is) (6 tools)

| Tool | Purpose |
|------|---------|
| `search_markdown` | Keyword search on markdown corpus |
| `search_ip_docs` | Vector search on IP documents |
| `get_breakthrough_docs` | Docs with BREAKTHROUGH energy |
| `get_deep_docs` | High depth-score documents |
| `get_project_docs` | Docs for a specific project |
| `get_open_todos` | Documents with open TODOs |

### Tier 7: Existing Analytics (kept, minor updates) (5 tools)

| Tool | Purpose |
|------|---------|
| `query_tool_stacks` | Technology stack patterns |
| `query_problem_resolution` | Debugging patterns |
| `query_spend` | Cost breakdown |
| `query_timeline` | Multi-source timeline for a date |
| `query_problem_chains` | Problem resolution chains |

---

## Tool Count Summary

| Category | v5 | v2 | New | Upgraded | Kept |
|----------|----|----|-----|----------|------|
| Core Search | 4 | 6 | 2 | 1 | 3 |
| Synthesis | 4 | 6 | 3 | 2 | 1 |
| Analytics | 5 | 7 | 3 | 2 | 2 |
| Connections | 0 | 3 | 3 | 0 | 0 |
| Raw Data | 6 | 6 | 0 | 0 | 6 |
| Markdown | 5 | 5 | 0 | 0 | 5 |
| GitHub | 4 | 4 | 0 | 0 | 4 |
| Meta/Principles | 2 | 2 | 0 | 0 | 2 |
| Analytics (query_*) | 5 | 5 | 0 | 0 | 5 |
| **Total** | **31** | **40** | **11** | **5** | **24** |

31 → 40 tools (+11 new, 5 upgraded, 24 unchanged)

---

## Implementation Plan

### Phase 1: Foundation (update existing)
1. Update `search_summaries` to use new `brain_summaries.lance/summary` table
2. Update `brain_stats` with v6 fields
3. Update `what_do_i_think` to use structured v6 data
4. Update `find_precedent` and `thinking_trajectory`
5. Test all existing tools still work

### Phase 2: New Search Tools
6. Implement `search_open_questions`
7. Implement `search_decisions`
8. Test with real queries

### Phase 3: New Synthesis Tools
9. Implement `quote_me`
10. Implement `open_questions_report`
11. Implement `decision_trail`

### Phase 4: New Analytics
12. Implement `domain_map`
13. Implement `thinking_pulse`
14. Implement `cognitive_profile`

### Phase 5: Connections
15. Implement `find_connections`
16. Implement `concept_cluster`
17. Implement `unfinished_threads`

### Phase 6: Integration
18. Update mcporter config with new tool descriptions
19. Update TOOLS.md with new tool list
20. Run full test suite
21. Update sync pipeline to maintain v6 summaries for new conversations

---

## Technical Notes

### LanceDB Table Schema (v6 summary)

```
summary (LanceDB table in brain_summaries.lance)
├── conversation_id: string
├── source: string
├── title: string (nullable)
├── summary: string
├── importance: string (routine|significant|breakthrough)
├── domain_primary: string (25 categories)
├── thinking_stage: string (exploring|crystallizing|refining|executing)
├── concepts: string (JSON array)
├── open_questions: string (JSON array)
├── decisions: string (JSON array)
├── quotable: string (JSON array)
├── embedding_text: string
└── vector: float32[768] (nomic-embed-text-v1.5)
```

### Parquet Schema (v6 full)

Same as LanceDB plus:
- `domain_secondary`, `key_insights`, `connections_to`
- `people_mentioned`, `projects_mentioned`
- `emotional_tone`, `privacy_level`
- `prompting_pattern`, `cognitive_pattern`, `problem_solving_approach`
- `content_category`, `msg_count`
- `summary_hash`, `summarized_at`

### Query Patterns

**Vector search (semantic):**
```python
table.search(embedding).where(f"domain_primary = '{domain}'").limit(n)
```

**SQL aggregation (DuckDB on parquet):**
```python
duckdb.sql(f"""
  SELECT domain_primary, thinking_stage, COUNT(*) 
  FROM '{parquet_path}'
  GROUP BY 1, 2
""")
```

**Hybrid (vector + filter + rerank):**
```python
results = table.search(query, query_type="hybrid").where(filters).limit(n*3)
reranked = cross_encoder.predict([(query, r.summary) for r in results])
```

---

## Incremental Sync (for new conversations)

New conversations need to be summarized and added to v6:
1. `sync_clawdbot.py` runs hourly → adds to `all_conversations.parquet`
2. New: detect conversations not in `brain_summaries_v6.parquet`
3. Send to Gemini Flash for v6 summarization (same prompt as initial run)
4. Normalize domains using keyword rules + AI fallback
5. Append to parquet + embed to LanceDB

This should be a new pipeline: `pipelines/sync_summaries_v6.py`

---

## Success Criteria

1. All 31 existing tools still work (backward compatible)
2. `search_summaries` returns cleaner results with domain filters
3. `search_open_questions` finds relevant unresolved questions
4. `quote_me` returns authentic Mordechai quotes
5. `domain_map` gives accurate overview of intellectual landscape
6. `unfinished_threads` surfaces things worth revisiting
7. New tools respond in < 2 seconds (LanceDB + DuckDB)
8. Incremental sync handles new conversations automatically
