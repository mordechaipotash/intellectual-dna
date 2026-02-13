# Proactive Context Injection — Specification

**The killer feature.** Don't wait for Mordechai to ask "what was I thinking about X?" — detect the topic and inject the save state automatically.

---

## The Mechanism

```
User message arrives
        │
        ▼
┌──────────────────┐
│ Topic Extraction  │  Extract domain/topic from message
│ (keyword + fuzzy) │  Match against 25 canonical domains
└────────┬─────────┘
         │
    ┌────┴────┐
    │ Match?  │
    └────┬────┘
     yes │ no → standard response (no injection)
         │
         ▼
┌──────────────────┐
│ tunnel_state()   │  Load save state for matched domain
│ via Brain MCP    │  thinking_stage, open_questions, concepts
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Format Injection │  1-3 lines of context
│ 💭 prefix        │  Prepend to response
└──────────────────┘
```

---

## When to Inject

### ALWAYS inject (session start)
- First user message of every session
- Query: `tunnel_state` for the topic + `search_summaries` for related context
- Purpose: "Here's where you left off"

### Inject on topic shift
- User mentions a domain they haven't talked about in this session
- Detection: keyword match against 25 canonical domains
- Query: `tunnel_state` for that domain
- Purpose: "You were in the refining stage on this, with 3 open questions"

### Inject on decision language
- "Should I...", "thinking about...", "I need to decide..."
- Query: `alignment_check` + `search_summaries(extract="decisions")`
- Purpose: "You've made 4 previous decisions about this. Last one was X"

### Inject on return language
- "I should get back to...", "revisiting...", "picking up..."
- Query: `context_recovery` (full brief)
- Purpose: "Here's your full re-entry brief"

### Inject on switching language
- "Let me switch to...", "moving on to...", "let me look at..."
- Query: `switching_cost` (current domain → target domain)
- Purpose: "Heads up: you have 5 open questions in the current domain"

### DON'T inject
- Pure commands ("set a reminder", "play music")
- Continuation of same thread (already injected this topic)
- Explicit "don't search"
- Sub-agent/group chat contexts (only main session)

---

## Injection Formats

### Standard (first message / topic shift)
```
💭 *{domain}: {thinking_stage} stage, {n} open questions. 
Last insight: "{quotable_or_insight}". {days} days since last visit.*

[Normal response here]
```

### Decision context
```
💭 *You've made {n} decisions about {topic}. Most recent: "{decision}". 
{alignment_note if relevant}.*

[Normal response here]
```

### Return/recovery (rich)
```
💭 *Re-entering {domain} ({days} days dormant):*
*Stage: {thinking_stage} | {n} open questions | {m} decisions made*
*Top open: "{open_question_1}"*
*Last direction: "{recent_insight}"*

[Normal response here]
```

### Switching warning
```
💭 *Switch cost {domain_a} → {domain_b}: {score}/1.0*
*You'd leave behind {n} open questions in {domain_a}.*
*{domain_b} last touched {days} days ago, {shared} shared concepts.*

[Normal response here]
```

---

## Implementation: Steve (Clawdbot Agent)

This runs in Steve's AGENTS.md auto-inject logic, NOT in the MCP server.
The MCP server provides the tools; Steve orchestrates when to call them.

### Updated AGENTS.md Brain Auto-Inject Section

```markdown
## 🧠 Automatic Context Injection — MANDATORY (v2: Prosthetic-Powered)

### Trigger → Tool Mapping (updated)

| Trigger | Tool | Format |
|---------|------|--------|
| First message of session | `tunnel_state(topic)` | Standard |
| Topic shift (new domain) | `tunnel_state(domain)` | Standard |
| Decision language | `search_summaries(topic, extract="decisions")` + `alignment_check` | Decision |
| Return language ("revisiting X") | `context_recovery(domain)` | Recovery (rich) |
| Switching language ("moving to X") | `switching_cost(current, target)` | Switch warning |
| "What was I working on?" | `open_threads()` | List format |
| "What do I think about X?" | `what_do_i_think(topic)` | Synthesis |

### Domain Detection (keyword → canonical)

Map user message keywords to the 25 canonical domains:

| Keywords | Domain |
|----------|--------|
| torah, talmud, halacha, shabbos, bracha | torah |
| supabase, database, schema, sql, migration | database |
| next.js, react, frontend, css, dashboard | frontend-dev |
| webhook, api, backend, typescript, server | backend-dev |
| deploy, railway, vercel, docker, git | devops |
| data, etl, pipeline, cleaning, parquet | data-engineering |
| python, script | python |
| ai, llm, agent, mcp, embedding, rag | ai-dev |
| prompt, context engineering | prompt-engineering |
| wotc, tax credit | wotc |
| optilab, lens, ophthalm | optilab |
| brain, prosthetic, monotropic, cognitive | cognitive-architecture |
| job, career, resume, linkedin, interview | career |
| business, strategy, marketing, product | business-strategy |
| adhd, medication, health | health |
| ... (remaining domains) | ... |

If no keyword match, fall back to `semantic_search` on the message text.
```

---

## Implementation: mcporter Calls

Steve calls these via mcporter:

```bash
# Standard context injection
mcporter call brain.tunnel_state domain="torah"

# Full recovery
mcporter call brain.context_recovery domain="ai-dev"

# Switching cost
mcporter call brain.switching_cost current_domain="torah" target_domain="ai-dev"

# Open threads overview
mcporter call brain.open_threads

# Trust dashboard (periodic, not per-message)
mcporter call brain.trust_dashboard
```

---

## Proactive Cron Jobs (Background)

### Daily morning brief
```bash
clawdbot cron add \
  --name "morning-brain-brief" \
  --cron "0 9 * * *" \
  --tz "Asia/Jerusalem" \
  --session isolated \
  --message "🧠 Morning brain brief: Run trust_dashboard + dormant_contexts(days_inactive=7) + open_threads(max_domains=5). Summarize what needs attention." \
  --deliver --channel webchat
```

### Weekly thread review
```bash
clawdbot cron add \
  --name "weekly-thread-review" \
  --cron "0 10 * * 0" \
  --tz "Asia/Jerusalem" \
  --session isolated \
  --message "🧵 Weekly thread review: Run dormant_contexts(days_inactive=14) + cognitive_patterns(). What threads are dying? What patterns are emerging?" \
  --deliver --channel webchat
```

---

## The Key Insight

The prosthetic report nails it:

> "The idle state IS the product. When the user is NOT using the tool, the tool is working (by existing as a trusted safety net)."

Proactive injection means:
1. **You never have to ask** — context surfaces automatically
2. **You never lose context** — every domain has a save state
3. **You never switch blind** — switching cost is quantified
4. **You never forget open threads** — dormant contexts surface

The injection ISN'T the value. The **knowledge that injection will happen** is the value. It's the Masicampo & Baumeister finding: knowing you have a plan eliminates intrusive thoughts, even though the plan isn't executed yet.

---

## Phased Rollout

### Phase 1 (now): Tool-based injection
- Steve calls Brain MCP tools on first message + topic shifts
- Manual keyword → domain mapping
- Format: 💭 prefix, 1-3 lines

### Phase 2 (next): Smarter detection
- Use the v6 summary embeddings for fuzzy topic matching
- Detect domain transitions within a conversation
- Inject switching cost warnings proactively

### Phase 3 (future): Ambient awareness
- Background cron checks for dormant contexts
- Morning briefs with open threads
- Weekly cognitive pattern reports
- Trust dashboard as persistent widget

### Phase 4 (aspirational): Context recovery UX
- When returning to a topic, show full recovery brief
- "Last time you were here, you were investigating X. Your hypothesis was Y. Next step was Z."
- The experience should feel like waking up, not searching a database
