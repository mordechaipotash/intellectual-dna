# Experiment 02: Monotropic Prosthetic

## The Question
Can I use a serverless knowledge graph to track my monotropic history and have it available on-hand as a kind of prosthetic memory?

---

## What You Already Built (Dec 2-6, 2025)

**Location**: `/Users/mordechai/Mordechai Dev 2025/Sparkii/mordetropic/docs/genesis-2025-12-02/`

### Already Done:
1. **FalkorDB + Graphiti Python environment** - `external-brain-env/` with both installed
2. **Entity extraction script** - `extract_entities_to_falkordb_04-11-25-21-05.py` that:
   - Connects to Supabase to pull conversation semantics
   - Extracts: Technology, Problem, Solution, Concept nodes
   - Creates relationships: USES, DISCUSSES, ADDRESSES, PROPOSES, HELPS_SOLVE, SOLVED_BY, RELATES_TO
3. **KG Landscape Analysis** - `KG-LANDSCAPE-ANALYSIS-DEC-2025.md` (comprehensive market research)
4. **Brain backups** - `brain_backup_latest.json` (633KB of extracted knowledge)
5. **Seeds** - `mordetropic-seed-comprehensive-v1.json`, `SEED-MORDETROPIC-128KB-MASTER.json`

### The Architecture You Designed:
```
Supabase (conversations + embeddings)
    ↓
FalkorDB (knowledge graph layer)
    ↓
GraphRAG (hybrid search: vector + graph)
    ↓
Query: "What was I working on before X?"
```

### Key Insight from Your Nov 4 Analysis:
> "FalkorDB is a **complementary technology** - keep pgvector for vector search, add FalkorDB for knowledge graph relationships. Best of both worlds."

---

## The Gap: FalkorDBLite for Local/Portable

What you built requires Docker + server. **FalkorDBLite** would let you:
- Run embedded (no server)
- Portable `.db` file (iCloud syncable)
- Same API as your server code
- Works offline

---

## The Tools

### Option 1: FalkorDBLite (Embedded, Zero-Config)

**What it is**: Embedded Python graph database. No server, no Docker, just `pip install`.

```bash
pip install falkordblite
```

**Requirements**: Python >=3.12, libomp (`brew install libomp` on Mac)

**How it works**:
- Runs as subprocess with Unix socket communication
- Zero network latency
- File-based persistence (like SQLite for graphs)
- OpenCypher query language

**Basic usage**:
```python
from redislite.falkordb_client import FalkorDB

db = FalkorDB('/tmp/monotropic.db')
g = db.select_graph('focus_history')

# Track a focus session
g.query('''
CREATE (f:Focus {
    topic: "SeedGarden HPI",
    started: datetime(),
    intensity: 9,
    context: "transportable context webhook design"
})
RETURN f
''')

# Link related focuses
g.query('''
MATCH (f1:Focus {topic: "SeedGarden HPI"}),
      (f2:Focus {topic: "intellectual DNA extraction"})
CREATE (f1)-[r:INTERRUPTED_BY]->(f2)
RETURN r
''')
```

**Pros**:
- Zero infrastructure
- Works in Jupyter notebooks
- Local file = portable
- Same API as production FalkorDB

**Cons**:
- Python 3.12+ only
- Mac needs libomp
- Not cloud-synced by default

---

### Option 2: Graphiti + Zep (Temporal Knowledge Graph)

**What it is**: Framework for temporally-aware knowledge graphs. Built for AI agent memory.

```bash
pip install graphiti-core
# or with FalkorDB backend:
pip install graphiti-core-falkordb
```

**The killer feature**: **Bi-temporal model** - tracks when something happened AND when you learned about it.

**Why this matters for monotropic tracking**:
- Records when you switched focus (event time)
- Records when you realized you switched (ingestion time)
- Can query "what was I working on last Tuesday at 3pm?"
- Can query "what did I know about X before I got distracted?"

**Performance**:
- P95 latency: 300ms
- 94.8% accuracy on memory retrieval benchmark
- Hybrid search: semantic + keyword + graph traversal

**MCP Server**: There's an [MCP server for Graphiti](https://github.com/getzep/graphiti) that gives Claude direct access to your knowledge graph.

---

### Option 3: Obsidian + Graph View (Manual but ADHD-Friendly)

**What it is**: Markdown files with backlinks, visualized as a graph.

**ADHD-specific insight from research**:
> "A second brain acts as an external hard drive for your mind. It's not just about dumping notes; it's about designing a system that catches every fleeting spark and hands it back when you need fire."

**Tools mentioned**:
- [Saner.AI](https://www.saner.ai/) - ADHD-focused, captures notes/emails/tasks/Slack, AI organizes
- [Constella App](https://www.kosmik.app/blog/best-second-brain-apps) - Infinite canvas, auto-organization, AI assistant

---

## The ADHD Pattern Recognition Insight

From [ADDPMP (Attention Deficit Disorder Prosthetic Memory Program)](https://www.spaziomaiocchi.com/addpmp-attention-deficit-disorder-prosthetic-memory-program/):

> "There's a pattern in chaos. For a long time I believed that ADHD made me think in a very disorganised way — making these books helped me figure out a formula I've been unconsciously applying to my work since day one."

This is exactly what you're trying to do:
- Track the chaos (monotropic focus jumps)
- Find the patterns
- Have them on-hand as prosthetic memory
- Use the patterns to predict/guide future focus

---

## Architecture for Monotropic Prosthetic

```
┌─────────────────────────────────────────────────────────────┐
│                    MONOTROPIC PROSTHETIC                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INPUT LAYER (captures focus switches)                      │
│  ├── Claude conversation exports (.jsonl)                   │
│  ├── Manual annotations ("switching to X")                  │
│  ├── Time-based triggers (what were you doing at 3am?)      │
│  └── Project activity (git commits, file changes)           │
│                                                             │
│  STORAGE LAYER (temporal knowledge graph)                   │
│  ├── FalkorDBLite (local, embedded, portable)               │
│  └── Graphiti (bi-temporal, conflict resolution)            │
│                                                             │
│  QUERY LAYER (prosthetic memory retrieval)                  │
│  ├── "What was I working on before X?"                      │
│  ├── "What patterns emerge in my focus history?"            │
│  ├── "What got interrupted and never resumed?"              │
│  └── "What's the relationship between X and Y projects?"    │
│                                                             │
│  OUTPUT LAYER (available on-hand)                           │
│  ├── MCP server → Claude has direct access                  │
│  ├── CLI tool → query from terminal                         │
│  └── Webhook → other tools can query                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Experiment: Minimum Viable Prosthetic

### Step 1: Install FalkorDBLite
```bash
brew install libomp  # Mac only
pip install falkordblite
```

### Step 2: Create Focus Tracking Schema
```cypher
// Node types
(:Focus {topic, started, ended, intensity, context, interrupted_by})
(:Project {name, description, status})
(:Insight {content, discovered, source_focus})
(:Connection {from_topic, to_topic, relationship_type})

// Relationships
(Focus)-[:PART_OF]->(Project)
(Focus)-[:INTERRUPTED_BY]->(Focus)
(Focus)-[:GENERATED]->(Insight)
(Focus)-[:CONNECTED_TO]->(Focus)
```

### Step 3: Build Ingestion from Claude Exports
Your `.jsonl` conversation exports contain the raw material:
- Topic switches (detected by content analysis)
- Timestamps (when each message happened)
- Intensity signals (message length, question depth)
- Context (what was being discussed)

### Step 4: Query Interface
```python
def what_was_i_working_on(before_topic):
    """Find what got interrupted when I started this topic"""
    return g.query(f'''
        MATCH (f1:Focus)-[:INTERRUPTED_BY]->(f2:Focus {{topic: "{before_topic}"}})
        RETURN f1.topic, f1.context, f1.started
        ORDER BY f1.started DESC
    ''')

def find_patterns():
    """Find recurring topic clusters"""
    return g.query('''
        MATCH (f1:Focus)-[:CONNECTED_TO]-(f2:Focus)
        RETURN f1.topic, f2.topic, count(*) as frequency
        ORDER BY frequency DESC
        LIMIT 10
    ''')
```

---

## The Vision

You're not building a note-taking app. You're building a **cognitive prosthetic** that:

1. **Captures** your monotropic focus history automatically
2. **Finds patterns** in your chaos (the ADDPMP insight)
3. **Retrieves** context when you need it ("what was I working on?")
4. **Predicts** likely interruption paths ("you usually switch to X after Y")
5. **Is transportable** - the graph file goes with you

This is the **infrastructure** for the HPI concept - tracking cognitive architecture through observed patterns, not surveys.

---

## Sources

- [FalkorDBLite PyPI](https://pypi.org/project/falkordblite/)
- [FalkorDBLite Blog](https://www.falkordb.com/blog/falkordblite-embedded-python-graph-database/)
- [Graphiti GitHub](https://github.com/getzep/graphiti)
- [Zep Paper (arXiv)](https://arxiv.org/abs/2501.13956)
- [Graphiti + FalkorDB](https://blog.getzep.com/graphiti-knowledge-graphs-falkordb-support/)
- [ADDPMP](https://www.spaziomaiocchi.com/addpmp-attention-deficit-disorder-prosthetic-memory-program/)
- [Saner.AI](https://www.saner.ai/blogs/the-second-brain)
- [Second Brain for ADHD](https://medium.com/@theo-james/second-brain-strategies-for-adhd-users-that-actually-stick-83a785290a08)
