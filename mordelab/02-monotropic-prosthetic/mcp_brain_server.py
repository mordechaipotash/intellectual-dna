#!/usr/bin/env python3
"""
Brain MCP Server - Your intellectual DNA, queryable as Claude Code tools.

Core data sources:
- 353K conversation messages (2023-2025) in DuckDB/Parquet
- 106K embedded messages with semantic search (768-dim nomic)
- 8 SEED principles (foundational mental models)
- GitHub repos + commits (live-synced daily)
- GitHub repos + commits (code history)

Start: ./mcp-env/bin/python mcp_brain_server.py
"""

from mcp.server.fastmcp import FastMCP
from pathlib import Path
import duckdb
import lancedb
import json
import sys

# Add parent directories to path for shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import paths from central config
from config import (
    BASE,
    PARQUET_PATH as CONVERSATIONS_PARQUET,
    EMBEDDINGS_DB,
    LANCE_PATH,
    GITHUB_REPOS_PARQUET,
    GITHUB_COMMITS_PARQUET,
    MARKDOWN_PARQUET,
    SEED_PATH as SEED_FILE,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    SUMMARIES_V6_PARQUET,
    SUMMARIES_V6_LANCE,
    SUMMARIES_V6_TABLE,
)

# Embedding model - PRE-WARMED at server startup for instant queries
_embedding_model = None

def get_embedding_model():
    """Get cached embedding model (pre-warmed at startup)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    return _embedding_model

def get_embedding(text: str) -> list[float] | None:
    """Get embedding for text."""
    try:
        model = get_embedding_model()
        embedding = model.encode(text[:8000], convert_to_numpy=True)
        return embedding.tolist()
    except Exception:
        return None

# LAZY LOAD: Model loads on first semantic search (fast MCP startup)
# Pre-warming disabled - was causing 12s startup timeout
# try:
#     print("Pre-warming embedding model...", file=sys.stderr)
#     _embedding_model = get_embedding_model()
#     print("Embedding model ready!", file=sys.stderr)
# except Exception as e:
#     print(f"Warning: Could not pre-warm embedding model: {e}", file=sys.stderr)

# Create MCP server
mcp = FastMCP(
    "brain",
    instructions="""You are interfacing with Mordechai's intellectual DNA.
    This server provides access to:
    - 353K raw conversation messages (2023-2025)
    - 106K embedded messages with semantic search
    - 8 SEED principles (foundational mental models)
    - GitHub repos + commits (live-synced daily)
    - GitHub repos + commits (code history)

    Use these tools to understand what Mordechai thinks, find precedents,
    check alignment with his principles, and mine his conversation history."""
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTION MANAGEMENT (lazy-loaded, cached)
# ═══════════════════════════════════════════════════════════════════════════════

_seed_data = None
_conversations_db = None
_embeddings_db = None
_lance_db = None  # LanceDB for vector search (2026 migration)
_github_db = None
_interpretations_db = None
_markdown_db = None


def get_seed():
    """Load SEED principles (the 8 universal principles)."""
    global _seed_data
    if _seed_data is None:
        if SEED_FILE.exists():
            with open(SEED_FILE) as f:
                _seed_data = json.load(f)
        else:
            _seed_data = {}
    return _seed_data


def get_conversations():
    """Get cached DuckDB connection for conversations."""
    global _conversations_db
    if _conversations_db is None:
        _conversations_db = duckdb.connect()
        _conversations_db.execute(f"""
            CREATE VIEW IF NOT EXISTS conversations
            AS SELECT * FROM read_parquet('{CONVERSATIONS_PARQUET}')
        """)
    return _conversations_db


def get_embeddings_db():
    """Get cached connection to embeddings database with VSS loaded.
    DEPRECATED: Use get_lance_db() instead for 32x smaller storage."""
    global _embeddings_db
    if _embeddings_db is None:
        if not EMBEDDINGS_DB.exists():
            return None
        _embeddings_db = duckdb.connect(str(EMBEDDINGS_DB), read_only=True)
        _embeddings_db.execute("INSTALL vss; LOAD vss;")
    return _embeddings_db


def get_lance_db():
    """Get cached LanceDB connection for vector search.
    2026 migration: 32x smaller than DuckDB VSS (440MB vs 14GB)."""
    global _lance_db
    if _lance_db is None:
        if not LANCE_PATH.exists():
            return None
        _lance_db = lancedb.connect(str(LANCE_PATH))
    return _lance_db


def lance_search(embedding, table: str = "message", limit: int = 10, min_sim: float = 0.0):
    """Search LanceDB with embedding vector.
    Returns list of (conversation_title, content, year, month, similarity) tuples."""
    db = get_lance_db()
    if not db:
        return []
    try:
        tbl = db.open_table(table)
        results = tbl.search(embedding).limit(limit).to_pandas()
        # Convert distance to similarity (LanceDB returns L2 distance)
        output = []
        for _, row in results.iterrows():
            sim = 1 / (1 + row.get('_distance', 0))  # Convert distance to similarity
            if sim >= min_sim:
                output.append((
                    row.get('conversation_title', 'Untitled'),
                    row.get('content', ''),
                    row.get('year', 0),
                    row.get('month', 0),
                    sim
                ))
        return output
    except Exception:
        return []


def lance_count(table: str = "message") -> int:
    """Get row count from LanceDB table."""
    db = get_lance_db()
    if not db:
        return 0
    try:
        return db.open_table(table).count_rows()
    except:
        return 0


_summaries_lance = None
_summaries_db = None

def get_summaries_lance():
    """LanceDB for v6 summary vectors."""
    global _summaries_lance
    if _summaries_lance is None:
        _summaries_lance = lancedb.connect(str(SUMMARIES_V6_LANCE))
    return _summaries_lance

def get_summaries_db():
    """DuckDB for v6 summary parquet queries."""
    global _summaries_db
    if _summaries_db is None:
        _summaries_db = duckdb.connect()
        _summaries_db.execute(f"""
            CREATE VIEW IF NOT EXISTS summaries
            AS SELECT * FROM read_parquet('{SUMMARIES_V6_PARQUET}')
        """)
    return _summaries_db

def _parse_json_field(value):
    """Safely parse a JSON string field from parquet. Returns list or empty list."""
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
        return [str(parsed)]
    except (json.JSONDecodeError, TypeError):
        return [str(value)] if value else []


def get_github_db():
    """Get cached DuckDB connection for GitHub data."""
    global _github_db
    if _github_db is None:
        _github_db = duckdb.connect()
        if GITHUB_REPOS_PARQUET.exists():
            _github_db.execute(f"CREATE VIEW IF NOT EXISTS github_repos AS SELECT * FROM read_parquet('{GITHUB_REPOS_PARQUET}')")
        if GITHUB_COMMITS_PARQUET.exists():
            _github_db.execute(f"CREATE VIEW IF NOT EXISTS github_commits AS SELECT * FROM read_parquet('{GITHUB_COMMITS_PARQUET}')")
    return _github_db


# get_youtube_db() — REMOVED 2026-02-05


def get_interpretations_db():
    """Get cached DuckDB connection for interpretations."""
    global _interpretations_db
    if _interpretations_db is None:
        _interpretations_db = duckdb.connect()
    return _interpretations_db


# ═══════════════════════════════════════════════════════════════════════════════
# SEED PRINCIPLES (the 8 foundational mental models)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def list_principles() -> str:
    """
    List the 8 universal principles from SEED.
    These are Mordechai's foundational mental models.
    """
    seed = get_seed()
    section = seed.get('SECTION_2_THE_EIGHT_UNIVERSAL_PRINCIPLES_DETAILED', {})

    if not section:
        return "SEED principles not loaded"

    output = ["## The 8 Universal Principles\n"]
    for key in sorted(section.keys()):
        principle = section[key]
        name = principle.get('name', 'Unknown')
        definition = principle.get('definition', '')[:150]
        freq = principle.get('frequency_mentions', 0)
        output.append(f"**{key.replace('principle_', '').upper()}. {name}**")
        output.append(f"> {definition}...")
        output.append(f"_Mentions: {freq}_\n")

    return "\n".join(output)


@mcp.tool()
def get_principle(name: str) -> str:
    """
    Get detailed info about a specific SEED principle.
    Names: inversion, compression, bottleneck, agency, seeds, translation, temporal, cognitive_architecture
    """
    seed = get_seed()
    section = seed.get('SECTION_2_THE_EIGHT_UNIVERSAL_PRINCIPLES_DETAILED', {})

    # Find matching principle
    name_lower = name.lower()
    for key, principle in section.items():
        principle_name = principle.get('name', '').lower()
        if name_lower in principle_name or name_lower in key:
            output = [f"## {principle.get('name', 'Unknown')}\n"]
            output.append(f"**Definition**: {principle.get('definition', 'N/A')}\n")
            output.append(f"**First Appearance**: {principle.get('first_appearance', 'N/A')}")
            output.append(f"**Frequency**: {principle.get('frequency_mentions', 0)} mentions\n")

            # Core formula if present
            if 'core_formula' in principle:
                output.append(f"**Formula**: `{principle['core_formula']}`\n")
            if 'implementation_formula' in principle:
                output.append(f"**Implementation**: `{principle['implementation_formula']}`\n")

            # Applications
            apps = principle.get('applications', {})
            if apps:
                output.append("### Applications")
                for domain, app in apps.items():
                    output.append(f"\n**{domain}**:")
                    if isinstance(app, dict):
                        for k, v in app.items():
                            output.append(f"- {k}: {v}")

            # Market positioning
            if 'market_positioning' in principle:
                output.append(f"\n**Market**: {principle['market_positioning']}")
            if 'addressable_market_usd' in principle:
                market = principle['addressable_market_usd']
                output.append(f"**TAM**: ${market:,}")

            return "\n".join(output)

    return f"Principle '{name}' not found. Try: inversion, compression, bottleneck, agency, seeds, translation, temporal, cognitive_architecture"


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSATION MINING TOOLS (353K raw messages)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_conversations(term: str = "", limit: int = 15, role: str = None) -> str:
    """
    Full-text search across all conversation messages.

    Args:
        term: Search term (keyword). If empty with role="user", finds recent user questions.
        limit: Max results (default 15)
        role: Filter by role — "user" for Mordechai's words, "assistant" for AI responses.
              With role="user" and empty term, returns recent questions asked (was find_user_questions).
    """
    con = get_conversations()

    # Special mode: find recent user questions when no term and role=user
    if (not term or not term.strip()) and role == "user":
        results = con.execute("""
            SELECT substr(content, 1, 200) as question,
                   conversation_title, source, created
            FROM conversations
            WHERE has_question = 1 AND role = 'user'
            ORDER BY created DESC
            LIMIT ?
        """, [limit]).fetchall()

        output = ["## Recent Questions Asked\n"]
        for question, title, source, created in results:
            output.append(f"**[{created}]** {question}")
            output.append(f"  _From: {title or 'Untitled'} ({source})_\n")
        return "\n".join(output)

    pattern = f"%{term}%"

    # Parameterized query to prevent SQL injection
    if role:
        results = con.execute("""
            SELECT source, model, conversation_title, role,
                   substr(content, 1, 200) as preview,
                   created, conversation_id
            FROM conversations
            WHERE content ILIKE ? AND role = ?
            ORDER BY created DESC
            LIMIT ?
        """, [pattern, role, limit]).fetchall()
    else:
        results = con.execute("""
            SELECT source, model, conversation_title, role,
                   substr(content, 1, 200) as preview,
                   created, conversation_id
            FROM conversations
            WHERE content ILIKE ?
            ORDER BY created DESC
            LIMIT ?
        """, [pattern, limit]).fetchall()

    if not results:
        return f"No conversations found containing '{term}'"

    output = [f"## Conversations containing '{term}' ({len(results)} found)\n"]
    for source, model, title, msg_role, preview, created, conv_id in results:
        output.append(f"**[{created}]** {title or 'Untitled'}")
        output.append(f"  {msg_role}: {preview}...")
        output.append(f"  _ID: {conv_id[:20]}... | {source}/{model}_\n")
    return "\n".join(output)


@mcp.tool()
def get_conversation(conversation_id: str) -> str:
    """
    Get the full content of a specific conversation by ID.
    Use search_conversations() first to find conversation IDs.
    """
    con = get_conversations()
    messages = con.execute("""
        SELECT role, content, msg_timestamp
        FROM conversations
        WHERE conversation_id = ?
        ORDER BY msg_index ASC
    """, [conversation_id]).fetchall()

    if not messages:
        return f"Conversation not found: {conversation_id}"

    output = [f"## Conversation ({len(messages)} messages)\n"]
    for role, content, ts in messages[:20]:
        output.append(f"### {role.upper()} [{ts}]")
        output.append(str(content)[:1000] if content else "(empty)")
        if content and len(str(content)) > 1000:
            output.append(f"_... ({len(str(content))} chars total)_")
        output.append("")

    if len(messages) > 20:
        output.append(f"_... {len(messages) - 20} more messages_")

    return "\n".join(output)


# find_user_questions — MERGED into search_conversations(term="", role="user")


@mcp.tool()
def conversations_by_date(date: str, limit: int = 30) -> str:
    """
    Get conversations from a specific date (YYYY-MM-DD format).
    """
    con = get_conversations()
    results = con.execute("""
        SELECT DISTINCT conversation_id, conversation_title, source, model, created
        FROM conversations
        WHERE CAST(created AS DATE) = CAST(? AS DATE)
        ORDER BY created DESC
        LIMIT ?
    """, [date, limit]).fetchall()

    if not results:
        return f"No conversations found on {date}"

    output = [f"## Conversations on {date} ({len(results)} found)\n"]
    for conv_id, title, source, model, _ in results:
        output.append(f"- **{title or 'Untitled'}**")
        output.append(f"  {source}/{model} | ID: {conv_id[:20]}...")
    return "\n".join(output)


@mcp.tool()
def brain_stats(view: str = "overview") -> str:
    """
    Brain overview, domain distribution, and thinking pulse.

    Args:
        view: What to display:
            - "overview" (default): Stats across all data sources
            - "domains": Domain breakdown with counts, %, breakthroughs, top concepts (was domain_map)
            - "pulse": Domain × thinking_stage matrix — what's crystallizing vs exploring (was thinking_pulse)
            - "conversations": Detailed conversation stats
            - "embeddings": Embedding coverage stats
            - "github": Repository and commit stats
            - "markdown": Document corpus stats
    """
    view = view.lower().strip()

    if view == "domains":
        return _domain_map_view()
    elif view == "pulse":
        return _thinking_pulse_view()
    elif view == "conversations":
        return _conversation_stats()
    elif view == "embeddings":
        return _embedding_stats()
    elif view == "github":
        return _github_stats()
    elif view == "markdown":
        return _markdown_stats()
    elif view == "overview":
        # Quick overview of all sources
        output = ["## Brain Data Overview\n"]

        # Conversations
        try:
            con = get_conversations()
            total = con.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            user = con.execute("SELECT COUNT(*) FROM conversations WHERE role = 'user'").fetchone()[0]
            output.append(f"**Conversations**: {total:,} messages ({user:,} user)")
        except:
            output.append("**Conversations**: unavailable")

        # Embeddings (LanceDB)
        try:
            embedded = lance_count("message")
            if embedded:
                output.append(f"**Embeddings**: {embedded:,} vectors (768d, LanceDB)")
            else:
                output.append("**Embeddings**: unavailable")
        except:
            output.append("**Embeddings**: unavailable")

        # v6 Summaries
        try:
            sdb = get_summaries_db()
            v6_total = sdb.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
            v6_breakthroughs = sdb.execute("SELECT COUNT(*) FROM summaries WHERE importance = 'breakthrough'").fetchone()[0]
            v6_decisions = sdb.execute("SELECT COUNT(*) FROM summaries WHERE decisions IS NOT NULL AND decisions != '[]' AND decisions NOT LIKE '%none identified%'").fetchone()[0]
            v6_open_q = sdb.execute("SELECT COUNT(*) FROM summaries WHERE open_questions IS NOT NULL AND open_questions != '[]' AND open_questions NOT LIKE '%none identified%'").fetchone()[0]
            output.append(f"**v6 Summaries**: {v6_total:,} conversations summarized")
            output.append(f"  - {v6_breakthroughs:,} breakthroughs | {v6_decisions:,} with decisions | {v6_open_q:,} with open questions")

            # Domain distribution (top 10)
            domains = sdb.execute("""
                SELECT domain_primary, COUNT(*) as cnt
                FROM summaries
                GROUP BY domain_primary
                ORDER BY cnt DESC
                LIMIT 10
            """).fetchall()
            if domains:
                output.append("  - **Top domains**: " + ", ".join(f"{d[0]} ({d[1]})" for d in domains))

            # Thinking stage distribution
            stages = sdb.execute("""
                SELECT thinking_stage, COUNT(*) as cnt
                FROM summaries
                WHERE thinking_stage IS NOT NULL
                GROUP BY thinking_stage
                ORDER BY cnt DESC
            """).fetchall()
            if stages:
                output.append("  - **Thinking stages**: " + ", ".join(f"{s[0]} ({s[1]})" for s in stages))
        except:
            output.append("**v6 Summaries**: unavailable")

        # GitHub
        try:
            if GITHUB_REPOS_PARQUET.exists():
                gdb = get_github_db()
                repos = gdb.execute("SELECT COUNT(*) FROM github_repos").fetchone()[0]
                commits = gdb.execute("SELECT COUNT(*) FROM github_commits").fetchone()[0] if GITHUB_COMMITS_PARQUET.exists() else 0
                output.append(f"**GitHub**: {repos} repos, {commits:,} commits")
        except:
            pass

        # Markdown
        try:
            mdb = get_markdown_db()
            if mdb:
                docs = mdb.execute("SELECT COUNT(*) FROM markdown_docs").fetchone()[0]
                output.append(f"**Markdown Docs**: {docs:,}")
        except:
            pass

        output.append("\n_Use brain_stats(view='...') for details. Views: overview, domains, pulse, conversations, embeddings, github, markdown_")
        return "\n".join(output)
    else:
        return f"Unknown view: {view}. Use: overview, domains, pulse, conversations, embeddings, github, markdown"


def _conversation_stats() -> str:
    """Internal: Get statistics about the raw conversation archive."""
    con = get_conversations()

    total_messages = con.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    total_convs = con.execute("SELECT COUNT(DISTINCT conversation_id) FROM conversations").fetchone()[0]
    date_range = con.execute("SELECT MIN(created), MAX(created) FROM conversations").fetchone()
    sources = con.execute("SELECT source, COUNT(*) FROM conversations GROUP BY source ORDER BY COUNT(*) DESC").fetchall()
    models = con.execute("SELECT model, COUNT(*) FROM conversations GROUP BY model ORDER BY COUNT(*) DESC LIMIT 10").fetchall()

    output = [
        "## Conversation Archive Statistics\n",
        f"**Total Messages**: {total_messages:,}",
        f"**Total Conversations**: {total_convs:,}",
        f"**Date Range**: {date_range[0]} to {date_range[1]}\n",
        "### By Source:"
    ]
    for source, count in sources:
        output.append(f"- {source}: {count:,}")

    output.append("\n### Top Models:")
    for model, count in models:
        output.append(f"- {model}: {count:,}")

    return "\n".join(output)


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPORAL TOOLS (time-based analysis of intellectual evolution)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def what_was_i_thinking(month: str) -> str:
    """
    Time-travel snapshot: What was on Mordechai's mind during a specific month?
    Format: YYYY-MM (e.g., '2024-08' or '2025-07')
    Returns: top themes, activity level, key questions asked, emerging concepts.
    """
    con = get_conversations()

    # Parse month
    try:
        year, mon = month.split('-')
        year, mon = int(year), int(mon)
    except:
        return f"Invalid month format. Use YYYY-MM (e.g., '2024-08')"

    # Activity stats for this month
    stats = con.execute("""
        SELECT
            COUNT(*) as total_msgs,
            COUNT(DISTINCT conversation_id) as convos,
            SUM(CASE WHEN role='user' THEN 1 ELSE 0 END) as user_msgs,
            SUM(CASE WHEN has_question=1 AND role='user' THEN 1 ELSE 0 END) as questions_asked
        FROM conversations
        WHERE year = ? AND month = ?
    """, [year, mon]).fetchone()
    total_msgs, convos, user_msgs, questions = stats

    if total_msgs == 0:
        return f"No data found for {month}"

    # Get overall average for comparison
    avg_query = """
    SELECT AVG(monthly_count) FROM (
        SELECT COUNT(*) as monthly_count
        FROM conversations
        GROUP BY year, month
    )
    """
    avg_monthly = con.execute(avg_query).fetchone()[0] or 0

    # Top conversation titles (proxy for themes)
    top_titles = con.execute("""
        SELECT conversation_title, COUNT(*) as msg_count
        FROM conversations
        WHERE year = ? AND month = ?
          AND conversation_title IS NOT NULL
          AND conversation_title != ''
          AND conversation_title != 'Untitled'
        GROUP BY conversation_title
        ORDER BY msg_count DESC
        LIMIT 10
    """, [year, mon]).fetchall()

    # Sample questions asked
    sample_questions = con.execute("""
        SELECT substr(content, 1, 150) as question, created
        FROM conversations
        WHERE year = ? AND month = ?
          AND role = 'user' AND has_question = 1
        ORDER BY created DESC
        LIMIT 5
    """, [year, mon]).fetchall()

    # Sources breakdown
    sources = con.execute("""
        SELECT source, COUNT(*) as count
        FROM conversations
        WHERE year = ? AND month = ?
        GROUP BY source
        ORDER BY count DESC
    """, [year, mon]).fetchall()

    # Build output
    activity_level = "🔥 HIGH" if total_msgs > avg_monthly * 1.5 else "📊 NORMAL" if total_msgs > avg_monthly * 0.5 else "📉 LOW"

    output = [
        f"## What Was I Thinking: {month}\n",
        f"### Activity Level: {activity_level}",
        f"- **{total_msgs:,}** messages ({int(total_msgs/avg_monthly*100)}% of average)",
        f"- **{convos:,}** conversations",
        f"- **{user_msgs:,}** messages from you",
        f"- **{questions:,}** questions asked\n",
        "### Sources:"
    ]
    for source, count in sources:
        output.append(f"- {source}: {count:,}")

    if top_titles:
        output.append("\n### Top Conversations (themes):")
        for title, count in top_titles[:7]:
            output.append(f"- {title} ({count} msgs)")

    if sample_questions:
        output.append("\n### Sample Questions You Asked:")
        for q, _ in sample_questions:
            output.append(f"- \"{q}...\"")

    return "\n".join(output)


def _concept_velocity_view(term: str, granularity: str = "month") -> str:
    """Internal: Track how often a concept appears over time."""
    con = get_conversations()

    if granularity == "quarter":
        time_group = "year || '-Q' || ((month-1)/3 + 1)"
        time_label = "quarter"
    else:
        time_group = "year || '-' || LPAD(CAST(month AS VARCHAR), 2, '0')"
        time_label = "month"

    pattern = f"%{term}%"
    # Note: time_group is safe as it's built from constants, not user input
    results = con.execute(f"""
        SELECT
            {time_group} as period,
            COUNT(*) as mentions,
            COUNT(DISTINCT conversation_id) as conversations
        FROM conversations
        WHERE content ILIKE ? AND role = 'user'
        GROUP BY {time_group}
        ORDER BY period ASC
    """, [pattern]).fetchall()

    if not results:
        return f"No mentions of '{term}' found in your conversations"

    # Find peak and calculate trend
    max_mentions = max(r[1] for r in results)
    peak_period = [r[0] for r in results if r[1] == max_mentions][0]

    # Calculate simple trend (compare last 3 periods to first 3)
    if len(results) >= 6:
        early_avg = sum(r[1] for r in results[:3]) / 3
        recent_avg = sum(r[1] for r in results[-3:]) / 3
        if recent_avg > early_avg * 1.5:
            trend = "📈 ACCELERATING"
        elif recent_avg < early_avg * 0.5:
            trend = "📉 DECLINING"
        else:
            trend = "➡️ STABLE"
    else:
        trend = "📊 INSUFFICIENT DATA"

    output = [
        f"## Concept Velocity: '{term}'\n",
        f"**Trend**: {trend}",
        f"**Peak**: {peak_period} ({max_mentions} mentions)",
        f"**Total mentions**: {sum(r[1] for r in results)} across {sum(r[2] for r in results)} conversations\n",
        f"### Timeline by {time_label}:"
    ]

    for period, mentions, _ in results:
        bar = "█" * min(mentions, 30)  # Visual bar, max 30 chars
        peak_marker = " ← PEAK" if period == peak_period else ""
        output.append(f"{period}: {mentions:>3} {bar}{peak_marker}")

    return "\n".join(output)


def _first_mention_view(term: str) -> str:
    """Internal: Find when a concept first appeared."""
    con = get_conversations()
    pattern = f"%{term}%"

    # Find first mention
    first = con.execute("""
        SELECT
            created,
            conversation_title,
            conversation_id,
            substr(content, 1, 300) as preview,
            source
        FROM conversations
        WHERE content ILIKE ? AND role = 'user'
        ORDER BY created ASC
        LIMIT 1
    """, [pattern]).fetchone()

    if not first:
        return f"No mentions of '{term}' found in your conversations"

    created, title, conv_id, preview, source = first

    # Count total mentions
    total_mentions, total_convos = con.execute("""
        SELECT COUNT(*), COUNT(DISTINCT conversation_id)
        FROM conversations
        WHERE content ILIKE ? AND role = 'user'
    """, [pattern]).fetchone()

    # Find most recent mention for comparison
    last = con.execute("""
        SELECT created, conversation_title
        FROM conversations
        WHERE content ILIKE ? AND role = 'user'
        ORDER BY created DESC
        LIMIT 1
    """, [pattern]).fetchone()
    last_created, last_title = last

    # Calculate time span
    time_span = last_created - created
    days_span = time_span.days if hasattr(time_span, 'days') else 0

    output = [
        f"## First Mention: '{term}'\n",
        f"### Genesis Moment",
        f"**Date**: {created}",
        f"**Conversation**: {title or 'Untitled'}",
        f"**Source**: {source}",
        f"**Context**:",
        f"> {preview}...\n",
        f"### Journey Since",
        f"- **First**: {str(created)[:10]}",
        f"- **Latest**: {str(last_created)[:10]} ({last_title or 'Untitled'})",
        f"- **Span**: {days_span} days",
        f"- **Total mentions**: {total_mentions} across {total_convos} conversations",
        f"\n_Conversation ID: {conv_id}_"
    ]

    return "\n".join(output)


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHESIS TOOLS (combine semantic + keyword search)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def what_do_i_think(topic: str, mode: str = "synthesize") -> str:
    """
    Synthesize what Mordechai thinks about a topic, or find similar past situations.

    Args:
        topic: The topic or situation to analyze
        mode: Analysis mode:
            - "synthesize" (default): Full synthesis with decisions, open questions, quotes
            - "precedent": Find similar past situations with context and decisions made (was find_precedent)
    """
    if mode == "precedent":
        return _find_precedent_view(topic)
    output = [f"## What do I think about: {topic}\n"]

    # Vector search v6 summaries
    embedding = get_embedding(f"search_query: {topic}")
    if not embedding:
        return "Could not generate embedding for topic."

    try:
        tbl = get_summaries_lance().open_table(SUMMARIES_V6_TABLE)
        results = tbl.search(embedding).limit(20).to_list()
    except Exception as e:
        return f"Summary search error: {e}"

    if not results:
        output.append("_No structured thoughts found on this topic._")
        return "\n".join(output)

    # Prioritize breakthrough > significant > routine
    importance_order = {"breakthrough": 0, "significant": 1, "routine": 2}
    results.sort(key=lambda r: importance_order.get(r.get("importance", "routine"), 2))

    # Collect structured data
    all_decisions = []
    all_open_questions = []
    all_quotables = []
    summaries_shown = 0

    output.append("### Summary of Thinking\n")
    for r in results[:10]:
        title = r.get("title", "Untitled") or "Untitled"
        summary = r.get("summary", "")
        importance = r.get("importance", "?")
        domain = r.get("domain_primary", "?")
        stage = r.get("thinking_stage", "?")
        conv_id = r.get("conversation_id", "?")

        if summary and summaries_shown < 5:
            imp_icon = {"breakthrough": "🔥", "significant": "⭐", "routine": "📝"}.get(importance, "📝")
            output.append(f"{imp_icon} **{title}** [{domain} | {stage}]")
            output.append(f"> {summary[:300]}{'...' if len(summary) > 300 else ''}")
            output.append(f"_ID: {conv_id[:20]}..._\n")
            summaries_shown += 1

        # Collect decisions
        decisions = _parse_json_field(r.get("decisions"))
        for d in decisions:
            if d and "none identified" not in str(d).lower():
                all_decisions.append((d, title, conv_id))

        # Collect open questions
        questions = _parse_json_field(r.get("open_questions"))
        for q in questions:
            if q and "none identified" not in str(q).lower():
                all_open_questions.append((q, title, conv_id))

        # Collect quotables
        quotes = _parse_json_field(r.get("quotable"))
        for q in quotes:
            if q and "none identified" not in str(q).lower():
                all_quotables.append((q, title))

    # Key Decisions
    if all_decisions:
        output.append("### Key Decisions\n")
        seen = set()
        for decision, title, _ in all_decisions[:10]:
            d_key = decision[:80].lower()
            if d_key not in seen:
                seen.add(d_key)
                output.append(f"- {decision[:200]}")
                output.append(f"  _From: {title}_")

    # Still Open
    if all_open_questions:
        output.append("\n### Still Open\n")
        seen = set()
        for question, title, _ in all_open_questions[:8]:
            q_key = question[:80].lower()
            if q_key not in seen:
                seen.add(q_key)
                output.append(f"- {question[:200]}")
                output.append(f"  _From: {title}_")

    # Authentic Quotes
    if all_quotables:
        output.append("\n### Authentic Quotes\n")
        for quote, title in all_quotables[:5]:
            output.append(f"> \"{quote[:250]}\"")
            output.append(f"> — _{title}_\n")

    return "\n".join(output)


def _find_precedent_view(situation: str) -> str:
    """Internal: Find similar situations Mordechai has dealt with before."""
    embedding = get_embedding(f"search_query: {situation}")
    if not embedding:
        return "Could not generate embedding."

    try:
        tbl = get_summaries_lance().open_table(SUMMARIES_V6_TABLE)
        results = tbl.search(embedding).limit(15).to_list()
    except Exception as e:
        return f"Search error: {e}"

    if not results:
        return f"No precedents found for: {situation}"

    output = [f"## Precedents for: {situation}\n"]
    output.append(f"_Found {len(results)} similar past situations_\n")

    for i, r in enumerate(results[:10]):
        title = r.get("title", "Untitled") or "Untitled"
        summary = r.get("summary", "")
        importance = r.get("importance", "?")
        domain = r.get("domain_primary", "?")
        stage = r.get("thinking_stage", "?")
        conv_id = r.get("conversation_id", "?")
        source = r.get("source", "?")

        imp_icon = {"breakthrough": "🔥", "significant": "⭐", "routine": "📝"}.get(importance, "📝")
        output.append(f"### {i+1}. {imp_icon} {title}")
        output.append(f"**Domain**: {domain} | **Stage**: {stage} | **Source**: {source}")
        output.append(f"> {summary[:350]}{'...' if len(summary) > 350 else ''}")

        # Show decisions if present
        decisions = _parse_json_field(r.get("decisions"))
        real_decisions = [d for d in decisions if d and "none identified" not in str(d).lower()]
        if real_decisions:
            output.append("**Decisions made**:")
            for d in real_decisions[:3]:
                output.append(f"  - {d[:150]}")

        output.append(f"_ID: {conv_id[:20]}..._\n")

    return "\n".join(output)


@mcp.tool()
def alignment_check(decision: str) -> str:
    """
    Check if a decision aligns with Mordechai's principles.
    Searches SEED principles and semantic history for guidance.
    """
    decision_lower = decision.lower()
    seed = get_seed()
    output = [f"## Alignment Check: {decision}\n"]

    # 1. SEED PRINCIPLES (the 8 universal principles) - Primary source
    seed_section = seed.get('SECTION_2_THE_EIGHT_UNIVERSAL_PRINCIPLES_DETAILED', {})
    relevant_principles = []
    for _, principle in seed_section.items():
        name = principle.get('name', '').lower()
        definition = principle.get('definition', '').lower()
        if (decision_lower in name or decision_lower in definition or
            any(word in definition for word in decision_lower.split() if len(word) > 4)):
            relevant_principles.append({
                'name': principle.get('name'),
                'definition': principle.get('definition'),
                'formula': principle.get('core_formula') or principle.get('implementation_formula')
            })

    if relevant_principles:
        output.append("### SEED Principles (foundational guidance):\n")
        for p in relevant_principles[:3]:
            output.append(f"**{p['name']}**")
            output.append(f"> {p['definition'][:200]}...")
            if p['formula']:
                output.append(f"_Formula: {p['formula']}_\n")

    # 2. SEMANTIC SEARCH for related past decisions - LanceDB
    embedding = get_embedding(decision)
    if embedding and LANCE_PATH.exists():
        results = lance_search(embedding, limit=5, min_sim=0.35)
        if results:
            output.append("### Related Past Thinking:\n")
            for title, content, year, month, sim in results:
                preview = content[:200] + "..." if len(content) > 200 else content
                output.append(f"**[{year}-{month:02d}]** {title or 'Untitled'} (sim: {sim:.2f})")
                output.append(f"> {preview}\n")

    if len(output) == 1:
        output.append("_No direct alignment guidance found. Try rephrasing or use semantic_search._")

    return "\n".join(output)


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC SEARCH (vector embeddings)
# Uses get_embedding() and get_embeddings_db() from CONNECTION MANAGEMENT above
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
def semantic_search(query: str, limit: int = 10) -> str:
    """
    Search conversations using semantic similarity (vector embeddings).
    Finds messages that are conceptually similar to your query,
    even if they don't contain the exact words.

    More powerful than keyword search for finding related ideas.
    Uses sentence-transformers locally with LanceDB (32x smaller than DuckDB).
    """
    # Check if LanceDB exists
    if not LANCE_PATH.exists():
        return "LanceDB not found. Run migrate_to_lancedb.py first."

    # Get query embedding
    embedding = get_embedding(query)
    if not embedding:
        return "Could not generate embedding for query."

    # Search LanceDB
    db = get_lance_db()
    if not db:
        return "Could not connect to LanceDB."

    try:
        tbl = db.open_table("message")
        results = tbl.search(embedding).limit(limit).to_pandas()
    except Exception as e:
        return f"Search error: {e}"

    if results.empty:
        return "No results found."

    output = [f"## Semantic Search: '{query}'\n"]
    output.append(f"_Found {len(results)} semantically similar messages_\n")

    for i, row in results.iterrows():
        title = row.get('conversation_title') or 'Untitled'
        content = str(row.get('content', ''))
        year = row.get('year', 0)
        month = row.get('month', 0)
        distance = row.get('_distance', 0)

        # Truncate content for display
        preview = content[:400] + "..." if len(content) > 400 else content

        output.append(f"### {i+1}. [{year}-{month:02d}] {title}")
        output.append(f"**Similarity**: {distance:.4f}")
        output.append(f"> {preview}\n")

    return "\n".join(output)


@mcp.tool()
def search_summaries(
    query: str,
    extract: str = "summary",
    limit: int = 10,
    domain: str = None,
    importance: str = None,
    thinking_stage: str = None,
    source: str = None,
    mode: str = "hybrid"
) -> str:
    """
    Search v6 conversation SUMMARIES with hybrid vector + keyword search.

    This is the BEST search tool for finding conversations by topic, concept, or domain.
    Searches structured summaries with 25 normalized domains, concepts, decisions, and insights.

    Args:
        query: Search query
        extract: What to extract from results:
            - "summary" (default): Full summary with metadata
            - "questions": Open questions from matching conversations (was search_open_questions)
            - "decisions": Decisions made in matching conversations (was search_decisions)
            - "quotes": Quotable phrases from matching conversations (was quote_me)
        limit: Max results (default 10)
        domain: Filter by domain (e.g. "torah", "ai-dev", "wotc", "cognitive-architecture")
        importance: Filter by importance ("breakthrough", "significant", "routine")
        thinking_stage: Filter by stage ("exploring", "crystallizing", "refining", "executing")
        source: Filter by source ("claude-code", "chatgpt", "clawdbot", "claude_desktop", "gemini")
        mode: Search mode — "hybrid" (default), "vector", "fts"

    Examples:
        search_summaries("bottleneck amplifier", importance="breakthrough")
        search_summaries("WOTC", extract="decisions")
        search_summaries("Torah", extract="questions", domain="torah")
        search_summaries("monotropic", extract="quotes")
    """
    # Route to specialized extraction if needed
    if extract == "questions":
        return _extract_open_questions(query, domain=domain, importance=importance, thinking_stage=thinking_stage, source=source, limit=limit, mode=mode)
    elif extract == "decisions":
        return _extract_decisions(query, domain=domain, importance=importance, thinking_stage=thinking_stage, source=source, limit=limit, mode=mode)
    elif extract == "quotes":
        return _extract_quotes(query, domain=domain, importance=importance, thinking_stage=thinking_stage, source=source, limit=limit, mode=mode)
    try:
        tbl = get_summaries_lance().open_table(SUMMARIES_V6_TABLE)
    except Exception as e:
        return f"v6 summary table not found ({e}). Check brain_summaries.lance/summary exists."

    # Build SQL filter
    filters = []
    if domain:
        filters.append(f"domain_primary = '{domain}'")
    if importance:
        filters.append(f"importance = '{importance}'")
    if thinking_stage:
        filters.append(f"thinking_stage = '{thinking_stage}'")
    if source:
        filters.append(f"source = '{source}'")
    where_clause = " AND ".join(filters) if filters else None

    try:
        if mode == "hybrid":
            search = tbl.search(query, query_type="hybrid").limit(limit * 3)
            if where_clause:
                search = search.where(where_clause)
            results = search.to_list()
        elif mode == "fts":
            search = tbl.search(query, query_type="fts").limit(limit)
            if where_clause:
                search = search.where(where_clause)
            results = search.to_list()
        else:
            # Vector search
            embedding = get_embedding(f"search_query: {query}")
            if not embedding:
                return "Could not generate embedding."
            search = tbl.search(embedding).limit(limit)
            if where_clause:
                search = search.where(where_clause)
            results = search.to_list()
    except Exception:
        # Fallback to vector-only (hybrid/fts may not have index)
        embedding = get_embedding(f"search_query: {query}")
        if not embedding:
            return "Could not generate embedding."
        search = tbl.search(embedding).limit(limit)
        if where_clause:
            search = search.where(where_clause)
        results = search.to_list()

    if not results:
        return f"No summaries found for: {query}"

    # Cross-encoder reranking (if available)
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, r.get("summary", "")) for r in results]
        scores = reranker.predict(pairs)
        for i, r in enumerate(results):
            r["rerank_score"] = float(scores[i])
        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    except Exception:
        pass  # Reranking optional

    results = results[:limit]

    output = [f"## Summary Search: '{query}'\n"]
    if where_clause:
        output.append(f"_Filters: {where_clause}_\n")
    output.append(f"_Found {len(results)} matching conversation summaries_\n")

    for i, r in enumerate(results):
        title = (r.get("title") or "Untitled")[:80]
        summary = (r.get("summary") or "")[:400]
        imp = r.get("importance", "?")
        domain_p = r.get("domain_primary", "?")
        stage = r.get("thinking_stage", "?")
        src = r.get("source", "?")
        concepts_raw = _parse_json_field(r.get("concepts"))
        concepts_str = ", ".join(concepts_raw[:8]) if concepts_raw else ""
        rerank = r.get("rerank_score")
        conv_id = r.get("conversation_id", "?")

        imp_icon = {"breakthrough": "🔥", "significant": "⭐", "routine": "📝"}.get(imp, "📝")
        output.append(f"### {i+1}. {imp_icon} {title}")
        output.append(f"**Domain**: {domain_p} | **Stage**: {stage} | **Source**: {src} | **Importance**: {imp}")
        if concepts_str:
            output.append(f"**Concepts**: {concepts_str}")
        if rerank is not None:
            output.append(f"**Relevance**: {rerank:.2f} (reranked)")
        output.append(f"> {summary}")
        output.append(f"_Conversation ID: {conv_id}_\n")

    return "\n".join(output)


def _summary_search_core(query, domain=None, importance=None, thinking_stage=None, source=None, limit=10, mode="hybrid"):
    """Shared search logic for all extract modes. Returns list of result dicts."""
    try:
        tbl = get_summaries_lance().open_table(SUMMARIES_V6_TABLE)
    except Exception as e:
        return []

    filters = []
    if domain:
        filters.append(f"domain_primary = '{domain}'")
    if importance:
        filters.append(f"importance = '{importance}'")
    if thinking_stage:
        filters.append(f"thinking_stage = '{thinking_stage}'")
    if source:
        filters.append(f"source = '{source}'")
    where_clause = " AND ".join(filters) if filters else None

    try:
        if mode == "hybrid":
            search = tbl.search(query, query_type="hybrid").limit(limit * 3)
            if where_clause:
                search = search.where(where_clause)
            return search.to_list()
        elif mode == "fts":
            search = tbl.search(query, query_type="fts").limit(limit * 2)
            if where_clause:
                search = search.where(where_clause)
            return search.to_list()
        else:
            embedding = get_embedding(f"search_query: {query}")
            if not embedding:
                return []
            search = tbl.search(embedding).limit(limit * 2)
            if where_clause:
                search = search.where(where_clause)
            return search.to_list()
    except Exception:
        embedding = get_embedding(f"search_query: {query}")
        if not embedding:
            return []
        search = tbl.search(embedding).limit(limit * 2)
        if where_clause:
            search = search.where(where_clause)
        return search.to_list()


def _extract_open_questions(query, **kwargs):
    """Extract open questions from matching summaries."""
    limit = kwargs.pop("limit", 20)
    results = _summary_search_core(query, limit=limit, **kwargs)
    if not results:
        return f"No results found for: {query}"

    output = [f"## Open Questions about: {query}\n"]
    question_count = 0
    for r in results:
        questions = _parse_json_field(r.get("open_questions"))
        real_questions = [q for q in questions if q and "none identified" not in str(q).lower()]
        if not real_questions:
            continue
        title = r.get("title", "Untitled") or "Untitled"
        domain_p = r.get("domain_primary", "?")
        importance = r.get("importance", "?")
        conv_id = r.get("conversation_id", "?")
        summary = (r.get("summary") or "")[:150]
        output.append(f"### {title}")
        output.append(f"_Domain: {domain_p} | Importance: {importance} | ID: {conv_id[:20]}..._")
        output.append(f"_Context: {summary}..._\n")
        for q in real_questions:
            output.append(f"  ❓ {q[:250]}")
            question_count += 1
        output.append("")
        if question_count >= limit:
            break
    if question_count == 0:
        output.append("_No open questions found matching this topic._")
    else:
        output.append(f"\n_Total: {question_count} open questions found_")
    return "\n".join(output)


def _extract_decisions(query, **kwargs):
    """Extract decisions from matching summaries."""
    limit = kwargs.pop("limit", 20)
    results = _summary_search_core(query, limit=limit, **kwargs)
    if not results:
        return f"No results found for: {query}"

    output = [f"## Decisions about: {query}\n"]
    decision_count = 0
    for r in results:
        decisions = _parse_json_field(r.get("decisions"))
        real_decisions = [d for d in decisions if d and "none identified" not in str(d).lower()]
        if not real_decisions:
            continue
        title = r.get("title", "Untitled") or "Untitled"
        domain_p = r.get("domain_primary", "?")
        imp = r.get("importance", "?")
        stage = r.get("thinking_stage", "?")
        conv_id = r.get("conversation_id", "?")
        source = r.get("source", "?")
        imp_icon = {"breakthrough": "🔥", "significant": "⭐", "routine": "📝"}.get(imp, "📝")
        output.append(f"### {imp_icon} {title}")
        output.append(f"_Domain: {domain_p} | Stage: {stage} | Source: {source}_")
        for d in real_decisions:
            output.append(f"  ✅ {d[:250]}")
            decision_count += 1
        output.append(f"_ID: {conv_id[:20]}..._\n")
        if decision_count >= limit:
            break
    if decision_count == 0:
        output.append("_No decisions found matching this topic._")
    else:
        output.append(f"\n_Total: {decision_count} decisions found_")
    return "\n".join(output)


def _extract_quotes(query, **kwargs):
    """Extract quotable phrases from matching summaries."""
    limit = kwargs.pop("limit", 10)
    results = _summary_search_core(query, limit=limit, **kwargs)
    if not results:
        return f"No results found for: {query}"

    output = [f"## Quotes from Mordechai on: {query}\n"]
    quote_count = 0
    for r in results:
        quotes = _parse_json_field(r.get("quotable"))
        real_quotes = [q for q in quotes if q and "none identified" not in str(q).lower()]
        if not real_quotes:
            continue
        title = r.get("title", "Untitled") or "Untitled"
        domain_p = r.get("domain_primary", "?")
        source = r.get("source", "?")
        for q in real_quotes:
            output.append(f"> \"{q[:300]}\"")
            output.append(f"> — _{title}_ ({domain_p}, {source})\n")
            quote_count += 1
            if quote_count >= limit:
                break
        if quote_count >= limit:
            break
    if quote_count == 0:
        output.append("_No quotable phrases found matching this topic._")
    else:
        output.append(f"_Total: {quote_count} quotes found_")
    return "\n".join(output)


def _search_ip_docs_view(query: str, limit: int = 10) -> str:
    """Internal: Vector search on curated IP documents."""
    if not LANCE_PATH.exists():
        return "LanceDB not found. Run migrate_to_lancedb.py first."

    embedding = get_embedding(query)
    if not embedding:
        return "Could not generate embedding."

    db = get_lance_db()
    if not db:
        return "Could not connect to LanceDB."

    try:
        tbl = db.open_table("markdown")
        results = tbl.search(embedding).limit(limit).to_pandas()
    except Exception as e:
        return f"Search error: {e}"

    if results.empty:
        return "No IP documents found."

    output = [f"## IP Document Search: '{query}'\n"]
    output.append(f"_Found {len(results)} matching intellectual property documents_\n")

    for i, row in results.iterrows():
        filename = row.get('filename', 'Unknown')
        ip_type = row.get('ip_type', 'unknown')
        depth = row.get('depth_score', 0)
        energy = row.get('energy', 'unknown')
        words = row.get('word_count', 0)
        preview = row.get('content_preview', '')[:300]
        distance = row.get('_distance', 0)

        output.append(f"### {i+1}. {filename}")
        output.append(f"**Type**: {ip_type} | **Depth**: {depth} | **Energy**: {energy} | **Words**: {words:,}")
        output.append(f"**Similarity**: {distance:.4f}")
        output.append(f"> {preview}...\n")

    return "\n".join(output)


@mcp.tool()
def thinking_trajectory(topic: str, view: str = "full") -> str:
    """
    Track the evolution of thinking about a topic over time.

    Args:
        topic: The concept/term to track
        view: What to show:
            - "full" (default): Complete trajectory with genesis, temporal pattern, semantic matches, thinking stages
            - "velocity": How often the concept appears over time with trend analysis (was concept_velocity)
            - "first": When the concept first appeared — the genesis moment (was first_mention)
    """
    if view == "velocity":
        return _concept_velocity_view(topic)
    elif view == "first":
        return _first_mention_view(topic)
    output = [f"## Thinking Trajectory: '{topic}'\n"]

    # 1. Get semantic matches from embeddings - LanceDB
    embedding = get_embedding(topic)
    semantic_results = []

    if embedding and LANCE_PATH.exists():
        lance_results = lance_search(embedding, limit=20, min_sim=0.3)
        # Reformat to match expected tuple structure (year, month, title, content, sim)
        semantic_results = [(r[2], r[3], r[0], r[1], r[4]) for r in lance_results]

    # 2. Get keyword matches with temporal distribution
    conv_con = get_conversations()
    pattern = f"%{topic}%"
    temporal_dist = conv_con.execute("""
        SELECT
            strftime(created, '%Y-%m') as period,
            COUNT(*) as mentions
        FROM conversations
        WHERE content ILIKE ? AND role = 'user'
        GROUP BY period
        ORDER BY period
    """, [pattern]).fetchall()

    # 3. Find first mention
    first_mention = conv_con.execute("""
        SELECT created, conversation_title, substr(content, 1, 200) as preview
        FROM conversations
        WHERE content ILIKE ? AND role = 'user'
        ORDER BY created ASC
        LIMIT 1
    """, [pattern]).fetchone()

    # 4. Build output
    if first_mention:
        output.append("### Genesis")
        output.append(f"**First appeared**: {str(first_mention[0])[:10]}")
        output.append(f"**Context**: {first_mention[1] or 'Untitled'}")
        output.append(f"> {first_mention[2]}...\n")

    if temporal_dist:
        output.append("### Temporal Pattern")
        total = sum(t[1] for t in temporal_dist)
        peak = max(temporal_dist, key=lambda x: x[1])
        output.append(f"**Total keyword mentions**: {total}")
        output.append(f"**Peak period**: {peak[0]} ({peak[1]} mentions)")

        # Show recent trend
        output.append("\n**Recent activity**:")
        for period, count in temporal_dist[-6:]:
            bar = "█" * min(count, 20)
            output.append(f"  {period}: {bar} ({count})")

    if semantic_results:
        output.append("\n### Semantically Related Thoughts")
        output.append("_Messages conceptually similar, even without exact keyword match_\n")

        # Group by year-month
        by_period = {}
        for year, month, title, content, sim in semantic_results[:10]:
            period = f"{year}-{month:02d}"
            if period not in by_period:
                by_period[period] = []
            by_period[period].append((title, content[:150], sim))

        for period in sorted(by_period.keys(), reverse=True)[:4]:
            items = by_period[period]
            output.append(f"**{period}** ({len(items)} related thoughts):")
            for title, preview, sim in items[:2]:
                output.append(f"  - [{sim:.2f}] {title or 'Untitled'}: {preview}...")

    # 5. v6 Thinking Stage Progression
    try:
        v6_embedding = get_embedding(f"search_query: {topic}")
        if v6_embedding:
            tbl = get_summaries_lance().open_table(SUMMARIES_V6_TABLE)
            v6_results = tbl.search(v6_embedding).limit(20).to_list()
            if v6_results:
                stage_order = {"exploring": 0, "crystallizing": 1, "refining": 2, "executing": 3}
                stage_items = []
                for r in v6_results:
                    stage = r.get("thinking_stage", "")
                    if stage in stage_order:
                        stage_items.append((
                            stage_order[stage],
                            stage,
                            r.get("title", "Untitled") or "Untitled",
                            r.get("importance", "?"),
                            r.get("domain_primary", "?")
                        ))
                if stage_items:
                    stage_items.sort(key=lambda x: x[0])
                    output.append("\n### Thinking Stage Progression (v6)")
                    stage_icons = {"exploring": "🔍", "crystallizing": "💎", "refining": "🔧", "executing": "🚀"}
                    current_stage = None
                    for _, stage, title, imp, domain in stage_items:
                        if stage != current_stage:
                            current_stage = stage
                            output.append(f"\n{stage_icons.get(stage, '📝')} **{stage.upper()}**")
                        output.append(f"  - {title} [{domain} | {imp}]")
    except Exception:
        pass

    if not (temporal_dist or semantic_results):
        output.append("_No trajectory data found for this topic._")

    return "\n".join(output)


def _embedding_stats() -> str:
    """Internal: Get statistics about the embeddings database (LanceDB)."""
    if not LANCE_PATH.exists():
        return "LanceDB not found. Run migrate_to_lancedb.py to create it."

    db = get_lance_db()
    if not db:
        return "Could not connect to LanceDB."

    try:
        tbl = db.open_table("message")
        total = tbl.count_rows()
        # Get by-year stats
        df = tbl.to_pandas()[['year']].value_counts().reset_index()
        df.columns = ['year', 'count']
        by_year = sorted(df.values.tolist(), key=lambda x: x[0])
    except Exception as e:
        return f"Error: {e}"

    output = [
        "## Embedding Statistics (LanceDB)\n",
        f"**Total embedded messages**: {total:,}",
        f"**Embedding model**: {EMBEDDING_MODEL}",
        f"**Dimensions**: {EMBEDDING_DIM}",
        f"**Storage**: ~440MB (32x smaller than DuckDB VSS)\n",
        "### By Year:"
    ]

    for year, count in by_year:
        output.append(f"- {int(year)}: {count:,} messages")

    return "\n".join(output)


# ═══════════════════════════════════════════════════════════════════════════════
# GITHUB INTEGRATION TOOLS (cross-reference code with conversations)
# Uses get_github_db() from CONNECTION MANAGEMENT above
# ═══════════════════════════════════════════════════════════════════════════════

def get_github_repos_df():
    """Load GitHub repos as DataFrame (uses cached connection)."""
    if not GITHUB_REPOS_PARQUET.exists():
        return None
    return get_github_db().execute("SELECT * FROM github_repos").fetchdf()


def get_github_commits_df():
    """Load GitHub commits as DataFrame (uses cached connection)."""
    if not GITHUB_COMMITS_PARQUET.exists():
        return None
    return get_github_db().execute("SELECT * FROM github_commits").fetchdf()


@mcp.tool()
def github_search(query: str = "", project: str = None, mode: str = "timeline", limit: int = 10) -> str:
    """
    Search GitHub repos, commits, and cross-reference with conversations.

    Args:
        query: Search query (used for code semantic search or as conversation_id for validate mode)
        project: Project/repo name (used for timeline and conversations modes)
        mode: Search mode:
            - "timeline" (default): Project creation date, commits, activity windows (was github_project_timeline)
            - "conversations": Find conversations mentioning a project (was conversation_project_context)
            - "code": Semantic search across commits AND conversations (was code_to_conversation)
            - "validate": Check conversation date validity via GitHub evidence. Pass conversation_id as query (was validate_date_with_github)
        limit: Max results (default 10)
    """
    if mode == "conversations":
        return _conversation_project_context_view(project or query, limit)
    elif mode == "code":
        return _code_to_conversation_view(query, limit)
    elif mode == "validate":
        return _validate_date_with_github_view(query)

    # Default: timeline mode
    project_name = project or query
    if not GITHUB_REPOS_PARQUET.exists():
        return "GitHub data not imported. Run import_github_data.py first."

    con = get_github_db()
    pattern = f"%{project_name.lower()}%"

    # Find matching repo
    repos = con.execute("""
        SELECT repo_name, created_at, pushed_at, description, language, is_private, stars, url
        FROM github_repos
        WHERE LOWER(repo_name) LIKE ?
        ORDER BY created_at DESC
        LIMIT 5
    """, [pattern]).fetchall()

    if not repos:
        return f"No GitHub project found matching '{project_name}'"

    output = [f"## GitHub Project Timeline: '{project_name}'\n"]

    for repo in repos:
        name, created, pushed, desc, lang, private, stars, url = repo
        output.append(f"### {name}")
        output.append(f"**Created**: {str(created)[:10]}")
        output.append(f"**Last pushed**: {str(pushed)[:10] if pushed else 'N/A'}")
        output.append(f"**Language**: {lang or 'N/A'}")
        output.append(f"**Private**: {'Yes' if private else 'No'}")
        output.append(f"**Stars**: {stars}")
        if desc:
            output.append(f"**Description**: {desc[:100]}")
        output.append(f"**URL**: {url}\n")

        # Get commits for this repo
        if GITHUB_COMMITS_PARQUET.exists():
            commits = con.execute("""
                SELECT timestamp, message, author
                FROM github_commits
                WHERE repo_name = ?
                ORDER BY timestamp DESC
                LIMIT 10
            """, [name]).fetchall()

            if commits:
                output.append("**Recent Commits**:")
                for ts, msg, _ in commits:
                    msg_preview = msg.split('\n')[0][:60]
                    output.append(f"  - [{str(ts)[:10]}] {msg_preview}...")

                # Commit distribution by month
                monthly = con.execute("""
                    SELECT strftime(timestamp, '%Y-%m') as month, COUNT(*) as count
                    FROM github_commits
                    WHERE repo_name = ?
                    GROUP BY 1
                    ORDER BY 1
                """, [name]).fetchall()

                if monthly:
                    output.append("\n**Activity by Month**:")
                    for month, count in monthly[-6:]:
                        bar = "█" * min(count, 20)
                        output.append(f"  {month}: {bar} ({count})")

    return "\n".join(output)


def _conversation_project_context_view(project: str, limit: int = 10) -> str:
    """Internal: Find conversations mentioning a specific GitHub project."""
    con = get_conversations()
    pattern = f"%{project.lower()}%"

    # Search for project mentions
    results = con.execute("""
        SELECT
            conversation_title,
            substr(content, 1, 250) as preview,
            created,
            role,
            conversation_id
        FROM conversations
        WHERE (LOWER(content) LIKE ? OR LOWER(conversation_title) LIKE ?)
          AND role = 'user'
        ORDER BY created DESC
        LIMIT ?
    """, [pattern, pattern, limit]).fetchall()

    if not results:
        return f"No conversations found mentioning '{project}'"

    # Get project creation date for validation
    project_created = None
    if GITHUB_REPOS_PARQUET.exists():
        repo_result = get_github_db().execute("""
            SELECT repo_name, created_at
            FROM github_repos
            WHERE LOWER(repo_name) LIKE ?
            LIMIT 1
        """, [pattern]).fetchone()
        if repo_result:
            project_created = repo_result[1]

    output = [f"## Conversations about: '{project}'\n"]

    if project_created:
        output.append(f"_GitHub project created: {str(project_created)[:10]}_\n")

    for title, preview, created, _, conv_id in results:
        # Check if conversation predates project
        date_flag = ""
        if project_created:
            try:
                if created < project_created:
                    date_flag = " ⚠️ PREDATES PROJECT"
            except:
                pass

        output.append(f"**[{str(created)[:10]}]** {title or 'Untitled'}{date_flag}")
        output.append(f"> {preview}...")
        output.append(f"_ID: {conv_id[:20]}..._\n")

    return "\n".join(output)


def _code_to_conversation_view(query: str, limit: int = 10) -> str:
    """Internal: Semantic search across BOTH commits and conversations."""
    embedding = get_embedding(query)
    if not embedding:
        return "Could not generate embedding. Is Ollama running?"

    output = [f"## Code ↔ Conversation Search: '{query}'\n"]

    # 1. Search commit embeddings - LanceDB
    if LANCE_PATH.exists():
        db = get_lance_db()
        if db:
            try:
                # Search commits table if exists
                if "commit" in db.table_names():
                    tbl = db.open_table("commit")
                    commit_df = tbl.search(embedding).limit(limit // 2).to_pandas()
                    if not commit_df.empty:
                        output.append("### Related Commits")
                        for _, row in commit_df.iterrows():
                            repo = row.get('repo_name', 'unknown')
                            msg = str(row.get('message', ''))
                            ts = row.get('timestamp', '')
                            sim = 1 / (1 + row.get('_distance', 0))
                            msg_preview = msg.split('\n')[0][:80]
                            output.append(f"**[{repo}]** {msg_preview}")
                            output.append(f"  {str(ts)[:10]} | Similarity: {sim:.3f}\n")

                # Search message embeddings
                conv_results = lance_search(embedding, limit=limit // 2)
                if conv_results:
                    output.append("### Related Conversations")
                    for title, content, year, month, sim in conv_results:
                        preview = content[:150]
                        output.append(f"**[{year}-{month:02d}]** {title or 'Untitled'}")
                        output.append(f"> {preview}...")
                        output.append(f"Similarity: {sim:.3f}\n")
            except Exception as e:
                output.append(f"_Search error: {e}_")

    if len(output) == 1:
        output.append("_No embeddings found. Run embed_commits.py and embed_messages.py first._")

    return "\n".join(output)


def _validate_date_with_github_view(conversation_id: str) -> str:
    """Internal: Check conversation date validity via GitHub evidence."""
    import re

    con = get_conversations()

    # Get conversation details
    conv = con.execute("""
        SELECT
            conversation_title,
            MIN(created) as first_msg,
            MAX(created) as last_msg,
            COUNT(*) as msg_count,
            MAX(timestamp_is_fallback) as has_fallback
        FROM conversations
        WHERE conversation_id = ?
        GROUP BY conversation_title
    """, [conversation_id]).fetchone()

    if not conv:
        return f"Conversation not found: {conversation_id}"

    title, first_msg, _, msg_count, has_fallback = conv

    output = [f"## Date Validation: {title or 'Untitled'}\n"]
    output.append(f"**Conversation ID**: {conversation_id[:30]}...")
    output.append(f"**Recorded date**: {str(first_msg)[:10]}")
    output.append(f"**Messages**: {msg_count}")
    output.append(f"**Fallback timestamp**: {'Yes ⚠️' if has_fallback else 'No ✓'}\n")

    # Get content to scan for project mentions
    content = con.execute("""
        SELECT content FROM conversations
        WHERE conversation_id = ? AND role = 'user'
        LIMIT 50
    """, [conversation_id]).fetchall()

    all_content = ' '.join([c[0] for c in content if c[0]])

    # Load repos for validation
    if not GITHUB_REPOS_PARQUET.exists():
        output.append("_GitHub data not available for validation._")
        return "\n".join(output)

    repos = get_github_db().execute("""
        SELECT repo_name, created_at
        FROM github_repos
    """).fetchall()

    # Check for project mentions
    issues = []
    validations = []

    for repo_name, created_at in repos:
        # Check if repo is mentioned in conversation
        if re.search(rf'\b{re.escape(repo_name)}\b', all_content, re.IGNORECASE):
            if first_msg < created_at:
                issues.append({
                    'project': repo_name,
                    'project_created': str(created_at)[:10],
                    'conv_date': str(first_msg)[:10],
                    'days_before': (created_at - first_msg).days
                })
            else:
                validations.append({
                    'project': repo_name,
                    'project_created': str(created_at)[:10]
                })

    if issues:
        output.append("### ⚠️ Date Conflicts Found")
        output.append("_Conversation mentions projects that didn't exist yet_\n")
        for issue in issues:
            output.append(f"- **{issue['project']}** created {issue['project_created']}")
            output.append(f"  But conversation dated {issue['conv_date']} ({issue['days_before']} days before!)")

    if validations:
        output.append("\n### ✓ Valid Project References")
        for v in validations[:5]:
            output.append(f"- {v['project']} (created {v['project_created']})")

    if not issues and not validations:
        output.append("_No GitHub project references found in this conversation._")

    # Verdict
    output.append("\n### Verdict")
    if issues:
        output.append("🔴 **DATE LIKELY INCORRECT** - Conversation references projects that didn't exist yet.")
        output.append(f"   Earliest valid date: {max(i['project_created'] for i in issues)}")
    elif has_fallback:
        output.append("🟡 **UNCERTAIN** - Uses fallback timestamp, but no conflicting evidence found.")
    else:
        output.append("🟢 **LIKELY VALID** - No date conflicts detected.")

    return "\n".join(output)


def _github_stats() -> str:
    """Internal: Get statistics about imported GitHub data."""
    output = ["## GitHub Data Statistics\n"]

    if not GITHUB_REPOS_PARQUET.exists():
        return "GitHub data not imported. Run import_github_data.py first."

    con = get_github_db()

    # Repo stats
    repo_stats = con.execute("""
        SELECT
            COUNT(*) as total_repos,
            SUM(CASE WHEN is_private THEN 1 ELSE 0 END) as private_repos,
            SUM(CASE WHEN NOT is_private THEN 1 ELSE 0 END) as public_repos,
            MIN(created_at) as earliest_repo,
            MAX(pushed_at) as latest_push
        FROM github_repos
    """).fetchone()

    output.append(f"### Repositories")
    output.append(f"**Total**: {repo_stats[0]}")
    output.append(f"**Private**: {repo_stats[1]}")
    output.append(f"**Public**: {repo_stats[2]}")
    output.append(f"**Earliest created**: {str(repo_stats[3])[:10]}")
    output.append(f"**Latest push**: {str(repo_stats[4])[:10]}\n")

    # Language breakdown
    languages = con.execute("""
        SELECT language, COUNT(*) as count
        FROM github_repos
        WHERE language IS NOT NULL
        GROUP BY language
        ORDER BY count DESC
        LIMIT 10
    """).fetchall()

    output.append("### Languages")
    for lang, count in languages:
        output.append(f"- {lang}: {count}")

    # Commit stats
    if GITHUB_COMMITS_PARQUET.exists():
        commit_stats = con.execute("""
            SELECT
                COUNT(*) as total_commits,
                COUNT(DISTINCT repo_name) as repos_with_commits,
                MIN(timestamp) as earliest_commit,
                MAX(timestamp) as latest_commit
            FROM github_commits
        """).fetchone()

        output.append(f"\n### Commits")
        output.append(f"**Total**: {commit_stats[0]:,}")
        output.append(f"**Repos with commits**: {commit_stats[1]}")
        output.append(f"**Date range**: {str(commit_stats[2])[:10]} to {str(commit_stats[3])[:10]}")

        # Commits by year
        by_year = con.execute("""
            SELECT strftime(timestamp, '%Y') as year, COUNT(*) as count
            FROM github_commits
            GROUP BY 1 ORDER BY 1
        """).fetchall()

        output.append("\n**By Year**:")
        for year, count in by_year:
            output.append(f"- {year}: {count:,}")

        # Most active repos
        top_repos = con.execute(f"""
            SELECT repo_name, COUNT(*) as commits
            FROM read_parquet('{GITHUB_COMMITS_PARQUET}')
            GROUP BY repo_name
            ORDER BY commits DESC
            LIMIT 10
        """).fetchall()

        output.append("\n### Most Active Repos")
        for repo, commits in top_repos:
            output.append(f"- {repo}: {commits} commits")

    # Embedding stats - LanceDB
    if LANCE_PATH.exists():
        try:
            db = get_lance_db()
            if db and "commit" in db.table_names():
                commit_emb = db.open_table("commit").count_rows()
                output.append(f"\n### Embeddings (LanceDB)")
                output.append(f"**Commit messages embedded**: {commit_emb:,}")
        except:
            pass

    return "\n".join(output)


# ═══════════════════════════════════════════════════════════════════════════════
# YOUTUBE TOOLS — REMOVED 2026-02-05 (static data, not live-synced)

# LAYER QUERY TOOLS (Phase 6 - query new architecture)
# ═══════════════════════════════════════════════════════════════════════════════

FACTS_DIR = BASE / "data" / "facts"
INTERP_DIR = BASE / "data" / "interpretations"
DATA_DIR = BASE / "data"


@mcp.tool()
def query_analytics(view: str = "timeline", date: str = None, month: str = None, source: str = None, limit: int = 15) -> str:
    """
    Query analytics across timeline, tool stacks, problem resolution, spend, and conversation summary.

    Args:
        view: What to analyze:
            - "timeline" (default): What happened on a specific date across all sources (was query_timeline)
            - "stacks": Technology stack patterns over time (was query_tool_stacks)
            - "problems": Debugging and problem resolution patterns (was query_problem_resolution)
            - "spend": Cost breakdown by source/time (was query_spend)
            - "summary": Comprehensive conversation analysis summary (was query_conversation_summary)
        date: Date in YYYY-MM-DD format (used with view="timeline")
        month: YYYY-MM filter (used with stacks, problems, spend views)
        source: Source filter for spend (e.g., "openrouter", "claude_code")
        limit: Max results (default 15)
    """
    if view == "timeline":
        if not date:
            return "Please provide a date (YYYY-MM-DD) for timeline view."
        return _query_timeline_view(date)
    elif view == "stacks":
        return _query_tool_stacks_view(month, limit)
    elif view == "problems":
        return _query_problem_resolution_view(month, limit)
    elif view == "spend":
        return _query_spend_view(month, source)
    elif view == "summary":
        return _query_conversation_summary_view()
    else:
        return f"Unknown view: {view}. Use: timeline, stacks, problems, spend, summary"


def _query_tool_stacks_view(month: str = None, limit: int = 12) -> str:
    """Internal: Query technology stack patterns."""
    path = INTERP_DIR / "tool_stacks" / "v2" / "monthly.parquet"
    if not path.exists():
        return "tool_stacks/v2 not built. Run: python pipelines/build_tool_stacks_v2.py"

    con = get_interpretations_db()

    if month:
        query = f"""
            SELECT month_start, stack_count, dominant_stack,
                   new_adoptions, abandonments, all_techs
            FROM '{path}'
            WHERE CAST(month_start AS VARCHAR) LIKE ?
            ORDER BY month_start DESC
            LIMIT {limit}
        """
        results = con.execute(query, [f"{month}%"]).fetchall()
    else:
        query = f"""
            SELECT month_start, stack_count, dominant_stack,
                   new_adoptions, abandonments, all_techs
            FROM '{path}'
            ORDER BY month_start DESC
            LIMIT {limit}
        """
        results = con.execute(query).fetchall()

    if not results:
        return f"No tool stack data found{' for ' + month if month else ''}"

    output = [f"🔧 Tool Stack Evolution{' for ' + month if month else ''}\n"]
    for row in results:
        month_start, count, dominant, new_tech, dropped, _ = row
        output.append(f"**{month_start}** - {count} stacks identified")
        output.append(f"  Dominant: {dominant[:60]}{'...' if dominant and len(dominant) > 60 else ''}")
        if new_tech:
            output.append(f"  ➕ New: {new_tech}")
        if dropped:
            output.append(f"  ➖ Dropped: {dropped}")
        output.append("")

    return "\n".join(output)


def _query_problem_resolution_view(month: str = None, limit: int = 12) -> str:
    """Internal: Query debugging and problem resolution patterns."""
    path = INTERP_DIR / "problem_resolution" / "v2" / "monthly.parquet"
    if not path.exists():
        return "problem_resolution/v2 not built. Run: python pipelines/build_problem_resolution_v2.py"

    con = get_interpretations_db()

    if month:
        query = f"""
            SELECT month_start, chain_count, domains, difficulties,
                   debugging_patterns, hardest_problem, aha_quotes
            FROM '{path}'
            WHERE CAST(month_start AS VARCHAR) LIKE ?
            ORDER BY month_start DESC
            LIMIT {limit}
        """
        results = con.execute(query, [f"{month}%"]).fetchall()
    else:
        query = f"""
            SELECT month_start, chain_count, domains, difficulties,
                   debugging_patterns, hardest_problem, aha_quotes
            FROM '{path}'
            ORDER BY month_start DESC
            LIMIT {limit}
        """
        results = con.execute(query).fetchall()

    if not results:
        return f"No problem resolution data found{' for ' + month if month else ''}"

    output = [f"🔍 Problem Resolution Patterns{' for ' + month if month else ''}\n"]
    for row in results:
        month_start, chains, domains, difficulties, _, hardest, aha = row
        output.append(f"**{month_start}** - {chains} resolution chains")
        output.append(f"  Domains: {domains}")
        output.append(f"  Difficulties: {difficulties}")
        if hardest:
            output.append(f"  Hardest: {hardest[:80]}{'...' if len(hardest) > 80 else ''}")
        if aha:
            output.append(f"  💡 Aha: {aha[:80]}{'...' if len(aha) > 80 else ''}")
        output.append("")

    return "\n".join(output)


def _query_spend_view(month: str = None, source: str = None) -> str:
    """Internal: Query spend data from facts/spend layers."""
    monthly_path = FACTS_DIR / "spend" / "monthly.parquet"
    daily_path = FACTS_DIR / "spend" / "daily.parquet"

    if not monthly_path.exists():
        return "Spend data not built. Run: python pipelines/build_facts_spend.py"

    con = get_interpretations_db()
    output = []

    if month:
        # Show daily breakdown for specific month
        query = f"""
            SELECT date, source, cost_usd, tokens_total
            FROM '{daily_path}'
            WHERE CAST(date AS VARCHAR) LIKE ?
            {'AND source = ?' if source else ''}
            ORDER BY date DESC
        """
        params = [f"{month}%"]
        if source:
            params.append(source)
        results = con.execute(query, params).fetchall()

        output.append(f"## Daily Spend for {month}\n")
        total_cost = 0
        for date, src, cost, tokens in results:
            output.append(f"  {date} | {src}: ${cost:.2f} ({tokens:,} tokens)")
            total_cost += cost or 0
        output.append(f"\n**Total**: ${total_cost:.2f}")
    else:
        # Show monthly summary
        query = f"""
            SELECT month, source, cost_usd, tokens_total
            FROM '{monthly_path}'
            {'WHERE source = ?' if source else ''}
            ORDER BY month DESC, cost_usd DESC
            LIMIT 30
        """
        params = [source] if source else []
        results = con.execute(query, params).fetchall()

        output.append("## Monthly Spend Summary\n")
        current_month = None
        for mo, src, cost, tokens in results:
            if mo != current_month:
                current_month = mo
                output.append(f"\n**{mo}**")
            output.append(f"  {src}: ${cost:.2f} ({tokens:,} tokens)")

    return "\n".join(output)


def _query_timeline_view(date: str) -> str:
    """Internal: What happened on a specific date across all sources."""
    temporal_path = FACTS_DIR / "temporal_dim.parquet"
    focus_path = INTERP_DIR / "focus" / "v1" / "daily.parquet"

    if not temporal_path.exists():
        return "Temporal data not built. Run: python pipelines/build_temporal_dim.py"

    con = get_interpretations_db()

    # Get temporal dim data
    temporal = con.execute(f"""
        SELECT
            date, year, quarter, month, day_of_week, is_weekend,
            messages_sent, convos_started, videos_watched, commits_made,
            cost_total, tokens_used, was_active
        FROM '{temporal_path}'
        WHERE CAST(date AS VARCHAR) = ?
    """, [date]).fetchone()

    if not temporal:
        return f"No data for {date}"

    output = [f"## Timeline for {date}\n"]
    output.append(f"**{temporal[4]}** (Q{temporal[2]} {temporal[1]})")
    output.append(f"Weekend: {'Yes' if temporal[5] else 'No'}\n")

    output.append("### Activity")
    output.append(f"  Messages: {temporal[6] or 0}")
    output.append(f"  Conversations: {temporal[7] or 0}")
    output.append(f"  Videos Watched: {temporal[8] or 0}")
    output.append(f"  Commits: {temporal[9] or 0}")

    if temporal[10]:
        output.append(f"\n### Spend")
        output.append(f"  Cost: ${temporal[10]:.2f}")
        output.append(f"  Tokens: {temporal[11]:,}")

    # Get focus keywords if available
    if focus_path.exists():
        focus = con.execute(f"""
            SELECT top_keyword, keywords, focus_score
            FROM '{focus_path}'
            WHERE CAST(date AS VARCHAR) = ?
        """, [date]).fetchone()

        if focus:
            import json
            try:
                kw_list = json.loads(focus[1])[:5]
            except:
                kw_list = []
            output.append(f"\n### Focus")
            output.append(f"  Top: {focus[0]} (score: {focus[2]:.2f})")
            output.append(f"  Keywords: {', '.join(kw_list)}")

    return "\n".join(output)


# ═══════════════════════════════════════════════════════════════════════════════
# GOOGLE BROWSING TOOLS — REMOVED 2026-02-05 (static data, not live-synced)

# GITHUB FILE CHANGES — REMOVED 2026-02-10 (orphaned code from deleted tool)


# ═══════════════════════════════════════════════════════════════════════════════
# YOUTUBE SEARCHES — REMOVED 2026-02-05 (static data, not live-synced)

# INTERPRETATION QUERY TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CORPUS (39K harvested documents)
# ═══════════════════════════════════════════════════════════════════════════════

def get_markdown_db():
    """Get cached DuckDB connection for markdown corpus."""
    global _markdown_db
    if _markdown_db is None and MARKDOWN_PARQUET.exists():
        _markdown_db = duckdb.connect()
        _markdown_db.execute(f"""
            CREATE VIEW IF NOT EXISTS markdown_docs
            AS SELECT * FROM read_parquet('{MARKDOWN_PARQUET}')
        """)
    return _markdown_db


@mcp.tool()
def search_docs(query: str = "", filter: str = None, project: str = None, limit: int = 15, min_depth: int = 70) -> str:
    """
    Search markdown corpus and IP documents with various filters.

    Args:
        query: Search query (keyword for markdown, semantic for IP docs)
        filter: What to search/filter:
            - None (default): Keyword search on markdown corpus (was search_markdown)
            - "ip": Vector search on curated IP documents — frameworks, SHELET, etc. (was search_ip_docs)
            - "breakthrough": Documents with BREAKTHROUGH energy (was get_breakthrough_docs)
            - "deep": High depth-score documents (was get_deep_docs)
            - "project": Documents for a specific project (was get_project_docs)
            - "todos": Documents with open TODOs (was get_open_todos)
        project: Project name (used with filter="project" or filter="todos")
        limit: Max results (default 15)
        min_depth: Minimum depth score (used with filter="deep", default 70)
    """
    if filter == "ip":
        return _search_ip_docs_view(query, limit)
    elif filter == "breakthrough":
        return _get_breakthrough_docs_view(limit)
    elif filter == "deep":
        return _get_deep_docs_view(min_depth, limit)
    elif filter == "project":
        return _get_project_docs_view(project or query, limit)
    elif filter == "todos":
        return _get_open_todos_view(project, limit)

    # Default: keyword search on markdown
    db = get_markdown_db()
    if not db:
        return "Markdown corpus not found. Run harvest_markdown_v3.py first."

    pattern = f"%{query}%"
    results = db.execute("""
        SELECT filename, project, voice, energy, depth_score, harvest_score,
               decision_count, word_count, first_line
        FROM markdown_docs
        WHERE content ILIKE ? OR title ILIKE ? OR filename ILIKE ?
        ORDER BY depth_score DESC, harvest_score DESC
        LIMIT ?
    """, [pattern, pattern, pattern, limit]).fetchall()

    if not results:
        return f"No markdown documents found for '{query}'"

    output = [f"## Markdown Search: '{query}'\n", f"_Found {len(results)} documents_\n"]
    for r in results:
        fname, proj, voice, energy, depth, harvest, decisions, words, preview = r
        output.append(f"**{fname}**")
        output.append(f"  Project: {proj or 'unassigned'} | Voice: {voice} | Energy: {energy}")
        output.append(f"  Depth: {depth} | Harvest: {harvest} | Decisions: {decisions} | Words: {words:,}")
        output.append(f"  > {preview[:100]}...\n")

    return "\n".join(output)


def _get_breakthrough_docs_view(limit: int = 20) -> str:
    """Internal: Get documents with BREAKTHROUGH energy."""
    db = get_markdown_db()
    if not db:
        return "Markdown corpus not found."

    results = db.execute("""
        SELECT filename, project, depth_score, harvest_score, decision_count,
               word_count, first_line
        FROM markdown_docs
        WHERE energy = 'BREAKTHROUGH'
        ORDER BY depth_score DESC, harvest_score DESC
        LIMIT ?
    """, [limit]).fetchall()

    if not results:
        return "No breakthrough documents found."

    output = [f"## Breakthrough Documents\n", f"_Found {len(results)} aha moments_\n"]
    for r in results:
        fname, proj, depth, harvest, decisions, words, preview = r
        output.append(f"**{fname}** (depth: {depth}, harvest: {harvest})")
        output.append(f"  Project: {proj or 'unassigned'} | Decisions: {decisions} | Words: {words:,}")
        output.append(f"  > {preview[:120]}...\n")

    return "\n".join(output)


def _get_deep_docs_view(min_depth: int = 70, limit: int = 20) -> str:
    """Internal: Get documents with high depth scores."""
    db = get_markdown_db()
    if not db:
        return "Markdown corpus not found."

    results = db.execute("""
        SELECT filename, project, voice, energy, depth_score, harvest_score,
               decision_count, seed_concepts, word_count
        FROM markdown_docs
        WHERE depth_score >= ?
        ORDER BY depth_score DESC
        LIMIT ?
    """, [min_depth, limit]).fetchall()

    if not results:
        return f"No documents with depth >= {min_depth}"

    output = [f"## Deep Documents (depth >= {min_depth})\n"]
    for r in results:
        fname, proj, voice, energy, depth, _, decisions, seeds, words = r
        output.append(f"**{fname}** (depth: {depth})")
        output.append(f"  Project: {proj or 'unassigned'} | Voice: {voice} | Energy: {energy}")
        output.append(f"  Decisions: {decisions} | SEED: {seeds or 'none'} | Words: {words:,}\n")

    return "\n".join(output)


def _get_project_docs_view(project: str, limit: int = 20) -> str:
    """Internal: Get documents for a specific project."""
    db = get_markdown_db()
    if not db:
        return "Markdown corpus not found."

    results = db.execute("""
        SELECT filename, voice, energy, depth_score, harvest_score,
               decision_count, todos_open, word_count, first_line
        FROM markdown_docs
        WHERE project = ?
        ORDER BY depth_score DESC
        LIMIT ?
    """, [project.lower(), limit]).fetchall()

    if not results:
        return f"No documents found for project '{project}'"

    output = [f"## Project: {project}\n", f"_Found {len(results)} documents_\n"]
    for r in results:
        fname, voice, energy, depth, harvest, decisions, todos, _, preview = r
        output.append(f"**{fname}** (depth: {depth}, harvest: {harvest})")
        output.append(f"  Voice: {voice} | Energy: {energy} | Decisions: {decisions} | TODOs: {todos}")
        output.append(f"  > {preview[:100]}...\n")

    return "\n".join(output)


def _get_open_todos_view(project: str = None, limit: int = 20) -> str:
    """Internal: Get documents with the most open TODOs."""
    db = get_markdown_db()
    if not db:
        return "Markdown corpus not found."

    if project:
        results = db.execute("""
            SELECT filename, project, todos_open, todos_done, draft_status, depth_score
            FROM markdown_docs
            WHERE todos_open > 0 AND project = ?
            ORDER BY todos_open DESC
            LIMIT ?
        """, [project.lower(), limit]).fetchall()
    else:
        results = db.execute("""
            SELECT filename, project, todos_open, todos_done, draft_status, depth_score
            FROM markdown_docs
            WHERE todos_open > 5
            ORDER BY todos_open DESC
            LIMIT ?
        """, [limit]).fetchall()

    if not results:
        return "No documents with open TODOs found."

    output = [f"## Open TODOs{' (' + project + ')' if project else ''}\n"]
    for r in results:
        fname, proj, opens, done, status, depth = r
        pct = round(100 * done / (opens + done)) if (opens + done) > 0 else 0
        output.append(f"**{fname}** ({opens} open, {done} done = {pct}% complete)")
        output.append(f"  Project: {proj or 'unassigned'} | Status: {status} | Depth: {depth}\n")

    return "\n".join(output)


def _markdown_stats() -> str:
    """Internal: Get statistics about the markdown corpus."""
    db = get_markdown_db()
    if not db:
        return "Markdown corpus not found."

    stats = db.execute("""
        SELECT
            COUNT(*) as total_files,
            SUM(word_count) as total_words,
            SUM(CASE WHEN voice = 'FIRST_PERSON' THEN 1 ELSE 0 END) as first_person,
            SUM(CASE WHEN energy = 'BREAKTHROUGH' THEN 1 ELSE 0 END) as breakthroughs,
            SUM(CASE WHEN depth_score >= 70 THEN 1 ELSE 0 END) as deep_docs,
            SUM(decision_count) as total_decisions,
            SUM(todos_open) as open_todos,
            SUM(CASE WHEN is_duplicate THEN 1 ELSE 0 END) as duplicates
        FROM markdown_docs
    """).fetchone()

    projects = db.execute("""
        SELECT COALESCE(NULLIF(project, ''), 'unassigned') as proj, COUNT(*) as cnt
        FROM markdown_docs
        GROUP BY 1
        ORDER BY cnt DESC
        LIMIT 10
    """).fetchall()

    output = ["## Markdown Corpus Statistics\n"]
    output.append(f"**Total Files**: {stats[0]:,}")
    output.append(f"**Total Words**: {stats[1]:,}")
    output.append(f"**First Person Docs**: {stats[2]:,}")
    output.append(f"**Breakthrough Docs**: {stats[3]:,}")
    output.append(f"**Deep Docs (70+)**: {stats[4]:,}")
    output.append(f"**Total Decisions**: {stats[5]:,}")
    output.append(f"**Open TODOs**: {stats[6]:,}")
    output.append(f"**Duplicates**: {stats[7]:,}")
    output.append("\n### By Project:")
    for proj, cnt in projects:
        output.append(f"  {proj}: {cnt:,}")

    return "\n".join(output)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED CROSS-SOURCE SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def unified_search(query: str, limit: int = 15) -> str:
    """
    Search across ALL sources: conversations, GitHub, markdown.
    Returns integrated timeline of thinking on a topic.
    """
    results = []

    # 1. Conversation embeddings (semantic search) - LanceDB
    try:
        embedding = get_embedding(query)
        if embedding and LANCE_PATH.exists():
            lance_results = lance_search(embedding, limit=5)
            for title, content, year, month, sim in lance_results:
                date = f"{year}-{month:02d}"
                results.append(('conversation', title or 'Untitled', content, date, sim))
    except Exception:
        pass


    # 3. GitHub commits (keyword search)
    try:
        gh_db = get_github_db()
        if GITHUB_COMMITS_PARQUET.exists():
            gh_results = gh_db.execute("""
                SELECT 'github' as source,
                       repo_name || ': ' || LEFT(message, 80) as title,
                       message as content,
                       CAST(timestamp AS VARCHAR) as date,
                       0.4 as score
                FROM github_commits
                WHERE message ILIKE ? OR repo_name ILIKE ?
                ORDER BY timestamp DESC
                LIMIT 3
            """, [f'%{query}%', f'%{query}%']).fetchall()
            results.extend(gh_results)
    except Exception:
        pass

    # 4. Markdown docs (keyword search)
    try:
        md_db = get_markdown_db()
        if md_db:
            md_results = md_db.execute("""
                SELECT 'markdown' as source,
                       COALESCE(title, filename) as title,
                       LEFT(content, 500) as content,
                       CAST(modified_at AS VARCHAR) as date,
                       0.45 as score
                FROM markdown_docs
                WHERE content ILIKE ? OR title ILIKE ? OR filename ILIKE ?
                ORDER BY depth_score DESC
                LIMIT 3
            """, [f'%{query}%', f'%{query}%', f'%{query}%']).fetchall()
            results.extend(md_results)
    except Exception:
        pass

    if not results:
        return f"No results found across any source for: {query}"

    # Sort by score (semantic results first)
    results.sort(key=lambda x: -x[4])

    # Format output
    output = [f"## Unified Search: \"{query}\"\n"]
    output.append(f"Found {len(results)} results across sources:\n")

    current_source = None
    for source, title, content, date, _ in results[:limit]:
        if source != current_source:
            output.append(f"\n### {source.upper()}")
            current_source = source

        date_str = date[:10] if date else "unknown"
        preview = (content[:150] + "...") if content and len(content) > 150 else (content or "")
        output.append(f"**[{date_str}]** {title}")
        if preview:
            output.append(f"> {preview}")
        output.append("")

    return "\n".join(output)


# ═══════════════════════════════════════════════════════════════════════════════
# HIDDEN INTERPRETATION LAYERS (Phase 1 55x Mining - Jan 2026)
# ═══════════════════════════════════════════════════════════════════════════════


def query_problem_chains(month: str = None, limit: int = 15) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/problem-chains.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/problem-chains.md\n\nUse semantic_search() or search_conversations() for fresh queries."


# ═══════════════════════════════════════════════════════════════════════════════
# SPEND ANALYSIS TOOLS (Phase 2 55x Mining - Jan 2026)
# Mining 732K OpenRouter API calls for insights
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSATION ANALYSIS TOOLS (Phase 3 55x Mining - Jan 2026)
# Mining 353K messages for threads, Q&A patterns, and corrections
# ═══════════════════════════════════════════════════════════════════════════════


def _query_conversation_summary_view() -> str:
    """Internal: Comprehensive conversation analysis summary."""
    stats_path = INTERP_DIR / "conversation_stats" / "v1" / "conversations.parquet"
    qa_path = INTERP_DIR / "conversation_qa" / "v1" / "questions.parquet"
    corr_path = INTERP_DIR / "conversation_corrections" / "v1" / "corrections.parquet"
    deep_path = INTERP_DIR / "conversation_threads" / "v1" / "deep_conversations.parquet"

    if not stats_path.exists():
        return "Conversation analysis not built. Run: python pipelines/build_conversation_analysis.py"

    con = get_interpretations_db()

    output = ["💬 CONVERSATION ANALYSIS SUMMARY\n"]
    output.append("=" * 50)

    # Overall stats
    totals = con.execute(f"""
        SELECT
            COUNT(*) as conversations,
            SUM(message_count) as messages,
            SUM(total_words) as words,
            SUM(questions_asked) as questions,
            SUM(code_messages) as code_msgs,
            AVG(message_count) as avg_length
        FROM '{stats_path}'
    """).fetchone()

    output.append(f"\n📊 TOTALS")
    output.append(f"  Conversations: {totals[0]:,}")
    output.append(f"  Messages: {totals[1]:,}")
    output.append(f"  Total Words: {totals[2]:,}")
    output.append(f"  Questions Asked: {totals[3]:,}")
    output.append(f"  Code Messages: {totals[4]:,}")
    output.append(f"  Avg Conversation Length: {totals[5]:.0f} messages")

    # Deep conversations
    if deep_path.exists():
        deep = con.execute(f"SELECT COUNT(*) FROM '{deep_path}'").fetchone()[0]
        output.append(f"\n🌊 DEEP CONVERSATIONS")
        output.append(f"  100+ message conversations: {deep}")

    # Q&A stats
    if qa_path.exists():
        q_count = con.execute(f"SELECT COUNT(*) FROM '{qa_path}'").fetchone()[0]
        output.append(f"\n❓ Q&A PATTERNS")
        output.append(f"  User questions: {q_count:,}")

    # Correction stats
    if corr_path.exists():
        c_count = con.execute(f"SELECT COUNT(*) FROM '{corr_path}'").fetchone()[0]
        output.append(f"\n🔄 CORRECTIONS")
        output.append(f"  Correction messages: {c_count:,}")

    # Top conversations by messages
    top = con.execute(f"""
        SELECT conversation_title, message_count
        FROM '{stats_path}'
        ORDER BY message_count DESC
        LIMIT 5
    """).fetchall()

    output.append(f"\n🏆 LONGEST CONVERSATIONS")
    for title, msgs in top:
        t = (title or "untitled")[:35]
        output.append(f"  {msgs:>5,} msgs - {t}")

    return "\n".join(output)


# ═══════════════════════════════════════════════════════════════════════════════
# BEHAVIORAL ARCHAEOLOGY TOOLS (Phase 4 55x Mining - Jan 2026)
# Mining 112K Google searches/visits for behavioral patterns
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# CODE PRODUCTIVITY TOOLS (Phase 5 55x Mining - Jan 2026)
# Mining 1.4K commits + 132 repos for productivity patterns
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN KNOWLEDGE TOOLS (Phase 6 55x Mining - Jan 2026)
# Mining 5.5K documents (6.3M words) for knowledge patterns
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7: CROSS-DIMENSIONAL SYNTHESIS (2026-01-11)
# Productivity matrix, learning arcs, project success, unified timeline
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 9: DISCOVERY PIPELINES (2026-01-11)
# Anomalies, trends, recommendations, weekly synthesis
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# V6 STRUCTURED SUMMARY TOOLS (Phase 2 - New tools)
# ═══════════════════════════════════════════════════════════════════════════════


# search_open_questions — MERGED into search_summaries(extract="questions")


# search_decisions — MERGED into search_summaries(extract="decisions")


# quote_me — MERGED into search_summaries(extract="quotes")


def _domain_map_view(source: str = None, importance: str = None) -> str:
    """Internal: Overview of thinking distribution across all 25 domains."""
    sdb = get_summaries_db()

    # Build filter
    where_parts = []
    params = []
    if source:
        where_parts.append("source = ?")
        params.append(source)
    if importance:
        where_parts.append("importance = ?")
        params.append(importance)
    where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    # Domain breakdown
    domains = sdb.execute(f"""
        SELECT
            domain_primary,
            COUNT(*) as count,
            COUNT(CASE WHEN importance = 'breakthrough' THEN 1 END) as breakthroughs,
            COUNT(CASE WHEN importance = 'significant' THEN 1 END) as significant,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) as pct
        FROM summaries
        {where_clause}
        GROUP BY domain_primary
        ORDER BY count DESC
    """, params).fetchall()

    if not domains:
        return "No domain data available."

    total = sum(d[1] for d in domains)
    filter_desc = ""
    if source:
        filter_desc += f" source={source}"
    if importance:
        filter_desc += f" importance={importance}"

    output = [f"## Domain Map ({total:,} conversations{filter_desc})\n"]

    for domain, count, bt, sig, pct in domains:
        bar = "█" * int(pct / 2)
        bt_str = f" 🔥{bt}" if bt else ""
        sig_str = f" ⭐{sig}" if sig else ""
        output.append(f"**{domain}**: {count:,} ({pct}%) {bar}{bt_str}{sig_str}")

    # Top concepts per top 5 domains
    output.append("\n### Top Concepts by Domain\n")
    for domain, count, _, _, _ in domains[:5]:
        rows = sdb.execute("""
            SELECT concepts FROM summaries
            WHERE domain_primary = ? AND concepts IS NOT NULL
            LIMIT 50
        """, [domain]).fetchall()
        concept_counts = {}
        for row in rows:
            for c in _parse_json_field(row[0]):
                if c:
                    concept_counts[c] = concept_counts.get(c, 0) + 1
        top_concepts = sorted(concept_counts.items(), key=lambda x: -x[1])[:8]
        if top_concepts:
            output.append(f"**{domain}**: {', '.join(c[0] for c in top_concepts)}")

    return "\n".join(output)


def _thinking_pulse_view(domain: str = None) -> str:
    """Internal: Domain × thinking_stage matrix."""
    sdb = get_summaries_db()

    if domain:
        # Focused view for one domain
        stages = sdb.execute("""
            SELECT thinking_stage, COUNT(*) as cnt,
                   COUNT(CASE WHEN importance = 'breakthrough' THEN 1 END) as breakthroughs
            FROM summaries
            WHERE domain_primary = ? AND thinking_stage IS NOT NULL
            GROUP BY thinking_stage
            ORDER BY cnt DESC
        """, [domain]).fetchall()

        if not stages:
            return f"No data for domain: {domain}"

        output = [f"## Thinking Pulse: {domain}\n"]
        stage_icons = {"exploring": "🔍", "crystallizing": "💎", "refining": "🔧", "executing": "🚀"}
        for stage, cnt, bt in stages:
            icon = stage_icons.get(stage, "📝")
            bt_str = f" (🔥{bt} breakthroughs)" if bt else ""
            output.append(f"{icon} **{stage}**: {cnt} conversations{bt_str}")

        return "\n".join(output)

    # Full crosstab view
    crosstab = sdb.execute("""
        SELECT
            domain_primary,
            COUNT(CASE WHEN thinking_stage = 'exploring' THEN 1 END) as exploring,
            COUNT(CASE WHEN thinking_stage = 'crystallizing' THEN 1 END) as crystallizing,
            COUNT(CASE WHEN thinking_stage = 'refining' THEN 1 END) as refining,
            COUNT(CASE WHEN thinking_stage = 'executing' THEN 1 END) as executing,
            COUNT(*) as total
        FROM summaries
        WHERE thinking_stage IS NOT NULL
        GROUP BY domain_primary
        ORDER BY total DESC
        LIMIT 25
    """).fetchall()

    if not crosstab:
        return "No thinking pulse data available."

    output = ["## Thinking Pulse (all domains)\n"]
    output.append(f"{'Domain':<25s} {'🔍 Expl':>8s} {'💎 Cryst':>8s} {'🔧 Refn':>8s} {'🚀 Exec':>8s} {'Total':>7s}")
    output.append("-" * 72)

    crystallizing_domains = []
    exploring_domains = []

    for domain, expl, cryst, refn, exec_, total in crosstab:
        output.append(f"{domain:<25s} {expl:>8d} {cryst:>8d} {refn:>8d} {exec_:>8d} {total:>7d}")
        if cryst > expl and cryst > 0:
            crystallizing_domains.append((domain, cryst))
        if expl > cryst + refn + exec_ and expl > 0:
            exploring_domains.append((domain, expl))

    # Highlights
    if crystallizing_domains:
        output.append("\n### 💎 Crystallizing (ready to ship)")
        for domain, cnt in sorted(crystallizing_domains, key=lambda x: -x[1])[:5]:
            output.append(f"  - {domain} ({cnt} crystallizing)")

    if exploring_domains:
        output.append("\n### 🔍 Still Exploring")
        for domain, cnt in sorted(exploring_domains, key=lambda x: -x[1])[:5]:
            output.append(f"  - {domain} ({cnt} exploring)")

    return "\n".join(output)


@mcp.tool()
def unfinished_threads(domain: str = None, importance: str = "significant") -> str:
    """
    Find conversations worth revisiting: exploring/crystallizing stage with open questions.
    These are unfinished intellectual threads that might deserve attention.

    Args:
        domain: Optional domain filter
        importance: Minimum importance level (default "significant")
    """
    sdb = get_summaries_db()

    importance_filter = {
        "breakthrough": "importance = 'breakthrough'",
        "significant": "importance IN ('breakthrough', 'significant')",
        "routine": "1=1",  # all
    }.get(importance, "importance IN ('breakthrough', 'significant')")

    where_parts = [
        "thinking_stage IN ('exploring', 'crystallizing')",
        importance_filter,
        "open_questions IS NOT NULL",
        "open_questions != '[]'",
        "open_questions NOT LIKE '%none identified%'",
    ]
    params = []
    if domain:
        where_parts.append("domain_primary = ?")
        params.append(domain)

    where_clause = " AND ".join(where_parts)

    results = sdb.execute(f"""
        SELECT conversation_id, title, source, domain_primary,
               thinking_stage, importance, open_questions, summary, msg_count
        FROM summaries
        WHERE {where_clause}
        ORDER BY
            CASE importance WHEN 'breakthrough' THEN 0 WHEN 'significant' THEN 1 ELSE 2 END,
            msg_count DESC
        LIMIT 25
    """, params).fetchall()

    if not results:
        return f"No unfinished threads found{' in ' + domain if domain else ''}."

    filter_desc = f" in {domain}" if domain else ""
    output = [f"## Unfinished Threads{filter_desc} (importance >= {importance})\n"]
    output.append(f"_Conversations still exploring/crystallizing with open questions_\n")

    for conv_id, title, source, dom, stage, imp, oq_raw, summary, msg_count in results:
        questions = _parse_json_field(oq_raw)
        real_questions = [q for q in questions if q and "none identified" not in str(q).lower()]
        if not real_questions:
            continue

        imp_icon = {"breakthrough": "🔥", "significant": "⭐", "routine": "📝"}.get(imp, "📝")
        stage_icon = {"exploring": "🔍", "crystallizing": "💎"}.get(stage, "📝")

        output.append(f"### {imp_icon} {stage_icon} {title or 'Untitled'}")
        output.append(f"_Domain: {dom} | Source: {source} | {msg_count} msgs_")
        output.append(f"> {(summary or '')[:200]}...")
        output.append("**Open questions**:")
        for q in real_questions[:3]:
            output.append(f"  ❓ {q[:200]}")
        if len(real_questions) > 3:
            output.append(f"  _... and {len(real_questions) - 3} more_")
        output.append(f"_ID: {conv_id[:20]}..._\n")

    return "\n".join(output)


# ═══════════════════════════════════════════════════════════════════════════════
# COGNITIVE PROSTHETIC TOOLS (tunnel state, context preservation, switching cost)
# These are the SOUL of the prosthetic — they turn a search engine into a
# monotropic cognitive aid that preserves context across hyperfocus tunnels.
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
def tunnel_state(domain: str, limit: int = 10) -> str:
    """Reconstruct cognitive save-state for a domain — where you left off.
    Returns: thinking stage, open questions, decisions, concepts, emotional tone.
    The 'load game' button for a monotropic mind."""
    db = get_summaries_db()
    rows = db.execute(f"""
        SELECT summary, thinking_stage, importance, emotional_tone,
               open_questions, decisions, concepts, key_insights, connections_to,
               cognitive_pattern, problem_solving_approach, msg_count, title, source
        FROM summaries
        WHERE domain_primary = ?
        ORDER BY summarized_at DESC
        LIMIT ?
    """, [domain, limit]).fetchall()
    if not rows:
        return f"No conversations found for domain: {domain}"
    cols = ['summary','thinking_stage','importance','emotional_tone','open_questions',
            'decisions','concepts','key_insights','connections_to','cognitive_pattern',
            'problem_solving_approach','msg_count','title','source']
    # Latest state
    latest = dict(zip(cols, rows[0]))
    # Aggregate across all rows
    all_oq, all_dec, all_concepts, all_insights, all_connections = [], [], set(), [], set()
    importance_counts = {"breakthrough":0,"significant":0,"routine":0}
    stage_counts = {}
    for row in rows:
        r = dict(zip(cols, row))
        for q in _parse_json_field(r['open_questions']):
            if q and q.lower() != 'none identified' and q not in all_oq:
                all_oq.append(q)
        for d in _parse_json_field(r['decisions']):
            if d and d not in all_dec:
                all_dec.append(d)
        for c in _parse_json_field(r['concepts']):
            all_concepts.add(c)
        for i in _parse_json_field(r['key_insights']):
            if i and i not in all_insights:
                all_insights.append(i)
        for c in _parse_json_field(r['connections_to']):
            all_connections.add(c)
        imp = r['importance'] or 'routine'
        importance_counts[imp] = importance_counts.get(imp, 0) + 1
        stage = r['thinking_stage'] or ''
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    output = [f"## 🧠 Tunnel State: {domain}\n"]
    output.append(f"**Current stage:** {latest['thinking_stage'] or 'unknown'}")
    output.append(f"**Emotional tone:** {latest['emotional_tone'] or 'unknown'}")
    output.append(f"**Conversations:** {len(rows)} (last {limit})")
    bt = importance_counts.get('breakthrough', 0)
    if bt: output.append(f"**Breakthroughs:** {bt} 💎")
    output.append(f"**Cognitive pattern:** {latest['cognitive_pattern'] or 'unknown'}")
    output.append(f"**Problem solving:** {latest['problem_solving_approach'] or 'unknown'}")
    if all_oq:
        output.append(f"\n### ❓ Open Questions ({len(all_oq)})")
        for q in all_oq[:10]:
            output.append(f"  - {q}")
        if len(all_oq) > 10: output.append(f"  _... and {len(all_oq)-10} more_")
    if all_dec:
        output.append(f"\n### ✅ Decisions ({len(all_dec)})")
        for d in all_dec[:7]:
            output.append(f"  - {d}")
        if len(all_dec) > 7: output.append(f"  _... and {len(all_dec)-7} more_")
    if all_concepts:
        output.append(f"\n### 🏷️ Active Concepts ({len(all_concepts)})")
        output.append(f"  {', '.join(sorted(all_concepts)[:15])}")
    if all_insights:
        output.append(f"\n### 💡 Key Insights")
        for i in all_insights[:5]:
            output.append(f"  - {i}")
    if all_connections:
        output.append(f"\n### 🔗 Connected Domains")
        output.append(f"  {', '.join(sorted(all_connections)[:10])}")
    if stage_counts:
        output.append(f"\n### 📊 Thinking Stage History")
        for s, c in sorted(stage_counts.items(), key=lambda x:-x[1]):
            output.append(f"  {s or 'unknown'}: {c}")
    return "\n".join(output)


@mcp.tool()
def dormant_contexts(min_importance: str = "significant", limit: int = 20) -> str:
    """Find abandoned tunnels — domains with open questions you haven't resolved.
    The 'what have I forgotten?' alarm."""
    db = get_summaries_db()
    rows = db.execute("""
        SELECT domain_primary, COUNT(*) as conv_count,
               GROUP_CONCAT(open_questions, '|||') as all_oq,
               GROUP_CONCAT(importance, ',') as importances,
               MAX(thinking_stage) as latest_stage
        FROM summaries
        WHERE domain_primary != '' AND domain_primary IS NOT NULL
        GROUP BY domain_primary
        ORDER BY conv_count DESC
    """).fetchall()
    if not rows:
        return "No domain data found."
    importance_rank = {"breakthrough": 3, "significant": 2, "routine": 1}
    min_rank = importance_rank.get(min_importance, 1)
    results = []
    for domain, count, all_oq_str, importances_str, stage in rows:
        # Check importance threshold
        imps = (importances_str or '').split(',')
        max_imp = max(importance_rank.get(i.strip(), 0) for i in imps if i.strip())
        if max_imp < min_rank:
            continue
        # Collect open questions
        questions = []
        for chunk in (all_oq_str or '').split('|||'):
            for q in _parse_json_field(chunk):
                if q and q.lower() != 'none identified' and q not in questions:
                    questions.append(q)
        if not questions:
            continue
        bt_count = sum(1 for i in imps if i.strip() == 'breakthrough')
        results.append((domain, count, questions, stage, bt_count))
    results.sort(key=lambda x: (-x[4], -len(x[2]), -x[1]))
    output = [f"## 🔴 Dormant Contexts (importance >= {min_importance})\n"]
    output.append(f"_Domains with unresolved open questions_\n")
    for domain, count, questions, stage, bt in results[:limit]:
        bt_marker = " 💎" if bt else ""
        output.append(f"### {domain}{bt_marker}")
        output.append(f"_{count} conversations | Stage: {stage or 'unknown'}_")
        output.append(f"**{len(questions)} open questions:**")
        for q in questions[:5]:
            output.append(f"  ❓ {q}")
        if len(questions) > 5:
            output.append(f"  _... and {len(questions)-5} more_")
        output.append("")
    output.append(f"_Total: {len(results)} domains with open questions_")
    return "\n".join(output)


@mcp.tool()
def context_recovery(domain: str, summary_count: int = 5) -> str:
    """Full 'waking up' brief for re-entering a domain.
    Returns recent summaries + accumulated state — everything needed to resume work.
    The prosthetic's core value: making re-entry cheap."""
    db = get_summaries_db()
    rows = db.execute("""
        SELECT title, source, summary, thinking_stage, importance,
               emotional_tone, open_questions, decisions, key_insights,
               concepts, connections_to, quotable, cognitive_pattern,
               problem_solving_approach, msg_count
        FROM summaries
        WHERE domain_primary = ?
        ORDER BY summarized_at DESC
        LIMIT ?
    """, [domain, summary_count + 10]).fetchall()
    if not rows:
        return f"No conversations found for domain: {domain}"
    cols = ['title','source','summary','thinking_stage','importance',
            'emotional_tone','open_questions','decisions','key_insights',
            'concepts','connections_to','quotable','cognitive_pattern',
            'problem_solving_approach','msg_count']
    all_oq, all_dec, all_insights, all_quotes = [], [], [], []
    for row in rows:
        r = dict(zip(cols, row))
        for q in _parse_json_field(r['open_questions']):
            if q and q.lower() != 'none identified' and q not in all_oq: all_oq.append(q)
        for d in _parse_json_field(r['decisions']):
            if d and d not in all_dec: all_dec.append(d)
        for i in _parse_json_field(r['key_insights']):
            if i and i not in all_insights: all_insights.append(i)
        for q in _parse_json_field(r['quotable']):
            if q and q not in all_quotes: all_quotes.append(q)
    latest = dict(zip(cols, rows[0]))
    output = [f"## 🔄 Context Recovery: {domain}\n"]
    output.append(f"**Stage:** {latest['thinking_stage'] or '?'} | **Tone:** {latest['emotional_tone'] or '?'} | **Conversations:** {len(rows)}")
    output.append(f"\n### 📋 Recent Summaries\n")
    for row in rows[:summary_count]:
        r = dict(zip(cols, row))
        title = (r['title'] or 'Untitled')[:60]
        imp_icon = "💎" if r['importance'] == 'breakthrough' else "⭐" if r['importance'] == 'significant' else "·"
        output.append(f"**{imp_icon} {title}** ({r['source']}, {r['msg_count']} msgs)")
        output.append(f"> {(r['summary'] or '')[:300]}{'...' if len(r['summary'] or '') > 300 else ''}")
        output.append("")
    if all_oq:
        output.append(f"### ❓ Accumulated Open Questions ({len(all_oq)})")
        for q in all_oq[:10]: output.append(f"  - {q}")
        if len(all_oq) > 10: output.append(f"  _... and {len(all_oq)-10} more_")
    if all_dec:
        output.append(f"\n### ✅ Key Decisions ({len(all_dec)})")
        for d in all_dec[:7]: output.append(f"  - {d}")
    if all_insights:
        output.append(f"\n### 💡 Key Insights")
        for i in all_insights[:5]: output.append(f"  - {i}")
    if all_quotes:
        output.append(f"\n### 💬 Quotable")
        output.append(f'  > "{all_quotes[0][:200]}"')
    return "\n".join(output)


@mcp.tool()
def tunnel_history(domain: str) -> str:
    """Meta-view of your engagement with a domain over time.
    Shows total conversations, thinking stage distribution, importance peaks,
    and cognitive patterns."""
    db = get_summaries_db()
    rows = db.execute("""
        SELECT thinking_stage, importance, emotional_tone,
               cognitive_pattern, problem_solving_approach, concepts, source
        FROM summaries
        WHERE domain_primary = ?
    """, [domain]).fetchall()
    if not rows:
        return f"No conversations found for domain: {domain}"
    cols = ['thinking_stage','importance','emotional_tone','cognitive_pattern',
            'problem_solving_approach','concepts','source']
    stage_counts, imp_counts, tone_counts = {}, {}, {}
    pattern_counts, approach_counts, source_counts = {}, {}, {}
    all_concepts = {}
    for row in rows:
        r = dict(zip(cols, row))
        s = r['thinking_stage'] or 'unknown'
        stage_counts[s] = stage_counts.get(s, 0) + 1
        i = r['importance'] or 'routine'
        imp_counts[i] = imp_counts.get(i, 0) + 1
        t = r['emotional_tone'] or ''
        if t: tone_counts[t] = tone_counts.get(t, 0) + 1
        p = r['cognitive_pattern'] or ''
        if p: pattern_counts[p] = pattern_counts.get(p, 0) + 1
        a = r['problem_solving_approach'] or ''
        if a: approach_counts[a] = approach_counts.get(a, 0) + 1
        src = r['source'] or ''
        if src: source_counts[src] = source_counts.get(src, 0) + 1
        for c in _parse_json_field(r['concepts']):
            all_concepts[c] = all_concepts.get(c, 0) + 1
    output = [f"## 📊 Tunnel History: {domain}\n"]
    output.append(f"**Total conversations:** {len(rows)}")
    bt = imp_counts.get('breakthrough', 0)
    sig = imp_counts.get('significant', 0)
    output.append(f"**Importance:** {bt} breakthrough, {sig} significant, {imp_counts.get('routine',0)} routine")
    output.append(f"\n### Thinking Stages")
    for s, c in sorted(stage_counts.items(), key=lambda x:-x[1]):
        pct = c/len(rows)*100
        bar = "█" * int(pct/5)
        output.append(f"  {s}: {c} ({pct:.0f}%) {bar}")
    if source_counts:
        output.append(f"\n### Sources")
        for s, c in sorted(source_counts.items(), key=lambda x:-x[1]):
            output.append(f"  {s}: {c}")
    if pattern_counts:
        output.append(f"\n### Cognitive Patterns")
        for p, c in sorted(pattern_counts.items(), key=lambda x:-x[1])[:7]:
            output.append(f"  {p}: {c}")
    if approach_counts:
        output.append(f"\n### Problem Solving Approaches")
        for a, c in sorted(approach_counts.items(), key=lambda x:-x[1])[:7]:
            output.append(f"  {a}: {c}")
    if tone_counts:
        output.append(f"\n### Emotional Tones")
        for t, c in sorted(tone_counts.items(), key=lambda x:-x[1])[:5]:
            output.append(f"  {t}: {c}")
    if all_concepts:
        output.append(f"\n### Top Concepts ({len(all_concepts)} total)")
        for c, n in sorted(all_concepts.items(), key=lambda x:-x[1])[:10]:
            output.append(f"  {c}: {n}")
    return "\n".join(output)


@mcp.tool()
def switching_cost(current_domain: str, target_domain: str) -> str:
    """Estimate cognitive cost of switching between domains.
    Factors: open questions left behind (abandonment), shared concepts (overlap discount).
    Returns 0-1 score where lower = cheaper switch."""
    db = get_summaries_db()
    # Current domain state
    cur_rows = db.execute("""
        SELECT open_questions, concepts, thinking_stage
        FROM summaries WHERE domain_primary = ?
    """, [current_domain]).fetchall()
    # Target domain state
    tgt_rows = db.execute("""
        SELECT open_questions, concepts, thinking_stage
        FROM summaries WHERE domain_primary = ?
    """, [target_domain]).fetchall()
    if not cur_rows:
        return f"No data for current domain: {current_domain}"
    if not tgt_rows:
        return f"No data for target domain: {target_domain}"
    # Aggregate current domain
    cur_oq = set()
    cur_concepts = set()
    for row in cur_rows:
        for q in _parse_json_field(row[0]):
            if q and q.lower() != 'none identified': cur_oq.add(q)
        for c in _parse_json_field(row[1]): cur_concepts.add(c)
    cur_stage = cur_rows[0][2] or 'unknown'
    # Aggregate target domain
    tgt_concepts = set()
    for row in tgt_rows:
        for c in _parse_json_field(row[1]): tgt_concepts.add(c)
    tgt_stage = tgt_rows[0][2] or 'unknown'
    # Calculate cost
    shared = cur_concepts & tgt_concepts
    oq_cost = min(len(cur_oq) / 10.0, 1.0)
    overlap_discount = min(len(shared) / max(len(cur_concepts), 1), 1.0)
    # Stage cost: leaving executing/refining is expensive
    stage_cost = {"executing": 0.8, "refining": 0.6, "crystallizing": 0.4, "exploring": 0.2}.get(cur_stage, 0.3)
    score = round((oq_cost * 0.35) + (stage_cost * 0.35) - (overlap_discount * 0.3), 3)
    score = max(0.0, min(1.0, score))
    if score < 0.3: rec = "✅ Low cost — go for it"
    elif score < 0.6: rec = "⚠️ Moderate — consider noting current open questions first"
    else: rec = "🔴 High cost — significant unfinished work in current domain"
    output = [f"## 🔀 Switching Cost: {current_domain} → {target_domain}\n"]
    output.append(f"### Score: **{score}** / 1.0  ({rec})\n")
    output.append(f"**Current domain:** {current_domain}")
    output.append(f"  Stage: {cur_stage}")
    output.append(f"  Open questions: {len(cur_oq)}")
    output.append(f"  Concepts: {len(cur_concepts)}")
    output.append(f"\n**Target domain:** {target_domain}")
    output.append(f"  Stage: {tgt_stage}")
    output.append(f"  Conversations: {len(tgt_rows)}")
    output.append(f"  Concepts: {len(tgt_concepts)}")
    output.append(f"\n**Overlap:** {len(shared)} shared concepts")
    if shared:
        output.append(f"  {', '.join(sorted(shared)[:10])}")
    output.append(f"\n**Cost breakdown:**")
    output.append(f"  Abandonment (open Qs): {oq_cost:.2f}")
    output.append(f"  Stage penalty ({cur_stage}): {stage_cost:.2f}")
    output.append(f"  Overlap discount: -{overlap_discount:.2f}")
    if cur_oq:
        output.append(f"\n**Questions you'd leave behind:**")
        for q in list(cur_oq)[:5]: output.append(f"  ❓ {q}")
    return "\n".join(output)


@mcp.tool()
def cognitive_patterns(domain: str = None) -> str:
    """Analyze cognitive patterns and problem-solving approaches.
    Answers: 'When do I think best?' with data."""
    db = get_summaries_db()
    where = f"WHERE domain_primary = '{domain}'" if domain else ""
    rows = db.execute(f"""
        SELECT cognitive_pattern, problem_solving_approach, importance,
               emotional_tone, thinking_stage, content_category
        FROM summaries {where}
    """).fetchall()
    if not rows:
        return f"No data found{f' for domain: {domain}' if domain else ''}"
    cols = ['cognitive_pattern','problem_solving_approach','importance',
            'emotional_tone','thinking_stage','content_category']
    pattern_counts, approach_counts, tone_counts = {}, {}, {}
    bt_patterns, bt_approaches, bt_tones = {}, {}, {}
    category_counts = {}
    total, bt_total = len(rows), 0
    for row in rows:
        r = dict(zip(cols, row))
        p = r['cognitive_pattern'] or ''
        if p: pattern_counts[p] = pattern_counts.get(p, 0) + 1
        a = r['problem_solving_approach'] or ''
        if a: approach_counts[a] = approach_counts.get(a, 0) + 1
        t = r['emotional_tone'] or ''
        if t: tone_counts[t] = tone_counts.get(t, 0) + 1
        cat = r['content_category'] or ''
        if cat: category_counts[cat] = category_counts.get(cat, 0) + 1
        if r['importance'] == 'breakthrough':
            bt_total += 1
            if p: bt_patterns[p] = bt_patterns.get(p, 0) + 1
            if a: bt_approaches[a] = bt_approaches.get(a, 0) + 1
            if t: bt_tones[t] = bt_tones.get(t, 0) + 1
    output = [f"## 🧬 Cognitive Patterns{f' ({domain})' if domain else ''}\n"]
    output.append(f"_Analyzed {total} conversations ({bt_total} breakthroughs)_\n")
    output.append(f"### Cognitive Patterns")
    for p, c in sorted(pattern_counts.items(), key=lambda x:-x[1])[:10]:
        bt = bt_patterns.get(p, 0)
        bt_mark = f" (💎×{bt})" if bt else ""
        output.append(f"  {p}: {c} ({c/total*100:.0f}%){bt_mark}")
    output.append(f"\n### Problem Solving Approaches")
    for a, c in sorted(approach_counts.items(), key=lambda x:-x[1])[:10]:
        bt = bt_approaches.get(a, 0)
        bt_mark = f" (💎×{bt})" if bt else ""
        output.append(f"  {a}: {c} ({c/total*100:.0f}%){bt_mark}")
    output.append(f"\n### Emotional Tones")
    for t, c in sorted(tone_counts.items(), key=lambda x:-x[1])[:8]:
        bt = bt_tones.get(t, 0)
        bt_mark = f" (💎×{bt})" if bt else ""
        output.append(f"  {t}: {c}{bt_mark}")
    if category_counts:
        output.append(f"\n### Content Categories")
        for cat, c in sorted(category_counts.items(), key=lambda x:-x[1])[:8]:
            output.append(f"  {cat}: {c}")
    if bt_patterns:
        top_bt = max(bt_patterns, key=bt_patterns.get)
        output.append(f"\n### 💎 Breakthrough Insight")
        output.append(f"Your breakthroughs most associate with **{top_bt}** thinking")
        if bt_tones:
            top_tone = max(bt_tones, key=bt_tones.get)
            output.append(f"and tend to happen when you're in a **{top_tone}** emotional state.")
    return "\n".join(output)


@mcp.tool()
def open_threads(limit_per_domain: int = 5, max_domains: int = 20) -> str:
    """Global inventory of ALL open questions across ALL domains.
    The 'unfinished business' dashboard."""
    db = get_summaries_db()
    rows = db.execute("""
        SELECT domain_primary, open_questions, importance, thinking_stage
        FROM summaries
        WHERE domain_primary != '' AND domain_primary IS NOT NULL
    """).fetchall()
    if not rows:
        return "No data found."
    domain_data = {}
    for domain, oq_str, importance, stage in rows:
        if domain not in domain_data:
            domain_data[domain] = {"questions": [], "count": 0, "bt": 0, "stage": stage}
        domain_data[domain]["count"] += 1
        if importance == "breakthrough": domain_data[domain]["bt"] += 1
        for q in _parse_json_field(oq_str):
            if q and q.lower() != 'none identified' and q not in domain_data[domain]["questions"]:
                domain_data[domain]["questions"].append(q)
    # Filter domains with open questions, sort by question count
    active = [(d, v) for d, v in domain_data.items() if v["questions"]]
    active.sort(key=lambda x: (-x[1]["bt"], -len(x[1]["questions"])))
    total_q = sum(len(v["questions"]) for _, v in active)
    output = [f"## 🧵 Open Threads\n"]
    output.append(f"**{total_q} open questions** across **{len(active)} domains**\n")
    for domain, data in active[:max_domains]:
        bt = f" 💎×{data['bt']}" if data['bt'] else ""
        output.append(f"### {domain}{bt} ({len(data['questions'])} questions, {data['count']} convos)")
        for q in data["questions"][:limit_per_domain]:
            output.append(f"  ❓ {q}")
        if len(data["questions"]) > limit_per_domain:
            output.append(f"  _... and {len(data['questions'])-limit_per_domain} more_")
        output.append("")
    return "\n".join(output)


@mcp.tool()
def trust_dashboard() -> str:
    """System-wide stats proving the prosthetic works.
    Shows everything that's preserved: conversations, domains, questions, decisions.
    The 'everything is okay' view."""
    db = get_summaries_db()
    # Total stats
    stats = db.execute("""
        SELECT COUNT(*) as total,
               COUNT(DISTINCT domain_primary) as domains,
               COUNT(CASE WHEN importance = 'breakthrough' THEN 1 END) as breakthroughs
        FROM summaries
    """).fetchone()
    total, domains, breakthroughs = stats
    # Count open questions and decisions
    rows = db.execute("SELECT open_questions, decisions FROM summaries").fetchall()
    total_oq, total_dec = 0, 0
    for oq_str, dec_str in rows:
        oqs = _parse_json_field(oq_str)
        total_oq += sum(1 for q in oqs if q and q.lower() != 'none identified')
        total_dec += len(_parse_json_field(dec_str))
    # Source breakdown
    sources = db.execute("""
        SELECT source, COUNT(*) as count FROM summaries GROUP BY source ORDER BY count DESC
    """).fetchall()
    # Domain status
    domain_rows = db.execute("""
        SELECT domain_primary, COUNT(*) as count,
               MAX(thinking_stage) as stage,
               COUNT(CASE WHEN importance='breakthrough' THEN 1 END) as bt
        FROM summaries
        WHERE domain_primary != '' AND domain_primary IS NOT NULL
        GROUP BY domain_primary
        ORDER BY count DESC
    """).fetchall()
    # Count domains with open questions
    oq_domains = set()
    for row in db.execute("SELECT domain_primary, open_questions FROM summaries").fetchall():
        for q in _parse_json_field(row[1]):
            if q and q.lower() != 'none identified':
                oq_domains.add(row[0])
                break
    output = [f"## 🛡️ Trust Dashboard\n"]
    output.append(f"_Your cognitive safety net — proof that nothing is lost_\n")
    output.append(f"### 📊 Global Metrics")
    output.append(f"  **Conversations indexed:** {total:,}")
    output.append(f"  **Domains tracked:** {domains}")
    output.append(f"  **Open questions preserved:** {total_oq:,}")
    output.append(f"  **Decisions preserved:** {total_dec:,}")
    output.append(f"  **Breakthroughs captured:** {breakthroughs} 💎")
    output.append(f"  **Domains with active threads:** {len(oq_domains)}")
    output.append(f"\n### 📡 Sources")
    for src, count in sources:
        output.append(f"  {src}: {count:,}")
    output.append(f"\n### 🗺️ Domain Coverage (top 15)")
    for domain, count, stage, bt in domain_rows[:15]:
        bt_mark = f" 💎×{bt}" if bt else ""
        has_oq = " 🔴" if domain in oq_domains else " ✅"
        output.append(f"  {domain}: {count} convos ({stage or '?'}){bt_mark}{has_oq}")
    output.append(f"\n### 🔑 Safety Net Status")
    output.append(f"  {'🟢' if total > 5000 else '🟡'} Coverage: {total:,} conversations")
    output.append(f"  {'🟢' if domains > 15 else '🟡'} Breadth: {domains} domains")
    output.append(f"  {'🟢' if breakthroughs > 10 else '🟡'} Depth: {breakthroughs} breakthroughs captured")
    output.append(f"  {'🔴' if len(oq_domains) > 10 else '🟢'} Open threads: {len(oq_domains)} domains need attention")
    return "\n".join(output)


# ═══════════════════════════════════════════════════════════════════════════════
# RESOURCES (static data endpoints)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.resource("brain://stats")
def resource_stats() -> str:
    """Current brain statistics."""
    return brain_stats("overview")

@mcp.resource("brain://principles")
def resource_principles() -> str:
    """List of SEED principles."""
    return list_principles()

@mcp.resource("brain://embeddings")
def resource_embeddings() -> str:
    """Embedding statistics."""
    return brain_stats(view="embeddings")


# Pre-warm models on startup for fast first query
def _prewarm():
    """Pre-load embedding model and LanceDB connection."""
    import sys
    print("Pre-warming MCP Brain...", file=sys.stderr)

    # Load embedding model (8s cold start → cached)
    model = get_embedding_model()

    # Warm LanceDB connection
    db = get_lance_db()
    if db:
        db.open_table("message")

    # Do a dummy embed to fully initialize
    if model:
        model.encode("warmup", convert_to_numpy=True)

    print("MCP Brain ready (model + LanceDB warmed)", file=sys.stderr)


def _prewarm_async():
    """Pre-warm in background thread so MCP starts immediately."""
    import threading
    t = threading.Thread(target=_prewarm, daemon=True)
    t.start()


# Run server
if __name__ == "__main__":
    _prewarm_async()
    mcp.run(transport="stdio")
