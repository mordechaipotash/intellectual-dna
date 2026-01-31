#!/usr/bin/env python3
"""
Brain MCP Server - Your intellectual DNA, queryable as Claude Code tools.

Core data sources:
- 353K conversation messages (2023-2025) in DuckDB/Parquet
- 106K embedded messages with semantic search (768-dim nomic)
- 8 SEED principles (foundational mental models)
- 31K YouTube videos (consumption patterns)
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
    YOUTUBE_PARQUET,
    GITHUB_REPOS_PARQUET,
    GITHUB_COMMITS_PARQUET,
    MARKDOWN_PARQUET,
    SEED_PATH as SEED_FILE,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
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
    - 31K YouTube videos (consumption patterns)
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
_youtube_db = None
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


def get_youtube_db():
    """Get cached DuckDB connection for YouTube data."""
    global _youtube_db
    if _youtube_db is None and YOUTUBE_PARQUET.exists():
        _youtube_db = duckdb.connect()
        _youtube_db.execute(f"CREATE VIEW IF NOT EXISTS youtube AS SELECT * FROM read_parquet('{YOUTUBE_PARQUET}')")
    return _youtube_db


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
def search_conversations(term: str, limit: int = 15, role: str = None) -> str:
    """
    Full-text search across 247K conversation messages.
    Optionally filter by role ('user' for Mordechai's words, 'assistant' for AI responses).
    Returns previews with conversation context.
    """
    con = get_conversations()
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
    for source, model, title, role, preview, created, conv_id in results:
        output.append(f"**[{created}]** {title or 'Untitled'}")
        output.append(f"  {role}: {preview}...")
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


@mcp.tool()
def find_user_questions(limit: int = 20) -> str:
    """
    Find questions Mordechai has asked across all conversations.
    Useful for understanding his inquiry patterns.
    """
    con = get_conversations()
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
def brain_stats(source: str = "all") -> str:
    """
    Get statistics about brain data sources.

    Args:
        source: Which source to show stats for. Options:
            - "all" (default): Overview of all sources
            - "conversations": Detailed conversation stats
            - "embeddings": Embedding coverage stats
            - "github": Repository and commit stats
            - "youtube": Video viewing stats
            - "google": Search and visit stats
            - "markdown": Document corpus stats
    """
    source = source.lower().strip()

    if source == "conversations":
        return _conversation_stats()
    elif source == "embeddings":
        return _embedding_stats()
    elif source == "github":
        return _github_stats()
    elif source == "youtube":
        return _youtube_stats()
    elif source == "google":
        return _google_stats()
    elif source == "markdown":
        return _markdown_stats()
    elif source == "all":
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

        # GitHub
        try:
            if GITHUB_REPOS_PARQUET.exists():
                gdb = get_github_db()
                repos = gdb.execute("SELECT COUNT(*) FROM github_repos").fetchone()[0]
                commits = gdb.execute("SELECT COUNT(*) FROM github_commits").fetchone()[0] if GITHUB_COMMITS_PARQUET.exists() else 0
                output.append(f"**GitHub**: {repos} repos, {commits:,} commits")
        except:
            pass

        # YouTube
        try:
            if YOUTUBE_PARQUET.exists():
                ydb = get_youtube_db()
                vids = ydb.execute("SELECT COUNT(*) FROM youtube").fetchone()[0]
                watched = ydb.execute("SELECT COUNT(*) FROM youtube WHERE watched_date IS NOT NULL").fetchone()[0]
                output.append(f"**YouTube**: {vids:,} videos ({watched:,} watched)")
        except:
            pass

        # Google
        try:
            searches_path = DATA_DIR / "google_searches.parquet"
            visits_path = DATA_DIR / "google_visits.parquet"
            idb = get_interpretations_db()
            if searches_path.exists():
                searches = idb.execute(f"SELECT COUNT(*) FROM '{searches_path}'").fetchone()[0]
                output.append(f"**Google Searches**: {searches:,}")
            if visits_path.exists():
                visits = idb.execute(f"SELECT COUNT(*) FROM '{visits_path}'").fetchone()[0]
                output.append(f"**Google Visits**: {visits:,}")
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

        output.append("\n_Use brain_stats('source') for details on specific source_")
        return "\n".join(output)
    else:
        return f"Unknown source: {source}. Use: all, conversations, embeddings, github, youtube, google, markdown"


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


@mcp.tool()
def concept_velocity(term: str, granularity: str = "month") -> str:
    """
    Track how often a concept appears over time.
    Shows acceleration/deceleration of ideas.
    Granularity: 'month' or 'quarter'
    """
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


@mcp.tool()
def first_mention(term: str) -> str:
    """
    Find when a concept first appeared in your conversations.
    Shows the genesis moment of an idea.
    """
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
def what_do_i_think(topic: str) -> str:
    """
    Synthesize what Mordechai thinks about a topic.
    Combines semantic search with keyword matches.
    Returns both conceptually similar and exact mentions.
    """
    output = [f"## What do I think about: {topic}\n"]

    # 1. SEMANTIC SEARCH (conceptually related) - LanceDB
    embedding = get_embedding(topic)
    if embedding and LANCE_PATH.exists():
        semantic_results = lance_search(embedding, limit=5, min_sim=0.3)
        if semantic_results:
            output.append("### Semantically Related Thoughts:\n")
            for title, content, year, month, sim in semantic_results:
                preview = content[:250] + "..." if len(content) > 250 else content
                output.append(f"**[{year}-{month:02d}]** {title or 'Untitled'} (sim: {sim:.2f})")
                output.append(f"> {preview}\n")

    # 2. KEYWORD SEARCH (exact mentions)
    con = get_conversations()
    pattern = f"%{topic}%"
    conv_results = con.execute("""
        SELECT substr(content, 1, 300) as preview, created, conversation_title
        FROM conversations
        WHERE content ILIKE ? AND role = 'user'
        ORDER BY created DESC LIMIT 10
    """, [pattern]).fetchall()

    if conv_results:
        output.append("\n### Direct Mentions:\n")
        for preview, created, title in conv_results:
            output.append(f"**[{created}]** _{title or 'Untitled'}_")
            output.append(f"{preview}...\n")

    if len(output) == 1:
        output.append("_No thoughts found on this topic._")

    return "\n".join(output)


@mcp.tool()
def find_precedent(situation: str) -> str:
    """
    Find similar situations Mordechai has dealt with before.
    Searches both semantic embeddings and conversation history.
    """
    # Search conversations for similar situations
    con = get_conversations()
    pattern = f"%{situation}%"
    results = con.execute("""
        SELECT conversation_title, substr(content, 1, 300) as preview,
               created, conversation_id
        FROM conversations
        WHERE content ILIKE ? AND role = 'user'
        ORDER BY created DESC
        LIMIT 15
    """, [pattern]).fetchall()

    if not results:
        return f"No precedents found for: {situation}"

    output = [f"## Precedents for: {situation}\n"]
    for title, preview, created, conv_id in results:
        output.append(f"**[{created}]** {title or 'Untitled'}")
        output.append(f"{preview}...")
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
def search_ip_docs(query: str, limit: int = 10) -> str:
    """
    Search the distilled intellectual property documents (137 high-value docs).
    These contain Mordechai's core frameworks: SHELET, MORDETROPIC, Bottleneck=Amplifier,
    SeedGarden, 8 Principles, Translation Layer, Cognitive Prosthetics.

    Use this to find the original thinking and frameworks, not just conversation history.
    """
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
def thinking_trajectory(topic: str) -> str:
    """
    Track the evolution of thinking about a topic over time.
    Combines semantic search with temporal analysis to show
    how ideas developed, when interest peaked, and what
    related concepts emerged alongside.

    This is the most powerful tool for understanding intellectual evolution.
    """
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
def github_project_timeline(project_name: str) -> str:
    """
    Get project creation date, commit history, activity windows.
    Shows when a GitHub project was created and its development timeline.
    Useful for validating conversation dates and understanding project context.
    """
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


@mcp.tool()
def conversation_project_context(project: str, limit: int = 10) -> str:
    """
    Find conversations mentioning a specific GitHub project.
    Cross-references conversation content with project names.
    Helps understand what was discussed about a project.
    """
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


@mcp.tool()
def code_to_conversation(query: str, limit: int = 10) -> str:
    """
    Semantic search across BOTH commits and conversations.
    Links code decisions to design discussions.
    Finds conceptually related content across code and chat history.
    """
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


@mcp.tool()
def validate_date_with_github(conversation_id: str) -> str:
    """
    Check if a conversation date is valid based on GitHub evidence.
    Uses project mentions and GitHub repo creation dates to validate timestamps.
    Identifies potentially mislabeled conversation dates.
    """
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
# YOUTUBE TOOLS (consumption patterns - what you watched)
# Uses get_youtube_db() from CONNECTION MANAGEMENT above
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
def youtube_search(query: str, limit: int = 15) -> str:
    """
    Search 31K YouTube videos you've watched by keyword.
    Searches titles, channel names, and transcripts.
    Shows what content you were consuming about a topic.
    """
    if not YOUTUBE_PARQUET.exists():
        return "YouTube data not found. Export youtube_rows.parquet first."

    con = get_youtube_db()
    pattern = f"%{query}%"

    results = con.execute("""
        SELECT
            title,
            channel_name,
            youtube_id,
            watched_date,
            watch_count,
            CASE WHEN full_transcript IS NOT NULL THEN length(full_transcript) ELSE 0 END as transcript_len,
            view_count,
            duration
        FROM youtube
        WHERE title ILIKE ?
           OR channel_name ILIKE ?
           OR full_transcript ILIKE ?
        ORDER BY watched_date DESC NULLS LAST
        LIMIT ?
    """, [pattern, pattern, pattern, limit]).fetchall()

    if not results:
        return f"No YouTube videos found matching '{query}'"

    output = [f"## YouTube Videos: '{query}' ({len(results)} found)\n"]
    for title, channel, yt_id, watched, watch_count, transcript_len, views, duration in results:
        duration_str = f"{duration//60}:{duration%60:02d}" if duration else "?"
        transcript_marker = "📝" if transcript_len and transcript_len > 100 else ""
        watch_info = f"(watched {watch_count}x)" if watch_count and watch_count > 1 else ""

        output.append(f"**{title or 'Untitled'}** {transcript_marker}")
        output.append(f"  📺 {channel or 'Unknown'} | ⏱️ {duration_str} | 👁️ {views or 0:,}")
        output.append(f"  🗓️ {str(watched)[:10] if watched else 'Unknown'} {watch_info}")
        output.append(f"  🔗 https://youtube.com/watch?v={yt_id}\n")

    return "\n".join(output)


@mcp.tool()
def youtube_semantic_search(query: str, limit: int = 10) -> str:
    """
    Search YouTube videos with transcript focus.
    Currently uses keyword matching on transcripts (768 vs 1024 dim mismatch).
    Prioritizes videos with transcripts for deeper content matching.

    Note: True semantic search pending re-embedding YouTube with nomic model.
    """
    if not YOUTUBE_PARQUET.exists():
        return "YouTube data not found."

    # Get query embedding (need to use a compatible model)
    # YouTube uses 1024-dim embeddings, we have 768-dim from nomic
    # For now, we'll use keyword search as fallback until embeddings are aligned
    embedding = get_embedding(query)
    if not embedding:
        return "Could not generate embedding. Is Ollama running?"

    # Since embedding dimensions don't match (768 vs 1024), use DuckDB's text search
    # TODO: Re-embed YouTube with nomic-embed-text for compatibility
    con = get_youtube_db()
    pattern = f"%{query}%"

    # Use full-text search on transcripts as semantic proxy
    results = con.execute("""
        SELECT
            title,
            channel_name,
            youtube_id,
            watched_date,
            substr(full_transcript, 1, 300) as transcript_preview,
            view_count,
            duration
        FROM youtube
        WHERE full_transcript IS NOT NULL
          AND length(full_transcript) > 100
          AND (full_transcript ILIKE ? OR title ILIKE ?)
        ORDER BY watched_date DESC NULLS LAST
        LIMIT ?
    """, [pattern, pattern, limit]).fetchall()

    if not results:
        return f"No videos with transcripts found matching '{query}'. Note: True semantic search requires re-embedding with compatible model."

    output = [f"## YouTube Semantic Search: '{query}'\n"]
    output.append("_Note: Using transcript keyword matching. Full vector search requires embedding alignment._\n")

    for title, channel, yt_id, watched, transcript, views, duration in results:
        duration_str = f"{duration//60}:{duration%60:02d}" if duration else "?"
        output.append(f"**{title or 'Untitled'}**")
        output.append(f"  📺 {channel or 'Unknown'} | ⏱️ {duration_str} | 👁️ {views or 0:,}")
        output.append(f"  🗓️ {str(watched)[:10] if watched else 'Unknown'}")
        if transcript:
            output.append(f"  > {transcript}...")
        output.append(f"  🔗 https://youtube.com/watch?v={yt_id}\n")

    return "\n".join(output)


def _youtube_stats() -> str:
    """Internal: Get statistics about YouTube viewing patterns."""
    if not YOUTUBE_PARQUET.exists():
        return "YouTube data not found."

    con = get_youtube_db()

    # Basic stats
    basic = con.execute("""
        SELECT
            COUNT(*) as total_videos,
            COUNT(DISTINCT channel_name) as unique_channels,
            SUM(CASE WHEN full_transcript IS NOT NULL AND length(full_transcript) > 100 THEN 1 ELSE 0 END) as with_transcripts,
            SUM(CASE WHEN content_embedding_1024 IS NOT NULL THEN 1 ELSE 0 END) as with_embeddings,
            MIN(watched_date) as earliest,
            MAX(watched_date) as latest,
            SUM(duration) / 3600 as total_hours
        FROM youtube
    """).fetchone()

    total, channels, transcripts, embeddings, earliest, latest, hours = basic

    output = [
        "## YouTube Viewing Statistics\n",
        f"**Total Videos**: {total:,}",
        f"**Unique Channels**: {channels:,}",
        f"**With Transcripts**: {transcripts:,} ({transcripts*100//total}%)",
        f"**With Embeddings**: {embeddings:,} ({embeddings*100//total}%)",
        f"**Date Range**: {str(earliest)[:10] if earliest else '?'} to {str(latest)[:10] if latest else '?'}",
        f"**Total Watch Time**: ~{int(hours or 0):,} hours\n"
    ]

    # Top channels
    top_channels = con.execute("""
        SELECT channel_name, COUNT(*) as count
        FROM youtube
        WHERE channel_name IS NOT NULL AND channel_name != ''
        GROUP BY channel_name
        ORDER BY count DESC
        LIMIT 10
    """).fetchall()

    output.append("### Top Channels")
    for channel, count in top_channels:
        output.append(f"- {channel}: {count} videos")

    # Viewing by year
    by_year = con.execute("""
        SELECT
            EXTRACT(YEAR FROM watched_date) as year,
            COUNT(*) as count
        FROM youtube
        WHERE watched_date IS NOT NULL
        GROUP BY year
        ORDER BY year
    """).fetchall()

    if by_year:
        output.append("\n### By Year")
        for year, count in by_year:
            if year:
                bar = "█" * min(count // 100, 30)
                output.append(f"  {int(year)}: {bar} ({count:,})")

    # Videos with multiple watches
    rewatched = con.execute("""
        SELECT COUNT(*) FROM youtube WHERE watch_count > 1
    """).fetchone()[0]

    if rewatched:
        output.append(f"\n**Rewatched**: {rewatched:,} videos watched multiple times")

    return "\n".join(output)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER QUERY TOOLS (Phase 6 - query new architecture)
# ═══════════════════════════════════════════════════════════════════════════════

FACTS_DIR = BASE / "data" / "facts"
INTERP_DIR = BASE / "data" / "interpretations"
DATA_DIR = BASE / "data"


@mcp.tool()
def query_focus(month: str = None, limit: int = 10) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/daily-focus-v1.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/daily-focus-v1.md\n\nUse semantic_search() or search_conversations() for fresh queries."


@mcp.tool()
def query_focus_v2(month: str = None, limit: int = 10) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/daily-focus-v2.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/daily-focus-v2.md\n\nUse semantic_search() or search_conversations() for fresh queries."


@mcp.tool()
def query_mvp_velocity(month: str = None, limit: int = 12) -> str:
    """BROKEN: Connection error during materialization. Use search_conversations('MVP') instead."""
    return "⚠️ BROKEN: Connection error during data extraction.\n\nUse semantic_search('mvp development') or search_conversations('MVP') instead."


@mcp.tool()
def query_tool_stacks(month: str = None, limit: int = 12) -> str:
    """
    Query technology stack patterns from tool_stacks/v2.
    Shows which tools Mordechai uses together and how they evolve.

    Args:
        month: Optional YYYY-MM to filter (e.g., '2025-12')
        limit: Max months to return (default 12)
    """
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
    con.close()

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


@mcp.tool()
def query_problem_resolution(month: str = None, limit: int = 12) -> str:
    """
    Query debugging and problem resolution patterns from problem_resolution/v2.
    Shows how Mordechai investigates and solves problems.

    Args:
        month: Optional YYYY-MM to filter (e.g., '2025-12')
        limit: Max months to return (default 12)
    """
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
    con.close()

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


@mcp.tool()
def query_spend(month: str = None, source: str = None) -> str:
    """
    Query spend data from facts/spend layers.
    Shows cost breakdown by source and time.

    Args:
        month: Optional YYYY-MM to filter (e.g., '2025-12')
        source: Optional source filter (e.g., 'openrouter', 'claude_code')
    """
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


@mcp.tool()
def query_timeline(date: str) -> str:
    """
    Query what happened on a specific date across all sources.
    Joins temporal_dim with focus and spend data.

    Args:
        date: Date in YYYY-MM-DD format
    """
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
# GOOGLE BROWSING TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_google_searches(query: str, limit: int = 20) -> str:
    """
    Search through 52K Google searches Mordechai has made.
    Reveals what he was curious about and researching.

    Args:
        query: Search term to find in Google search queries
        limit: Max results to return (default 20)
    """
    path = DATA_DIR / "google_searches.parquet"
    if not path.exists():
        return "Google searches data not found."

    con = get_interpretations_db()
    results = con.execute(f"""
        SELECT query, timestamp, has_question
        FROM '{path}'
        WHERE query ILIKE ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, [f"%{query}%", limit]).fetchall()

    if not results:
        return f"No Google searches matching '{query}'"

    output = [f"🔍 Google Searches matching '{query}' ({len(results)} results)\n"]
    for q, ts, has_q in results:
        q_mark = "❓" if has_q else ""
        output.append(f"  [{str(ts)[:10]}] {q_mark} {q}")

    return "\n".join(output)


@mcp.tool()
def search_google_visits(query: str, limit: int = 20) -> str:
    """
    Search through 58K websites Mordechai has visited.
    Shows browsing patterns and information sources.

    Args:
        query: Search term to find in page titles or URLs
        limit: Max results to return (default 20)
    """
    path = DATA_DIR / "google_visits.parquet"
    if not path.exists():
        return "Google visits data not found."

    con = get_interpretations_db()
    results = con.execute(f"""
        SELECT title, url, timestamp
        FROM '{path}'
        WHERE title ILIKE ? OR url ILIKE ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, [f"%{query}%", f"%{query}%", limit]).fetchall()

    if not results:
        return f"No website visits matching '{query}'"

    output = [f"🌐 Website Visits matching '{query}' ({len(results)} results)\n"]
    for title, url, ts in results:
        title_short = (title[:60] + "...") if title and len(title) > 60 else (title or "No title")
        output.append(f"  [{str(ts)[:10]}] {title_short}")
        output.append(f"    → {url[:80]}")

    return "\n".join(output)


def _google_stats() -> str:
    """Internal: Get statistics about Google browsing data."""
    searches_path = DATA_DIR / "google_searches.parquet"
    visits_path = DATA_DIR / "google_visits.parquet"

    con = get_interpretations_db()
    output = ["📊 Google Browsing Statistics\n"]

    if searches_path.exists():
        stats = con.execute(f"""
            SELECT
                COUNT(*) as total,
                MIN(timestamp) as first,
                MAX(timestamp) as last,
                SUM(CASE WHEN has_question THEN 1 ELSE 0 END) as questions
            FROM '{searches_path}'
        """).fetchone()
        output.append(f"### Searches")
        output.append(f"  Total: {stats[0]:,}")
        output.append(f"  Questions: {stats[3]:,}")
        output.append(f"  Range: {str(stats[1])[:10]} → {str(stats[2])[:10]}")

    if visits_path.exists():
        stats = con.execute(f"""
            SELECT
                COUNT(*) as total,
                MIN(timestamp) as first,
                MAX(timestamp) as last
            FROM '{visits_path}'
        """).fetchone()
        output.append(f"\n### Visits")
        output.append(f"  Total: {stats[0]:,}")
        output.append(f"  Range: {str(stats[1])[:10]} → {str(stats[2])[:10]}")

    return "\n".join(output)


# ═══════════════════════════════════════════════════════════════════════════════
# GITHUB FILE CHANGES
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_file_changes(query: str, limit: int = 30) -> str:
    """
    Search GitHub file changes by filename or repo.
    Shows what files were modified in commits.

    Args:
        query: Search term for filename or repo
        limit: Max results (default 30)
    """
    path = DATA_DIR / "github_file_changes.parquet"
    if not path.exists():
        return "GitHub file changes data not found."

    con = get_interpretations_db()
    results = con.execute(f"""
        SELECT sha, repo_name, filename, status, additions, deletions
        FROM '{path}'
        WHERE filename ILIKE ? OR repo_name ILIKE ?
        ORDER BY additions + deletions DESC
        LIMIT ?
    """, [f"%{query}%", f"%{query}%", limit]).fetchall()

    if not results:
        return f"No file changes matching '{query}'"

    output = [f"📝 File Changes matching '{query}' ({len(results)} results)\n"]
    for sha, repo, fname, status, adds, dels in results:
        output.append(f"  [{repo}] {fname}")
        output.append(f"    {status}: +{adds} -{dels} (sha: {sha[:7]})")

    return "\n".join(output)


# ═══════════════════════════════════════════════════════════════════════════════
# YOUTUBE SEARCHES
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_youtube_searches(query: str, limit: int = 20) -> str:
    """
    Search through 1.2K YouTube searches Mordechai has made.
    Shows what video content he was looking for.

    Args:
        query: Search term to find in YouTube search queries
        limit: Max results (default 20)
    """
    path = DATA_DIR / "youtube_searches.parquet"
    if not path.exists():
        return "YouTube searches data not found."

    con = get_interpretations_db()
    results = con.execute(f"""
        SELECT query, timestamp, has_question
        FROM '{path}'
        WHERE query ILIKE ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, [f"%{query}%", limit]).fetchall()

    if not results:
        return f"No YouTube searches matching '{query}'"

    output = [f"🎬 YouTube Searches matching '{query}' ({len(results)} results)\n"]
    for q, ts, has_q in results:
        q_mark = "❓" if has_q else ""
        output.append(f"  [{str(ts)[:10]}] {q_mark} {q}")

    return "\n".join(output)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERPRETATION QUERY TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def query_signature_phrases(category: str = None, limit: int = 50) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/signature-phrases.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/signature-phrases.md\n\nUse semantic_search() or search_conversations() for fresh queries."


@mcp.tool()
def query_insights(month: str = None, category: str = None, limit: int = 20) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/insights.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/insights.md\n\nUse semantic_search() or search_conversations() for fresh queries."


@mcp.tool()
def query_questions(month: str = None, category: str = None, limit: int = 20) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/questions.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/questions.md\n\nUse semantic_search() or search_conversations() for fresh queries."


@mcp.tool()
def query_monthly_themes(limit: int = 12) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/monthly-themes.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/monthly-themes.md\n\nUse semantic_search() or search_conversations() for fresh queries."


@mcp.tool()
def query_intellectual_evolution(limit: int = 11) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/intellectual-evolution.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/intellectual-evolution.md\n\nUse semantic_search() or search_conversations() for fresh queries."


@mcp.tool()
def query_weekly_summaries(month: str = None, limit: int = 10) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/weekly-synthesis.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/weekly-synthesis.md\n\nUse semantic_search() or search_conversations() for fresh queries."


@mcp.tool()
def query_mood(month: str = None, limit: int = 14) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/mood-patterns.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/mood-patterns.md\n\nUse semantic_search() or search_conversations() for fresh queries."


# ═══════════════════════════════════════════════════════════════════════════════
# HIDDEN GEM INTERPRETATION TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def query_accomplishments(month: str = None, limit: int = 10) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/accomplishments.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/accomplishments.md\n\nUse semantic_search() or search_conversations() for fresh queries."


@mcp.tool()
def query_glossary(term: str = None) -> str:
    """DEPRECATED: Materialized to wiki. Read content/07-meta/glossary.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/07-meta/glossary.md\n\nUse semantic_search() or search_conversations() for fresh queries."


@mcp.tool()
def query_phrase_context(phrase: str = None, limit: int = 20) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/phrase-context.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/phrase-context.md\n\nUse semantic_search() or search_conversations() for fresh queries."


@mcp.tool()
def query_project_arcs(project: str = None, limit: int = 20) -> str:
    """DEPRECATED: Materialized to wiki. Read content/03-case-studies/project-arcs.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/03-case-studies/project-arcs.md\n\nUse semantic_search() or search_conversations() for fresh queries."


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
def search_markdown(query: str, limit: int = 15) -> str:
    """
    Search 39K harvested markdown documents by keyword.
    Searches content, titles, and filenames.
    Returns documents with metadata (project, energy, depth, etc.)
    """
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


@mcp.tool()
def get_breakthrough_docs(limit: int = 20) -> str:
    """
    Get documents with BREAKTHROUGH energy.
    These are aha moments, insights, and realizations.
    Sorted by depth score for maximum intellectual value.
    """
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


@mcp.tool()
def get_deep_docs(min_depth: int = 70, limit: int = 20) -> str:
    """
    Get documents with high depth scores (substantive content).
    Depth combines: length, structure, SEED concepts, decisions, voice, energy.
    """
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


@mcp.tool()
def get_project_docs(project: str, limit: int = 20) -> str:
    """
    Get documents for a specific project (sparkii, wotc, intellectual_dna, etc.)
    Sorted by depth score to surface the best content.
    """
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


@mcp.tool()
def get_open_todos(project: str = None, limit: int = 20) -> str:
    """
    Get documents with the most open TODOs (unfinished work).
    Optionally filter by project.
    """
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
    Search across ALL sources: conversations, YouTube, GitHub, markdown.
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

    # 2. YouTube (keyword search)
    try:
        yt_db = get_youtube_db()
        if yt_db:
            yt_results = yt_db.execute("""
                SELECT 'youtube' as source, title,
                       COALESCE(LEFT(full_transcript, 500), '') as content,
                       CAST(watched_date AS VARCHAR) as date,
                       0.5 as score
                FROM youtube
                WHERE title ILIKE ?
                   OR channel_name ILIKE ?
                   OR full_transcript ILIKE ?
                ORDER BY watched_date DESC NULLS LAST
                LIMIT 3
            """, [f'%{query}%', f'%{query}%', f'%{query}%']).fetchall()
            results.extend(yt_results)
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

@mcp.tool()
def query_conversation_titles(month: str = None, limit: int = 20) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/conversation-titles.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/conversation-titles.md\n\nUse semantic_search() or search_conversations() for fresh queries."


@mcp.tool()
def query_weekly_expertise(month: str = None, limit: int = 12) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/weekly-expertise.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/weekly-expertise.md\n\nUse semantic_search() or search_conversations() for fresh queries."


@mcp.tool()
def query_tool_preferences(month: str = None, limit: int = 12) -> str:
    """BROKEN: Only returns message counts, no actual preference data."""
    return "⚠️ BROKEN: This tool only returns message counts, not actual preferences.\n\nUse query_tool_stacks() for technology stack evolution."


@mcp.tool()
def query_tool_combos(limit: int = 10) -> str:
    """BROKEN: Empty data - no stack combinations recorded."""
    return "⚠️ BROKEN: No stack combinations were recorded in the data.\n\nUse query_tool_stacks() for technology stack evolution."


@mcp.tool()
def query_intellectual_themes(theme: str = None) -> str:
    """DEPRECATED: Materialized to wiki. Read content/01-frameworks/themes.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/01-frameworks/themes.md\n\nUse semantic_search() or search_conversations() for fresh queries."


@mcp.tool()
def query_youtube_links(month: str = None, limit: int = 15) -> str:
    """BROKEN: Tool returns parse errors. Use youtube_search() instead."""
    return "⚠️ BROKEN: This tool returns parse errors for all data.\n\nUse youtube_search() or youtube_semantic_search() instead."


@mcp.tool()
def query_problem_chains(month: str = None, limit: int = 15) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/problem-chains.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/problem-chains.md\n\nUse semantic_search() or search_conversations() for fresh queries."


# ═══════════════════════════════════════════════════════════════════════════════
# SPEND ANALYSIS TOOLS (Phase 2 55x Mining - Jan 2026)
# Mining 732K OpenRouter API calls for insights
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def query_model_efficiency(top_n: int = 20, sort_by: str = "cost") -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/model-efficiency.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/model-efficiency.md\n\nUse semantic_search() or search_conversations() for fresh queries."


@mcp.tool()
def query_provider_performance(month: str = None) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/provider-performance.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/provider-performance.md\n\nUse semantic_search() or search_conversations() for fresh queries."


@mcp.tool()
def query_spend_temporal(month: str = None, view: str = "daily") -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/spend-temporal.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/spend-temporal.md\n\nUse semantic_search() or search_conversations() for fresh queries."


@mcp.tool()
def query_spend_summary() -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/spend-analysis.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/spend-analysis.md\n\nUse semantic_search() or search_conversations() for fresh queries."


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSATION ANALYSIS TOOLS (Phase 3 55x Mining - Jan 2026)
# Mining 353K messages for threads, Q&A patterns, and corrections
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def query_conversation_stats(top_n: int = 20, sort_by: str = "messages") -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/conversation-stats.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/conversation-stats.md\n\nUse semantic_search() or search_conversations() for fresh queries."


@mcp.tool()
def query_deep_conversations(min_messages: int = 100, limit: int = 20) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/deep-conversations.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/deep-conversations.md\n\nUse semantic_search() or search_conversations() for fresh queries."


@mcp.tool()
def query_question_patterns(month: str = None) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/question-patterns.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/question-patterns.md\n\nUse semantic_search() or search_conversations() for fresh queries."


@mcp.tool()
def query_correction_patterns(month: str = None) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/correction-patterns.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/correction-patterns.md\n\nUse semantic_search() or search_conversations() for fresh queries."


@mcp.tool()
def query_conversation_summary() -> str:
    """
    Get a comprehensive conversation analysis summary.
    """
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

@mcp.tool()
def query_search_patterns(month: str = None, intent: str = None) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/search-patterns.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/search-patterns.md\n\nUse search_google_searches() for fresh queries."


@mcp.tool()
def query_browsing_patterns(category: str = None) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/browsing-patterns.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/browsing-patterns.md\n\nUse search_google_visits() for fresh queries."


@mcp.tool()
def query_research_velocity(month: str = None) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/research-velocity.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/research-velocity.md\n\nUse search_google_searches() for fresh queries."


@mcp.tool()
def query_curiosity_terms(month: str = None, limit: int = 30) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/curiosity-terms.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/curiosity-terms.md\n\nUse search_google_searches() for fresh queries."


@mcp.tool()
def query_behavioral_summary() -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/behavioral-patterns.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/behavioral-patterns.md\n\nUse search_google_searches() or search_google_visits() for fresh queries."


# ═══════════════════════════════════════════════════════════════════════════════
# CODE PRODUCTIVITY TOOLS (Phase 5 55x Mining - Jan 2026)
# Mining 1.4K commits + 132 repos for productivity patterns
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def query_code_velocity(month: str = None, view: str = "monthly") -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/code-velocity.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/code-velocity.md\n\nUse github_project_timeline() or code_to_conversation() for fresh queries."


@mcp.tool()
def query_repo_stats(active_only: bool = True, limit: int = 20) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/repo-stats.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/repo-stats.md\n\nUse github_project_timeline() for fresh queries."


@mcp.tool()
def query_language_stats() -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/language-stats.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/language-stats.md\n\nUse github_project_timeline() for fresh queries."


@mcp.tool()
def query_commit_patterns(commit_type: str = None) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/commit-patterns.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/commit-patterns.md\n\nUse search_file_changes() for fresh queries."


@mcp.tool()
def query_high_productivity_days(limit: int = 20) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/high-productivity-days.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/high-productivity-days.md\n\nUse search_file_changes() for fresh queries."


@mcp.tool()
def query_code_summary() -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/code-productivity.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/code-productivity.md\n\nUse search_file_changes() or github_project_timeline() for fresh queries."


# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN KNOWLEDGE TOOLS (Phase 6 55x Mining - Jan 2026)
# Mining 5.5K documents (6.3M words) for knowledge patterns
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def query_markdown_stats(category: str = None, limit: int = 20) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/markdown-stats.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/markdown-stats.md\n\nUse search_markdown() for fresh queries."


@mcp.tool()
def query_markdown_categories() -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/markdown-categories.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/markdown-categories.md\n\nUse search_markdown() for fresh queries."


@mcp.tool()
def query_markdown_projects() -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/markdown-projects.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/markdown-projects.md\n\nUse search_markdown() for fresh queries."


@mcp.tool()
def query_curated_docs(collection: str = "ip") -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/curated-docs.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/curated-docs.md\n\nUse search_markdown() for fresh queries."


@mcp.tool()
def query_markdown_summary() -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/markdown-corpus.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/markdown-corpus.md\n\nUse search_markdown() for fresh queries."


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7: CROSS-DIMENSIONAL SYNTHESIS (2026-01-11)
# Productivity matrix, learning arcs, project success, unified timeline
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def query_productivity_matrix(month: str = None, view: str = "daily", top_n: int = 20) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/productivity-matrix.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/productivity-matrix.md\n\nUse query_accomplishments() or search_conversations() for fresh queries."


@mcp.tool()
def query_learning_arcs(month: str = None, topic: str = None, limit: int = 20) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/learning-arcs.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/learning-arcs.md\n\nUse youtube_search() or search_google_searches() for fresh queries."


@mcp.tool()
def query_project_success(category: str = None, limit: int = 20) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/project-success.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/project-success.md\n\nUse github_project_timeline() for fresh queries."


@mcp.tool()
def query_unified_timeline(year: str = None, limit: int = 24) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/unified-timeline.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/unified-timeline.md\n\nUse unified_search() for fresh queries."


@mcp.tool()
def query_cross_dimensional_summary() -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/cross-dimensional.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/cross-dimensional.md\n\nUse unified_search() for fresh queries."


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 9: DISCOVERY PIPELINES (2026-01-11)
# Anomalies, trends, recommendations, weekly synthesis
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def query_anomalies(anomaly_type: str = None, limit: int = 20) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/anomalies.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/anomalies.md\n\nUse unified_search() for fresh queries."


@mcp.tool()
def query_trends(trend_type: str = None, limit: int = 20) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/trends.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/trends.md\n\nUse semantic_search() for fresh queries."


@mcp.tool()
def query_recommendations(rec_type: str = None, limit: int = 20) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/recommendations.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/recommendations.md\n\nUse semantic_search() for fresh queries."


@mcp.tool()
def query_weekly_synthesis(weeks: int = 10) -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/weekly-synthesis.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/weekly-synthesis.md\n\nUse search_conversations() for fresh queries."


@mcp.tool()
def query_discovery_summary() -> str:
    """DEPRECATED: Materialized to wiki. Read content/02-evidence/discovery-insights.md"""
    return "📚 DEPRECATED: This data is now in the wiki.\n\nRead: content/02-evidence/discovery-insights.md\n\nUse unified_search() for fresh queries."


# ═══════════════════════════════════════════════════════════════════════════════
# RESOURCES (static data endpoints)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.resource("brain://stats")
def resource_stats() -> str:
    """Current brain statistics."""
    return brain_stats("all")

@mcp.resource("brain://principles")
def resource_principles() -> str:
    """List of SEED principles."""
    return list_principles()

@mcp.resource("brain://embeddings")
def resource_embeddings() -> str:
    """Embedding statistics."""
    return brain_stats("embeddings")


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
