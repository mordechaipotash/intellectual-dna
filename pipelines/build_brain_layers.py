#!/usr/bin/env python3
"""
Build brain layers (L0-L2) using the Onion Skin Architecture:
- L0 (index): Event pointers only - "What exists?"
- L1 (summary): Previews + embeddings - "Quick glance"
- L2 (content): Full text - "Read it"

L3 (deep) is handled separately as raw source files.
"""

import duckdb
from pathlib import Path

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
DATA_DIR = BASE_DIR / "data"
BRAIN_DIR = DATA_DIR / "facts" / "brain"
EMBEDDINGS_DB = BASE_DIR / "mordelab" / "02-monotropic-prosthetic" / "embeddings.duckdb"


def build_l0_index():
    """
    Build L0: Unified event index across all sources.
    Contains: event_id, event_type, timestamp, source, preview (50 chars)
    """
    print("\n" + "=" * 60)
    print("Building L0: index.parquet (unified event pointers)")
    print("=" * 60)

    con = duckdb.connect()
    output_path = BRAIN_DIR / "index.parquet"

    # Build unified index from all sources
    con.execute(f"""
        COPY (
            -- Conversation messages
            SELECT
                COALESCE(message_id, conversation_id || '_' || CAST(msg_index AS VARCHAR)) as event_id,
                'message' as event_type,
                COALESCE(msg_timestamp, created) as timestamp,
                source,
                role as subtype,
                conversation_id as parent_id,
                LEFT(COALESCE(content, ''), 50) as preview
            FROM '{DATA_DIR}/all_conversations.parquet'
            WHERE role = 'user'  -- Only index user messages for brain

            UNION ALL

            -- YouTube watches
            SELECT
                youtube_id as event_id,
                'youtube_watch' as event_type,
                watched_date as timestamp,
                'youtube' as source,
                channel_name as subtype,
                NULL as parent_id,
                LEFT(title, 50) as preview
            FROM '{DATA_DIR}/youtube_rows.parquet'
            WHERE watched_date IS NOT NULL

            UNION ALL

            -- GitHub commits
            SELECT
                sha as event_id,
                'commit' as event_type,
                timestamp,
                'github' as source,
                repo_name as subtype,
                repo_name as parent_id,
                LEFT(message, 50) as preview
            FROM '{DATA_DIR}/github_commits.parquet'

            UNION ALL

            -- Claude Code sessions
            SELECT
                conversation_id as event_id,
                'code_session' as event_type,
                first_timestamp as timestamp,
                'claude_code' as source,
                project as subtype,
                project as parent_id,
                LEFT(COALESCE(project, 'Unknown project'), 50) as preview
            FROM '{DATA_DIR}/claude_code_metadata.parquet'
            WHERE first_timestamp IS NOT NULL

            UNION ALL

            -- Google Searches
            SELECT
                search_id as event_id,
                'google_search' as event_type,
                timestamp,
                'google' as source,
                NULL as subtype,
                NULL as parent_id,
                LEFT(query, 50) as preview
            FROM '{DATA_DIR}/google_searches.parquet'
            WHERE timestamp IS NOT NULL

            UNION ALL

            -- YouTube Searches
            SELECT
                search_id as event_id,
                'youtube_search' as event_type,
                timestamp,
                'youtube' as source,
                NULL as subtype,
                NULL as parent_id,
                LEFT(query, 50) as preview
            FROM '{DATA_DIR}/youtube_searches.parquet'
            WHERE timestamp IS NOT NULL

            ORDER BY timestamp DESC
        ) TO '{output_path}' (FORMAT PARQUET)
    """)

    # Stats
    stats = con.execute(f"""
        SELECT
            event_type,
            COUNT(*) as count,
            MIN(timestamp) as earliest,
            MAX(timestamp) as latest
        FROM '{output_path}'
        GROUP BY event_type
        ORDER BY count DESC
    """).fetchall()

    print("\nEvent counts by type:")
    total = 0
    for event_type, count, earliest, latest in stats:
        print(f"  {event_type:<15} {count:>10,}  ({earliest} → {latest})")
        total += count
    print(f"  {'TOTAL':<15} {total:>10,}")

    return total


def build_l1_summary():
    """
    Build L1: Summary layer with previews and embeddings.
    Contains: event_id, event_type, preview (200 chars), embedding, metadata
    """
    print("\n" + "=" * 60)
    print("Building L1: summary.parquet (previews + embeddings)")
    print("=" * 60)

    con = duckdb.connect()
    output_path = BRAIN_DIR / "summary.parquet"
    index_path = BRAIN_DIR / "index.parquet"

    # Check if embeddings database exists
    if not EMBEDDINGS_DB.exists():
        print(f"  ⚠️ Embeddings DB not found: {EMBEDDINGS_DB}")
        print("  Building L1 without embeddings...")
        has_embeddings = False
    else:
        has_embeddings = True
        # Attach embeddings database
        con.execute(f"ATTACH '{EMBEDDINGS_DB}' AS emb (READ_ONLY)")

    if has_embeddings:
        # Build with embeddings joined
        con.execute(f"""
            COPY (
                SELECT
                    idx.event_id,
                    idx.event_type,
                    idx.timestamp,
                    idx.source,
                    idx.subtype,
                    idx.parent_id,
                    -- Extended preview (200 chars)
                    CASE
                        WHEN idx.event_type = 'message' THEN
                            LEFT((SELECT content FROM '{DATA_DIR}/all_conversations.parquet' c
                                  WHERE COALESCE(c.message_id, c.conversation_id || '_' || CAST(c.msg_index AS VARCHAR)) = idx.event_id
                                  LIMIT 1), 200)
                        WHEN idx.event_type = 'youtube_watch' THEN
                            (SELECT title FROM '{DATA_DIR}/youtube_rows.parquet' y
                             WHERE y.youtube_id = idx.event_id LIMIT 1)
                        WHEN idx.event_type = 'commit' THEN
                            (SELECT message FROM '{DATA_DIR}/github_commits.parquet' g
                             WHERE g.sha = idx.event_id LIMIT 1)
                        ELSE idx.preview
                    END as preview,
                    -- Embedding (if available)
                    emb.message_embeddings.embedding as embedding,
                    -- Has embedding flag
                    CASE WHEN emb.message_embeddings.embedding IS NOT NULL THEN true ELSE false END as has_embedding
                FROM '{index_path}' idx
                LEFT JOIN emb.message_embeddings ON idx.event_id = emb.message_embeddings.message_id
                ORDER BY idx.timestamp DESC
            ) TO '{output_path}' (FORMAT PARQUET)
        """)
    else:
        # Build without embeddings
        con.execute(f"""
            COPY (
                SELECT
                    idx.event_id,
                    idx.event_type,
                    idx.timestamp,
                    idx.source,
                    idx.subtype,
                    idx.parent_id,
                    idx.preview,
                    NULL as embedding,
                    false as has_embedding
                FROM '{index_path}' idx
                ORDER BY idx.timestamp DESC
            ) TO '{output_path}' (FORMAT PARQUET)
        """)

    # Stats
    count = con.execute(f"SELECT COUNT(*) FROM '{output_path}'").fetchone()[0]
    embedded = con.execute(f"SELECT COUNT(*) FROM '{output_path}' WHERE has_embedding = true").fetchone()[0]
    print(f"\n  Total events:    {count:,}")
    print(f"  With embeddings: {embedded:,} ({100*embedded/count:.1f}%)")

    return count


def build_l2_content():
    """
    Build L2: Full content layer.
    Contains: event_id, event_type, full_content, word_count, metadata
    """
    print("\n" + "=" * 60)
    print("Building L2: content.parquet (full text)")
    print("=" * 60)

    con = duckdb.connect()
    output_path = BRAIN_DIR / "content.parquet"
    index_path = BRAIN_DIR / "index.parquet"

    # Build content layer - full text for each event type
    con.execute(f"""
        COPY (
            -- Messages with full content
            SELECT
                COALESCE(c.message_id, c.conversation_id || '_' || CAST(c.msg_index AS VARCHAR)) as event_id,
                'message' as event_type,
                c.content as full_content,
                c.word_count,
                c.char_count,
                c.has_code,
                c.has_url,
                c.has_question
            FROM '{DATA_DIR}/all_conversations.parquet' c
            WHERE c.role = 'user'

            UNION ALL

            -- YouTube with full transcript
            SELECT
                y.youtube_id as event_id,
                'youtube_watch' as event_type,
                y.full_transcript as full_content,
                y.transcript_word_count as word_count,
                LENGTH(y.full_transcript) as char_count,
                false as has_code,
                true as has_url,
                false as has_question
            FROM '{DATA_DIR}/youtube_rows.parquet' y

            UNION ALL

            -- Commits with full message
            SELECT
                g.sha as event_id,
                'commit' as event_type,
                g.message as full_content,
                LENGTH(g.message) - LENGTH(REPLACE(g.message, ' ', '')) + 1 as word_count,
                LENGTH(g.message) as char_count,
                true as has_code,
                false as has_url,
                false as has_question
            FROM '{DATA_DIR}/github_commits.parquet' g

            UNION ALL

            -- Google Searches
            SELECT
                s.search_id as event_id,
                'google_search' as event_type,
                s.query as full_content,
                s.word_count,
                s.char_count,
                false as has_code,
                false as has_url,
                s.has_question
            FROM '{DATA_DIR}/google_searches.parquet' s

            UNION ALL

            -- YouTube Searches
            SELECT
                ys.search_id as event_id,
                'youtube_search' as event_type,
                ys.query as full_content,
                ys.word_count,
                ys.char_count,
                false as has_code,
                false as has_url,
                ys.has_question
            FROM '{DATA_DIR}/youtube_searches.parquet' ys

        ) TO '{output_path}' (FORMAT PARQUET)
    """)

    # Stats
    count = con.execute(f"SELECT COUNT(*) FROM '{output_path}'").fetchone()[0]
    total_words = con.execute(f"SELECT SUM(word_count) FROM '{output_path}'").fetchone()[0]
    print(f"\n  Total events: {count:,}")
    print(f"  Total words:  {total_words:,}")

    return count


def verify_layers():
    """Verify all layers are consistent."""
    print("\n" + "=" * 60)
    print("Verifying layer consistency")
    print("=" * 60)

    con = duckdb.connect()

    l0_count = con.execute(f"SELECT COUNT(*) FROM '{BRAIN_DIR}/index.parquet'").fetchone()[0]
    l1_count = con.execute(f"SELECT COUNT(*) FROM '{BRAIN_DIR}/summary.parquet'").fetchone()[0]
    l2_count = con.execute(f"SELECT COUNT(*) FROM '{BRAIN_DIR}/content.parquet'").fetchone()[0]

    print(f"\n  L0 (index):   {l0_count:,} events")
    print(f"  L1 (summary): {l1_count:,} events")
    print(f"  L2 (content): {l2_count:,} events")

    # Check sizes
    import os
    l0_size = os.path.getsize(BRAIN_DIR / "index.parquet") / 1024 / 1024
    l1_size = os.path.getsize(BRAIN_DIR / "summary.parquet") / 1024 / 1024
    l2_size = os.path.getsize(BRAIN_DIR / "content.parquet") / 1024 / 1024

    print(f"\n  L0 size: {l0_size:,.1f} MB")
    print(f"  L1 size: {l1_size:,.1f} MB")
    print(f"  L2 size: {l2_size:,.1f} MB")

    if l0_count == l1_count:
        print("\n  ✅ L0 and L1 counts match!")
    else:
        print(f"\n  ⚠️ L0 ({l0_count}) and L1 ({l1_count}) counts differ")

    return True


if __name__ == '__main__':
    BRAIN_DIR.mkdir(parents=True, exist_ok=True)

    build_l0_index()
    build_l1_summary()
    build_l2_content()
    verify_layers()
