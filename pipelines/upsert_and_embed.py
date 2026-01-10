#!/usr/bin/env python3
"""
Complete upsert and embed pipeline for JSONL conversations.
Handles import, deduplication, and embedding in one script.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import duckdb
import pandas as pd
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.utils.embedding_utils import get_embedding, get_embedding_model
from pipelines.utils.id_utils import EFFECTIVE_ID_SQL
from pipelines.import_claude_code import (
    find_recent_jsonl,
    parse_jsonl_file,
    extract_project_name
)
from config import (
    PARQUET_PATH,
    EMBEDDINGS_DB,
    CLAUDE_CODE_IMPORT_PARQUET,
    DATA_DIR
)


def upsert_conversations(days: Optional[int] = None, dry_run: bool = True) -> dict:
    """
    Import and upsert conversations from JSONL files.

    Args:
        days: Only import files modified in last N days (None = all files)
        dry_run: If True, show what would be imported without writing

    Returns:
        dict with stats: {files_found, messages_parsed, new_messages, duplicates}
    """
    print(f"{'='*60}")
    print(f"PHASE 1: IMPORT JSONL → PARQUET")
    print(f"{'='*60}\n")

    # Step 1: Find JSONL files
    print(f"🔍 Scanning for JSONL files...")
    if days:
        print(f"   Looking for files modified in last {days} days")

    jsonl_files = find_recent_jsonl(days=days)
    print(f"✅ Found {len(jsonl_files)} JSONL files\n")

    if len(jsonl_files) == 0:
        print("⚠️  No files to import. Exiting.")
        return {"files_found": 0, "messages_parsed": 0, "new_messages": 0, "duplicates": 0}

    # Step 2: Parse all JSONL files
    print(f"📖 Parsing JSONL files...")
    all_messages = []
    files_parsed = 0

    for jsonl_file in jsonl_files:
        try:
            messages = parse_jsonl_file(jsonl_file)
            all_messages.extend(messages)
            files_parsed += 1

            if files_parsed % 100 == 0:
                print(f"   Parsed {files_parsed}/{len(jsonl_files)} files ({len(all_messages)} messages)...")
        except Exception as e:
            print(f"   ⚠️  Error parsing {jsonl_file.name}: {e}")

    print(f"✅ Parsed {files_parsed} files → {len(all_messages)} messages\n")

    if len(all_messages) == 0:
        print("⚠️  No messages extracted. Exiting.")
        return {"files_found": len(jsonl_files), "messages_parsed": 0, "new_messages": 0, "duplicates": 0}

    # Step 3: Convert to DataFrame
    df_new = pd.DataFrame(all_messages)

    # Step 4: Deduplication and merge
    print(f"🔄 Deduplicating and merging with archive...")

    con = duckdb.connect(":memory:")

    # Register new data
    con.register("new_import", df_new)

    # Load existing archive if it exists
    if PARQUET_PATH.exists():
        print(f"   Loading existing archive: {PARQUET_PATH}")
        existing_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{PARQUET_PATH}')").fetchone()[0]
        print(f"   Existing messages: {existing_count:,}")

        # Merge with deduplication
        merge_query = f"""
        WITH combined AS (
            SELECT * FROM read_parquet('{PARQUET_PATH}')
            UNION ALL
            SELECT * FROM new_import
        ),
        deduplicated AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY COALESCE(message_id, md5(content || conversation_id || role))
                    ORDER BY created
                ) as rn
            FROM combined
        )
        SELECT * EXCLUDE (rn) FROM deduplicated WHERE rn = 1
        ORDER BY created
        """

        df_merged = con.execute(merge_query).df()
        new_count = len(df_merged)
        duplicates = (existing_count + len(df_new)) - new_count
        new_messages = new_count - existing_count

        print(f"   After deduplication: {new_count:,} total messages")
        print(f"   New messages: {new_messages:,}")
        print(f"   Duplicates removed: {duplicates:,}\n")
    else:
        print(f"   No existing archive found, creating new one")
        df_merged = df_new
        new_messages = len(df_new)
        duplicates = 0
        print(f"   Total messages: {len(df_merged):,}\n")

    # Step 5: Write to parquet
    if dry_run:
        print(f"🔍 DRY RUN - Would write to: {PARQUET_PATH}")
        print(f"   Use --import flag to actually write\n")
    else:
        # Backup existing file
        if PARQUET_PATH.exists():
            backup_path = PARQUET_PATH.parent / f"all_conversations.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            print(f"📦 Creating backup: {backup_path.name}")
            import shutil
            shutil.copy(PARQUET_PATH, backup_path)

        # Write merged data
        print(f"💾 Writing to: {PARQUET_PATH}")
        df_merged.to_parquet(PARQUET_PATH, index=False)
        print(f"✅ Successfully wrote {len(df_merged):,} messages\n")

    stats = {
        "files_found": len(jsonl_files),
        "messages_parsed": len(all_messages),
        "new_messages": new_messages,
        "duplicates": duplicates
    }

    return stats


def embed_new_messages(batch_limit: Optional[int] = None, batch_size: int = 50) -> dict:
    """
    Embed messages that haven't been embedded yet.

    Args:
        batch_limit: Max batches to process (None = all remaining)
        batch_size: Messages per batch

    Returns:
        dict with stats: {unembedded_found, batches_processed, messages_embedded}
    """
    print(f"{'='*60}")
    print(f"PHASE 2: EMBED NEW MESSAGES")
    print(f"{'='*60}\n")

    # Pre-warm the model
    print(f"🔥 Pre-warming embedding model...")
    model = get_embedding_model()
    print(f"✅ Model ready: {model}\n")

    # Connect to embeddings database
    con = duckdb.connect(str(EMBEDDINGS_DB))
    con.execute("INSTALL vss")
    con.execute("LOAD vss")
    con.execute("SET hnsw_enable_experimental_persistence = true")

    # Ensure embeddings table exists
    con.execute("""
        CREATE TABLE IF NOT EXISTS message_embeddings (
            message_id VARCHAR PRIMARY KEY,
            conversation_id VARCHAR,
            conversation_title VARCHAR,
            content TEXT,
            role VARCHAR,
            year INTEGER,
            month INTEGER,
            embedding FLOAT[768],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Get unembedded messages
    print(f"🔍 Finding unembedded messages...")

    # Get existing embedded IDs
    existing_ids = set()
    for row in con.execute("SELECT message_id FROM message_embeddings").fetchall():
        existing_ids.add(row[0])

    print(f"   Already embedded: {len(existing_ids):,} messages")

    # Query source parquet for user messages
    query = f"""
        SELECT
            {EFFECTIVE_ID_SQL} as effective_id,
            conversation_id,
            conversation_title,
            content,
            role,
            year,
            month
        FROM read_parquet('{PARQUET_PATH}')
        WHERE role = 'user'
        AND content IS NOT NULL
        AND char_count > 20
        ORDER BY msg_timestamp DESC
    """

    unembedded = []
    for row in con.execute(query).fetchall():
        effective_id = row[0]
        if effective_id not in existing_ids:
            unembedded.append({
                "message_id": effective_id,
                "conversation_id": row[1],
                "conversation_title": row[2],
                "content": row[3],
                "role": row[4],
                "year": row[5],
                "month": row[6]
            })

    print(f"✅ Found {len(unembedded):,} unembedded messages\n")

    if len(unembedded) == 0:
        print("✅ All messages are already embedded!")
        return {"unembedded_found": 0, "batches_processed": 0, "messages_embedded": 0}

    # Process in batches
    total_batches = (len(unembedded) + batch_size - 1) // batch_size
    if batch_limit:
        total_batches = min(total_batches, batch_limit)

    print(f"🚀 Processing {total_batches} batches ({batch_size} messages/batch)...")

    messages_embedded = 0
    batches_processed = 0

    for i in range(0, len(unembedded), batch_size):
        if batch_limit and batches_processed >= batch_limit:
            break

        batch = unembedded[i:i + batch_size]
        batch_num = batches_processed + 1

        print(f"   Batch {batch_num}/{total_batches}: ", end="", flush=True)

        for msg in batch:
            try:
                # Generate embedding
                embedding = get_embedding(msg["content"])

                if embedding:
                    # Insert into database
                    con.execute("""
                        INSERT OR REPLACE INTO message_embeddings
                        (message_id, conversation_id, conversation_title,
                         content, role, year, month, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        msg["message_id"],
                        msg["conversation_id"],
                        msg["conversation_title"],
                        msg["content"][:5000],  # Truncate for storage
                        msg["role"],
                        msg["year"],
                        msg["month"],
                        embedding
                    ])
                    messages_embedded += 1
            except Exception as e:
                print(f"\n   ⚠️  Error embedding message {msg['message_id']}: {e}")

        batches_processed += 1
        print(f"{len(batch)} messages embedded")

    print(f"\n✅ Embedded {messages_embedded:,} messages in {batches_processed} batches\n")

    # Create/update HNSW index
    print(f"🔧 Creating HNSW index for fast search...")
    try:
        con.execute("DROP INDEX IF EXISTS idx_embeddings_hnsw")
        con.execute("""
            CREATE INDEX idx_embeddings_hnsw
            ON message_embeddings
            USING HNSW (embedding)
            WITH (metric = 'cosine')
        """)
        print(f"✅ HNSW index created\n")
    except Exception as e:
        print(f"⚠️  Could not create index: {e}\n")

    con.close()

    stats = {
        "unembedded_found": len(unembedded),
        "batches_processed": batches_processed,
        "messages_embedded": messages_embedded
    }

    return stats


def main():
    """Main entry point for complete upsert and embed pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Upsert and embed JSONL conversations")
    parser.add_argument("--days", type=int, help="Import files from last N days (default: all)")
    parser.add_argument("--import", dest="do_import", action="store_true", help="Actually import (default: dry run)")
    parser.add_argument("--embed", action="store_true", help="Run embedding phase after import")
    parser.add_argument("--embed-only", action="store_true", help="Skip import, only embed")
    parser.add_argument("--batches", type=int, help="Max batches to embed (default: all)")
    parser.add_argument("--batch-size", type=int, default=50, help="Messages per batch (default: 50)")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"COMPLETE UPSERT & EMBED PIPELINE")
    print(f"{'='*60}\n")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Phase 1: Import (unless embed-only)
    if not args.embed_only:
        import_stats = upsert_conversations(
            days=args.days,
            dry_run=not args.do_import
        )

        print(f"📊 IMPORT SUMMARY:")
        print(f"   Files found: {import_stats['files_found']}")
        print(f"   Messages parsed: {import_stats['messages_parsed']}")
        print(f"   New messages: {import_stats['new_messages']}")
        print(f"   Duplicates: {import_stats['duplicates']}\n")

        # Only continue to embed if we actually imported
        if not args.do_import and args.embed:
            print(f"⚠️  Skipping embed phase (dry run mode)")
            print(f"   Use --import --embed to actually upsert and embed\n")
            return

    # Phase 2: Embed (if requested)
    if args.embed or args.embed_only:
        embed_stats = embed_new_messages(
            batch_limit=args.batches,
            batch_size=args.batch_size
        )

        print(f"📊 EMBEDDING SUMMARY:")
        print(f"   Unembedded found: {embed_stats['unembedded_found']}")
        print(f"   Batches processed: {embed_stats['batches_processed']}")
        print(f"   Messages embedded: {embed_stats['messages_embedded']}\n")

    print(f"{'='*60}")
    print(f"✅ PIPELINE COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
