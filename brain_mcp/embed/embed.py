#!/usr/bin/env python3
"""
brain-mcp — Embedding pipeline.

Reads all_conversations.parquet and embeds user messages into LanceDB
for semantic search. Supports incremental (only new messages), full
re-embed, rebuild, and stats modes.

Uses fastembed via the EmbeddingProvider ABC for lightweight,
PyTorch-free embedding.

All paths and model settings come from config.toml via config.py.

Usage:
    python -m embed.embed           # Incremental (default)
    python -m embed.embed full      # Re-embed everything
    python -m embed.embed stats     # Show current stats
"""

import sys
import os
import gc
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

import pandas as pd
import lancedb

from brain_mcp.config import get_config
from brain_mcp.ingest.noise_filter import is_noise_message


def get_provider(force_provider=None):
    """Get the embedding provider (cached after first call).

    Args:
        force_provider: Force 'fastembed' or 'sentence-transformers'.
    """
    from brain_mcp.embed.provider import get_provider as _get_provider
    cfg = get_config()
    print(f"Loading embedding model...", flush=True)
    provider = _get_provider(
        model_name=cfg.embedding.model,
        force_provider=force_provider,
    )
    print(f"Model loaded! ({provider.provider_name})", flush=True)
    return provider


def get_existing_ids(db) -> set:
    """Get message IDs already in LanceDB."""
    try:
        tbl = db.open_table("message")
        df = tbl.to_pandas()[["message_id"]]
        ids = set(df["message_id"].tolist())
        del df
        gc.collect()
        return ids
    except Exception as e:
        print(f"Warning: Could not load existing IDs: {e}", flush=True)
        return set()


def rebuild_table(db):
    """Drop and recreate the LanceDB message table."""
    try:
        if "message" in db.table_names():
            db.drop_table("message")
            print("Dropped existing 'message' table.", flush=True)
    except Exception as e:
        print(f"Warning: Could not drop table: {e}", flush=True)


def _check_dimension_mismatch(db, provider):
    """Check if existing vectors have a different dimension than current provider.

    Returns (mismatch: bool, existing_dim: int or None).
    """
    try:
        if "message" not in db.table_names():
            return False, None
        tbl = db.open_table("message")
        if tbl.count_rows() == 0:
            return False, None
        # Sample one row to get embedding dimension
        sample = tbl.to_pandas().head(1)
        if "embedding" in sample.columns:
            existing_emb = sample["embedding"].iloc[0]
            existing_dim = len(existing_emb) if existing_emb is not None else None
            if existing_dim and existing_dim != provider.dimension:
                return True, existing_dim
    except Exception:
        pass
    return False, None


def embed_messages(full: bool = False, rebuild: bool = False, force_provider: str = None):
    """Embed user messages into LanceDB."""
    cfg = get_config()

    parquet_path = cfg.parquet_path
    lance_path = cfg.lance_path
    batch_size = cfg.embedding.batch_size
    max_chars = cfg.embedding.max_chars
    chunk_size = min(batch_size * 5, 250)  # Smaller chunks for memory safety

    if not parquet_path.exists():
        print(f"Error: {parquet_path} not found. Run ingest first.", flush=True)
        sys.exit(1)

    # Memory warning
    import shutil
    _, _, free_disk = shutil.disk_usage(str(parquet_path.parent))
    free_gb = free_disk / (1024**3)
    if free_gb < 2:
        print(f"⚠️  Low disk space: {free_gb:.1f}GB free. Embedding needs ~1-2GB.", flush=True)
        sys.exit(1)

    # Load conversations — only columns we need
    print(f"Loading {parquet_path}...", flush=True)
    columns = [
        "conversation_id", "conversation_title", "message_id",
        "msg_index", "year", "month", "content", "source", "role",
    ]
    df = pd.read_parquet(parquet_path, columns=columns)

    # Filter to user messages only
    user_df = df[df["role"] == "user"].copy()
    del df
    gc.collect()

    print(f"Total user messages: {len(user_df)}", flush=True)

    # Deduplicate by message_id
    if "message_id" in user_df.columns:
        before = len(user_df)
        user_df = user_df.drop_duplicates(subset=["message_id"], keep="first")
        if before != len(user_df):
            print(f"Deduped: {before} → {len(user_df)}", flush=True)

    gc.collect()

    # Connect to LanceDB
    lance_path.parent.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(lance_path))

    # Load provider early to check dimensions
    provider = get_provider(force_provider=force_provider)

    # Check for dimension mismatch BEFORE proceeding
    if not rebuild:
        mismatch, existing_dim = _check_dimension_mismatch(db, provider)
        if mismatch:
            print(
                f"\n⚠️  DIMENSION MISMATCH: existing vectors are {existing_dim}d, "
                f"but current provider produces {provider.dimension}d vectors.",
                flush=True,
            )
            print(
                "   Existing vectors will NOT be deleted automatically.",
                flush=True,
            )
            print(
                "   To rebuild with the new provider, run:",
                flush=True,
            )
            print(
                "     brain-mcp embed --rebuild",
                flush=True,
            )
            print(
                "\n   Skipping embedding to protect existing vectors.\n",
                flush=True,
            )
            return

    # Handle --rebuild: drop and recreate the table
    if rebuild:
        rebuild_table(db)
        full = True  # Rebuild implies full re-embed

    # Filter noise messages
    print("Filtering noise messages...", flush=True)
    user_df["_is_noise"] = user_df["content"].apply(is_noise_message)
    noise_count = user_df["_is_noise"].sum()
    user_df = user_df[~user_df["_is_noise"]]
    user_df = user_df.drop(columns=["_is_noise"])
    print(f"Filtered {noise_count:,} noise messages, {len(user_df):,} remain", flush=True)

    # Generate message_id if missing
    if "message_id" not in user_df.columns:
        user_df["message_id"] = user_df.apply(
            lambda r: f"{r['conversation_id']}_{r.get('msg_index', 0)}", axis=1
        )

    # Incremental: skip already-embedded
    if not full:
        existing = get_existing_ids(db)
        print(f"Already embedded: {len(existing)}", flush=True)
        user_df = user_df[~user_df["message_id"].isin(existing)]
        del existing
        gc.collect()
        print(f"New messages to embed: {len(user_df)}", flush=True)

    if len(user_df) == 0:
        print("Nothing to embed!", flush=True)
        return

    # Process in chunks
    total_embedded = 0
    total_rows = len(user_df)

    for chunk_start in range(0, total_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_rows)
        chunk_df = user_df.iloc[chunk_start:chunk_end]

        print(f"\n--- Chunk {chunk_start}-{chunk_end} of {total_rows} ---", flush=True)

        records = []
        texts = []

        for _, row in chunk_df.iterrows():
            content = str(row.get("content", ""))[:max_chars]
            if len(content) < 10:
                continue

            texts.append(content)
            records.append({
                "message_id": row.get("message_id"),
                "conversation_id": row.get("conversation_id"),
                "conversation_title": row.get("conversation_title"),
                "content": content,
                "role": "user",
                "year": int(row.get("year", 0)),
                "month": int(row.get("month", 0)),
            })

        if not texts:
            continue

        print(f"Embedding {len(texts)} messages...", flush=True)

        # Batch embed using provider
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                embeddings = provider.embed_batch(batch)
                all_embeddings.extend(embeddings)
            except Exception as e:
                print(f"  Error in batch {i}: {e}", flush=True)
                continue

        # Attach embeddings to records
        from datetime import datetime
        now = datetime.now()
        for i, record in enumerate(records[:len(all_embeddings)]):
            record["embedding"] = all_embeddings[i]
            record["created_at"] = now

        records = records[:len(all_embeddings)]

        # Write to LanceDB
        try:
            if (full or "message" not in db.table_names()) and chunk_start == 0:
                db.create_table("message", records, mode="overwrite")
            else:
                tbl = db.open_table("message")
                tbl.add(records)

            total_embedded += len(records)
            print(f"  Written {len(records)} vectors (total: {total_embedded})", flush=True)
        except Exception as e:
            print(f"  Error writing to LanceDB: {e}", flush=True)

        # Memory cleanup
        del records, texts, all_embeddings, chunk_df
        gc.collect()

    print(f"\n✅ Done! Embedded {total_embedded} messages.", flush=True)

    # Final stats
    try:
        tbl = db.open_table("message")
        print(f"Total vectors in message table: {tbl.count_rows()}", flush=True)
    except Exception:
        pass


def stats():
    """Show embedding stats."""
    cfg = get_config()
    lance_path = cfg.lance_path

    if not lance_path.exists():
        print(f"No LanceDB found at {lance_path}")
        return

    db = lancedb.connect(str(lance_path))

    print(f"LanceDB at: {lance_path}")
    tables = db.table_names() if hasattr(db, "table_names") else list(db.list_tables())
    print(f"Tables: {tables}")

    for table_name in tables:
        tbl = db.open_table(table_name)
        count = tbl.count_rows()
        print(f"\n{table_name}: {count:,} vectors")

        if table_name == "message":
            df = tbl.to_pandas()
            if "year" in df.columns:
                print("  By year:")
                for yr, cnt in sorted(df["year"].value_counts().items()):
                    print(f"    {int(yr)}: {cnt:,}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "incremental"

    if mode == "stats":
        stats()
    elif mode == "full":
        embed_messages(full=True)
    elif mode == "rebuild":
        embed_messages(rebuild=True)
    else:
        embed_messages(full=False)
