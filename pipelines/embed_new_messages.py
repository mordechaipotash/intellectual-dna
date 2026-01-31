#!/usr/bin/env python3
"""
Embed new messages into LanceDB (incremental).

Reads: data/all_conversations.parquet
Updates: vectors/brain.lance/message.lance

Only embeds messages not already in LanceDB.
Run after sync_clawdbot.py to update vectors.

Usage:
  python pipelines/embed_new_messages.py          # Incremental
  python pipelines/embed_new_messages.py full     # Re-embed all
  python pipelines/embed_new_messages.py stats    # Show stats
"""

import sys
import os
import gc
import resource
from pathlib import Path

# === MEMORY LIMITS ===
MAX_MEMORY_GB = 8  # Hard limit - kill if exceeded
resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY_GB * 1024**3, MAX_MEMORY_GB * 1024**3))

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

import pandas as pd
import lancedb

# Add parent for config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PARQUET_PATH,
    LANCE_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
)

EMBEDDING_MAX_CHARS = 8000
EMBEDDING_BATCH_SIZE = 25  # Smaller batches for stability
CHUNK_SIZE = 5000  # Process in chunks to limit memory

# Embedding model (lazy loaded)
_model = None

def get_model():
    global _model
    if _model is None:
        print(f"Loading {EMBEDDING_MODEL}...", flush=True)
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
        print("Model loaded!", flush=True)
    return _model


def get_existing_ids(db) -> set:
    """Get message IDs already in LanceDB."""
    try:
        tbl = db.open_table("message")
        # Only fetch the columns we need
        df = tbl.to_pandas()[["conversation_id", "msg_index"]]
        ids = set(
            f"{row['conversation_id']}_{row.get('msg_index', 0)}" 
            for _, row in df.iterrows()
        )
        del df
        gc.collect()
        return ids
    except Exception:
        return set()


def embed_messages(full: bool = False):
    """Embed messages into LanceDB."""
    
    # Load conversations - only columns we need
    print(f"Loading {PARQUET_PATH}...", flush=True)
    columns = ["conversation_id", "conversation_title", "msg_index", "year", "month", "content", "source", "role"]
    df = pd.read_parquet(PARQUET_PATH, columns=columns)
    
    # Filter to user messages only (we embed what Mordechai said)
    user_df = df[df["role"] == "user"].copy()
    del df
    gc.collect()
    
    print(f"Total user messages: {len(user_df)}", flush=True)
    
    # Connect to LanceDB
    LANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(LANCE_PATH))
    
    # Get existing IDs (for incremental)
    if not full:
        existing = get_existing_ids(db)
        print(f"Already embedded: {len(existing)}", flush=True)
        
        # Create composite key for filtering
        user_df["_key"] = user_df.apply(
            lambda r: f"{r['conversation_id']}_{r.get('msg_index', 0)}", axis=1
        )
        user_df = user_df[~user_df["_key"].isin(existing)]
        user_df = user_df.drop(columns=["_key"])
        del existing
        gc.collect()
        
        print(f"New messages to embed: {len(user_df)}", flush=True)
    
    if len(user_df) == 0:
        print("Nothing to embed!", flush=True)
        return
    
    # Load model
    model = get_model()
    
    # Process in chunks to limit memory
    total_embedded = 0
    total_rows = len(user_df)
    
    for chunk_start in range(0, total_rows, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_rows)
        chunk_df = user_df.iloc[chunk_start:chunk_end]
        
        print(f"\n--- Chunk {chunk_start}-{chunk_end} of {total_rows} ---", flush=True)
        
        # Prepare data for embedding
        records = []
        texts = []
        
        for _, row in chunk_df.iterrows():
            content = str(row.get("content", ""))[:EMBEDDING_MAX_CHARS]
            if len(content) < 10:  # Skip tiny messages
                continue
            
            texts.append(content)
            records.append({
                "conversation_id": row.get("conversation_id"),
                "conversation_title": row.get("conversation_title"),
                "msg_index": row.get("msg_index", 0),
                "year": int(row.get("year", 0)),
                "month": int(row.get("month", 0)),
                "content": content,
                "source": row.get("source", "unknown"),
            })
        
        if not texts:
            continue
            
        print(f"Embedding {len(texts)} messages...", flush=True)
        
        # Batch embed
        all_embeddings = []
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i:i + EMBEDDING_BATCH_SIZE]
            try:
                embeddings = model.encode(batch, convert_to_numpy=True)
                all_embeddings.extend(embeddings.tolist())
            except Exception as e:
                print(f"  Error in batch {i}: {e}", flush=True)
                continue
        
        # Add embeddings to records
        for i, record in enumerate(records[:len(all_embeddings)]):
            record["vector"] = all_embeddings[i]
        
        records = records[:len(all_embeddings)]
        
        # Write chunk to LanceDB
        try:
            if (full or "message" not in db.table_names()) and chunk_start == 0:
                tbl = db.create_table("message", records, mode="overwrite")
            else:
                tbl = db.open_table("message")
                tbl.add(records)
            
            total_embedded += len(records)
            print(f"  Written {len(records)} vectors (total: {total_embedded})", flush=True)
        except Exception as e:
            print(f"  Error writing to LanceDB: {e}", flush=True)
        
        # Clean up chunk memory
        del records, texts, all_embeddings, chunk_df
        gc.collect()
    
    print(f"\n✅ Done! Embedded {total_embedded} messages.", flush=True)
    
    # Final stats
    try:
        tbl = db.open_table("message")
        print(f"Total vectors in message table: {tbl.count_rows()}", flush=True)
    except:
        pass


def stats():
    """Show embedding stats."""
    db = lancedb.connect(str(LANCE_PATH))
    
    print(f"LanceDB at: {LANCE_PATH}")
    tables = db.table_names() if hasattr(db, 'table_names') else list(db.list_tables())
    print(f"Tables: {tables}")
    
    for table_name in tables:
        tbl = db.open_table(table_name)
        count = tbl.count_rows()
        print(f"\n{table_name}: {count:,} vectors")
        
        if table_name == "message":
            df = tbl.to_pandas()
            if "source" in df.columns:
                print("  By source:")
                for src, cnt in df["source"].value_counts().items():
                    print(f"    {src}: {cnt:,}")
            if "year" in df.columns:
                print("  By year:")
                for yr, cnt in sorted(df["year"].value_counts().items()):
                    print(f"    {int(yr)}: {cnt:,}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        stats()
    elif len(sys.argv) > 1 and sys.argv[1] == "full":
        embed_messages(full=True)
    else:
        embed_messages(full=False)
