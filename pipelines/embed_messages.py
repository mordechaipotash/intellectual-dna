#!/usr/bin/env python3
"""
Embedding Pipeline for Intellectual DNA

Embeds user messages using sentence-transformers (nomic-embed-text-v1.5).
Stores in DuckDB with HNSW vector index for fast similarity search.

Usage:
    python -m pipelines embed                    # Embed 5 batches
    python -m pipelines embed --batches 100      # Embed 100 batches
    python -m pipelines embed search "query"     # Search embeddings
"""

import sys
import time
from pathlib import Path

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
from config import PARQUET_PATH, EMBEDDINGS_DB, EMBEDDING_DIM, EMBEDDING_BATCH_SIZE
from pipelines.utils.id_utils import EFFECTIVE_ID_SQL
from pipelines.utils.embedding_utils import get_embedding, get_embedding_model


def setup_database(con: duckdb.DuckDBPyConnection):
    """Create embeddings table with vector support."""
    con.execute("INSTALL vss")
    con.execute("LOAD vss")
    con.execute("SET hnsw_enable_experimental_persistence = true")

    con.execute(f"""
        CREATE TABLE IF NOT EXISTS message_embeddings (
            message_id VARCHAR PRIMARY KEY,
            conversation_id VARCHAR,
            conversation_title VARCHAR,
            content TEXT,
            role VARCHAR,
            year INTEGER,
            month INTEGER,
            embedding FLOAT[{EMBEDDING_DIM}],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    print("Database setup complete")


def create_hnsw_index(con: duckdb.DuckDBPyConnection):
    """Create HNSW index for fast similarity search."""
    print("Creating HNSW index...")
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
        ON message_embeddings
        USING HNSW (embedding)
        WITH (metric = 'cosine')
    """)
    print("HNSW index created")


def get_unembedded_messages(con: duckdb.DuckDBPyConnection, limit: int = 1000) -> list[dict]:
    """Get user messages that haven't been embedded yet."""
    # Load existing message IDs
    try:
        existing = set(row[0] for row in con.execute(
            "SELECT message_id FROM message_embeddings"
        ).fetchall())
    except:
        existing = set()

    # Query source parquet - get all user messages with synthetic ID for NULLs
    source_con = duckdb.connect()

    query = f'''
        SELECT
            {EFFECTIVE_ID_SQL} as effective_id,
            conversation_id,
            conversation_title,
            content,
            role,
            year,
            month
        FROM read_parquet("{PARQUET_PATH}")
        WHERE role = 'user'
        AND content IS NOT NULL
        AND char_count > 20
        ORDER BY msg_timestamp DESC
    '''

    rows = source_con.execute(query).fetchall()

    # Filter out already embedded and collect results
    results = []
    for row in rows:
        effective_id = row[0]
        if effective_id not in existing:
            results.append({
                "message_id": effective_id,
                "conversation_id": row[1],
                "conversation_title": row[2],
                "content": row[3],
                "role": row[4],
                "year": row[5],
                "month": row[6]
            })
            if len(results) >= limit:
                break

    return results


def embed_batch(con: duckdb.DuckDBPyConnection, messages: list[dict]) -> int:
    """Embed a batch of messages and store in database."""
    embedded = 0

    for msg in messages:
        embedding = get_embedding(msg["content"])
        if embedding:
            con.execute("""
                INSERT OR REPLACE INTO message_embeddings
                (message_id, conversation_id, conversation_title, content, role, year, month, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                msg["message_id"],
                msg["conversation_id"],
                msg["conversation_title"],
                msg["content"][:5000],
                msg["role"],
                msg["year"],
                msg["month"],
                embedding
            ])
            embedded += 1

    return embedded


def get_stats(con: duckdb.DuckDBPyConnection) -> dict:
    """Get embedding statistics."""
    total = con.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0]
    by_year = con.execute("""
        SELECT year, COUNT(*)
        FROM message_embeddings
        GROUP BY year
        ORDER BY year
    """).fetchall()

    return {
        "total_embedded": total,
        "by_year": dict(by_year)
    }


def run_embedding_pipeline(batch_count: int = 10):
    """Run the embedding pipeline."""
    from config import EMBEDDING_MODEL

    print(f"Starting embedding pipeline...")
    print(f"Database: {EMBEDDINGS_DB}")
    print(f"Model: {EMBEDDING_MODEL}")

    con = duckdb.connect(str(EMBEDDINGS_DB))
    setup_database(con)

    total_embedded = 0

    for batch_num in range(batch_count):
        messages = get_unembedded_messages(con, limit=EMBEDDING_BATCH_SIZE)

        if not messages:
            print("No more messages to embed")
            break

        print(f"\nBatch {batch_num + 1}: Embedding {len(messages)} messages...")
        start = time.time()

        embedded = embed_batch(con, messages)
        total_embedded += embedded

        elapsed = time.time() - start
        rate = embedded / elapsed if elapsed > 0 else 0

        print(f"  Embedded: {embedded} | Rate: {rate:.1f} msg/s | Total: {total_embedded}")

    if total_embedded > 0:
        create_hnsw_index(con)

    stats = get_stats(con)
    print(f"\n=== Final Stats ===")
    print(f"Total embedded: {stats['total_embedded']:,}")
    print(f"By year: {stats['by_year']}")

    con.close()
    return stats


def search_similar(query: str, limit: int = 10) -> list[dict]:
    """Search for similar messages using vector similarity."""
    embedding = get_embedding(query)
    if not embedding:
        return []

    con = duckdb.connect(str(EMBEDDINGS_DB))
    con.execute("LOAD vss")

    results = con.execute(f"""
        SELECT
            message_id,
            conversation_id,
            conversation_title,
            content,
            year,
            month,
            array_cosine_similarity(embedding, ?::FLOAT[{EMBEDDING_DIM}]) as similarity
        FROM message_embeddings
        ORDER BY similarity DESC
        LIMIT ?
    """, [embedding, limit]).fetchall()

    con.close()

    return [
        {
            "message_id": r[0],
            "conversation_id": r[1],
            "title": r[2],
            "content": r[3][:500] + "..." if len(r[3]) > 500 else r[3],
            "year": r[4],
            "month": r[5],
            "similarity": round(r[6], 4)
        }
        for r in results
    ]


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Embedding pipeline')
    parser.add_argument('command', nargs='?', default='embed', choices=['embed', 'search', 'stats'])
    parser.add_argument('query', nargs='*', help='Search query (for search command)')
    parser.add_argument('--batches', type=int, default=5, help='Number of batches to embed')
    args = parser.parse_args()

    if args.command == 'search':
        query = ' '.join(args.query) if args.query else "What is SHELET?"
        print(f"Searching for: {query}\n")
        results = search_similar(query, limit=5)
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['year']}-{r['month']:02d}] {r['title']}")
            print(f"   Similarity: {r['similarity']}")
            print(f"   {r['content'][:200]}...")
            print()
    elif args.command == 'stats':
        con = duckdb.connect(str(EMBEDDINGS_DB), read_only=True)
        stats = get_stats(con)
        print(f"Total embedded: {stats['total_embedded']:,}")
        print(f"By year: {stats['by_year']}")
        con.close()
    else:
        run_embedding_pipeline(batch_count=args.batches)


if __name__ == "__main__":
    main()
