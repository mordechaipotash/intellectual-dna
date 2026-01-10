#!/usr/bin/env python3
"""
Migrate embeddings from DuckDB VSS to LanceDB.

This migrates:
- message_embeddings (106K)
- commit_embeddings (1K)
- markdown_embeddings (137)

From 14GB DuckDB to ~500MB LanceDB.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import lancedb
import pyarrow as pa
import numpy as np
from tqdm import tqdm

from config import BASE, EMBEDDINGS_DB, EMBEDDING_DIM

# Paths
DUCKDB_PATH = EMBEDDINGS_DB.parent / "embeddings.duckdb.nosync"
LANCE_PATH = BASE / "vectors" / "brain.lance"


def migrate_table(source_con, lance_db, table_name: str, batch_size: int = 10000):
    """Migrate a single table from DuckDB to LanceDB."""

    # Get count
    count = source_con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"\nMigrating {table_name}: {count:,} rows")

    if count == 0:
        print(f"  Skipping empty table")
        return

    # Get schema
    schema_info = source_con.execute(f"DESCRIBE {table_name}").fetchall()
    print(f"  Schema: {[col[0] for col in schema_info]}")

    # Export to Arrow in batches
    all_batches = []

    for offset in tqdm(range(0, count, batch_size), desc=f"  Exporting"):
        # Fetch batch
        batch = source_con.execute(f"""
            SELECT * FROM {table_name}
            LIMIT {batch_size} OFFSET {offset}
        """).fetch_arrow_table()

        all_batches.append(batch)

    # Concatenate all batches
    full_table = pa.concat_tables(all_batches)
    print(f"  Exported {full_table.num_rows:,} rows")

    # Create LanceDB table
    lance_table_name = table_name.replace("_embeddings", "")

    try:
        # Drop if exists
        lance_db.drop_table(lance_table_name, ignore_missing=True)
    except:
        pass

    # Create table
    tbl = lance_db.create_table(lance_table_name, full_table)
    print(f"  Created LanceDB table: {lance_table_name}")

    # Create vector index
    print(f"  Creating IVF_PQ index...")
    try:
        tbl.create_index(
            "embedding",
            index_type="IVF_PQ",
            num_partitions=min(256, count // 100 + 1),  # Scale partitions to data size
            num_sub_vectors=min(96, EMBEDDING_DIM // 8),  # 768/8 = 96
        )
        print(f"  Index created!")
    except Exception as e:
        print(f"  Index creation failed (will use brute force): {e}")

    return tbl


def main():
    print("=" * 60)
    print("MIGRATING DUCKDB VSS → LANCEDB")
    print("=" * 60)

    # Verify source exists
    if not DUCKDB_PATH.exists():
        print(f"ERROR: Source not found: {DUCKDB_PATH}")
        return

    print(f"\nSource: {DUCKDB_PATH}")
    print(f"  Size: {DUCKDB_PATH.stat().st_size / 1024 / 1024 / 1024:.2f} GB")

    # Create destination directory
    LANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nDestination: {LANCE_PATH}")

    # Connect to source
    source_con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    source_con.execute("LOAD vss")

    # List tables
    tables = source_con.execute("SHOW TABLES").fetchall()
    print(f"\nTables to migrate: {[t[0] for t in tables]}")

    # Connect to LanceDB
    lance_db = lancedb.connect(str(LANCE_PATH))

    # Migrate each table
    for (table_name,) in tables:
        if table_name.endswith("_embeddings"):
            migrate_table(source_con, lance_db, table_name)

    source_con.close()

    # Report final size
    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)

    # Calculate LanceDB size
    lance_size = sum(f.stat().st_size for f in LANCE_PATH.rglob("*") if f.is_file())
    print(f"\nDuckDB size: {DUCKDB_PATH.stat().st_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"LanceDB size: {lance_size / 1024 / 1024:.1f} MB")
    print(f"Compression ratio: {DUCKDB_PATH.stat().st_size / lance_size:.1f}x smaller")

    # List created tables
    print(f"\nLanceDB tables:")
    for name in lance_db.table_names():
        tbl = lance_db.open_table(name)
        print(f"  {name}: {tbl.count_rows():,} rows")


if __name__ == "__main__":
    main()
