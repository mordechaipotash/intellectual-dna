"""
Brain Status - Show current state of intellectual DNA.

Reports on:
- Total messages in parquet
- Embedding coverage
- Date ranges
- Recent activity
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
from config import PARQUET_PATH, EMBEDDINGS_DB, YOUTUBE_PARQUET
from pipelines.utils.id_utils import EFFECTIVE_ID_SQL


def show_status():
    """Display comprehensive brain status."""
    print("=" * 60)
    print("INTELLECTUAL DNA - BRAIN STATUS")
    print("=" * 60)
    print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check parquet
    if not PARQUET_PATH.exists():
        print(f"ERROR: Parquet not found at {PARQUET_PATH}")
        return

    con = duckdb.connect()

    # === CONVERSATIONS ===
    print("=== CONVERSATIONS ===")

    total_msgs = con.execute(f'''
        SELECT COUNT(*) FROM read_parquet("{PARQUET_PATH}")
    ''').fetchone()[0]
    print(f"Total messages: {total_msgs:,}")

    # By role
    roles = con.execute(f'''
        SELECT role, COUNT(*) as cnt
        FROM read_parquet("{PARQUET_PATH}")
        GROUP BY role
        ORDER BY cnt DESC
    ''').fetchall()
    for role, cnt in roles:
        print(f"  {role}: {cnt:,}")

    # Date range
    dates = con.execute(f'''
        SELECT MIN(created), MAX(created)
        FROM read_parquet("{PARQUET_PATH}")
    ''').fetchone()
    print(f"Date range: {dates[0]} to {dates[1]}")

    # Recent activity
    recent = con.execute(f'''
        SELECT COUNT(*)
        FROM read_parquet("{PARQUET_PATH}")
        WHERE created >= CURRENT_DATE - INTERVAL 7 DAY
    ''').fetchone()[0]
    print(f"Last 7 days: {recent:,} messages")

    # Unique conversations
    convos = con.execute(f'''
        SELECT COUNT(DISTINCT conversation_id)
        FROM read_parquet("{PARQUET_PATH}")
    ''').fetchone()[0]
    print(f"Unique conversations: {convos:,}")
    print()

    # === EMBEDDINGS ===
    print("=== EMBEDDINGS ===")

    if not EMBEDDINGS_DB.exists():
        print(f"Embeddings DB not found at {EMBEDDINGS_DB}")
    else:
        econ = duckdb.connect(str(EMBEDDINGS_DB), read_only=True)

        try:
            embedded = econ.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0]
            print(f"Embedded messages: {embedded:,}")

            # Embedding date range
            edates = econ.execute('''
                SELECT MIN(created_at), MAX(created_at)
                FROM message_embeddings
            ''').fetchone()
            print(f"Embedding dates: {edates[0]} to {edates[1]}")

            # By year
            by_year = econ.execute('''
                SELECT year, COUNT(*) as cnt
                FROM message_embeddings
                GROUP BY year
                ORDER BY year
            ''').fetchall()
            print("By year:", dict(by_year))

        except Exception as e:
            print(f"Error reading embeddings: {e}")
            embedded = 0

        econ.close()

    # Calculate gap
    user_msgs = con.execute(f'''
        SELECT COUNT(DISTINCT {EFFECTIVE_ID_SQL})
        FROM read_parquet("{PARQUET_PATH}")
        WHERE role = 'user' AND char_count > 20 AND content IS NOT NULL
    ''').fetchone()[0]
    print(f"\nUser messages (>20 chars): {user_msgs:,}")

    if EMBEDDINGS_DB.exists():
        gap = user_msgs - embedded
        pct = (embedded / user_msgs * 100) if user_msgs > 0 else 0
        print(f"Embedding coverage: {pct:.1f}%")
        if gap > 0:
            print(f"Gap to embed: {gap:,} messages")
            eta_mins = gap / 15 / 60  # ~15 msg/s
            print(f"Estimated time to complete: {eta_mins:.0f} minutes")
        else:
            print("All messages embedded!")
    print()

    # === YOUTUBE ===
    print("=== YOUTUBE ===")
    if YOUTUBE_PARQUET.exists():
        yt_count = con.execute(f'''
            SELECT COUNT(*) FROM read_parquet("{YOUTUBE_PARQUET}")
        ''').fetchone()[0]
        print(f"Videos watched: {yt_count:,}")
    else:
        print("YouTube data not found")
    print()

    # === SOURCES ===
    print("=== DATA SOURCES ===")
    sources = con.execute(f'''
        SELECT source, COUNT(*) as cnt
        FROM read_parquet("{PARQUET_PATH}")
        GROUP BY source
        ORDER BY cnt DESC
    ''').fetchall()
    for src, cnt in sources:
        print(f"  {src}: {cnt:,}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    show_status()
