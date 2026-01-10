#!/usr/bin/env python3
"""
Auto-sync Claude Code conversations to brain.

Called by:
- Claude Code Stop hook (real-time)
- launchd agent (hourly backup)

Usage:
    python sync.py                    # Sync recent (last 1 hour)
    python sync.py --source claude-code  # Same as above
    python sync.py --catch-up         # Sync last 24 hours
    python sync.py --all              # Full sync
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import lancedb
import pyarrow as pa

from config import (
    BASE,
    PARQUET_PATH,
    LANCE_PATH,
    CLAUDE_PROJECTS,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
)

# State tracking
STATE_DB = BASE / "live" / "sync_state.db"
LOG_FILE = BASE / "logs" / "sync.log"


def log(msg: str):
    """Log to file and stderr."""
    timestamp = datetime.now().isoformat()
    line = f"[{timestamp}] {msg}"
    print(line, file=sys.stderr)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_state_db():
    """Get SQLite connection for sync state tracking."""
    conn = sqlite3.connect(str(STATE_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS synced_files (
            filepath TEXT PRIMARY KEY,
            last_synced TEXT,
            message_count INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embedded_messages (
            message_id TEXT PRIMARY KEY,
            embedded_at TEXT
        )
    """)
    return conn


def find_new_jsonl(hours: int = 1) -> list[Path]:
    """Find JSONL files modified in the last N hours."""
    if not CLAUDE_PROJECTS.exists():
        return []

    cutoff = datetime.now() - timedelta(hours=hours)
    files = []

    for f in CLAUDE_PROJECTS.glob("**/*.jsonl"):
        if f.name.startswith("agent-"):  # Skip subagent files
            continue
        if datetime.fromtimestamp(f.stat().st_mtime) > cutoff:
            files.append(f)

    return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)


def parse_jsonl(filepath: Path) -> list[dict]:
    """Parse messages from a JSONL file."""
    records = []
    project_name = filepath.parent.name.replace("-Users-mordechai-", "")[-50:]

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        log(f"Error reading {filepath.name}: {e}")
        return []

    for line_num, line in enumerate(lines):
        try:
            data = json.loads(line.strip())
        except json.JSONDecodeError:
            continue

        msg_type = data.get("type", "")

        # Extract message content - types are "user" or "assistant"
        if msg_type in ("user", "assistant"):
            role = msg_type
            content = extract_content(data.get("message", data))

            if not content or len(content) < 10:
                continue

            # Get timestamp if available
            timestamp = data.get("timestamp") or datetime.now().isoformat()
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except:
                dt = datetime.now()

            record = {
                "message_id": f"{filepath.stem}_{line_num}",
                "conversation_id": filepath.stem,
                "conversation_title": project_name,
                "content": content,
                "role": role,
                "source": "claude-code",
                "model": data.get("message", {}).get("model", "unknown"),
                "created": timestamp,
                "year": dt.year,
                "month": dt.month,
            }
            records.append(record)

    return records


def extract_content(message: dict) -> str:
    """Extract text content from message structure."""
    content = message.get("content", "")

    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts)

    return str(content) if content else ""


def sync_to_parquet(records: list[dict]):
    """Append records to the main parquet file."""
    if not records:
        return 0

    # Load existing
    con = duckdb.connect()
    existing_ids = set()

    if PARQUET_PATH.exists():
        existing_ids = set(
            con.execute(f"""
                SELECT DISTINCT message_id
                FROM read_parquet('{PARQUET_PATH}')
                WHERE message_id IS NOT NULL
            """).fetchall()
        )
        existing_ids = {r[0] for r in existing_ids}

    # Filter new records
    new_records = [r for r in records if r["message_id"] not in existing_ids]

    if not new_records:
        return 0

    # Append to parquet
    # For now, just log - actual append needs schema alignment
    log(f"Would append {len(new_records)} new messages to parquet")

    return len(new_records)


def sync_to_lancedb(records: list[dict]):
    """Embed and add new messages to LanceDB."""
    if not records or not LANCE_PATH.exists():
        return 0

    # Filter to user messages only (for embedding)
    user_records = [r for r in records if r.get("role") == "user"]

    if not user_records:
        return 0

    # Check which are already embedded
    db = lancedb.connect(str(LANCE_PATH))
    tbl = db.open_table("message")

    # Get existing message IDs
    existing = tbl.to_pandas()["message_id"].tolist()
    existing_set = set(existing)

    # Filter new
    new_records = [r for r in user_records if r["message_id"] not in existing_set]

    if not new_records:
        return 0

    log(f"Embedding {len(new_records)} new messages...")

    # Load embedding model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)

    # Embed
    texts = [r["content"][:8000] for r in new_records]
    embeddings = model.encode(texts, show_progress_bar=True)

    # Prepare records for LanceDB
    lance_records = []
    for i, r in enumerate(new_records):
        lance_records.append({
            "message_id": r["message_id"],
            "conversation_id": r["conversation_id"],
            "conversation_title": r["conversation_title"],
            "content": r["content"],
            "role": r["role"],
            "year": r["year"],
            "month": r["month"],
            "embedding": embeddings[i].tolist(),
            "created_at": datetime.now(),
        })

    # Add to LanceDB
    tbl.add(lance_records)
    log(f"Added {len(lance_records)} messages to LanceDB")

    return len(lance_records)


def main():
    parser = argparse.ArgumentParser(description="Sync Claude Code to brain")
    parser.add_argument("--source", default="claude-code", help="Source type")
    parser.add_argument("--catch-up", action="store_true", help="Sync last 24 hours")
    parser.add_argument("--all", action="store_true", help="Full sync")
    args = parser.parse_args()

    log("=" * 50)
    log("Brain Sync Started")
    log("=" * 50)

    # Determine time range
    if args.all:
        hours = 24 * 365  # 1 year
        log("Mode: Full sync")
    elif args.catch_up:
        hours = 24
        log("Mode: Catch-up (24 hours)")
    else:
        hours = 1
        log("Mode: Recent (1 hour)")

    # Find new files
    files = find_new_jsonl(hours)
    log(f"Found {len(files)} modified JSONL files")

    if not files:
        log("Nothing to sync")
        return

    # Parse all files
    all_records = []
    for f in files:
        records = parse_jsonl(f)
        if records:
            log(f"  {f.name}: {len(records)} messages")
            all_records.extend(records)

    log(f"Total parsed: {len(all_records)} messages")

    # Sync to LanceDB (embedding)
    embedded = sync_to_lancedb(all_records)
    log(f"Embedded: {embedded} messages")

    log("=" * 50)
    log("Brain Sync Complete")
    log("=" * 50)


if __name__ == "__main__":
    main()
