#!/usr/bin/env python3
"""
brain-mcp — Generic JSONL conversation ingester.

Reads any JSONL file where each line has {role, content, timestamp}.
Optionally supports conversation_id, model, and other fields.
Maps everything to the canonical parquet schema.

Usage:
    python -m ingest.generic path/to/conversations.jsonl
    python -m ingest.generic path/to/directory/
"""

import json
import sys
import hashlib
from pathlib import Path
from datetime import datetime

from .schema import make_record, finalize_conversation


def parse_generic_jsonl(file_path: Path, source_name: str = "custom") -> list[dict]:
    """
    Parse a generic JSONL file into canonical records.

    Each line should be a JSON object with at least:
      - role: "user" or "assistant"
      - content: message text
      - timestamp: ISO format or unix epoch

    Optional fields:
      - conversation_id: groups messages into conversations
      - model: LLM model name
      - project: project/workspace name
      - title: conversation title
      - message_id: unique message identifier
      - parent_id: parent message id for threading
    """
    records = []
    conversations: dict[str, list[dict]] = {}

    with open(file_path, encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"  Warning: skipping invalid JSON at line {line_num}", file=sys.stderr)
                continue

            role = entry.get("role", "").lower()
            content = entry.get("content", "")
            if not role or not content:
                continue

            # Normalize role
            if role not in ("user", "assistant"):
                if role in ("system", "tool"):
                    continue  # skip system/tool messages
                role = "user"  # default unknown roles to user

            # Parse timestamp
            ts_raw = entry.get("timestamp") or entry.get("created_at") or entry.get("date")
            ts = _parse_timestamp(ts_raw)

            # Group by conversation
            conv_id = entry.get("conversation_id") or _generate_conv_id(file_path, source_name)

            if conv_id not in conversations:
                conversations[conv_id] = []

            record = make_record(
                source=source_name,
                conversation_id=conv_id,
                role=role,
                content=content,
                timestamp=ts,
                msg_index=len(conversations[conv_id]),
                model=entry.get("model"),
                project=entry.get("project"),
                conversation_title=entry.get("title") or file_path.stem,
                message_id=entry.get("message_id"),
                parent_id=entry.get("parent_id"),
            )

            if record:
                conversations[conv_id].append(record)

    # Finalize all conversations
    for conv_id, conv_records in conversations.items():
        records.extend(finalize_conversation(conv_records))

    return records


def _parse_timestamp(ts_raw) -> datetime:
    """Parse a timestamp from various formats."""
    if ts_raw is None:
        return datetime.now()

    if isinstance(ts_raw, (int, float)):
        try:
            # Unix epoch (seconds or milliseconds)
            if ts_raw > 1e12:
                return datetime.fromtimestamp(ts_raw / 1000)
            return datetime.fromtimestamp(ts_raw)
        except (ValueError, OSError):
            return datetime.now()

    if isinstance(ts_raw, str):
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                return datetime.strptime(ts_raw, fmt)
            except ValueError:
                continue

    return datetime.now()


def _generate_conv_id(file_path: Path, source_name: str) -> str:
    """Generate a stable conversation ID from file path."""
    key = f"{source_name}:{file_path.name}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def ingest_path(path: Path, source_name: str = "custom") -> list[dict]:
    """
    Ingest a file or directory of JSONL files.

    Args:
        path: Path to a .jsonl file or directory containing .jsonl files
        source_name: Name for this source in the parquet

    Returns:
        List of canonical message records
    """
    all_records = []

    if path.is_file():
        print(f"  Parsing {path.name}...", file=sys.stderr)
        all_records = parse_generic_jsonl(path, source_name)
    elif path.is_dir():
        files = sorted(path.glob("**/*.jsonl"))
        print(f"  Found {len(files)} JSONL files in {path}", file=sys.stderr)
        for f in files:
            records = parse_generic_jsonl(f, source_name)
            all_records.extend(records)
    else:
        print(f"  Error: {path} is not a file or directory", file=sys.stderr)

    print(f"  → {len(all_records)} messages from {source_name}", file=sys.stderr)
    return all_records


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ingest.generic <path> [source_name]")
        sys.exit(1)

    target = Path(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 else "custom"
    records = ingest_path(target, name)
    print(f"Total records: {len(records)}")
