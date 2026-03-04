"""
brain-mcp — Shared parquet schema for all conversation ingesters.

Every ingester produces records matching this schema. The embedder and
MCP server expect these columns.
"""

import re
from datetime import datetime


# The canonical columns for all_conversations.parquet
SCHEMA_COLUMNS = [
    "source",               # "claude-code", "chatgpt", "clawdbot", "custom"
    "model",                # LLM model name (if available)
    "project",              # Project/workspace name (if available)
    "conversation_id",      # Unique per conversation
    "conversation_title",   # Human-readable title
    "created",              # Timestamp of conversation start
    "updated",              # Timestamp of last message
    "year",                 # Integer year
    "month",                # Integer month
    "day_of_week",          # "Monday", "Tuesday", etc.
    "hour",                 # Integer hour (0-23)
    "message_id",           # Unique per message
    "parent_id",            # Parent message ID (for threading)
    "msg_index",            # Position in conversation (0-based)
    "msg_timestamp",        # Timestamp of this specific message
    "timestamp_is_fallback",  # 1 if timestamp was guessed, 0 if exact
    "role",                 # "user" or "assistant"
    "content_type",         # "text" (always for now)
    "is_first",             # 1 if first message in conversation
    "is_last",              # 1 if last message in conversation
    "word_count",           # Word count of content
    "char_count",           # Character count of content
    "conversation_msg_count",  # Total messages in this conversation
    "has_code",             # 1 if contains code blocks
    "has_url",              # 1 if contains URLs
    "has_question",         # 1 if contains "?"
    "has_attachment",       # 1 if has file attachment
    "has_citation",         # 1 if has citations
    "content",              # The actual message text
    "temporal_precision",   # "exact", "day", "approximate"
]


def make_record(
    source: str,
    conversation_id: str,
    role: str,
    content: str,
    timestamp: datetime,
    msg_index: int = 0,
    *,
    model: str = None,
    project: str = None,
    conversation_title: str = None,
    message_id: str = None,
    parent_id: str = None,
    temporal_precision: str = "exact",
) -> dict:
    """
    Create a single message record matching the canonical schema.

    This is the standard way for all ingesters to produce records.
    """
    content = (content or "").strip()
    if not content:
        return None

    # Sanitize surrogates (invalid UTF-8)
    try:
        content.encode("utf-8")
    except UnicodeEncodeError:
        content = content.encode("utf-8", errors="replace").decode("utf-8")

    ts = timestamp or datetime.now()

    return {
        "source": source,
        "model": model,
        "project": project,
        "conversation_id": conversation_id,
        "conversation_title": conversation_title or "Untitled",
        "created": ts,
        "updated": ts,
        "year": ts.year,
        "month": ts.month,
        "day_of_week": ts.strftime("%A"),
        "hour": ts.hour,
        "message_id": message_id or f"{conversation_id}_{msg_index}",
        "parent_id": parent_id,
        "msg_index": msg_index,
        "msg_timestamp": ts,
        "timestamp_is_fallback": 0 if temporal_precision == "exact" else 1,
        "role": role,
        "content_type": "text",
        "is_first": 1 if msg_index == 0 else 0,
        "is_last": 0,  # Fixed after all messages are collected
        "word_count": len(content.split()),
        "char_count": len(content),
        "conversation_msg_count": 0,  # Fixed after all messages are collected
        "has_code": 1 if "```" in content or re.search(r"def |function |class |import |const |let |var ", content) else 0,
        "has_url": 1 if re.search(r"https?://", content) else 0,
        "has_question": 1 if "?" in content else 0,
        "has_attachment": 0,
        "has_citation": 0,
        "content": content[:50000],  # Cap at 50K chars
        "temporal_precision": temporal_precision,
    }


def finalize_conversation(records: list[dict]) -> list[dict]:
    """
    Fix conversation-level fields after all messages are collected.
    Sets is_last and conversation_msg_count.
    """
    if not records:
        return records

    msg_count = len(records)
    for i, r in enumerate(records):
        r["conversation_msg_count"] = msg_count
        r["is_first"] = 1 if i == 0 else 0
        r["is_last"] = 1 if i == msg_count - 1 else 0
        r["msg_index"] = i

    return records
