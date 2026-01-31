#!/usr/bin/env python3
"""
Build conversation analysis interpretations from 353K messages.
Phase 3 of 55x Mining Plan.

Creates:
- interpretations/conversation_threads/v1/ - Thread depth and patterns
- interpretations/conversation_qa/v1/ - Q&A extraction
- interpretations/conversation_corrections/v1/ - Correction patterns
- interpretations/conversation_stats/v1/ - Per-conversation statistics
"""

import duckdb
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
PARQUET_PATH = BASE_DIR / "data" / "all_conversations.parquet"
INTERP_DIR = BASE_DIR / "data" / "interpretations"


def ensure_dir(path: Path):
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def build_conversation_stats():
    """
    Build per-conversation statistics.
    """
    print("\n=== Building Conversation Stats ===")

    out_dir = INTERP_DIR / "conversation_stats" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Per-conversation metrics
    con.execute(f"""
        COPY (
            SELECT
                conversation_id,
                conversation_title,
                source,
                MIN(created) as started,
                MAX(created) as ended,
                COUNT(*) as message_count,
                SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) as user_msgs,
                SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END) as assistant_msgs,
                SUM(word_count) as total_words,
                MAX(msg_index) as max_depth,
                SUM(CASE WHEN has_code THEN 1 ELSE 0 END) as code_messages,
                SUM(CASE WHEN has_question THEN 1 ELSE 0 END) as questions_asked,
                SUM(CASE WHEN has_url THEN 1 ELSE 0 END) as url_messages,
                COUNT(DISTINCT DATE_TRUNC('day', created)) as active_days,
                -- Threading metrics
                SUM(CASE WHEN parent_id IS NOT NULL THEN 1 ELSE 0 END) as threaded_msgs,
                -- Response ratio (assistant responses per user message)
                CASE
                    WHEN SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) > 0
                    THEN SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END) * 1.0 /
                         SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END)
                    ELSE 0
                END as response_ratio
            FROM '{PARQUET_PATH}'
            GROUP BY conversation_id, conversation_title, source
            ORDER BY message_count DESC
        ) TO '{out_dir}/conversations.parquet' (FORMAT PARQUET)
    """)

    count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/conversations.parquet'").fetchone()[0]
    print(f"  Written: {count} conversations analyzed")

    # Monthly conversation volume
    con.execute(f"""
        COPY (
            SELECT
                year || '-' || LPAD(CAST(month AS VARCHAR), 2, '0') as month_str,
                source,
                COUNT(DISTINCT conversation_id) as conversations,
                COUNT(*) as messages,
                SUM(word_count) as total_words
            FROM '{PARQUET_PATH}'
            GROUP BY year, month, source
            ORDER BY year, month, messages DESC
        ) TO '{out_dir}/monthly.parquet' (FORMAT PARQUET)
    """)

    monthly = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/monthly.parquet'").fetchone()[0]
    print(f"  Monthly records: {monthly}")

    return count


def build_qa_patterns():
    """
    Build Q&A extraction - user questions followed by assistant answers.
    """
    print("\n=== Building Q&A Patterns ===")

    out_dir = INTERP_DIR / "conversation_qa" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Extract user questions (has_question=true)
    con.execute(f"""
        COPY (
            SELECT
                conversation_id,
                message_id,
                msg_index,
                created as asked_at,
                content as question,
                word_count,
                source,
                year,
                month
            FROM '{PARQUET_PATH}'
            WHERE role = 'user'
              AND has_question = true
              AND word_count > 5  -- Filter noise
            ORDER BY created
        ) TO '{out_dir}/questions.parquet' (FORMAT PARQUET)
    """)

    q_count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/questions.parquet'").fetchone()[0]
    print(f"  User questions: {q_count}")

    # Question categories by keyword (simple heuristic)
    con.execute(f"""
        COPY (
            SELECT
                year || '-' || LPAD(CAST(month AS VARCHAR), 2, '0') as month_str,
                CASE
                    WHEN LOWER(content) LIKE '%how do%' OR LOWER(content) LIKE '%how to%' THEN 'how-to'
                    WHEN LOWER(content) LIKE '%what is%' OR LOWER(content) LIKE '%what are%' THEN 'definition'
                    WHEN LOWER(content) LIKE '%why%' THEN 'explanation'
                    WHEN LOWER(content) LIKE '%can you%' OR LOWER(content) LIKE '%could you%' THEN 'request'
                    WHEN LOWER(content) LIKE '%should%' THEN 'advice'
                    WHEN LOWER(content) LIKE '%error%' OR LOWER(content) LIKE '%bug%' OR LOWER(content) LIKE '%fix%' THEN 'debugging'
                    ELSE 'other'
                END as question_type,
                COUNT(*) as count
            FROM '{PARQUET_PATH}'
            WHERE role = 'user' AND has_question = true
            GROUP BY year, month, question_type
            ORDER BY year, month, count DESC
        ) TO '{out_dir}/question_types.parquet' (FORMAT PARQUET)
    """)

    types = con.execute(f"SELECT COUNT(DISTINCT question_type) FROM '{out_dir}/question_types.parquet'").fetchone()[0]
    print(f"  Question type categories: {types}")

    return q_count


def build_correction_patterns():
    """
    Build correction patterns - "no no no" and similar correction indicators.
    """
    print("\n=== Building Correction Patterns ===")

    out_dir = INTERP_DIR / "conversation_corrections" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Find correction indicators
    con.execute(f"""
        COPY (
            SELECT
                conversation_id,
                message_id,
                msg_index,
                created,
                content as correction_text,
                word_count,
                source,
                CASE
                    WHEN LOWER(content) LIKE '%no no no%' THEN 'strong_rejection'
                    WHEN LOWER(content) LIKE '%not what i%' OR LOWER(content) LIKE '%thats not%' THEN 'misunderstanding'
                    WHEN LOWER(content) LIKE '%try again%' OR LOWER(content) LIKE '%redo%' THEN 'retry_request'
                    WHEN LOWER(content) LIKE '%actually%' AND word_count < 50 THEN 'clarification'
                    WHEN LOWER(content) LIKE '%wrong%' THEN 'error_correction'
                    WHEN LOWER(content) LIKE '%dont%' OR LOWER(content) LIKE '%stop%' THEN 'stop_request'
                    ELSE 'other'
                END as correction_type
            FROM '{PARQUET_PATH}'
            WHERE role = 'user'
              AND (
                  LOWER(content) LIKE '%no no%'
                  OR LOWER(content) LIKE '%not what i%'
                  OR LOWER(content) LIKE '%try again%'
                  OR LOWER(content) LIKE '%thats not%'
                  OR LOWER(content) LIKE '%actually no%'
                  OR (LOWER(content) LIKE '%wrong%' AND word_count < 30)
                  OR LOWER(content) LIKE '%redo%'
              )
            ORDER BY created
        ) TO '{out_dir}/corrections.parquet' (FORMAT PARQUET)
    """)

    c_count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/corrections.parquet'").fetchone()[0]
    print(f"  Correction messages: {c_count}")

    # Monthly correction rate
    con.execute(f"""
        COPY (
            SELECT
                year || '-' || LPAD(CAST(month AS VARCHAR), 2, '0') as month_str,
                COUNT(CASE WHEN
                    LOWER(content) LIKE '%no no%'
                    OR LOWER(content) LIKE '%not what i%'
                    OR LOWER(content) LIKE '%try again%'
                    OR LOWER(content) LIKE '%thats not%'
                    OR LOWER(content) LIKE '%actually no%'
                    OR LOWER(content) LIKE '%redo%'
                THEN 1 END) as corrections,
                COUNT(*) as total_user_msgs,
                COUNT(CASE WHEN
                    LOWER(content) LIKE '%no no%'
                    OR LOWER(content) LIKE '%not what i%'
                    OR LOWER(content) LIKE '%try again%'
                    OR LOWER(content) LIKE '%thats not%'
                    OR LOWER(content) LIKE '%actually no%'
                    OR LOWER(content) LIKE '%redo%'
                THEN 1 END) * 100.0 / COUNT(*) as correction_rate
            FROM '{PARQUET_PATH}'
            WHERE role = 'user'
            GROUP BY year, month
            ORDER BY year, month
        ) TO '{out_dir}/monthly_rates.parquet' (FORMAT PARQUET)
    """)

    monthly = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/monthly_rates.parquet'").fetchone()[0]
    print(f"  Monthly correction rates: {monthly}")

    return c_count


def build_thread_analysis():
    """
    Build thread depth and follow-up analysis.
    """
    print("\n=== Building Thread Analysis ===")

    out_dir = INTERP_DIR / "conversation_threads" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Thread depth per conversation
    con.execute(f"""
        COPY (
            SELECT
                conversation_id,
                conversation_title,
                MAX(msg_index) as max_depth,
                COUNT(*) as message_count,
                SUM(CASE WHEN parent_id IS NOT NULL THEN 1 ELSE 0 END) as threaded_count,
                -- Threading ratio
                SUM(CASE WHEN parent_id IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as threading_pct,
                MIN(created) as started,
                MAX(created) as ended,
                source
            FROM '{PARQUET_PATH}'
            GROUP BY conversation_id, conversation_title, source
            HAVING message_count > 5  -- Filter trivial conversations
            ORDER BY max_depth DESC
        ) TO '{out_dir}/depth.parquet' (FORMAT PARQUET)
    """)

    depth_count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/depth.parquet'").fetchone()[0]
    print(f"  Conversations with depth analysis: {depth_count}")

    # Deep conversations (100+ messages)
    con.execute(f"""
        COPY (
            SELECT
                conversation_id,
                conversation_title,
                source,
                COUNT(*) as message_count,
                MAX(msg_index) as max_depth,
                SUM(word_count) as total_words,
                MIN(created) as started,
                MAX(created) as ended,
                COUNT(DISTINCT DATE_TRUNC('day', created)) as span_days
            FROM '{PARQUET_PATH}'
            GROUP BY conversation_id, conversation_title, source
            HAVING message_count >= 100
            ORDER BY message_count DESC
        ) TO '{out_dir}/deep_conversations.parquet' (FORMAT PARQUET)
    """)

    deep_count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/deep_conversations.parquet'").fetchone()[0]
    print(f"  Deep conversations (100+ msgs): {deep_count}")

    return depth_count


def build_all():
    """Build all conversation analysis layers."""
    print(f"Building conversation analysis from {PARQUET_PATH}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    if not PARQUET_PATH.exists():
        print(f"ERROR: Conversations not found at {PARQUET_PATH}")
        return

    # Build all layers
    build_conversation_stats()
    build_qa_patterns()
    build_correction_patterns()
    build_thread_analysis()

    print("\n✅ All conversation analysis layers built!")


if __name__ == "__main__":
    build_all()
