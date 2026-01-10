#!/usr/bin/env python3
"""
questions/v2 - Extract USER questions from conversations
Fixes v1 which incorrectly captured AI assistant questions
"""

import re
import json
import duckdb
from pathlib import Path
from datetime import datetime
from collections import Counter
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
CONVERSATIONS = BASE_DIR / "data/facts/sources/all_conversations.parquet"
OUTPUT_DIR = BASE_DIR / "data/interpretations/questions/v2"

# IDE/System noise patterns to filter out
NOISE_PATTERNS = [
    r'context window', r'tokens used', r'vscode', r'localhost',
    r'node_modules', r'package\.json', r'/Users/', r'/home/',
    r'Tool Result', r'user uploaded', r'interrupted by',
    r'null null', r'\.parquet', r'\.tsx', r'\.jsx',
]
NOISE_REGEX = re.compile('|'.join(NOISE_PATTERNS), re.IGNORECASE)

# Question categories based on first words
QUESTION_CATEGORIES = {
    'how': 'process',      # How do I, How can I
    'what': 'definition',  # What is, What are
    'why': 'reasoning',    # Why does, Why is
    'where': 'location',   # Where is, Where can
    'when': 'temporal',    # When should, When does
    'which': 'choice',     # Which one, Which is
    'can': 'capability',   # Can you, Can I
    'could': 'capability', # Could you, Could I
    'would': 'hypothetical', # Would it, Would you
    'should': 'advice',    # Should I, Should we
    'is': 'verification',  # Is it, Is this
    'are': 'verification', # Are you, Are there
    'do': 'verification',  # Do you, Do I
    'does': 'verification', # Does it, Does this
    'will': 'future',      # Will it, Will this
    'have': 'experience',  # Have you, Have I
    'who': 'identity',     # Who is, Who can
}


def is_genuine_question(text: str) -> bool:
    """Check if text is a genuine user question, not noise."""
    if not text or len(text) < 15 or len(text) > 300:
        return False
    if NOISE_REGEX.search(text):
        return False
    if '```' in text or '{}' in text or '`' in text:
        return False
    # Must end with ?
    if not text.strip().endswith('?'):
        return False
    # Must have at least 3 words (real questions are longer)
    words = text.split()
    if len(words) < 3:
        return False
    # Filter out AI-style questions
    ai_patterns = [
        'would you like', 'shall i', 'do you want me to',
        'would that help', 'does that make sense', 'any questions',
        'let me know if', 'feel free to', 'happy to help',
    ]
    text_lower = text.lower()
    if any(p in text_lower for p in ai_patterns):
        return False
    # Filter out file/path fragments
    if text.startswith('.') or '/' in text[:20]:
        return False
    return True


def extract_questions(text: str) -> list:
    """Extract question sentences from text."""
    # Split on sentence boundaries
    sentences = re.split(r'[.!]\s+|\n', text)
    questions = []
    for s in sentences:
        s = s.strip()
        if s.endswith('?') and is_genuine_question(s):
            questions.append(s)
    return questions


def categorize_question(q: str) -> str:
    """Categorize question by first word."""
    words = q.lower().split()
    if not words:
        return 'other'
    first = words[0]
    return QUESTION_CATEGORIES.get(first, 'other')


def build_questions_v2():
    """Build questions v2 extraction."""
    print("=" * 60)
    print("QUESTIONS V2 EXTRACTION")
    print("=" * 60)

    con = duckdb.connect()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load user messages with timestamps
    print("\n📥 Loading user messages...")
    messages = con.execute(f"""
        SELECT
            content,
            CAST(msg_timestamp AS DATE) as msg_date,
            DATE_TRUNC('month', msg_timestamp::TIMESTAMP) as month
        FROM '{CONVERSATIONS}'
        WHERE role = 'user'
          AND content IS NOT NULL
          AND LENGTH(content) BETWEEN 10 AND 2000
          AND msg_timestamp IS NOT NULL
    """).fetchall()
    print(f"   Total messages: {len(messages):,}")

    # Extract questions
    print("\n🔍 Extracting questions...")
    all_questions = []
    category_counts = Counter()

    for content, msg_date, month in messages:
        questions = extract_questions(content)
        for q in questions:
            cat = categorize_question(q)
            category_counts[cat] += 1
            all_questions.append({
                'question': q,
                'category': cat,
                'date': msg_date,
                'month': month,
            })

    print(f"   Questions found: {len(all_questions):,}")
    print(f"\n📊 By category:")
    for cat, count in category_counts.most_common():
        print(f"   {cat}: {count}")

    # Deduplicate exact matches
    print("\n🧹 Deduplicating...")
    seen = set()
    unique_questions = []
    for q in all_questions:
        key = q['question'].lower()
        if key not in seen:
            seen.add(key)
            unique_questions.append(q)
    print(f"   Unique questions: {len(unique_questions):,}")

    # Save to parquet
    print(f"\n💾 Saving to parquet...")
    table = pa.table({
        'question': [q['question'] for q in unique_questions],
        'category': [q['category'] for q in unique_questions],
        'date': [q['date'] for q in unique_questions],
        'month': [q['month'] for q in unique_questions],
        'processed_at': [datetime.now().isoformat() for _ in unique_questions],
    })
    pq.write_table(table, OUTPUT_DIR / "questions.parquet")

    # Save config
    config = {
        'version': 'v2',
        'created_at': datetime.now().isoformat(),
        'total_questions': len(unique_questions),
        'categories': dict(category_counts),
        'source': 'user messages only',
    }
    with open(OUTPUT_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Questions extracted: {len(unique_questions):,}")
    print(f"Output: {OUTPUT_DIR / 'questions.parquet'}")

    # Sample
    print(f"\n📝 SAMPLE QUESTIONS:")
    for q in unique_questions[:10]:
        print(f"  [{q['category']}] {q['question'][:70]}...")


if __name__ == "__main__":
    build_questions_v2()
