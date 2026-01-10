#!/usr/bin/env python3
"""
Build conversation_titles interpretation: LLM-generated titles for conversations.

Uses Gemini 2.5 Flash Lite to generate concise, descriptive titles
for each conversation session.
"""

import os
import json
import time
import duckdb
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(Path("/Users/mordechai/intellectual_dna/.env"))

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
DATA_DIR = BASE_DIR / "data"
FACTS_DIR = DATA_DIR / "facts"
INTERP_DIR = DATA_DIR / "interpretations" / "conversation_titles" / "v1"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash-lite"

REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE


def call_gemini(prompt: str, max_tokens: int = 50) -> str:
    """Call Gemini via OpenRouter API."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Intellectual DNA Conversation Titles"
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Generate a concise 3-8 word title for this conversation. Be specific about the main topic or task. Output ONLY the title, no quotes or punctuation."
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip().strip('"\'')
    except Exception as e:
        return f"[Error: {str(e)[:50]}]"


def build_conversation_titles(limit: int = None):
    """Build conversation titles using LLM."""
    print("Building conversation_titles/v1 interpretation...")
    print(f"  Model: {MODEL}")

    if not OPENROUTER_API_KEY:
        print("  ERROR: OPENROUTER_API_KEY not found")
        return

    con = duckdb.connect()
    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERP_DIR / "titles.parquet"

    # Check existing
    processed_ids = set()
    if output_path.exists():
        existing = con.execute(f"SELECT conversation_id FROM '{output_path}'").fetchall()
        processed_ids = {row[0] for row in existing}
        print(f"  Already processed: {len(processed_ids)} conversations")

    # Get daily sessions with their first few messages (no conversation_id available)
    conversations = con.execute(f"""
        SELECT
            CAST(idx.timestamp AS DATE) as session_date,
            MIN(idx.timestamp) as start_time,
            LIST(LEFT(c.full_content, 300) ORDER BY idx.timestamp) as previews,
            COUNT(*) as message_count
        FROM '{FACTS_DIR}/brain/content.parquet' c
        JOIN '{FACTS_DIR}/brain/index.parquet' idx ON c.event_id = idx.event_id
        WHERE c.event_type = 'message'
          AND c.full_content IS NOT NULL
        GROUP BY CAST(idx.timestamp AS DATE)
        HAVING COUNT(*) >= 3
        ORDER BY MIN(idx.timestamp)
    """).fetchall()

    to_process = [(str(sid), ts, prev[:5], cnt) for sid, ts, prev, cnt in conversations
                  if str(sid) not in processed_ids]

    if limit:
        to_process = to_process[:limit]

    print(f"  Conversations to process: {len(to_process)}")

    if not to_process:
        print("  Nothing to process!")
        return

    results = []
    total_cost = 0.0
    start_time = time.time()

    for i, (conv_id, timestamp, previews, msg_count) in enumerate(to_process):
        sample = "\n---\n".join(previews)
        prompt = f"""Conversation from {timestamp}:
{sample}

What is this conversation about?"""

        title = call_gemini(prompt)

        results.append({
            "conversation_id": conv_id,
            "title": title,
            "start_time": str(timestamp),
            "message_count": msg_count,
            "processed_at": datetime.now().isoformat()
        })

        # Estimate cost
        input_tokens = len(prompt) / 4
        output_tokens = len(title) / 4
        cost = (input_tokens * 0.075 / 1_000_000) + (output_tokens * 0.30 / 1_000_000)
        total_cost += cost

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{i+1}/{len(to_process)}] {rate:.1f} req/min - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if results:
        con.execute("""
            CREATE TABLE conv_titles (
                conversation_id VARCHAR,
                title VARCHAR,
                start_time VARCHAR,
                message_count INTEGER,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO conv_titles VALUES (?, ?, ?, ?, ?)",
                       [r["conversation_id"], r["title"], r["start_time"],
                        r["message_count"], r["processed_at"]])

        if output_path.exists():
            con.execute(f"INSERT INTO conv_titles SELECT * FROM '{output_path}'")

        con.execute(f"COPY conv_titles TO '{output_path}' (FORMAT PARQUET)")

        print(f"\n  Processed: {len(results)} conversations")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Wrote to: {output_path}")

    # Config
    config = {
        "name": "conversation_titles/v1",
        "description": "LLM-generated conversation titles",
        "model": MODEL,
        "created": datetime.now().strftime("%Y-%m-%d")
    }
    with open(INTERP_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return len(results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Max conversations')
    parser.add_argument('--test', action='store_true', help='Test with 5')
    args = parser.parse_args()

    if args.test:
        build_conversation_titles(limit=5)
    else:
        build_conversation_titles(limit=args.limit)
