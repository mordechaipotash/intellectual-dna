#!/usr/bin/env python3
"""
Build focus/v2 interpretation: LLM-powered daily focus summaries.

Uses Gemini 2.5 Flash Lite via OpenRouter to generate narrative summaries
of what Mordechai was thinking about each day.
"""

import os
import json
import time
import duckdb
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path("/Users/mordechai/intellectual_dna/.env"))

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
DATA_DIR = BASE_DIR / "data"
FACTS_DIR = DATA_DIR / "facts"
INTERP_DIR = DATA_DIR / "interpretations" / "focus" / "v2"

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash-lite"

# Rate limiting
REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE  # 2 seconds between requests


def call_gemini(prompt: str, max_tokens: int = 300) -> str:
    """Call Gemini via OpenRouter API."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Intellectual DNA Focus v2"
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Write a first-person summary starting with an action verb (Worked on, Explored, Debugged, etc). Be specific about projects, technologies, and concepts. 1-2 sentences. Example: 'Debugged Python turtle graphics for chessboard visualization while reviewing Hilchos Shabbos sources.'"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        return f"[API Error: {str(e)}]"
    except (KeyError, IndexError) as e:
        return f"[Parse Error: {str(e)}]"


def build_focus_v2(limit: int = None, start_date: str = None):
    """Build daily focus summaries using LLM."""
    print("Building focus/v2 interpretation (LLM-powered)...")
    print(f"  Model: {MODEL}")
    print(f"  Rate: {REQUESTS_PER_MINUTE} requests/minute")

    if not OPENROUTER_API_KEY:
        print("  ERROR: OPENROUTER_API_KEY not found")
        return

    con = duckdb.connect()

    # Check what days we already have processed
    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERP_DIR / "daily.parquet"

    processed_dates = set()
    if output_path.exists():
        existing = con.execute(f"SELECT date FROM '{output_path}'").fetchall()
        processed_dates = {str(row[0]) for row in existing}
        print(f"  Already processed: {len(processed_dates)} days")

    # Get days to process
    date_filter = ""
    if start_date:
        date_filter = f"AND idx.timestamp >= '{start_date}'"

    daily_messages = con.execute(f"""
        SELECT
            CAST(idx.timestamp AS DATE) as date,
            LIST(LEFT(c.full_content, 500)) as previews,
            COUNT(*) as message_count
        FROM '{FACTS_DIR}/brain/content.parquet' c
        JOIN '{FACTS_DIR}/brain/index.parquet' idx ON c.event_id = idx.event_id
        WHERE c.event_type = 'message'
          AND c.full_content IS NOT NULL
          AND idx.timestamp >= '2022-11-01'
          {date_filter}
        GROUP BY CAST(idx.timestamp AS DATE)
        HAVING COUNT(*) >= 3
        ORDER BY date
    """).fetchall()

    # Filter to unprocessed days
    to_process = [(date, previews, count) for date, previews, count in daily_messages
                  if str(date) not in processed_dates]

    if limit:
        to_process = to_process[:limit]

    print(f"  Days to process: {len(to_process)}")

    if not to_process:
        print("  Nothing to process!")
        return

    # Process each day
    results = []
    total_cost = 0.0
    start_time = time.time()

    for i, (date, previews, msg_count) in enumerate(to_process):
        # Create prompt from message previews
        sample_text = "\n---\n".join(previews[:10])  # First 10 messages
        prompt = f"""Date: {date}
Number of messages: {msg_count}

Sample messages from this day:
{sample_text}

What was the main focus or theme of thinking on this day? Be specific about topics, projects, or problems being worked on."""

        # Call LLM
        summary = call_gemini(prompt)

        results.append({
            "date": str(date),
            "summary": summary,
            "message_count": msg_count,
            "processed_at": datetime.now().isoformat()
        })

        # Estimate cost (Gemini Flash Lite is very cheap)
        # ~$0.075/1M input tokens, ~$0.30/1M output tokens
        input_tokens = len(prompt) / 4  # rough estimate
        output_tokens = len(summary) / 4
        cost = (input_tokens * 0.075 / 1_000_000) + (output_tokens * 0.30 / 1_000_000)
        total_cost += cost

        # Progress
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{i+1}/{len(to_process)}] {date} - {rate:.1f} req/min - ${total_cost:.4f}")

        # Rate limiting
        time.sleep(REQUEST_DELAY)

    # Save results
    if results:
        # Create or append to parquet
        con.execute("""
            CREATE TABLE IF NOT EXISTS focus_v2 (
                date DATE,
                summary VARCHAR,
                message_count INTEGER,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("""
                INSERT INTO focus_v2 VALUES (?, ?, ?, ?)
            """, [r["date"], r["summary"], r["message_count"], r["processed_at"]])

        # Merge with existing if any
        if output_path.exists():
            con.execute(f"""
                INSERT INTO focus_v2
                SELECT * FROM '{output_path}'
            """)

        con.execute(f"COPY focus_v2 TO '{output_path}' (FORMAT PARQUET)")

        print(f"\n  Processed: {len(results)} days")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Wrote to: {output_path}")

    # Update config
    config = {
        "name": "focus/v2",
        "description": "LLM-powered daily focus summaries",
        "version": "2.0.0",
        "created": datetime.now().strftime("%Y-%m-%d"),
        "algorithm": "gemini_summarization",
        "model": MODEL,
        "parameters": {
            "max_tokens": 300,
            "temperature": 0.3,
            "min_messages_per_day": 3
        },
        "sources": ["facts/brain/content.parquet", "facts/brain/index.parquet"],
        "outputs": ["daily.parquet"]
    }

    config_path = INTERP_DIR / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return len(results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build focus/v2 with LLM summaries')
    parser.add_argument('--limit', type=int, help='Max days to process')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--test', action='store_true', help='Test with 3 days')
    args = parser.parse_args()

    if args.test:
        build_focus_v2(limit=3)
    else:
        build_focus_v2(limit=args.limit, start_date=args.start)
