#!/usr/bin/env python3
"""
Build weekly_summaries interpretation: Roll up daily focus into weekly narratives.

Uses Gemini 2.5 Flash Lite to synthesize weekly themes from daily summaries.
Depends on focus/v2 daily summaries being built first.
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
INTERP_DIR = DATA_DIR / "interpretations" / "weekly_summaries" / "v1"
FOCUS_V2_PATH = DATA_DIR / "interpretations" / "focus" / "v2" / "daily.parquet"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash-lite"

REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE


def call_gemini(prompt: str, max_tokens: int = 400) -> str:
    """Call Gemini via OpenRouter API."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Intellectual DNA Weekly Summaries"
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Synthesize the daily focus summaries into a cohesive weekly narrative. Identify the main themes, key projects, and notable patterns. Be specific and concise. 2-4 sentences."
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
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Error: {str(e)[:50]}]"


def build_weekly_summaries(limit: int = None):
    """Build weekly summaries from daily focus data."""
    print("Building weekly_summaries/v1 interpretation...")
    print(f"  Model: {MODEL}")

    if not FOCUS_V2_PATH.exists():
        print(f"  ERROR: {FOCUS_V2_PATH} not found - run focus/v2 first")
        return

    if not OPENROUTER_API_KEY:
        print("  ERROR: OPENROUTER_API_KEY not found")
        return

    con = duckdb.connect()
    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERP_DIR / "weekly.parquet"

    # Check existing
    processed_weeks = set()
    if output_path.exists():
        existing = con.execute(f"SELECT week_start FROM '{output_path}'").fetchall()
        processed_weeks = {str(row[0]) for row in existing}
        print(f"  Already processed: {len(processed_weeks)} weeks")

    # Group daily summaries by week
    weeks = con.execute(f"""
        SELECT
            DATE_TRUNC('week', date) as week_start,
            LIST(summary ORDER BY date) as daily_summaries,
            SUM(message_count) as total_messages,
            COUNT(*) as days_with_data
        FROM '{FOCUS_V2_PATH}'
        WHERE summary NOT LIKE '[%Error%'
        GROUP BY DATE_TRUNC('week', date)
        HAVING COUNT(*) >= 3
        ORDER BY week_start
    """).fetchall()

    to_process = [(ws, ds, tm, dwd) for ws, ds, tm, dwd in weeks
                  if str(ws) not in processed_weeks]

    if limit:
        to_process = to_process[:limit]

    print(f"  Weeks to process: {len(to_process)}")

    if not to_process:
        print("  Nothing to process!")
        return

    results = []
    total_cost = 0.0
    start_time = time.time()

    for i, (week_start, summaries, total_msgs, days) in enumerate(to_process):
        daily_text = "\n".join([f"- {s}" for s in summaries])
        prompt = f"""Week of {week_start}:

Daily focus summaries:
{daily_text}

What were the main themes and activities this week?"""

        weekly_summary = call_gemini(prompt)

        results.append({
            "week_start": str(week_start),
            "summary": weekly_summary,
            "days_with_data": days,
            "total_messages": total_msgs,
            "processed_at": datetime.now().isoformat()
        })

        input_tokens = len(prompt) / 4
        output_tokens = len(weekly_summary) / 4
        cost = (input_tokens * 0.075 / 1_000_000) + (output_tokens * 0.30 / 1_000_000)
        total_cost += cost

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{i+1}/{len(to_process)}] {week_start} - {rate:.1f} req/min - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if results:
        con.execute("""
            CREATE TABLE weekly_summaries (
                week_start DATE,
                summary VARCHAR,
                days_with_data INTEGER,
                total_messages INTEGER,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO weekly_summaries VALUES (?, ?, ?, ?, ?)",
                       [r["week_start"], r["summary"], r["days_with_data"],
                        r["total_messages"], r["processed_at"]])

        if output_path.exists():
            con.execute(f"INSERT INTO weekly_summaries SELECT * FROM '{output_path}'")

        con.execute(f"COPY weekly_summaries TO '{output_path}' (FORMAT PARQUET)")

        print(f"\n  Processed: {len(results)} weeks")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Wrote to: {output_path}")

    config = {
        "name": "weekly_summaries/v1",
        "description": "LLM-synthesized weekly narratives from daily focus",
        "model": MODEL,
        "depends_on": ["focus/v2"],
        "created": datetime.now().strftime("%Y-%m-%d")
    }
    with open(INTERP_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return len(results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Max weeks')
    parser.add_argument('--test', action='store_true', help='Test with 3')
    args = parser.parse_args()

    if args.test:
        build_weekly_summaries(limit=3)
    else:
        build_weekly_summaries(limit=args.limit)
