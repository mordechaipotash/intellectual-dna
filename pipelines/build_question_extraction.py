#!/usr/bin/env python3
"""
Build question_extraction interpretation: Extract questions from conversations.

Uses Gemini 2.5 Flash Lite to identify and categorize questions asked,
revealing inquiry patterns and thinking style.
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
INTERP_DIR = DATA_DIR / "interpretations" / "questions" / "v1"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash-lite"

REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE


def call_gemini(prompt: str, max_tokens: int = 500) -> str:
    """Call Gemini via OpenRouter API."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Intellectual DNA Question Extraction"
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """Extract questions from the user's messages. For each question found:
1. Extract the exact question or paraphrase if implicit
2. Categorize: technical, conceptual, clarification, exploratory, philosophical, practical
3. Rate depth: surface (simple lookup), medium (requires reasoning), deep (requires synthesis)

Output JSON array: [{"question": "...", "category": "...", "depth": "..."}]
If no questions found, output: []"""
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Error: {str(e)[:50]}]"


def parse_questions(llm_output: str) -> list:
    """Parse LLM output to extract questions."""
    try:
        # Try to find JSON array in output
        import re
        match = re.search(r'\[.*\]', llm_output, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []
    except:
        return []


def build_question_extraction(limit: int = None, batch_size: int = 50):
    """Extract questions from user messages."""
    print("Building questions/v1 interpretation...")
    print(f"  Model: {MODEL}")

    if not OPENROUTER_API_KEY:
        print("  ERROR: OPENROUTER_API_KEY not found")
        return

    con = duckdb.connect()
    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERP_DIR / "questions.parquet"

    # Check existing
    processed_dates = set()
    if output_path.exists():
        existing = con.execute(f"SELECT DISTINCT date FROM '{output_path}'").fetchall()
        processed_dates = {str(row[0]) for row in existing}
        print(f"  Already processed: {len(processed_dates)} days")

    # Get user messages by day (messages with ? are likely questions)
    daily_questions = con.execute(f"""
        SELECT
            CAST(idx.timestamp AS DATE) as date,
            LIST(c.full_content) as messages
        FROM '{FACTS_DIR}/brain/content.parquet' c
        JOIN '{FACTS_DIR}/brain/index.parquet' idx ON c.event_id = idx.event_id
        WHERE c.event_type = 'message'
          AND c.full_content IS NOT NULL
          AND c.full_content LIKE '%?%'
          AND LENGTH(c.full_content) < 1000
        GROUP BY CAST(idx.timestamp AS DATE)
        ORDER BY date
    """).fetchall()

    to_process = [(date, msgs) for date, msgs in daily_questions
                  if str(date) not in processed_dates]

    if limit:
        to_process = to_process[:limit]

    print(f"  Days to process: {len(to_process)}")

    if not to_process:
        print("  Nothing to process!")
        return

    all_questions = []
    total_cost = 0.0
    start_time = time.time()

    for i, (date, messages) in enumerate(to_process):
        # Sample messages with questions
        question_msgs = [m for m in messages if '?' in m][:10]
        if not question_msgs:
            continue

        sample = "\n---\n".join(question_msgs)
        prompt = f"""Messages from {date}:
{sample}

Extract all questions asked."""

        response = call_gemini(prompt)
        questions = parse_questions(response)

        for q in questions:
            all_questions.append({
                "date": str(date),
                "question": q.get("question", ""),
                "category": q.get("category", "unknown"),
                "depth": q.get("depth", "unknown"),
                "processed_at": datetime.now().isoformat()
            })

        input_tokens = len(prompt) / 4
        output_tokens = len(response) / 4
        cost = (input_tokens * 0.075 / 1_000_000) + (output_tokens * 0.30 / 1_000_000)
        total_cost += cost

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{i+1}/{len(to_process)}] {date} - {len(questions)} Qs - {rate:.1f} req/min - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if all_questions:
        con.execute("""
            CREATE TABLE questions (
                date DATE,
                question VARCHAR,
                category VARCHAR,
                depth VARCHAR,
                processed_at VARCHAR
            )
        """)

        for q in all_questions:
            con.execute("INSERT INTO questions VALUES (?, ?, ?, ?, ?)",
                       [q["date"], q["question"], q["category"],
                        q["depth"], q["processed_at"]])

        if output_path.exists():
            con.execute(f"INSERT INTO questions SELECT * FROM '{output_path}'")

        con.execute(f"COPY questions TO '{output_path}' (FORMAT PARQUET)")

        print(f"\n  Extracted: {len(all_questions)} questions from {len(to_process)} days")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Wrote to: {output_path}")

        # Show category distribution
        categories = {}
        for q in all_questions:
            cat = q["category"]
            categories[cat] = categories.get(cat, 0) + 1
        print("\n  Categories:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:5]:
            print(f"    {cat}: {count}")

    config = {
        "name": "questions/v1",
        "description": "LLM-extracted questions with categorization",
        "model": MODEL,
        "created": datetime.now().strftime("%Y-%m-%d")
    }
    with open(INTERP_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return len(all_questions)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Max days')
    parser.add_argument('--test', action='store_true', help='Test with 10 days')
    args = parser.parse_args()

    if args.test:
        build_question_extraction(limit=10)
    else:
        build_question_extraction(limit=args.limit)
