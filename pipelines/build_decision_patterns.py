#!/usr/bin/env python3
"""
Build decision_patterns interpretation: Extract decision-making patterns.

Uses Gemini 2.5 Flash Lite to identify when and how decisions are made,
what triggers them, and what patterns emerge.
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
INTERP_DIR = DATA_DIR / "interpretations" / "decision_patterns" / "v1"

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
        "X-Title": "Intellectual DNA Decision Patterns"
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """Analyze decision-making patterns in these messages.
Look for:
1. Decision triggers ("I'm going to...", "Let's...", "I've decided...")
2. Decision types: technical, strategic, creative, abandonment, pivot
3. Decision speed: impulsive, deliberate, procrastinated
4. Confidence level: certain, tentative, experimental
5. What influenced the decision

Write in FIRST PERSON as if reflecting on your own decision-making.

Output JSON: {"decisions": [{"decision": "what I decided", "type": "technical|strategic|creative|abandon|pivot", "trigger": "what prompted this", "speed": "impulsive|deliberate|procrastinated", "confidence": "certain|tentative|experimental", "outcome_hint": "any indication of result"}], "pattern_note": "overall decision-making pattern observed"}"""
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


def parse_decisions(llm_output: str) -> dict:
    """Parse LLM output for decision data."""
    import re
    try:
        match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}
    except:
        return {}


def build_decision_patterns(limit: int = None):
    """Build decision pattern analysis."""
    print("Building decision_patterns/v1 interpretation...")
    print(f"  Model: {MODEL}")

    if not OPENROUTER_API_KEY:
        print("  ERROR: OPENROUTER_API_KEY not found")
        return

    con = duckdb.connect()
    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERP_DIR / "daily.parquet"

    # Check existing
    processed_dates = set()
    if output_path.exists():
        existing = con.execute(f"SELECT date FROM '{output_path}'").fetchall()
        processed_dates = {str(row[0]) for row in existing}
        print(f"  Already processed: {len(processed_dates)} days")

    # Get daily messages with decision-like content
    daily_messages = con.execute(f"""
        SELECT
            CAST(idx.timestamp::TIMESTAMP AS DATE) as date,
            LIST(c.full_content ORDER BY idx.timestamp::TIMESTAMP) as messages,
            COUNT(*) as message_count
        FROM '{FACTS_DIR}/brain/content.parquet' c
        JOIN '{FACTS_DIR}/brain/index.parquet' idx ON c.event_id = idx.event_id
        WHERE c.event_type = 'message'
          AND idx.subtype = 'user'
          AND c.full_content IS NOT NULL
          AND (
              c.full_content LIKE '%going to%'
              OR c.full_content LIKE '%let''s%'
              OR c.full_content LIKE '%decided%'
              OR c.full_content LIKE '%I will%'
              OR c.full_content LIKE '%I''ll%'
              OR c.full_content LIKE '%instead%'
              OR c.full_content LIKE '%switch%'
              OR c.full_content LIKE '%abandon%'
              OR c.full_content LIKE '%start%'
              OR c.full_content LIKE '%stop%'
          )
        GROUP BY CAST(idx.timestamp::TIMESTAMP AS DATE)
        HAVING COUNT(*) >= 3
        ORDER BY date
    """).fetchall()

    to_process = [(d, m, c) for d, m, c in daily_messages
                  if str(d) not in processed_dates]

    if limit:
        to_process = to_process[:limit]

    print(f"  Days to process: {len(to_process)}")

    if not to_process:
        print("  Nothing to process!")
        return

    results = []
    total_cost = 0.0
    start_time = time.time()

    for i, (date, messages, msg_count) in enumerate(to_process):
        sample = messages[:20] if len(messages) > 20 else messages
        sample_text = "\n---\n".join(sample)

        prompt = f"""Date: {date}
{msg_count} decision-related messages

Messages:
{sample_text}

What decisions did I make on this day? What patterns do you see in my decision-making?"""

        response = call_gemini(prompt)
        parsed = parse_decisions(response)

        decisions = parsed.get("decisions", [])
        pattern_note = parsed.get("pattern_note", "")

        results.append({
            "date": str(date),
            "decision_count": len(decisions),
            "decisions_json": json.dumps(decisions),
            "pattern_note": pattern_note,
            "message_count": msg_count,
            "processed_at": datetime.now().isoformat()
        })

        input_tokens = len(prompt) / 4
        output_tokens = len(response) / 4
        cost = (input_tokens * 0.075 / 1_000_000) + (output_tokens * 0.30 / 1_000_000)
        total_cost += cost

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{i+1}/{len(to_process)}] {date} - {len(decisions)} decisions - {rate:.1f} req/min - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if results:
        con.execute("""
            CREATE TABLE decision_patterns (
                date DATE,
                decision_count INTEGER,
                decisions_json VARCHAR,
                pattern_note VARCHAR,
                message_count INTEGER,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO decision_patterns VALUES (?, ?, ?, ?, ?, ?)",
                       [r["date"], r["decision_count"], r["decisions_json"],
                        r["pattern_note"], r["message_count"], r["processed_at"]])

        if output_path.exists():
            con.execute(f"INSERT INTO decision_patterns SELECT * FROM '{output_path}'")

        con.execute(f"COPY decision_patterns TO '{output_path}' (FORMAT PARQUET)")

        total_decisions = sum(r["decision_count"] for r in results)
        print(f"\n  Processed: {len(results)} days")
        print(f"  Total decisions found: {total_decisions}")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Wrote to: {output_path}")

    config = {
        "name": "decision_patterns/v1",
        "description": "Decision-making patterns and triggers",
        "model": MODEL,
        "voice": "first-person",
        "created": datetime.now().strftime("%Y-%m-%d")
    }
    with open(INTERP_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return len(results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Max days')
    parser.add_argument('--test', action='store_true', help='Test with 10 days')
    args = parser.parse_args()

    if args.test:
        build_decision_patterns(limit=10)
    else:
        build_decision_patterns(limit=args.limit)
