#!/usr/bin/env python3
"""
Build daily_accomplishments interpretation: What got DONE each day.
Proven approach from Round 3 testing.
"""

import os
import json
import re
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
INTERP_DIR = DATA_DIR / "interpretations" / "daily_accomplishments" / "v1"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash-lite"

REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE


def call_gemini(prompt: str, max_tokens: int = 500) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Intellectual DNA Accomplishments"
    }
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """Analyze these messages and extract what got DONE.
Write in first person as a daily log.

Look for evidence of:
- Code shipped/deployed/merged
- Bugs fixed
- Features built
- Problems solved
- Things created/generated
- Configurations set up
- Data processed/cleaned
- Documents/content created

Format:
✓ [what was done] (category)

Categories: shipped, fixed, built, solved, setup, created, processed, documented

If nothing concrete was completed: "Mostly exploration and discussion today."
Be specific about the actual work, not vague summaries."""
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Error: {str(e)[:50]}]"


def parse_accomplishments(text: str) -> list:
    """Parse accomplishments from LLM output."""
    accomplishments = []
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('✓') or line.startswith('*'):
            # Extract accomplishment and category
            match = re.search(r'[✓\*]\s*(.+?)\s*\((\w+)\)', line)
            if match:
                accomplishments.append({
                    "accomplishment": match.group(1).strip(),
                    "category": match.group(2).strip()
                })
            else:
                # No category parsed
                clean = line.lstrip('✓* ').strip()
                if clean and len(clean) > 10:
                    accomplishments.append({
                        "accomplishment": clean,
                        "category": "uncategorized"
                    })
    return accomplishments


def build_daily_accomplishments(limit: int = None):
    """Build daily accomplishments interpretation."""
    print("Building daily_accomplishments/v1 interpretation...")
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

    # Get daily messages
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
          AND LENGTH(c.full_content) BETWEEN 20 AND 600
        GROUP BY CAST(idx.timestamp::TIMESTAMP AS DATE)
        HAVING COUNT(*) >= 10
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
    total_accomplishments = 0
    start_time = time.time()

    for i, (date, messages, msg_count) in enumerate(to_process):
        sample = messages[:30]
        prompt = f"Date: {date}\n\n" + "\n---\n".join(sample)

        response = call_gemini(prompt)
        accomplishments = parse_accomplishments(response)
        total_accomplishments += len(accomplishments)

        results.append({
            "date": str(date),
            "accomplishments_json": json.dumps(accomplishments),
            "accomplishment_count": len(accomplishments),
            "raw_response": response[:500],
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
            print(f"  [{i+1}/{len(to_process)}] {date} - {len(accomplishments)} accomplishments - {rate:.1f} req/min - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if results:
        con.execute("""
            CREATE TABLE daily_accomplishments (
                date DATE,
                accomplishments_json VARCHAR,
                accomplishment_count INTEGER,
                raw_response VARCHAR,
                message_count INTEGER,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO daily_accomplishments VALUES (?, ?, ?, ?, ?, ?)",
                       [r["date"], r["accomplishments_json"], r["accomplishment_count"],
                        r["raw_response"], r["message_count"], r["processed_at"]])

        if output_path.exists():
            con.execute(f"INSERT INTO daily_accomplishments SELECT * FROM '{output_path}'")

        con.execute(f"COPY daily_accomplishments TO '{output_path}' (FORMAT PARQUET)")

        print(f"\n  Processed: {len(results)} days")
        print(f"  Total accomplishments: {total_accomplishments}")
        print(f"  Total cost: ${total_cost:.4f}")

    config = {
        "name": "daily_accomplishments/v1",
        "description": "What got done each day",
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
    parser.add_argument('--test', action='store_true', help='Test with 5 days')
    args = parser.parse_args()

    if args.test:
        build_daily_accomplishments(limit=5)
    else:
        build_daily_accomplishments(limit=args.limit)
