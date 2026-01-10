#!/usr/bin/env python3
"""
Build weekly_expertise interpretation: Track technologies and tools used confidently.
Proven approach from Round 3 testing - V3 format.
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
INTERP_DIR = DATA_DIR / "interpretations" / "weekly_expertise" / "v1"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash-lite"

REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE


def call_gemini(prompt: str, max_tokens: int = 600) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Intellectual DNA Expertise"
    }
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """Identify technologies and tools this person is working with confidently.
Look for specific mentions of:
- Programming languages (Python, TypeScript, etc.)
- Frameworks (React, FastAPI, etc.)
- Databases (DuckDB, PostgreSQL, etc.)
- Tools (git, docker, Claude, etc.)
- Libraries (pandas, numpy, etc.)
- Platforms (Supabase, Vercel, etc.)

For each technology found:
TECH: [name]
EVIDENCE: [brief quote or description of usage]
CONFIDENCE: high/medium

Only include technologies that are clearly being USED, not just mentioned.
If no clear technology usage found: "No technology signals this week."
Be specific - don't guess or hallucinate technologies not mentioned."""
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


def parse_expertise(text: str) -> list:
    """Parse expertise signals from LLM output."""
    expertise = []
    lines = text.split('\n')

    current = {}
    for line in lines:
        line = line.strip()
        if line.startswith('TECH:'):
            if current and 'tech' in current:
                expertise.append(current)
            current = {'tech': line[5:].strip()}
        elif line.startswith('EVIDENCE:') and current:
            current['evidence'] = line[9:].strip()
        elif line.startswith('CONFIDENCE:') and current:
            current['confidence'] = line[11:].strip().lower()

    # Don't forget the last one
    if current and 'tech' in current:
        expertise.append(current)

    return expertise


def build_weekly_expertise(limit: int = None):
    """Build weekly expertise interpretation."""
    print("Building weekly_expertise/v1 interpretation...")
    print(f"  Model: {MODEL}")

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

    # Get weekly messages
    weekly_messages = con.execute(f"""
        SELECT
            DATE_TRUNC('week', idx.timestamp::TIMESTAMP) as week_start,
            LIST(c.full_content ORDER BY idx.timestamp::TIMESTAMP) as messages,
            COUNT(*) as message_count
        FROM '{FACTS_DIR}/brain/content.parquet' c
        JOIN '{FACTS_DIR}/brain/index.parquet' idx ON c.event_id = idx.event_id
        WHERE c.event_type = 'message'
          AND idx.subtype = 'user'
          AND c.full_content IS NOT NULL
          AND LENGTH(c.full_content) BETWEEN 30 AND 800
        GROUP BY DATE_TRUNC('week', idx.timestamp::TIMESTAMP)
        HAVING COUNT(*) >= 20
        ORDER BY week_start
    """).fetchall()

    to_process = [(w, m, c) for w, m, c in weekly_messages
                  if str(w) not in processed_weeks]

    if limit:
        to_process = to_process[:limit]

    print(f"  Weeks to process: {len(to_process)}")

    if not to_process:
        print("  Nothing to process!")
        return

    results = []
    total_cost = 0.0
    total_tech = 0
    start_time = time.time()

    for i, (week_start, messages, msg_count) in enumerate(to_process):
        sample = messages[:40]
        prompt = f"Week of {week_start}\n\n" + "\n---\n".join(sample)

        response = call_gemini(prompt)
        expertise = parse_expertise(response)
        total_tech += len(expertise)

        # Extract unique techs for summary
        tech_list = list(set(e.get('tech', '') for e in expertise if e.get('tech')))

        results.append({
            "week_start": str(week_start),
            "expertise_json": json.dumps(expertise),
            "tech_count": len(expertise),
            "tech_list": json.dumps(tech_list),
            "raw_response": response[:600],
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
            techs = ', '.join(tech_list[:3]) if tech_list else 'none'
            print(f"  [{i+1}/{len(to_process)}] {week_start} - {len(expertise)} techs ({techs}) - {rate:.1f} req/min - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if results:
        con.execute("""
            CREATE TABLE weekly_expertise (
                week_start DATE,
                expertise_json VARCHAR,
                tech_count INTEGER,
                tech_list VARCHAR,
                raw_response VARCHAR,
                message_count INTEGER,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO weekly_expertise VALUES (?, ?, ?, ?, ?, ?, ?)",
                       [r["week_start"], r["expertise_json"], r["tech_count"],
                        r["tech_list"], r["raw_response"], r["message_count"], r["processed_at"]])

        if output_path.exists():
            con.execute(f"INSERT INTO weekly_expertise SELECT * FROM '{output_path}'")

        con.execute(f"COPY weekly_expertise TO '{output_path}' (FORMAT PARQUET)")

        print(f"\n  Processed: {len(results)} weeks")
        print(f"  Total tech signals: {total_tech}")
        print(f"  Total cost: ${total_cost:.4f}")

        # Show tech frequency
        all_techs = []
        for r in results:
            all_techs.extend(json.loads(r["tech_list"]))
        from collections import Counter
        top_techs = Counter(all_techs).most_common(10)
        if top_techs:
            print(f"\n  Top technologies:")
            for tech, count in top_techs:
                print(f"    {tech}: {count} weeks")

    config = {
        "name": "weekly_expertise/v1",
        "description": "Technologies and tools used each week",
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
    parser.add_argument('--limit', type=int, help='Max weeks')
    parser.add_argument('--test', action='store_true', help='Test with 5 weeks')
    args = parser.parse_args()

    if args.test:
        build_weekly_expertise(limit=5)
    else:
        build_weekly_expertise(limit=args.limit)
