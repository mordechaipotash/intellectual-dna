#!/usr/bin/env python3
"""
Build problem_resolutions interpretation: Track issues that got RESOLVED.
Proven approach from Round 3 testing - V4 format.
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
INTERP_DIR = DATA_DIR / "interpretations" / "problem_resolutions" / "v1"

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
        "X-Title": "Intellectual DNA Problems"
    }
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """Find any issues or blockers that were RESOLVED in these messages.
Look for patterns like:
- "fixed the bug where..."
- "finally got X working"
- "the issue was..."
- "solved by..."
- "turns out the problem was..."
- Error messages followed by solutions

Format each resolution as:
ISSUE: [what was broken/blocking]
RESOLUTION: [what fixed it]
DOMAIN: [tech area - e.g., python, api, database, ui, config, etc.]

Only include problems that were ACTUALLY SOLVED, not ongoing issues.
If no resolutions found: "No resolutions this period."
Be specific and quote from the messages when possible."""
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


def parse_resolutions(text: str) -> list:
    """Parse problem resolutions from LLM output."""
    resolutions = []
    lines = text.split('\n')

    current = {}
    for line in lines:
        line = line.strip()
        if line.startswith('ISSUE:'):
            if current and 'issue' in current:
                resolutions.append(current)
            current = {'issue': line[6:].strip()}
        elif line.startswith('RESOLUTION:') and current:
            current['resolution'] = line[11:].strip()
        elif line.startswith('DOMAIN:') and current:
            current['domain'] = line[7:].strip().lower()

    # Don't forget the last one
    if current and 'issue' in current and 'resolution' in current:
        resolutions.append(current)

    return resolutions


def build_problem_resolutions(limit: int = None):
    """Build problem resolutions interpretation."""
    print("Building problem_resolutions/v1 interpretation...")
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

    # Get weekly messages - focus on problem-solving language
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
          AND LENGTH(c.full_content) BETWEEN 30 AND 1000
        GROUP BY DATE_TRUNC('week', idx.timestamp::TIMESTAMP)
        HAVING COUNT(*) >= 15
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
    total_resolutions = 0
    start_time = time.time()

    for i, (week_start, messages, msg_count) in enumerate(to_process):
        # Sample messages, prefer those with problem-solving keywords
        sample = messages[:50]
        prompt = f"Week of {week_start}\n\n" + "\n---\n".join(sample)

        response = call_gemini(prompt)
        resolutions = parse_resolutions(response)
        total_resolutions += len(resolutions)

        # Extract domains for summary
        domains = list(set(r.get('domain', 'unknown') for r in resolutions if r.get('domain')))

        results.append({
            "week_start": str(week_start),
            "resolutions_json": json.dumps(resolutions),
            "resolution_count": len(resolutions),
            "domains": json.dumps(domains),
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
            domain_str = ', '.join(domains[:3]) if domains else 'none'
            print(f"  [{i+1}/{len(to_process)}] {week_start} - {len(resolutions)} resolutions ({domain_str}) - {rate:.1f} req/min - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if results:
        con.execute("""
            CREATE TABLE problem_resolutions (
                week_start DATE,
                resolutions_json VARCHAR,
                resolution_count INTEGER,
                domains VARCHAR,
                raw_response VARCHAR,
                message_count INTEGER,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO problem_resolutions VALUES (?, ?, ?, ?, ?, ?, ?)",
                       [r["week_start"], r["resolutions_json"], r["resolution_count"],
                        r["domains"], r["raw_response"], r["message_count"], r["processed_at"]])

        if output_path.exists():
            con.execute(f"INSERT INTO problem_resolutions SELECT * FROM '{output_path}'")

        con.execute(f"COPY problem_resolutions TO '{output_path}' (FORMAT PARQUET)")

        print(f"\n  Processed: {len(results)} weeks")
        print(f"  Total resolutions: {total_resolutions}")
        print(f"  Total cost: ${total_cost:.4f}")

        # Show domain frequency
        all_domains = []
        for r in results:
            all_domains.extend(json.loads(r["domains"]))
        from collections import Counter
        top_domains = Counter(all_domains).most_common(10)
        if top_domains:
            print(f"\n  Top problem domains:")
            for domain, count in top_domains:
                print(f"    {domain}: {count} weeks")

        # Show sample resolutions
        print(f"\n  Sample resolutions:")
        for r in results[-5:]:
            res_list = json.loads(r["resolutions_json"])
            if res_list:
                first = res_list[0]
                print(f"    {r['week_start'][:10]}: {first.get('issue', '?')[:50]}... → {first.get('resolution', '?')[:30]}...")

    config = {
        "name": "problem_resolutions/v1",
        "description": "Issues resolved each week",
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
        build_problem_resolutions(limit=5)
    else:
        build_problem_resolutions(limit=args.limit)
