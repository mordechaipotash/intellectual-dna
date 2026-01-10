#!/usr/bin/env python3
"""
Build MVP_VELOCITY interpretation: Track rapid prototyping patterns.
Extracts: oneshot builds, simplest way thinking, ASAP delivery, prototype→ship cycles.

Model: Gemini 3 Flash (Dec 2025) - Near-Pro quality, 3x faster
Budget: $5 (~6000 requests)
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
from collections import Counter

load_dotenv(Path("/Users/mordechai/intellectual_dna/.env"))

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
DATA_DIR = BASE_DIR / "data"
FACTS_DIR = DATA_DIR / "facts"
INTERP_DIR = DATA_DIR / "interpretations" / "mvp_velocity" / "v1"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3-flash-preview"

# Gemini 3 Flash: $0.50/1M input, $3.00/1M output
COST_PER_1M_INPUT = 0.50
COST_PER_1M_OUTPUT = 3.00

REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE

# Trigger phrases from 100x deep mining
MVP_TRIGGERS = [
    "simplest way", "oneshot", "mvp", "asap", "quick",
    "working by tomorrow", "build it", "prototype", "demo",
    "ship it", "just make it work", "fastest way", "minimum viable",
    "good enough", "iterate later", "v1", "first version"
]


def call_gemini(prompt: str, max_tokens: int = 600) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Intellectual DNA MVP Velocity"
    }
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """Extract MVP/rapid prototyping patterns from these messages.

Look for:
- "Simplest way" thinking - choosing minimal viable approach
- Oneshot builds - completing something in single focused session
- ASAP delivery pressure - urgency-driven development
- Prototype→ship cycles - quick iteration to working state
- "Good enough" decisions - accepting imperfection for speed
- V1 thinking - explicit versioning mindset

For each pattern found:
PATTERN: [oneshot|simplest-way|asap|prototype|good-enough|v1-thinking]
QUOTE: [exact phrase or close paraphrase from messages]
CONTEXT: [what was being built/solved]
VELOCITY: [hours|day|weekend|week] - estimated time to ship

Only include ACTUAL rapid development instances, not discussions about methodology.
If no MVP patterns found: "No rapid prototyping signals this period."
Be specific - quote the actual language used."""
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


def parse_mvp_patterns(text: str) -> list:
    """Parse MVP velocity patterns from LLM output."""
    patterns = []
    lines = text.split('\n')

    current = {}
    for line in lines:
        line = line.strip()
        if line.startswith('PATTERN:'):
            if current and 'pattern' in current:
                patterns.append(current)
            current = {'pattern': line[8:].strip().lower()}
        elif line.startswith('QUOTE:') and current:
            current['quote'] = line[6:].strip()
        elif line.startswith('CONTEXT:') and current:
            current['context'] = line[8:].strip()
        elif line.startswith('VELOCITY:') and current:
            current['velocity'] = line[9:].strip().lower()

    if current and 'pattern' in current and 'quote' in current:
        patterns.append(current)

    return patterns


def has_mvp_triggers(messages: list) -> bool:
    """Check if messages contain MVP trigger phrases."""
    text = ' '.join(messages).lower()
    return any(trigger in text for trigger in MVP_TRIGGERS)


def build_mvp_velocity(limit: int = None, budget: float = 5.0):
    """Build MVP velocity interpretation."""
    print("Building mvp_velocity/v1 interpretation...")
    print(f"  Model: {MODEL}")
    print(f"  Budget: ${budget:.2f}")

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
          AND LENGTH(c.full_content) BETWEEN 30 AND 1000
        GROUP BY DATE_TRUNC('week', idx.timestamp::TIMESTAMP)
        HAVING COUNT(*) >= 15
        ORDER BY week_start
    """).fetchall()

    # Filter to weeks with MVP triggers and not yet processed
    to_process = [(w, m, c) for w, m, c in weekly_messages
                  if str(w) not in processed_weeks and has_mvp_triggers(m)]

    if limit:
        to_process = to_process[:limit]

    print(f"  Weeks with MVP triggers: {len(to_process)}")

    if not to_process:
        print("  Nothing to process!")
        return

    results = []
    total_cost = 0.0
    total_patterns = 0
    start_time = time.time()

    for i, (week_start, messages, msg_count) in enumerate(to_process):
        # Check budget
        if total_cost >= budget:
            print(f"  Budget exhausted at ${total_cost:.4f}")
            break

        # Sample messages with MVP triggers preferentially
        trigger_msgs = [m for m in messages if any(t in m.lower() for t in MVP_TRIGGERS)]
        other_msgs = [m for m in messages if m not in trigger_msgs]
        sample = trigger_msgs[:30] + other_msgs[:20]

        prompt = f"Week of {week_start}\n\n" + "\n---\n".join(sample[:50])

        response = call_gemini(prompt)
        patterns = parse_mvp_patterns(response)
        total_patterns += len(patterns)

        # Extract pattern types for summary
        pattern_types = list(set(p.get('pattern', 'unknown') for p in patterns if p.get('pattern')))

        results.append({
            "week_start": str(week_start),
            "patterns_json": json.dumps(patterns),
            "pattern_count": len(patterns),
            "pattern_types": json.dumps(pattern_types),
            "raw_response": response[:600],
            "message_count": msg_count,
            "processed_at": datetime.now().isoformat()
        })

        # Calculate cost
        input_tokens = len(prompt) / 4
        output_tokens = len(response) / 4
        cost = (input_tokens * COST_PER_1M_INPUT / 1_000_000) + (output_tokens * COST_PER_1M_OUTPUT / 1_000_000)
        total_cost += cost

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            types_str = ', '.join(pattern_types[:3]) if pattern_types else 'none'
            print(f"  [{i+1}/{len(to_process)}] {week_start} - {len(patterns)} patterns ({types_str}) - {rate:.1f} req/min - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if results:
        con.execute("""
            CREATE TABLE mvp_velocity (
                week_start DATE,
                patterns_json VARCHAR,
                pattern_count INTEGER,
                pattern_types VARCHAR,
                raw_response VARCHAR,
                message_count INTEGER,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO mvp_velocity VALUES (?, ?, ?, ?, ?, ?, ?)",
                       [r["week_start"], r["patterns_json"], r["pattern_count"],
                        r["pattern_types"], r["raw_response"], r["message_count"], r["processed_at"]])

        if output_path.exists():
            con.execute(f"INSERT INTO mvp_velocity SELECT * FROM '{output_path}'")

        con.execute(f"COPY mvp_velocity TO '{output_path}' (FORMAT PARQUET)")

        print(f"\n  Processed: {len(results)} weeks")
        print(f"  Total MVP patterns: {total_patterns}")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Remaining budget: ${budget - total_cost:.4f}")

        # Show pattern frequency
        all_patterns = []
        for r in results:
            all_patterns.extend([p.get('pattern', 'unknown') for p in json.loads(r["patterns_json"])])
        top_patterns = Counter(all_patterns).most_common(10)
        if top_patterns:
            print(f"\n  Top MVP patterns:")
            for pattern, count in top_patterns:
                print(f"    {pattern}: {count} instances")

        # Show velocity distribution
        all_velocities = []
        for r in results:
            all_velocities.extend([p.get('velocity', 'unknown') for p in json.loads(r["patterns_json"]) if p.get('velocity')])
        velocity_dist = Counter(all_velocities).most_common()
        if velocity_dist:
            print(f"\n  Velocity distribution:")
            for vel, count in velocity_dist:
                print(f"    {vel}: {count}")

        # Show sample patterns
        print(f"\n  Sample MVP patterns:")
        for r in results[-5:]:
            pat_list = json.loads(r["patterns_json"])
            if pat_list:
                first = pat_list[0]
                quote = first.get('quote', '?')[:40]
                print(f"    {r['week_start'][:10]}: [{first.get('pattern', '?')}] \"{quote}...\"")

    config = {
        "name": "mvp_velocity/v1",
        "description": "Rapid prototyping and MVP patterns",
        "model": MODEL,
        "triggers": MVP_TRIGGERS,
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
    parser.add_argument('--budget', type=float, default=5.0, help='Max budget in USD')
    parser.add_argument('--test', action='store_true', help='Test with 5 weeks')
    args = parser.parse_args()

    if args.test:
        build_mvp_velocity(limit=5, budget=0.50)
    else:
        build_mvp_velocity(limit=args.limit, budget=args.budget)
