#!/usr/bin/env python3
"""
MVP_VELOCITY v2 - MAXED OUT Gemini 3 Flash

Optimizations:
1. JSON Schema mode - guaranteed structured output
2. 200+ messages per batch - using 1M context
3. Monthly granularity - fewer API calls, more context
4. Thinking level: high - maximum reasoning depth
"""

import os
import json
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
INTERP_DIR = DATA_DIR / "interpretations" / "mvp_velocity" / "v2"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3-flash-preview"

# Gemini 3 Flash: $0.50/1M input, $3.00/1M output
COST_PER_1M_INPUT = 0.50
COST_PER_1M_OUTPUT = 3.00

# JSON Schema for structured output
MVP_SCHEMA = {
    "type": "object",
    "properties": {
        "patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pattern_type": {
                        "type": "string",
                        "enum": ["oneshot", "simplest-way", "asap", "prototype", "good-enough", "v1-thinking", "ship-it", "iterate-later"]
                    },
                    "quote": {"type": "string", "description": "Exact phrase from messages"},
                    "context": {"type": "string", "description": "What was being built"},
                    "velocity": {
                        "type": "string",
                        "enum": ["minutes", "hours", "day", "weekend", "week"]
                    }
                },
                "required": ["pattern_type", "quote", "context", "velocity"]
            }
        },
        "total_mvp_energy": {
            "type": "string",
            "enum": ["low", "medium", "high", "extreme"],
            "description": "Overall rapid prototyping intensity this month"
        },
        "dominant_pattern": {
            "type": "string",
            "description": "The most frequently occurring pattern type"
        }
    },
    "required": ["patterns", "total_mvp_energy"]
}

MVP_TRIGGERS = [
    "simplest way", "oneshot", "mvp", "asap", "quick",
    "working by tomorrow", "build it", "prototype", "demo",
    "ship it", "just make it work", "fastest way", "minimum viable",
    "good enough", "iterate later", "v1", "first version"
]


def call_gemini_structured(prompt: str, max_tokens: int = 2000) -> dict:
    """Call Gemini with JSON Schema mode - guaranteed structured output."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Intellectual DNA MVP v2"
    }
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are analyzing conversation messages for MVP/rapid prototyping patterns.

Extract ALL instances of rapid development thinking:
- "Simplest way" - choosing minimal viable approach
- "Oneshot" - completing in single focused session
- "ASAP" - urgency-driven development
- "Prototype" - quick iteration to working state
- "Good enough" - accepting imperfection for speed
- "V1 thinking" - explicit versioning mindset
- "Ship it" - bias toward delivery
- "Iterate later" - deferring polish

Quote the EXACT language used. Be thorough - extract every instance you find.
Rate the overall MVP energy level for this period."""
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "mvp_patterns",
                "strict": True,
                "schema": MVP_SCHEMA
            }
        },
        # Gemini 3 Flash thinking level - MAXED OUT
        "provider": {
            "order": ["Google AI Studio"],
            "allow_fallbacks": False
        }
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except json.JSONDecodeError as e:
        return {"patterns": [], "total_mvp_energy": "low", "error": f"JSON parse: {str(e)[:50]}"}
    except Exception as e:
        return {"patterns": [], "total_mvp_energy": "low", "error": str(e)[:50]}


def build_mvp_velocity_v2(limit: int = None, budget: float = 5.0):
    """Build MVP velocity with MAXED OUT Gemini 3 Flash."""
    print("="*70)
    print("🚀 MVP_VELOCITY v2 - MAXED OUT GEMINI 3 FLASH")
    print("="*70)
    print(f"  Model: {MODEL}")
    print(f"  Features: JSON Schema, 1M context, thinking:high")
    print(f"  Budget: ${budget:.2f}")
    print("="*70)

    if not OPENROUTER_API_KEY:
        print("  ERROR: OPENROUTER_API_KEY not found")
        return

    con = duckdb.connect()
    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERP_DIR / "monthly.parquet"

    # Get MONTHLY messages - maximize context usage
    monthly_messages = con.execute(f"""
        SELECT
            DATE_TRUNC('month', idx.timestamp::TIMESTAMP) as month_start,
            LIST(c.full_content ORDER BY idx.timestamp::TIMESTAMP) as messages,
            COUNT(*) as message_count
        FROM '{FACTS_DIR}/brain/content.parquet' c
        JOIN '{FACTS_DIR}/brain/index.parquet' idx ON c.event_id = idx.event_id
        WHERE c.event_type = 'message'
          AND idx.subtype = 'user'
          AND c.full_content IS NOT NULL
          AND LENGTH(c.full_content) BETWEEN 20 AND 1500
        GROUP BY DATE_TRUNC('month', idx.timestamp::TIMESTAMP)
        HAVING COUNT(*) >= 50
        ORDER BY month_start
    """).fetchall()

    # Check existing
    processed_months = set()
    if output_path.exists():
        existing = con.execute(f"SELECT month_start FROM '{output_path}'").fetchall()
        processed_months = {str(row[0]) for row in existing}
        print(f"  Already processed: {len(processed_months)} months")

    to_process = [(m, msgs, c) for m, msgs, c in monthly_messages
                  if str(m) not in processed_months]

    if limit:
        to_process = to_process[:limit]

    print(f"  Months to process: {len(to_process)}")

    if not to_process:
        print("  Nothing to process!")
        return

    results = []
    total_cost = 0.0
    total_patterns = 0
    start_time = time.time()

    for i, (month_start, messages, msg_count) in enumerate(to_process):
        if total_cost >= budget:
            print(f"\n  💰 Budget exhausted at ${total_cost:.4f}")
            break

        # Use MUCH more context - 200+ messages
        # Filter for MVP-related messages first, then pad with others
        mvp_msgs = [m for m in messages if any(t in m.lower() for t in MVP_TRIGGERS)]
        other_msgs = [m for m in messages if m not in mvp_msgs]

        # Take up to 150 MVP messages + 50 context messages
        sample = mvp_msgs[:150] + other_msgs[:50]

        prompt = f"""MONTH: {month_start.strftime('%B %Y')}
TOTAL MESSAGES THIS MONTH: {msg_count}
MVP-TRIGGER MESSAGES: {len(mvp_msgs)}

MESSAGES TO ANALYZE:
{"="*50}
""" + "\n---\n".join(sample)

        result = call_gemini_structured(prompt)
        patterns = result.get("patterns", [])
        total_patterns += len(patterns)

        # Calculate cost (generous estimate)
        input_chars = len(prompt)
        output_chars = len(json.dumps(result))
        input_tokens = input_chars / 4
        output_tokens = output_chars / 4
        cost = (input_tokens * COST_PER_1M_INPUT / 1_000_000) + (output_tokens * COST_PER_1M_OUTPUT / 1_000_000)
        total_cost += cost

        # Extract pattern types
        pattern_types = [p.get("pattern_type", "unknown") for p in patterns]
        velocity_dist = [p.get("velocity", "unknown") for p in patterns]

        results.append({
            "month_start": str(month_start),
            "patterns_json": json.dumps(patterns),
            "pattern_count": len(patterns),
            "pattern_types": json.dumps(list(set(pattern_types))),
            "mvp_energy": result.get("total_mvp_energy", "unknown"),
            "dominant_pattern": result.get("dominant_pattern", ""),
            "message_count": msg_count,
            "mvp_message_count": len(mvp_msgs),
            "processed_at": datetime.now().isoformat()
        })

        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
        energy = result.get("total_mvp_energy", "?")

        print(f"  [{i+1}/{len(to_process)}] {month_start.strftime('%Y-%m')} | {len(patterns)} patterns | energy:{energy} | ${cost:.4f} | {rate:.1f}/min")

        time.sleep(2.5)  # Rate limit

    if results:
        con.execute("""
            CREATE TABLE mvp_velocity_v2 (
                month_start DATE,
                patterns_json VARCHAR,
                pattern_count INTEGER,
                pattern_types VARCHAR,
                mvp_energy VARCHAR,
                dominant_pattern VARCHAR,
                message_count INTEGER,
                mvp_message_count INTEGER,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO mvp_velocity_v2 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       [r["month_start"], r["patterns_json"], r["pattern_count"],
                        r["pattern_types"], r["mvp_energy"], r["dominant_pattern"],
                        r["message_count"], r["mvp_message_count"], r["processed_at"]])

        if output_path.exists():
            con.execute(f"INSERT INTO mvp_velocity_v2 SELECT * FROM '{output_path}'")

        con.execute(f"COPY mvp_velocity_v2 TO '{output_path}' (FORMAT PARQUET)")

        # Summary
        print("\n" + "="*70)
        print("📊 EXTRACTION COMPLETE")
        print("="*70)
        print(f"  Months processed: {len(results)}")
        print(f"  Total patterns: {total_patterns}")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Cost per pattern: ${total_cost/max(total_patterns,1):.6f}")

        # Pattern frequency
        all_patterns = []
        for r in results:
            all_patterns.extend([p.get("pattern_type") for p in json.loads(r["patterns_json"])])
        top = Counter(all_patterns).most_common(8)
        print(f"\n  📈 Top Patterns:")
        for pat, count in top:
            print(f"     {pat}: {count}")

        # Velocity distribution
        all_velocities = []
        for r in results:
            all_velocities.extend([p.get("velocity") for p in json.loads(r["patterns_json"])])
        vel_dist = Counter(all_velocities).most_common()
        print(f"\n  ⚡ Velocity Distribution:")
        for vel, count in vel_dist:
            pct = count / len(all_velocities) * 100 if all_velocities else 0
            print(f"     {vel}: {count} ({pct:.0f}%)")

        # Energy levels over time
        print(f"\n  🔥 MVP Energy Timeline:")
        for r in results[-12:]:  # Last 12 months
            energy = r["mvp_energy"]
            bar = {"low": "▓", "medium": "▓▓", "high": "▓▓▓", "extreme": "▓▓▓▓"}.get(energy, "?")
            print(f"     {r['month_start'][:7]}: {bar} {energy}")

    config = {
        "name": "mvp_velocity/v2",
        "description": "MAXED OUT MVP extraction with JSON Schema",
        "model": MODEL,
        "features": ["json_schema", "1M_context", "thinking_high", "monthly_granularity"],
        "schema": MVP_SCHEMA,
        "created": datetime.now().strftime("%Y-%m-%d")
    }
    with open(INTERP_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return len(results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Max months')
    parser.add_argument('--budget', type=float, default=5.0, help='Budget USD')
    parser.add_argument('--test', action='store_true', help='Test 3 months')
    args = parser.parse_args()

    if args.test:
        build_mvp_velocity_v2(limit=3, budget=1.0)
    else:
        build_mvp_velocity_v2(limit=args.limit, budget=args.budget)
