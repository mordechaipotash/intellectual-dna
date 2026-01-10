#!/usr/bin/env python3
"""
monthly_themes/v2 - Monthly gestalt synthesis with JSON Schema
Uses Gemini 2.0 Flash with guaranteed structured output
"""

import os
import json
import time
import duckdb
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import pyarrow as pa
import pyarrow.parquet as pq

load_dotenv(Path("/Users/mordechai/intellectual_dna/.env"))

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
CONVERSATIONS = BASE_DIR / "data/facts/sources/all_conversations.parquet"
OUTPUT_DIR = BASE_DIR / "data/interpretations/monthly_themes/v2"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3-flash-preview"  # Always use Gemini 3 Flash

# JSON Schema for guaranteed structured output
THEME_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "A poetic 3-6 word title capturing the month's essence"
        },
        "theme": {
            "type": "string",
            "description": "The dominant theme or preoccupation in 1-2 sentences"
        },
        "emotional_arc": {
            "type": "string",
            "description": "How energy and mood evolved through the month"
        },
        "breakthroughs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key realizations or wins (2-4 items)"
        },
        "struggles": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Recurring blockers or frustrations (2-4 items)"
        },
        "projects": {
            "type": "array",
            "items": {"type": "string"},
            "description": "What was being built or worked on (2-5 items)"
        },
        "narrative": {
            "type": "string",
            "description": "A 2-3 paragraph reflective synthesis of the month"
        }
    },
    "required": ["title", "theme", "emotional_arc", "breakthroughs", "struggles", "projects", "narrative"]
}

# Filter out system/IDE messages
NOISE_PATTERNS = [
    'context window', 'tokens used', 'vscode', 'localhost',
    'node_modules', 'Tool Result', 'user uploaded', 'interrupted by',
    'may not be related', 'the current task', 'null null'
]


def is_clean_message(content: str) -> bool:
    """Check if message is genuine conversation, not system noise."""
    if not content or len(content) < 20 or len(content) > 500:
        return False
    content_lower = content.lower()
    return not any(p in content_lower for p in NOISE_PATTERNS)


def call_gemini(prompt: str) -> dict:
    """Call Gemini with JSON Schema for structured output."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Monthly Themes v2"
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You synthesize the gestalt of a month from conversation samples.
Create a rich narrative that captures:
- The dominant theme or preoccupation
- Emotional arc (energy/mood shifts)
- Key breakthroughs or realizations
- Recurring struggles or blockers
- What was being built/created

Write as a thoughtful observer. Be specific and insightful, not generic.
Capture the unique texture of this particular month."""
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.4,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "monthly_theme",
                "strict": True,
                "schema": THEME_SCHEMA
            }
        }
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=90)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse: {str(e)[:50]}"}
    except Exception as e:
        return {"error": str(e)[:100]}


def build_monthly_themes_v2(limit: int = None):
    """Build monthly theme synthesis with guaranteed structured output."""
    print("=" * 60)
    print("MONTHLY THEMES v2")
    print("=" * 60)

    if not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY not found")
        return

    con = duckdb.connect()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get monthly message counts
    print("\n📥 Loading monthly data...")
    monthly_data = con.execute(f"""
        SELECT
            DATE_TRUNC('month', msg_timestamp::TIMESTAMP) as month_start,
            LIST(content) as messages,
            COUNT(*) as message_count,
            COUNT(DISTINCT CAST(msg_timestamp::TIMESTAMP AS DATE)) as days_active
        FROM '{CONVERSATIONS}'
        WHERE role = 'user'
          AND content IS NOT NULL
          AND LENGTH(content) BETWEEN 20 AND 500
          AND msg_timestamp IS NOT NULL
        GROUP BY DATE_TRUNC('month', msg_timestamp::TIMESTAMP)
        HAVING COUNT(*) >= 30
          AND month_start IS NOT NULL
        ORDER BY month_start
    """).fetchall()

    print(f"   Found {len(monthly_data)} months with data")

    if limit:
        monthly_data = monthly_data[:limit]
        print(f"   Processing {limit} months")

    results = []
    total_cost = 0.0
    start_time = time.time()

    for i, (month_start, messages, msg_count, days_active) in enumerate(monthly_data):
        month_str = month_start.strftime('%Y-%m')
        print(f"\n[{i+1}/{len(monthly_data)}] {month_start.strftime('%B %Y')} ({msg_count} msgs, {days_active} days)")

        # Filter to clean messages
        clean_msgs = [m for m in messages if is_clean_message(m)]
        print(f"   Clean messages: {len(clean_msgs)}/{len(messages)}")

        if len(clean_msgs) < 20:
            print("   ⚠️ Too few clean messages, skipping")
            continue

        # Sample from beginning, middle, end
        n = len(clean_msgs)
        if n > 60:
            samples = clean_msgs[:20] + clean_msgs[n//2-10:n//2+10] + clean_msgs[-20:]
        else:
            samples = clean_msgs

        sample_text = "\n---\n".join(samples[:60])

        prompt = f"""Month: {month_start.strftime('%B %Y')}
Total messages: {msg_count}
Days active: {days_active}

Sample messages from throughout the month:
{sample_text}

Synthesize the gestalt of this month. What was the dominant focus? What was the emotional texture? What breakthroughs and struggles occurred?"""

        result = call_gemini(prompt)

        if "error" in result:
            print(f"   ❌ {result['error']}")
            continue

        print(f"   ✓ \"{result.get('title', '?')}\"")
        if result.get('theme'):
            print(f"     {result['theme'][:70]}...")

        results.append({
            'month_start': month_start,
            'title': result.get('title', ''),
            'theme': result.get('theme', ''),
            'emotional_arc': result.get('emotional_arc', ''),
            'breakthroughs': json.dumps(result.get('breakthroughs', [])),
            'struggles': json.dumps(result.get('struggles', [])),
            'projects': json.dumps(result.get('projects', [])),
            'narrative': result.get('narrative', ''),
            'message_count': msg_count,
            'days_active': days_active,
            'processed_at': datetime.now().isoformat()
        })

        # Estimate cost (Gemini 2.0 Flash: ~$0.10/1M input, $0.40/1M output)
        input_tokens = len(prompt) / 4
        output_tokens = 800
        cost = (input_tokens * 0.10 / 1_000_000) + (output_tokens * 0.40 / 1_000_000)
        total_cost += cost

        # Rate limit
        time.sleep(1.5)

    if not results:
        print("\n❌ No results generated")
        return

    # Save to parquet
    print(f"\n💾 Saving {len(results)} months...")

    table = pa.table({
        'month_start': [r['month_start'] for r in results],
        'title': [r['title'] for r in results],
        'theme': [r['theme'] for r in results],
        'emotional_arc': [r['emotional_arc'] for r in results],
        'breakthroughs': [r['breakthroughs'] for r in results],
        'struggles': [r['struggles'] for r in results],
        'projects': [r['projects'] for r in results],
        'narrative': [r['narrative'] for r in results],
        'message_count': [r['message_count'] for r in results],
        'days_active': [r['days_active'] for r in results],
        'processed_at': [r['processed_at'] for r in results],
    })

    pq.write_table(table, OUTPUT_DIR / "monthly.parquet")

    # Save config
    config = {
        'version': 'v2',
        'model': MODEL,
        'created_at': datetime.now().isoformat(),
        'total_months': len(results),
        'total_cost': round(total_cost, 4),
    }
    with open(OUTPUT_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Months processed: {len(results)}")
    print(f"Estimated cost: ${total_cost:.4f}")
    print(f"Output: {OUTPUT_DIR / 'monthly.parquet'}")

    print(f"\n📅 MONTHLY TITLES:")
    for r in results:
        print(f"  {r['month_start'].strftime('%Y-%m')}: \"{r['title']}\"")


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    build_monthly_themes_v2(limit)
