#!/usr/bin/env python3
"""
intellectual_evolution/v2 - Track how thinking has changed over time
Uses Gemini 3 Flash with JSON Schema for guaranteed structured output
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
OUTPUT_DIR = BASE_DIR / "data/interpretations/intellectual_evolution/v2"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3-flash-preview"  # Always use Gemini 3 Flash

# JSON Schema for guaranteed structured output
EVOLUTION_SCHEMA = {
    "type": "object",
    "properties": {
        "evolved_beliefs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "belief": {"type": "string", "description": "What belief changed"},
                    "from_view": {"type": "string", "description": "Earlier perspective"},
                    "to_view": {"type": "string", "description": "Later perspective"},
                    "significance": {"type": "string", "description": "Why this matters"}
                },
                "required": ["belief", "from_view", "to_view", "significance"]
            },
            "description": "2-4 beliefs that evolved between periods"
        },
        "new_frameworks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "New mental models or frameworks adopted (2-4 items)"
        },
        "faded_interests": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Topics or interests that decreased (2-4 items)"
        },
        "emerged_interests": {
            "type": "array",
            "items": {"type": "string"},
            "description": "New interests or focus areas that appeared (2-4 items)"
        },
        "persistent_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Themes that remained constant across both periods (2-4 items)"
        },
        "sophistication_shift": {
            "type": "string",
            "description": "How thinking matured or deepened between periods"
        },
        "pivotal_insight": {
            "type": "string",
            "description": "The single biggest shift in perspective"
        }
    },
    "required": ["evolved_beliefs", "new_frameworks", "faded_interests",
                 "emerged_interests", "persistent_themes", "sophistication_shift", "pivotal_insight"]
}

# Filter out system/IDE noise
NOISE_PATTERNS = [
    'context window', 'tokens used', 'vscode', 'localhost',
    'node_modules', 'Tool Result', 'user uploaded', 'interrupted by',
    'may not be related', 'the current task', 'null null'
]


def is_clean_message(content: str) -> bool:
    """Check if message is genuine conversation, not system noise."""
    if not content or len(content) < 30 or len(content) > 600:
        return False
    content_lower = content.lower()
    return not any(p in content_lower for p in NOISE_PATTERNS)


def get_quarter_label(dt) -> str:
    """Get quarter label like 2024-Q1."""
    quarter = (dt.month - 1) // 3 + 1
    return f"{dt.year}-Q{quarter}"


def call_gemini(prompt: str) -> dict:
    """Call Gemini with JSON Schema for structured output."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Intellectual Evolution v2"
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You analyze intellectual evolution across two time periods.
Compare thinking patterns, beliefs, and frameworks between EARLIER and LATER periods.

Be specific and insightful. Look for:
- Beliefs that shifted or deepened
- New mental models adopted
- Interests that emerged or faded
- How sophistication of thinking changed
- The most pivotal shift in perspective

Write as a thoughtful observer noting genuine evolution, not superficial changes."""
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1200,
        "temperature": 0.4,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "intellectual_evolution",
                "strict": True,
                "schema": EVOLUTION_SCHEMA
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


def build_intellectual_evolution_v2(limit: int = None):
    """Build intellectual evolution with guaranteed structured output."""
    print("=" * 60)
    print("INTELLECTUAL EVOLUTION v2")
    print("=" * 60)

    if not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY not found")
        return

    con = duckdb.connect()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get quarterly data
    print("\n📥 Loading quarterly data...")
    quarterly_data = con.execute(f"""
        SELECT
            DATE_TRUNC('quarter', msg_timestamp::TIMESTAMP) as quarter_start,
            LIST(content) as messages,
            COUNT(*) as message_count,
            COUNT(DISTINCT CAST(msg_timestamp::TIMESTAMP AS DATE)) as days_active
        FROM '{CONVERSATIONS}'
        WHERE role = 'user'
          AND content IS NOT NULL
          AND LENGTH(content) BETWEEN 30 AND 600
          AND msg_timestamp IS NOT NULL
        GROUP BY DATE_TRUNC('quarter', msg_timestamp::TIMESTAMP)
        HAVING COUNT(*) >= 100
          AND quarter_start IS NOT NULL
        ORDER BY quarter_start
    """).fetchall()

    quarters = list(quarterly_data)
    print(f"   Found {len(quarters)} quarters with data")

    # Build comparison pairs (consecutive quarters)
    comparisons = []
    for i in range(1, len(quarters)):
        comparisons.append((quarters[i-1], quarters[i]))

    if limit:
        comparisons = comparisons[:limit]
        print(f"   Processing {limit} comparisons")

    print(f"   Total comparisons: {len(comparisons)}")

    if not comparisons:
        print("❌ No comparisons to process")
        return

    results = []
    total_cost = 0.0
    start_time = time.time()

    for i, ((q1_start, q1_msgs, q1_count, q1_days), (q2_start, q2_msgs, q2_count, q2_days)) in enumerate(comparisons):
        q1_label = get_quarter_label(q1_start)
        q2_label = get_quarter_label(q2_start)
        print(f"\n[{i+1}/{len(comparisons)}] {q1_label} → {q2_label} ({q1_count}→{q2_count} msgs)")

        # Filter to clean messages
        q1_clean = [m for m in q1_msgs if is_clean_message(m)]
        q2_clean = [m for m in q2_msgs if is_clean_message(m)]
        print(f"   Clean: {len(q1_clean)}/{len(q1_msgs)} → {len(q2_clean)}/{len(q2_msgs)}")

        if len(q1_clean) < 30 or len(q2_clean) < 30:
            print("   ⚠️ Too few clean messages, skipping")
            continue

        # Sample from beginning, middle, end of each quarter
        def sample_messages(msgs, n=45):
            if len(msgs) <= n:
                return msgs
            third = n // 3
            return msgs[:third] + msgs[len(msgs)//2-third//2:len(msgs)//2+third//2] + msgs[-third:]

        q1_sample = sample_messages(q1_clean)
        q2_sample = sample_messages(q2_clean)

        prompt = f"""Compare my intellectual evolution between two quarters:

EARLIER: {q1_label} ({q1_count} messages over {q1_days} days)
Sample messages:
{chr(10).join('- ' + m[:200] for m in q1_sample[:20])}

---

LATER: {q2_label} ({q2_count} messages over {q2_days} days)
Sample messages:
{chr(10).join('- ' + m[:200] for m in q2_sample[:20])}

---

Analyze how my thinking evolved between these periods. What beliefs changed? What new frameworks did I adopt? What interests emerged or faded?"""

        result = call_gemini(prompt)

        if "error" in result:
            print(f"   ❌ {result['error']}")
            continue

        # Show summary
        pivot = result.get('pivotal_insight', '?')[:60]
        print(f"   ✓ Pivotal: {pivot}...")

        n_beliefs = len(result.get('evolved_beliefs', []))
        n_frameworks = len(result.get('new_frameworks', []))
        print(f"     {n_beliefs} evolved beliefs, {n_frameworks} new frameworks")

        results.append({
            'earlier_quarter': q1_start,
            'later_quarter': q2_start,
            'period_label': f"{q1_label} → {q2_label}",
            'evolved_beliefs': json.dumps(result.get('evolved_beliefs', [])),
            'new_frameworks': json.dumps(result.get('new_frameworks', [])),
            'faded_interests': json.dumps(result.get('faded_interests', [])),
            'emerged_interests': json.dumps(result.get('emerged_interests', [])),
            'persistent_themes': json.dumps(result.get('persistent_themes', [])),
            'sophistication_shift': result.get('sophistication_shift', ''),
            'pivotal_insight': result.get('pivotal_insight', ''),
            'processed_at': datetime.now().isoformat()
        })

        # Estimate cost (Gemini 3 Flash: ~$0.10/1M input, $0.40/1M output)
        input_tokens = len(prompt) / 4
        output_tokens = 1000
        cost = (input_tokens * 0.10 / 1_000_000) + (output_tokens * 0.40 / 1_000_000)
        total_cost += cost

        # Rate limit
        time.sleep(2.0)

    if not results:
        print("\n❌ No results generated")
        return

    # Save to parquet
    print(f"\n💾 Saving {len(results)} comparisons...")

    table = pa.table({
        'earlier_quarter': [r['earlier_quarter'] for r in results],
        'later_quarter': [r['later_quarter'] for r in results],
        'period_label': [r['period_label'] for r in results],
        'evolved_beliefs': [r['evolved_beliefs'] for r in results],
        'new_frameworks': [r['new_frameworks'] for r in results],
        'faded_interests': [r['faded_interests'] for r in results],
        'emerged_interests': [r['emerged_interests'] for r in results],
        'persistent_themes': [r['persistent_themes'] for r in results],
        'sophistication_shift': [r['sophistication_shift'] for r in results],
        'pivotal_insight': [r['pivotal_insight'] for r in results],
        'processed_at': [r['processed_at'] for r in results],
    })

    pq.write_table(table, OUTPUT_DIR / "quarterly.parquet")

    # Save config
    config = {
        'version': 'v2',
        'model': MODEL,
        'created_at': datetime.now().isoformat(),
        'total_comparisons': len(results),
        'total_cost': round(total_cost, 4),
    }
    with open(OUTPUT_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Comparisons processed: {len(results)}")
    print(f"Estimated cost: ${total_cost:.4f}")
    print(f"Output: {OUTPUT_DIR / 'quarterly.parquet'}")

    print(f"\n📊 PIVOTAL INSIGHTS BY PERIOD:")
    for r in results:
        print(f"  {r['period_label']}")
        print(f"    → {r['pivotal_insight'][:80]}...")


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    build_intellectual_evolution_v2(limit)
