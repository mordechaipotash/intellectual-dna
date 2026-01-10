#!/usr/bin/env python3
"""
Build intellectual_evolution interpretation: Track how thinking has changed over time.

Uses Gemini 3 Flash Preview for high-quality synthesis - identifying how
beliefs, frameworks, and understanding have evolved.
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
INTERP_DIR = DATA_DIR / "interpretations" / "intellectual_evolution" / "v1"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash-preview-05-20"  # Flash 3 for synthesis

REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE


def call_gemini(prompt: str, max_tokens: int = 800) -> str:
    """Call Gemini Flash 3 via OpenRouter API."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Intellectual DNA Evolution"
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are analyzing intellectual evolution across two time periods.
Compare the thinking patterns, beliefs, and frameworks between EARLIER and LATER periods.

Identify:
1. Beliefs that changed or evolved
2. New frameworks or mental models adopted
3. Interests that emerged or faded
4. Sophistication changes in thinking
5. Recurring themes that persisted
6. Pivotal shifts in perspective

Write in FIRST PERSON as if reflecting on your own growth.

Output JSON: {
    "period": "comparison period label",
    "evolved_beliefs": [{"belief": "what changed", "from": "earlier view", "to": "later view", "significance": "why this matters"}],
    "new_frameworks": ["framework I adopted"],
    "faded_interests": ["what I stopped caring about"],
    "emerged_interests": ["what became important"],
    "persistent_themes": ["what stayed constant"],
    "sophistication_shift": "how my thinking matured",
    "pivotal_insight": "the biggest shift in perspective"
}"""
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.4
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Error: {str(e)[:50]}]"


def parse_evolution(llm_output: str) -> dict:
    """Parse LLM output for evolution data."""
    import re
    try:
        match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}
    except:
        return {}


def build_intellectual_evolution(limit: int = None):
    """Build intellectual evolution analysis."""
    print("Building intellectual_evolution/v1 interpretation...")
    print(f"  Model: {MODEL} (Flash 3 for synthesis)")

    if not OPENROUTER_API_KEY:
        print("  ERROR: OPENROUTER_API_KEY not found")
        return

    con = duckdb.connect()
    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERP_DIR / "quarterly.parquet"

    # Get quarterly data for comparison
    quarterly_data = con.execute(f"""
        SELECT
            DATE_TRUNC('quarter', idx.timestamp::TIMESTAMP) as quarter_start,
            LIST(c.full_content ORDER BY idx.timestamp::TIMESTAMP) as messages,
            COUNT(*) as message_count
        FROM '{FACTS_DIR}/brain/content.parquet' c
        JOIN '{FACTS_DIR}/brain/index.parquet' idx ON c.event_id = idx.event_id
        WHERE c.event_type = 'message'
          AND idx.subtype = 'user'
          AND c.full_content IS NOT NULL
          AND LENGTH(c.full_content) BETWEEN 50 AND 800
        GROUP BY DATE_TRUNC('quarter', idx.timestamp::TIMESTAMP)
        HAVING COUNT(*) >= 100
        ORDER BY quarter_start
    """).fetchall()

    quarters = list(quarterly_data)

    # Compare consecutive quarters
    comparisons = []
    for i in range(1, len(quarters)):
        comparisons.append((quarters[i-1], quarters[i]))

    if limit:
        comparisons = comparisons[:limit]

    print(f"  Quarter comparisons to analyze: {len(comparisons)}")

    if not comparisons:
        print("  Nothing to process!")
        return

    results = []
    total_cost = 0.0
    start_time = time.time()

    for i, ((q1_start, q1_msgs, q1_count), (q2_start, q2_msgs, q2_count)) in enumerate(comparisons):
        # Sample from each quarter
        q1_sample = q1_msgs[:30] if len(q1_msgs) > 30 else q1_msgs
        q2_sample = q2_msgs[:30] if len(q2_msgs) > 30 else q2_msgs

        prompt = f"""Compare my thinking between two periods:

EARLIER PERIOD: {q1_start.strftime('%B %Y')} ({q1_count} messages)
Sample messages:
{chr(10).join(q1_sample[:15])}

---

LATER PERIOD: {q2_start.strftime('%B %Y')} ({q2_count} messages)
Sample messages:
{chr(10).join(q2_sample[:15])}

---

How has my thinking evolved between these periods? What changed? What stayed the same?"""

        response = call_gemini(prompt)
        parsed = parse_evolution(response)

        results.append({
            "earlier_quarter": str(q1_start),
            "later_quarter": str(q2_start),
            "period_label": f"{q1_start.strftime('%Y-Q%q') if hasattr(q1_start, 'strftime') else str(q1_start)[:7]} → {q2_start.strftime('%Y-Q%q') if hasattr(q2_start, 'strftime') else str(q2_start)[:7]}",
            "evolved_beliefs": json.dumps(parsed.get("evolved_beliefs", [])),
            "new_frameworks": json.dumps(parsed.get("new_frameworks", [])),
            "faded_interests": json.dumps(parsed.get("faded_interests", [])),
            "emerged_interests": json.dumps(parsed.get("emerged_interests", [])),
            "persistent_themes": json.dumps(parsed.get("persistent_themes", [])),
            "sophistication_shift": parsed.get("sophistication_shift", ""),
            "pivotal_insight": parsed.get("pivotal_insight", ""),
            "processed_at": datetime.now().isoformat()
        })

        # Flash 3 pricing
        input_tokens = len(prompt) / 4
        output_tokens = len(response) / 4
        cost = (input_tokens * 0.50 / 1_000_000) + (output_tokens * 3.00 / 1_000_000)
        total_cost += cost

        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
        pivot = parsed.get("pivotal_insight", "?")[:50]
        print(f"  [{i+1}/{len(comparisons)}] {str(q1_start)[:7]} → {str(q2_start)[:7]} - \"{pivot}...\" - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if results:
        con.execute("""
            CREATE TABLE intellectual_evolution (
                earlier_quarter DATE,
                later_quarter DATE,
                period_label VARCHAR,
                evolved_beliefs VARCHAR,
                new_frameworks VARCHAR,
                faded_interests VARCHAR,
                emerged_interests VARCHAR,
                persistent_themes VARCHAR,
                sophistication_shift VARCHAR,
                pivotal_insight VARCHAR,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO intellectual_evolution VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       [r["earlier_quarter"], r["later_quarter"], r["period_label"],
                        r["evolved_beliefs"], r["new_frameworks"], r["faded_interests"],
                        r["emerged_interests"], r["persistent_themes"],
                        r["sophistication_shift"], r["pivotal_insight"], r["processed_at"]])

        con.execute(f"COPY intellectual_evolution TO '{output_path}' (FORMAT PARQUET)")

        print(f"\n  Processed: {len(results)} quarter comparisons")
        print(f"  Total cost: ${total_cost:.4f}")

        # Show pivotal insights
        print("\n  Pivotal insights by period:")
        for r in results[-6:]:
            if r["pivotal_insight"]:
                print(f"    {r['period_label']}: {r['pivotal_insight'][:60]}...")

    config = {
        "name": "intellectual_evolution/v1",
        "description": "Quarterly intellectual evolution analysis",
        "model": MODEL,
        "voice": "first-person",
        "tier": "synthesis",
        "created": datetime.now().strftime("%Y-%m-%d")
    }
    with open(INTERP_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return len(results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Max comparisons')
    parser.add_argument('--test', action='store_true', help='Test with 2 comparisons')
    args = parser.parse_args()

    if args.test:
        build_intellectual_evolution(limit=2)
    else:
        build_intellectual_evolution(limit=args.limit)
