#!/usr/bin/env python3
"""
Build mood_patterns interpretation: Detect emotional/energy patterns.

Uses Gemini 2.5 Flash Lite to analyze mood, energy, and emotional
patterns across daily conversations.
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
INTERP_DIR = DATA_DIR / "interpretations" / "mood" / "v1"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash-lite"

REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE


def call_gemini(prompt: str, max_tokens: int = 350) -> str:
    """Call Gemini via OpenRouter API."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Intellectual DNA Mood Patterns"
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """Analyze the emotional tone and energy level in these conversation excerpts.

Assess:
1. Primary mood: focused, frustrated, curious, excited, tired, anxious, calm, playful, determined
2. Energy level: 1-5 (1=exhausted, 2=low, 3=neutral, 4=energized, 5=highly activated)
3. Cognitive state: exploring, executing, stuck, learning, creating, debugging, planning
4. Stress indicators: none, mild, moderate, high
5. Brief explanation of your assessment

Output JSON: {"mood": "...", "energy": 3, "cognitive_state": "...", "stress": "...", "explanation": "..."}"""
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


def parse_mood(llm_output: str) -> dict:
    """Parse LLM output for mood assessment."""
    try:
        import re
        match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}
    except:
        return {}


def build_mood_patterns(limit: int = None):
    """Build mood pattern analysis."""
    print("Building mood/v1 interpretation...")
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
            CAST(idx.timestamp AS DATE) as date,
            LIST(LEFT(c.full_content, 400) ORDER BY idx.timestamp) as messages,
            COUNT(*) as message_count
        FROM '{FACTS_DIR}/brain/content.parquet' c
        JOIN '{FACTS_DIR}/brain/index.parquet' idx ON c.event_id = idx.event_id
        WHERE c.event_type = 'message'
          AND c.full_content IS NOT NULL
          AND LENGTH(c.full_content) BETWEEN 20 AND 800
        GROUP BY CAST(idx.timestamp AS DATE)
        HAVING COUNT(*) >= 5
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
        # Sample messages spread across the day
        sample_indices = [0, len(messages)//4, len(messages)//2, 3*len(messages)//4, -1]
        samples = [messages[min(idx, len(messages)-1)] for idx in sample_indices if idx < len(messages)]
        sample_text = "\n---\n".join(samples[:5])

        prompt = f"""Date: {date}
Total messages: {msg_count}

Sample messages from throughout the day:
{sample_text}

What was the emotional tone and energy level on this day?"""

        response = call_gemini(prompt)
        parsed = parse_mood(response)

        results.append({
            "date": str(date),
            "mood": parsed.get("mood", "unknown"),
            "energy": parsed.get("energy", 3),
            "cognitive_state": parsed.get("cognitive_state", "unknown"),
            "stress": parsed.get("stress", "unknown"),
            "explanation": parsed.get("explanation", ""),
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
            mood = parsed.get("mood", "?")
            energy = parsed.get("energy", "?")
            print(f"  [{i+1}/{len(to_process)}] {date} - {mood}/E{energy} - {rate:.1f} req/min - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if results:
        con.execute("""
            CREATE TABLE mood_daily (
                date DATE,
                mood VARCHAR,
                energy INTEGER,
                cognitive_state VARCHAR,
                stress VARCHAR,
                explanation VARCHAR,
                message_count INTEGER,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO mood_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       [r["date"], r["mood"], r["energy"], r["cognitive_state"],
                        r["stress"], r["explanation"], r["message_count"], r["processed_at"]])

        if output_path.exists():
            con.execute(f"INSERT INTO mood_daily SELECT * FROM '{output_path}'")

        con.execute(f"COPY mood_daily TO '{output_path}' (FORMAT PARQUET)")

        print(f"\n  Processed: {len(results)} days")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Wrote to: {output_path}")

        # Show mood distribution
        mood_counts = {}
        energy_sum = 0
        for r in results:
            m = r["mood"]
            mood_counts[m] = mood_counts.get(m, 0) + 1
            energy_sum += r["energy"] or 3

        print("\n  Mood distribution:")
        for mood, count in sorted(mood_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"    {mood}: {count}")
        print(f"\n  Average energy: {energy_sum / len(results):.1f}/5")

    config = {
        "name": "mood/v1",
        "description": "LLM-detected mood and energy patterns",
        "model": MODEL,
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
        build_mood_patterns(limit=10)
    else:
        build_mood_patterns(limit=args.limit)
