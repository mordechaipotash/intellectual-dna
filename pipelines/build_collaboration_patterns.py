#!/usr/bin/env python3
"""
Build collaboration_patterns interpretation: Analyze how AI collaboration works.

Uses Gemini 2.5 Flash Lite to identify collaboration patterns - how the user
works with AI (delegator, collaborator, rubber duck, teacher, etc.)
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
INTERP_DIR = DATA_DIR / "interpretations" / "collaboration_patterns" / "v1"

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
        "X-Title": "Intellectual DNA Collaboration Patterns"
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """Analyze how this person collaborates with AI based on their messages.
Collaboration modes:
1. Delegator: "Do this for me" - hands off tasks completely
2. Collaborator: "Let's figure this out together" - back and forth
3. Rubber Duck: Using AI to think out loud, clarify own thoughts
4. Teacher: Explaining concepts to AI, testing understanding
5. Learner: Asking questions, seeking to understand
6. Director: Precise instructions, AI as executor
7. Brainstormer: Exploring ideas, no fixed outcome
8. Debugger: Troubleshooting together

Write in FIRST PERSON as if reflecting on your own collaboration style.

Output JSON: {"primary_mode": "delegator|collaborator|rubber_duck|teacher|learner|director|brainstormer|debugger", "secondary_mode": "...", "autonomy_level": "high|medium|low", "trust_signals": "how I show trust/distrust in AI", "friction_points": "where collaboration breaks down", "effective_patterns": "what works well"}"""
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


def parse_collaboration(llm_output: str) -> dict:
    """Parse LLM output for collaboration data."""
    import re
    try:
        match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}
    except:
        return {}


def build_collaboration_patterns(limit: int = None):
    """Build collaboration pattern analysis."""
    print("Building collaboration_patterns/v1 interpretation...")
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
          AND LENGTH(c.full_content) BETWEEN 10 AND 500
        GROUP BY CAST(idx.timestamp::TIMESTAMP AS DATE)
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

    mode_counts = {}
    results = []
    total_cost = 0.0
    start_time = time.time()

    for i, (date, messages, msg_count) in enumerate(to_process):
        sample = messages[:20] if len(messages) > 20 else messages
        sample_text = "\n---\n".join(sample)

        prompt = f"""Date: {date}
{msg_count} messages with AI

Messages:
{sample_text}

How was I collaborating with AI on this day? What mode was I in?"""

        response = call_gemini(prompt)
        parsed = parse_collaboration(response)

        primary = parsed.get("primary_mode", "unknown")
        secondary = parsed.get("secondary_mode", "")
        autonomy = parsed.get("autonomy_level", "medium")

        # Track mode distribution
        mode_counts[primary] = mode_counts.get(primary, 0) + 1

        results.append({
            "date": str(date),
            "primary_mode": primary,
            "secondary_mode": secondary,
            "autonomy_level": autonomy,
            "trust_signals": parsed.get("trust_signals", ""),
            "friction_points": parsed.get("friction_points", ""),
            "effective_patterns": parsed.get("effective_patterns", ""),
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
            print(f"  [{i+1}/{len(to_process)}] {date} - {primary} - {rate:.1f} req/min - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if results:
        con.execute("""
            CREATE TABLE collaboration_daily (
                date DATE,
                primary_mode VARCHAR,
                secondary_mode VARCHAR,
                autonomy_level VARCHAR,
                trust_signals VARCHAR,
                friction_points VARCHAR,
                effective_patterns VARCHAR,
                message_count INTEGER,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO collaboration_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       [r["date"], r["primary_mode"], r["secondary_mode"],
                        r["autonomy_level"], r["trust_signals"], r["friction_points"],
                        r["effective_patterns"], r["message_count"], r["processed_at"]])

        if output_path.exists():
            con.execute(f"INSERT INTO collaboration_daily SELECT * FROM '{output_path}'")

        con.execute(f"COPY collaboration_daily TO '{output_path}' (FORMAT PARQUET)")

        print(f"\n  Processed: {len(results)} days")
        print(f"  Total cost: ${total_cost:.4f}")

        # Show mode distribution
        print("\n  Collaboration mode distribution:")
        for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(results)
            bar = "█" * int(pct / 5)
            print(f"    {mode:15}: {count:3} ({pct:5.1f}%) {bar}")

    config = {
        "name": "collaboration_patterns/v1",
        "description": "AI collaboration style patterns",
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
        build_collaboration_patterns(limit=10)
    else:
        build_collaboration_patterns(limit=args.limit)
