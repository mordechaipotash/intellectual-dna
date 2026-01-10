#!/usr/bin/env python3
"""
Build monthly_themes interpretation: Synthesize the gestalt of each month.

Uses Gemini 3 Flash Preview for high-quality synthesis - what was the
overarching theme, energy, and direction of each month?
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
INTERP_DIR = DATA_DIR / "interpretations" / "monthly_themes" / "v1"

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
        "X-Title": "Intellectual DNA Monthly Themes"
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are synthesizing the gestalt of a month from conversation samples.
Create a rich, narrative summary that captures:
1. The dominant theme or preoccupation
2. The emotional arc (how energy/mood shifted)
3. Key breakthroughs or realizations
4. Recurring struggles or blockers
5. What was being built/created
6. How this month connects to larger life patterns

Write in FIRST PERSON as a reflective journal entry. Be specific and insightful,
not generic. Capture the texture of this particular month.

Output JSON: {
    "title": "A poetic 3-5 word title for this month",
    "theme": "The dominant theme in 1-2 sentences",
    "emotional_arc": "How my energy/mood evolved this month",
    "breakthroughs": ["key realization 1", "key realization 2"],
    "struggles": ["recurring blocker 1", "recurring blocker 2"],
    "projects": ["what I was building"],
    "narrative": "A 3-4 paragraph reflective synthesis of the month"
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


def parse_theme(llm_output: str) -> dict:
    """Parse LLM output for theme data."""
    import re
    try:
        match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}
    except:
        return {}


def build_monthly_themes(limit: int = None):
    """Build monthly theme synthesis."""
    print("Building monthly_themes/v1 interpretation...")
    print(f"  Model: {MODEL} (Flash 3 for synthesis)")

    if not OPENROUTER_API_KEY:
        print("  ERROR: OPENROUTER_API_KEY not found")
        return

    con = duckdb.connect()
    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERP_DIR / "monthly.parquet"

    # Check existing
    processed_months = set()
    if output_path.exists():
        existing = con.execute(f"SELECT month_start FROM '{output_path}'").fetchall()
        processed_months = {str(row[0]) for row in existing}
        print(f"  Already processed: {len(processed_months)} months")

    # Get monthly data with samples from different parts of the month
    monthly_data = con.execute(f"""
        SELECT
            DATE_TRUNC('month', idx.timestamp::TIMESTAMP) as month_start,
            LIST(c.full_content ORDER BY idx.timestamp::TIMESTAMP) as messages,
            COUNT(*) as message_count,
            COUNT(DISTINCT CAST(idx.timestamp::TIMESTAMP AS DATE)) as days_active
        FROM '{FACTS_DIR}/brain/content.parquet' c
        JOIN '{FACTS_DIR}/brain/index.parquet' idx ON c.event_id = idx.event_id
        WHERE c.event_type = 'message'
          AND idx.subtype = 'user'
          AND c.full_content IS NOT NULL
          AND LENGTH(c.full_content) BETWEEN 30 AND 800
        GROUP BY DATE_TRUNC('month', idx.timestamp::TIMESTAMP)
        HAVING COUNT(*) >= 50
        ORDER BY month_start
    """).fetchall()

    to_process = [(m, msgs, cnt, days) for m, msgs, cnt, days in monthly_data
                  if str(m) not in processed_months]

    if limit:
        to_process = to_process[:limit]

    print(f"  Months to synthesize: {len(to_process)}")

    if not to_process:
        print("  Nothing to process!")
        return

    results = []
    total_cost = 0.0
    start_time = time.time()

    for i, (month_start, messages, msg_count, days_active) in enumerate(to_process):
        # Sample from beginning, middle, and end of month
        n = len(messages)
        samples = []
        if n > 60:
            samples = messages[:20] + messages[n//2-10:n//2+10] + messages[-20:]
        else:
            samples = messages[:min(60, n)]

        sample_text = "\n---\n".join(samples)

        prompt = f"""Month: {month_start.strftime('%B %Y')}
Total messages: {msg_count}
Days active: {days_active}

Sample messages from throughout the month:
{sample_text}

Synthesize the gestalt of this month. What was I really focused on? What was the emotional texture?"""

        response = call_gemini(prompt)
        parsed = parse_theme(response)

        results.append({
            "month_start": str(month_start),
            "title": parsed.get("title", ""),
            "theme": parsed.get("theme", ""),
            "emotional_arc": parsed.get("emotional_arc", ""),
            "breakthroughs": json.dumps(parsed.get("breakthroughs", [])),
            "struggles": json.dumps(parsed.get("struggles", [])),
            "projects": json.dumps(parsed.get("projects", [])),
            "narrative": parsed.get("narrative", ""),
            "message_count": msg_count,
            "days_active": days_active,
            "processed_at": datetime.now().isoformat()
        })

        # Flash 3 pricing: $0.50/1M input, $3.00/1M output
        input_tokens = len(prompt) / 4
        output_tokens = len(response) / 4
        cost = (input_tokens * 0.50 / 1_000_000) + (output_tokens * 3.00 / 1_000_000)
        total_cost += cost

        title = parsed.get("title", "?")
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
        print(f"  [{i+1}/{len(to_process)}] {month_start.strftime('%Y-%m')} - \"{title}\" - {rate:.1f} req/min - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if results:
        con.execute("""
            CREATE TABLE monthly_themes (
                month_start DATE,
                title VARCHAR,
                theme VARCHAR,
                emotional_arc VARCHAR,
                breakthroughs VARCHAR,
                struggles VARCHAR,
                projects VARCHAR,
                narrative VARCHAR,
                message_count INTEGER,
                days_active INTEGER,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO monthly_themes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       [r["month_start"], r["title"], r["theme"], r["emotional_arc"],
                        r["breakthroughs"], r["struggles"], r["projects"], r["narrative"],
                        r["message_count"], r["days_active"], r["processed_at"]])

        if output_path.exists():
            con.execute(f"INSERT INTO monthly_themes SELECT * FROM '{output_path}'")

        con.execute(f"COPY monthly_themes TO '{output_path}' (FORMAT PARQUET)")

        print(f"\n  Processed: {len(results)} months")
        print(f"  Total cost: ${total_cost:.4f}")

        # Show month titles
        print("\n  Monthly themes:")
        for r in results[-12:]:  # Last 12 months
            print(f"    {r['month_start'][:7]}: \"{r['title']}\"")
            if r['theme']:
                print(f"              {r['theme'][:70]}...")

    config = {
        "name": "monthly_themes/v1",
        "description": "Monthly gestalt synthesis",
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
    parser.add_argument('--limit', type=int, help='Max months')
    parser.add_argument('--test', action='store_true', help='Test with 2 months')
    args = parser.parse_args()

    if args.test:
        build_monthly_themes(limit=2)
    else:
        build_monthly_themes(limit=args.limit)
