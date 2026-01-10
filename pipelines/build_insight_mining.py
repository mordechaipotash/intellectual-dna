#!/usr/bin/env python3
"""
Build insight_mining interpretation: Extract notable insights and realizations.

Uses Gemini 2.5 Flash Lite to identify moments of insight, breakthrough thinking,
and notable realizations across conversations.
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
INTERP_DIR = DATA_DIR / "interpretations" / "insights" / "v1"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash-lite"

REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE

# Keywords that often signal insights
INSIGHT_SIGNALS = [
    "i realized", "i just realized", "aha", "eureka", "wait",
    "actually", "interesting", "key insight", "the thing is",
    "what if", "i wonder", "oh!", "makes sense", "got it",
    "so basically", "the pattern", "i think the", "core issue",
    "fundamental", "breakthrough", "clicked", "suddenly"
]


def call_gemini(prompt: str, max_tokens: int = 600) -> str:
    """Call Gemini via OpenRouter API."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Intellectual DNA Insight Mining"
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """Identify insights and realizations in these messages. Look for:
- Moments of clarity or understanding
- Novel connections between ideas
- Breakthrough thinking or problem-solving
- Self-discoveries or pattern recognition

For each insight:
1. Extract the core insight (paraphrase concisely)
2. Categorize: technical, conceptual, personal, strategic, creative
3. Rate significance: minor, moderate, significant, breakthrough

Output JSON: [{"insight": "...", "category": "...", "significance": "...", "context": "brief context"}]
If no clear insights, output: []"""
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


def parse_insights(llm_output: str) -> list:
    """Parse LLM output to extract insights."""
    try:
        import re
        match = re.search(r'\[.*\]', llm_output, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []
    except:
        return []


def build_insight_mining(limit: int = None):
    """Extract insights from user messages."""
    print("Building insights/v1 interpretation...")
    print(f"  Model: {MODEL}")

    if not OPENROUTER_API_KEY:
        print("  ERROR: OPENROUTER_API_KEY not found")
        return

    con = duckdb.connect()
    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERP_DIR / "insights.parquet"

    # Build pattern for insight signals
    like_patterns = " OR ".join([f"LOWER(c.full_content) LIKE '%{sig}%'" for sig in INSIGHT_SIGNALS[:10]])

    # Check existing
    processed_dates = set()
    if output_path.exists():
        existing = con.execute(f"SELECT DISTINCT date FROM '{output_path}'").fetchall()
        processed_dates = {str(row[0]) for row in existing}
        print(f"  Already processed: {len(processed_dates)} days")

    # Get messages with insight signals
    daily_insights = con.execute(f"""
        SELECT
            CAST(idx.timestamp AS DATE) as date,
            LIST(c.full_content) as messages
        FROM '{FACTS_DIR}/brain/content.parquet' c
        JOIN '{FACTS_DIR}/brain/index.parquet' idx ON c.event_id = idx.event_id
        WHERE c.event_type = 'message'
          AND c.full_content IS NOT NULL
          AND LENGTH(c.full_content) BETWEEN 50 AND 1500
          AND ({like_patterns})
        GROUP BY CAST(idx.timestamp AS DATE)
        HAVING COUNT(*) >= 1
        ORDER BY date
    """).fetchall()

    to_process = [(date, msgs) for date, msgs in daily_insights
                  if str(date) not in processed_dates]

    if limit:
        to_process = to_process[:limit]

    print(f"  Days to process: {len(to_process)}")

    if not to_process:
        print("  Nothing to process!")
        return

    all_insights = []
    total_cost = 0.0
    start_time = time.time()

    for i, (date, messages) in enumerate(to_process):
        sample = "\n---\n".join(messages[:8])
        prompt = f"""Messages from {date}:
{sample}

Identify any insights, realizations, or breakthrough moments."""

        response = call_gemini(prompt)
        insights = parse_insights(response)

        for ins in insights:
            all_insights.append({
                "date": str(date),
                "insight": ins.get("insight", ""),
                "category": ins.get("category", "unknown"),
                "significance": ins.get("significance", "unknown"),
                "context": ins.get("context", ""),
                "processed_at": datetime.now().isoformat()
            })

        input_tokens = len(prompt) / 4
        output_tokens = len(response) / 4
        cost = (input_tokens * 0.075 / 1_000_000) + (output_tokens * 0.30 / 1_000_000)
        total_cost += cost

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{i+1}/{len(to_process)}] {date} - {len(insights)} insights - {rate:.1f} req/min - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if all_insights:
        con.execute("""
            CREATE TABLE insights (
                date DATE,
                insight VARCHAR,
                category VARCHAR,
                significance VARCHAR,
                context VARCHAR,
                processed_at VARCHAR
            )
        """)

        for ins in all_insights:
            con.execute("INSERT INTO insights VALUES (?, ?, ?, ?, ?, ?)",
                       [ins["date"], ins["insight"], ins["category"],
                        ins["significance"], ins["context"], ins["processed_at"]])

        if output_path.exists():
            con.execute(f"INSERT INTO insights SELECT * FROM '{output_path}'")

        con.execute(f"COPY insights TO '{output_path}' (FORMAT PARQUET)")

        print(f"\n  Extracted: {len(all_insights)} insights from {len(to_process)} days")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Wrote to: {output_path}")

        # Show significance distribution
        sig_counts = {}
        for ins in all_insights:
            s = ins["significance"]
            sig_counts[s] = sig_counts.get(s, 0) + 1
        print("\n  Significance levels:")
        for sig, count in sorted(sig_counts.items(), key=lambda x: -x[1]):
            print(f"    {sig}: {count}")

    config = {
        "name": "insights/v1",
        "description": "LLM-mined insights and realizations",
        "model": MODEL,
        "created": datetime.now().strftime("%Y-%m-%d")
    }
    with open(INTERP_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return len(all_insights)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Max days')
    parser.add_argument('--test', action='store_true', help='Test with 10 days')
    args = parser.parse_args()

    if args.test:
        build_insight_mining(limit=10)
    else:
        build_insight_mining(limit=args.limit)
