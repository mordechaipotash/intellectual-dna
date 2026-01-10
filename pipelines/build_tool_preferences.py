#!/usr/bin/env python3
"""
Build tool_preferences interpretation: Extract tool and technology preferences.

Uses Gemini 2.5 Flash Lite to identify which tools, frameworks, and technologies
are preferred and how they're used.
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
INTERP_DIR = DATA_DIR / "interpretations" / "tool_preferences" / "v1"

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
        "X-Title": "Intellectual DNA Tool Preferences"
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """Identify tools, technologies, and frameworks mentioned in these messages.
Look for:
1. Programming languages (Python, TypeScript, etc.)
2. Frameworks (React, Next.js, FastAPI, etc.)
3. Databases (DuckDB, Supabase, PostgreSQL, etc.)
4. AI tools (Claude, GPT, Gemini, etc.)
5. Dev tools (VSCode, Git, Docker, etc.)
6. How they're being used (learning, building, debugging, etc.)

Write in FIRST PERSON as if reflecting on your own tool usage.

Output JSON: {"tools": [{"name": "tool name", "category": "language|framework|database|ai|devtool|service", "usage": "how I use it", "sentiment": "love|like|neutral|frustrated|abandoned", "context": "brief context"}], "stack_note": "overall technology preferences observed"}"""
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


def parse_tools(llm_output: str) -> dict:
    """Parse LLM output for tool data."""
    import re
    try:
        match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}
    except:
        return {}


def build_tool_preferences(limit: int = None):
    """Build tool preference analysis."""
    print("Building tool_preferences/v1 interpretation...")
    print(f"  Model: {MODEL}")

    if not OPENROUTER_API_KEY:
        print("  ERROR: OPENROUTER_API_KEY not found")
        return

    con = duckdb.connect()
    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERP_DIR / "monthly.parquet"

    # Analyze monthly for broader patterns
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
          AND LENGTH(c.full_content) BETWEEN 20 AND 500
        GROUP BY DATE_TRUNC('month', idx.timestamp::TIMESTAMP)
        HAVING COUNT(*) >= 20
        ORDER BY month_start
    """).fetchall()

    to_process = list(monthly_messages)
    if limit:
        to_process = to_process[:limit]

    print(f"  Months to analyze: {len(to_process)}")

    if not to_process:
        print("  Nothing to process!")
        return

    all_tools = {}  # tool_name -> aggregated data
    results = []
    total_cost = 0.0
    start_time = time.time()

    for i, (month_start, messages, msg_count) in enumerate(to_process):
        # Sample messages
        sample = messages[:40] if len(messages) > 40 else messages
        sample_text = "\n---\n".join(sample)

        prompt = f"""Month: {month_start}
{msg_count} total messages

Sample messages:
{sample_text}

What tools and technologies was I using this month? What's my relationship with them?"""

        response = call_gemini(prompt)
        parsed = parse_tools(response)

        tools = parsed.get("tools", [])
        stack_note = parsed.get("stack_note", "")

        # Aggregate tools
        for tool in tools:
            name = tool.get("name", "").lower()
            if name and len(name) > 1:
                if name in all_tools:
                    all_tools[name]["mentions"] += 1
                    all_tools[name]["months"].append(str(month_start))
                    if tool.get("sentiment"):
                        all_tools[name]["sentiments"].append(tool["sentiment"])
                else:
                    all_tools[name] = {
                        "name": tool.get("name", name),
                        "category": tool.get("category", "unknown"),
                        "mentions": 1,
                        "months": [str(month_start)],
                        "sentiments": [tool.get("sentiment", "neutral")],
                        "first_seen": str(month_start),
                        "usage_examples": [tool.get("usage", "")]
                    }

        results.append({
            "month_start": str(month_start),
            "tools_json": json.dumps(tools),
            "stack_note": stack_note,
            "message_count": msg_count,
            "processed_at": datetime.now().isoformat()
        })

        input_tokens = len(prompt) / 4
        output_tokens = len(response) / 4
        cost = (input_tokens * 0.075 / 1_000_000) + (output_tokens * 0.30 / 1_000_000)
        total_cost += cost

        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{i+1}/{len(to_process)}] {month_start} - {len(all_tools)} unique tools - {rate:.1f} req/min - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if results:
        # Save monthly analysis
        con.execute("""
            CREATE TABLE tool_monthly (
                month_start DATE,
                tools_json VARCHAR,
                stack_note VARCHAR,
                message_count INTEGER,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO tool_monthly VALUES (?, ?, ?, ?, ?)",
                       [r["month_start"], r["tools_json"], r["stack_note"],
                        r["message_count"], r["processed_at"]])

        con.execute(f"COPY tool_monthly TO '{output_path}' (FORMAT PARQUET)")

        # Save aggregated tools
        tools_path = INTERP_DIR / "tools.parquet"
        con.execute("""
            CREATE TABLE tools_agg (
                name VARCHAR,
                category VARCHAR,
                mentions INTEGER,
                months VARCHAR,
                primary_sentiment VARCHAR,
                first_seen DATE,
                processed_at VARCHAR
            )
        """)

        sorted_tools = sorted(all_tools.values(), key=lambda x: -x["mentions"])
        for t in sorted_tools:
            # Get most common sentiment
            sentiments = t["sentiments"]
            primary = max(set(sentiments), key=sentiments.count) if sentiments else "neutral"
            con.execute("INSERT INTO tools_agg VALUES (?, ?, ?, ?, ?, ?, ?)",
                       [t["name"], t["category"], t["mentions"],
                        json.dumps(t["months"]), primary, t["first_seen"],
                        datetime.now().isoformat()])

        con.execute(f"COPY tools_agg TO '{tools_path}' (FORMAT PARQUET)")

        print(f"\n  Processed: {len(results)} months")
        print(f"  Unique tools found: {len(all_tools)}")
        print(f"  Total cost: ${total_cost:.4f}")

        # Show top tools
        print("\n  Top 15 tools by mentions:")
        for t in sorted_tools[:15]:
            sent = max(set(t["sentiments"]), key=t["sentiments"].count) if t["sentiments"] else "?"
            print(f"    [{t['mentions']:2}x] {t['name']} ({t['category']}) - {sent}")

    config = {
        "name": "tool_preferences/v1",
        "description": "Tool and technology usage patterns",
        "model": MODEL,
        "voice": "first-person",
        "created": datetime.now().strftime("%Y-%m-%d")
    }
    with open(INTERP_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return len(all_tools)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Max months')
    parser.add_argument('--test', action='store_true', help='Test with 3 months')
    args = parser.parse_args()

    if args.test:
        build_tool_preferences(limit=3)
    else:
        build_tool_preferences(limit=args.limit)
