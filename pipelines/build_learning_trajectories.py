#!/usr/bin/env python3
"""
Build learning_trajectories interpretation: Track how topics evolve from question to mastery.

Uses Gemini 2.5 Flash Lite to identify learning patterns - how topics evolve
from initial questions through understanding to application.
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
INTERP_DIR = DATA_DIR / "interpretations" / "learning_trajectories" / "v1"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash-lite"

REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE


def call_gemini(prompt: str, max_tokens: int = 500) -> str:
    """Call Gemini via OpenRouter API."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Intellectual DNA Learning Trajectories"
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """Analyze learning patterns in these messages.
Identify topics being learned and their progression stage:
1. Discovery: First encounter, basic questions
2. Exploration: Trying things out, making mistakes
3. Understanding: Grasping concepts, fewer questions
4. Application: Using knowledge to build
5. Mastery: Teaching others, deep insights

Write in FIRST PERSON as if reflecting on your own learning journey.

Output JSON: {"learning_topics": [{"topic": "what I'm learning", "stage": "discovery|exploration|understanding|application|mastery", "indicators": "what shows this stage", "blockers": "what's confusing me", "breakthroughs": "aha moments if any"}], "learning_style_note": "observations about how I learn"}"""
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


def parse_learning(llm_output: str) -> dict:
    """Parse LLM output for learning data."""
    import re
    try:
        match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}
    except:
        return {}


def build_learning_trajectories(limit: int = None):
    """Build learning trajectory analysis."""
    print("Building learning_trajectories/v1 interpretation...")
    print(f"  Model: {MODEL}")

    if not OPENROUTER_API_KEY:
        print("  ERROR: OPENROUTER_API_KEY not found")
        return

    con = duckdb.connect()
    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERP_DIR / "weekly.parquet"

    # Analyze weekly for learning progression
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
          AND LENGTH(c.full_content) BETWEEN 20 AND 600
        GROUP BY DATE_TRUNC('week', idx.timestamp::TIMESTAMP)
        HAVING COUNT(*) >= 10
        ORDER BY week_start
    """).fetchall()

    to_process = list(weekly_messages)
    if limit:
        to_process = to_process[:limit]

    print(f"  Weeks to analyze: {len(to_process)}")

    if not to_process:
        print("  Nothing to process!")
        return

    all_topics = {}  # topic -> stage progression
    results = []
    total_cost = 0.0
    start_time = time.time()

    for i, (week_start, messages, msg_count) in enumerate(to_process):
        sample = messages[:25] if len(messages) > 25 else messages
        sample_text = "\n---\n".join(sample)

        prompt = f"""Week of {week_start}
{msg_count} total messages

Messages:
{sample_text}

What topics was I learning about this week? What stage am I at with each?"""

        response = call_gemini(prompt)
        parsed = parse_learning(response)

        topics = parsed.get("learning_topics", [])
        style_note = parsed.get("learning_style_note", "")

        # Track topic progression
        for topic in topics:
            name = topic.get("topic", "").lower()
            if name and len(name) > 2:
                if name not in all_topics:
                    all_topics[name] = {
                        "topic": topic.get("topic", name),
                        "stages": [],
                        "first_seen": str(week_start)
                    }
                all_topics[name]["stages"].append({
                    "week": str(week_start),
                    "stage": topic.get("stage", "unknown"),
                    "indicators": topic.get("indicators", ""),
                    "breakthroughs": topic.get("breakthroughs", "")
                })

        results.append({
            "week_start": str(week_start),
            "topics_json": json.dumps(topics),
            "style_note": style_note,
            "topic_count": len(topics),
            "message_count": msg_count,
            "processed_at": datetime.now().isoformat()
        })

        input_tokens = len(prompt) / 4
        output_tokens = len(response) / 4
        cost = (input_tokens * 0.075 / 1_000_000) + (output_tokens * 0.30 / 1_000_000)
        total_cost += cost

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{i+1}/{len(to_process)}] {week_start} - {len(all_topics)} topics tracked - {rate:.1f} req/min - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if results:
        # Save weekly data
        con.execute("""
            CREATE TABLE learning_weekly (
                week_start DATE,
                topics_json VARCHAR,
                style_note VARCHAR,
                topic_count INTEGER,
                message_count INTEGER,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO learning_weekly VALUES (?, ?, ?, ?, ?, ?)",
                       [r["week_start"], r["topics_json"], r["style_note"],
                        r["topic_count"], r["message_count"], r["processed_at"]])

        con.execute(f"COPY learning_weekly TO '{output_path}' (FORMAT PARQUET)")

        # Save topic trajectories
        topics_path = INTERP_DIR / "topics.parquet"
        con.execute("""
            CREATE TABLE learning_topics (
                topic VARCHAR,
                first_seen DATE,
                weeks_active INTEGER,
                stage_progression VARCHAR,
                current_stage VARCHAR,
                processed_at VARCHAR
            )
        """)

        for name, data in all_topics.items():
            stages = data["stages"]
            current = stages[-1]["stage"] if stages else "unknown"
            con.execute("INSERT INTO learning_topics VALUES (?, ?, ?, ?, ?, ?)",
                       [data["topic"], data["first_seen"], len(stages),
                        json.dumps([s["stage"] for s in stages]), current,
                        datetime.now().isoformat()])

        con.execute(f"COPY learning_topics TO '{topics_path}' (FORMAT PARQUET)")

        print(f"\n  Processed: {len(results)} weeks")
        print(f"  Unique learning topics: {len(all_topics)}")
        print(f"  Total cost: ${total_cost:.4f}")

        # Show topics with longest trajectories
        print("\n  Topics with longest learning trajectories:")
        sorted_topics = sorted(all_topics.items(), key=lambda x: -len(x[1]["stages"]))
        for name, data in sorted_topics[:10]:
            stages = [s["stage"] for s in data["stages"]]
            progression = " → ".join(stages[-3:]) if len(stages) > 3 else " → ".join(stages)
            print(f"    [{len(stages)}w] {data['topic']}: {progression}")

    config = {
        "name": "learning_trajectories/v1",
        "description": "Learning progression patterns by topic",
        "model": MODEL,
        "voice": "first-person",
        "created": datetime.now().strftime("%Y-%m-%d")
    }
    with open(INTERP_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return len(all_topics)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Max weeks')
    parser.add_argument('--test', action='store_true', help='Test with 5 weeks')
    args = parser.parse_args()

    if args.test:
        build_learning_trajectories(limit=5)
    else:
        build_learning_trajectories(limit=args.limit)
