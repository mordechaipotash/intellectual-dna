#!/usr/bin/env python3
"""
Build youtube_link interpretation: Connect YouTube videos to conversation topics.

Uses Gemini 2.5 Flash Lite to find semantic connections between
YouTube videos watched and conversation topics on the same/nearby days.
"""

import os
import json
import time
import duckdb
import requests
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv(Path("/Users/mordechai/intellectual_dna/.env"))

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
DATA_DIR = BASE_DIR / "data"
FACTS_DIR = DATA_DIR / "facts"
INTERP_DIR = DATA_DIR / "interpretations" / "youtube_link" / "v1"
YOUTUBE_PATH = DATA_DIR / "youtube_rows.parquet"
FOCUS_V2_PATH = DATA_DIR / "interpretations" / "focus" / "v2" / "daily.parquet"

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
        "X-Title": "Intellectual DNA YouTube Link"
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """Analyze connections between YouTube videos watched and conversation topics.

For each potential connection:
1. Describe the thematic link
2. Rate connection strength: strong (direct topic match), moderate (related themes), weak (tangential)
3. Classify: learning (skill acquisition), research (investigating topic), inspiration (creative influence), entertainment (relaxation), unknown

Output JSON: {"connections": [{"video_title": "...", "conversation_topic": "...", "link_description": "...", "strength": "...", "type": "..."}], "summary": "brief overall pattern"}
If no meaningful connections, output: {"connections": [], "summary": "No clear connections found"}"""
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


def parse_links(llm_output: str) -> dict:
    """Parse LLM output for connections."""
    try:
        import re
        match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"connections": [], "summary": "Parse error"}
    except:
        return {"connections": [], "summary": "Parse error"}


def build_youtube_link(limit: int = None):
    """Build YouTube-conversation link analysis."""
    print("Building youtube_link/v1 interpretation...")
    print(f"  Model: {MODEL}")

    if not YOUTUBE_PATH.exists():
        print(f"  ERROR: {YOUTUBE_PATH} not found")
        return

    if not FOCUS_V2_PATH.exists():
        print(f"  ERROR: {FOCUS_V2_PATH} not found - run focus/v2 first")
        return

    if not OPENROUTER_API_KEY:
        print("  ERROR: OPENROUTER_API_KEY not found")
        return

    con = duckdb.connect()
    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERP_DIR / "links.parquet"

    # Check existing
    processed_weeks = set()
    if output_path.exists():
        existing = con.execute(f"SELECT DISTINCT week_start FROM '{output_path}'").fetchall()
        processed_weeks = {str(row[0]) for row in existing}
        print(f"  Already processed: {len(processed_weeks)} weeks")

    # Get weeks with both YouTube and conversation data
    weeks = con.execute(f"""
        WITH youtube_weeks AS (
            SELECT
                DATE_TRUNC('week', watched_date) as week_start,
                LIST(title ORDER BY watched_date)[:15] as video_titles
            FROM '{YOUTUBE_PATH}'
            WHERE watched_date IS NOT NULL
            GROUP BY DATE_TRUNC('week', watched_date)
        ),
        focus_weeks AS (
            SELECT
                DATE_TRUNC('week', date) as week_start,
                LIST(summary ORDER BY date) as focus_summaries
            FROM '{FOCUS_V2_PATH}'
            WHERE summary NOT LIKE '[%Error%'
            GROUP BY DATE_TRUNC('week', date)
        )
        SELECT
            y.week_start,
            y.video_titles,
            f.focus_summaries
        FROM youtube_weeks y
        JOIN focus_weeks f ON y.week_start = f.week_start
        ORDER BY y.week_start
    """).fetchall()

    to_process = [(ws, vt, fs) for ws, vt, fs in weeks
                  if str(ws) not in processed_weeks]

    if limit:
        to_process = to_process[:limit]

    print(f"  Weeks to process: {len(to_process)}")

    if not to_process:
        print("  Nothing to process!")
        return

    all_links = []
    total_cost = 0.0
    start_time = time.time()

    for i, (week_start, videos, summaries) in enumerate(to_process):
        video_list = "\n".join([f"- {v}" for v in videos[:10]])
        summary_list = "\n".join([f"- {s}" for s in summaries[:7]])

        prompt = f"""Week of {week_start}:

YouTube videos watched:
{video_list}

Conversation topics (focus summaries):
{summary_list}

Find connections between what was watched and what was worked on/discussed."""

        response = call_gemini(prompt)
        parsed = parse_links(response)

        all_links.append({
            "week_start": str(week_start),
            "connections_json": json.dumps(parsed.get("connections", [])),
            "connection_count": len(parsed.get("connections", [])),
            "summary": parsed.get("summary", ""),
            "video_count": len(videos),
            "processed_at": datetime.now().isoformat()
        })

        input_tokens = len(prompt) / 4
        output_tokens = len(response) / 4
        cost = (input_tokens * 0.075 / 1_000_000) + (output_tokens * 0.30 / 1_000_000)
        total_cost += cost

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            conn_count = parsed.get("connections", [])
            print(f"  [{i+1}/{len(to_process)}] {week_start} - {len(conn_count)} links - {rate:.1f} req/min - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if all_links:
        con.execute("""
            CREATE TABLE youtube_links (
                week_start DATE,
                connections_json VARCHAR,
                connection_count INTEGER,
                summary VARCHAR,
                video_count INTEGER,
                processed_at VARCHAR
            )
        """)

        for link in all_links:
            con.execute("INSERT INTO youtube_links VALUES (?, ?, ?, ?, ?, ?)",
                       [link["week_start"], link["connections_json"], link["connection_count"],
                        link["summary"], link["video_count"], link["processed_at"]])

        if output_path.exists():
            con.execute(f"INSERT INTO youtube_links SELECT * FROM '{output_path}'")

        con.execute(f"COPY youtube_links TO '{output_path}' (FORMAT PARQUET)")

        print(f"\n  Processed: {len(all_links)} weeks")
        print(f"  Total connections found: {sum(l['connection_count'] for l in all_links)}")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Wrote to: {output_path}")

    config = {
        "name": "youtube_link/v1",
        "description": "LLM-analyzed YouTube-conversation connections",
        "model": MODEL,
        "depends_on": ["focus/v2"],
        "created": datetime.now().strftime("%Y-%m-%d")
    }
    with open(INTERP_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return len(all_links)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Max weeks')
    parser.add_argument('--test', action='store_true', help='Test with 5 weeks')
    args = parser.parse_args()

    if args.test:
        build_youtube_link(limit=5)
    else:
        build_youtube_link(limit=args.limit)
