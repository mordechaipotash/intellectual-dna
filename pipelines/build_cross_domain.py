#!/usr/bin/env python3
"""
Build cross_domain interpretation: Connect YouTube watching, conversations, and GitHub.

Uses Gemini 3 Flash Preview for high-quality synthesis - finding connections
between what was consumed (YouTube), discussed (conversations), and built (GitHub).
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
INTERP_DIR = DATA_DIR / "interpretations" / "cross_domain" / "v1"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash-preview-05-20"  # Flash 3 for synthesis

REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE


def call_gemini(prompt: str, max_tokens: int = 700) -> str:
    """Call Gemini Flash 3 via OpenRouter API."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Intellectual DNA Cross Domain"
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are analyzing connections across three domains:
1. YOUTUBE: What was being watched/consumed
2. CONVERSATIONS: What was being discussed with AI
3. GITHUB: What was being built (if available)

Find meaningful connections:
- Did watching something inspire a conversation or project?
- Did a conversation problem lead to seeking video tutorials?
- Did building something require learning from videos?
- What themes appear across all three domains?

Write in FIRST PERSON as if reflecting on your own intellectual journey.

Output JSON: {
    "connections": [{"type": "youtube_to_conversation|conversation_to_github|youtube_to_github|theme_across_all", "description": "what connected", "significance": "why this matters"}],
    "dominant_theme": "what theme spans all domains",
    "consumption_to_creation_pattern": "how watching relates to building",
    "learning_pipeline": "how information flows from consumption to creation",
    "synthesis": "2-3 sentence synthesis of cross-domain patterns"
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


def parse_cross_domain(llm_output: str) -> dict:
    """Parse LLM output for cross-domain data."""
    import re
    try:
        match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}
    except:
        return {}


def build_cross_domain(limit: int = None):
    """Build cross-domain connection analysis."""
    print("Building cross_domain/v1 interpretation...")
    print(f"  Model: {MODEL} (Flash 3 for synthesis)")

    if not OPENROUTER_API_KEY:
        print("  ERROR: OPENROUTER_API_KEY not found")
        return

    con = duckdb.connect()
    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERP_DIR / "weekly.parquet"

    # Check if YouTube data exists
    youtube_path = DATA_DIR / "youtube_rows.parquet"
    has_youtube = youtube_path.exists()

    # Check if GitHub data exists
    github_path = DATA_DIR / "github_commits.parquet"
    has_github = github_path.exists()

    print(f"  YouTube data: {'✓' if has_youtube else '✗'}")
    print(f"  GitHub data: {'✓' if has_github else '✗'}")

    # Get weekly conversation data
    weekly_convos = con.execute(f"""
        SELECT
            DATE_TRUNC('week', idx.timestamp::TIMESTAMP) as week_start,
            LIST(c.full_content ORDER BY idx.timestamp::TIMESTAMP) as messages,
            COUNT(*) as message_count
        FROM '{FACTS_DIR}/brain/content.parquet' c
        JOIN '{FACTS_DIR}/brain/index.parquet' idx ON c.event_id = idx.event_id
        WHERE c.event_type = 'message'
          AND idx.subtype = 'user'
          AND c.full_content IS NOT NULL
          AND LENGTH(c.full_content) BETWEEN 30 AND 600
        GROUP BY DATE_TRUNC('week', idx.timestamp::TIMESTAMP)
        HAVING COUNT(*) >= 10
        ORDER BY week_start
    """).fetchall()

    # Get YouTube data if available
    youtube_weekly = {}
    if has_youtube:
        try:
            yt_data = con.execute(f"""
                SELECT
                    DATE_TRUNC('week', watched_date) as week_start,
                    LIST(title) as titles,
                    LIST(channel_name) as channels,
                    COUNT(*) as video_count
                FROM '{youtube_path}'
                WHERE watched_date IS NOT NULL
                GROUP BY DATE_TRUNC('week', watched_date)
            """).fetchall()
            youtube_weekly = {str(w): (t, c, v) for w, t, c, v in yt_data}
        except Exception as e:
            print(f"  Warning: Could not load YouTube data: {e}")

    # Get GitHub data if available
    github_weekly = {}
    if has_github:
        try:
            gh_data = con.execute(f"""
                SELECT
                    DATE_TRUNC('week', committed_date) as week_start,
                    LIST(message) as messages,
                    LIST(repo_name) as repos,
                    COUNT(*) as commit_count
                FROM '{github_path}'
                WHERE committed_date IS NOT NULL
                GROUP BY DATE_TRUNC('week', committed_date)
            """).fetchall()
            github_weekly = {str(w): (m, r, c) for w, m, r, c in gh_data}
        except Exception as e:
            print(f"  Warning: Could not load GitHub data: {e}")

    to_process = list(weekly_convos)
    if limit:
        to_process = to_process[:limit]

    print(f"  Weeks to analyze: {len(to_process)}")

    if not to_process:
        print("  Nothing to process!")
        return

    results = []
    total_cost = 0.0
    start_time = time.time()

    for i, (week_start, messages, msg_count) in enumerate(to_process):
        week_key = str(week_start)

        # Get YouTube context for this week
        yt_context = ""
        if week_key in youtube_weekly:
            titles, channels, count = youtube_weekly[week_key]
            yt_titles = titles[:10] if len(titles) > 10 else titles
            yt_context = f"""
YOUTUBE ({count} videos watched):
{chr(10).join([f'- {t}' for t in yt_titles])}
"""

        # Get GitHub context for this week
        gh_context = ""
        if week_key in github_weekly:
            commits, repos, count = github_weekly[week_key]
            gh_commits = commits[:10] if len(commits) > 10 else commits
            gh_context = f"""
GITHUB ({count} commits):
Repos: {', '.join(set(repos[:5]))}
Commits: {chr(10).join([f'- {c[:80]}' for c in gh_commits])}
"""

        # Get conversation context
        conv_sample = messages[:20] if len(messages) > 20 else messages
        conv_context = f"""
CONVERSATIONS ({msg_count} messages):
{chr(10).join(conv_sample[:15])}
"""

        prompt = f"""Week of {week_start}

{yt_context if yt_context else "YOUTUBE: No data for this week"}

{gh_context if gh_context else "GITHUB: No data for this week"}

{conv_context}

Find connections between what I was watching, discussing, and building this week."""

        response = call_gemini(prompt)
        parsed = parse_cross_domain(response)

        connections = parsed.get("connections", [])

        results.append({
            "week_start": str(week_start),
            "has_youtube": week_key in youtube_weekly,
            "has_github": week_key in github_weekly,
            "connections_json": json.dumps(connections),
            "connection_count": len(connections),
            "dominant_theme": parsed.get("dominant_theme", ""),
            "consumption_to_creation": parsed.get("consumption_to_creation_pattern", ""),
            "learning_pipeline": parsed.get("learning_pipeline", ""),
            "synthesis": parsed.get("synthesis", ""),
            "message_count": msg_count,
            "processed_at": datetime.now().isoformat()
        })

        # Flash 3 pricing
        input_tokens = len(prompt) / 4
        output_tokens = len(response) / 4
        cost = (input_tokens * 0.50 / 1_000_000) + (output_tokens * 3.00 / 1_000_000)
        total_cost += cost

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            theme = parsed.get("dominant_theme", "?")[:30]
            print(f"  [{i+1}/{len(to_process)}] {str(week_start)[:10]} - {len(connections)} connections - \"{theme}\" - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if results:
        con.execute("""
            CREATE TABLE cross_domain (
                week_start DATE,
                has_youtube BOOLEAN,
                has_github BOOLEAN,
                connections_json VARCHAR,
                connection_count INTEGER,
                dominant_theme VARCHAR,
                consumption_to_creation VARCHAR,
                learning_pipeline VARCHAR,
                synthesis VARCHAR,
                message_count INTEGER,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO cross_domain VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       [r["week_start"], r["has_youtube"], r["has_github"],
                        r["connections_json"], r["connection_count"], r["dominant_theme"],
                        r["consumption_to_creation"], r["learning_pipeline"],
                        r["synthesis"], r["message_count"], r["processed_at"]])

        con.execute(f"COPY cross_domain TO '{output_path}' (FORMAT PARQUET)")

        total_connections = sum(r["connection_count"] for r in results)
        print(f"\n  Processed: {len(results)} weeks")
        print(f"  Total connections found: {total_connections}")
        print(f"  Total cost: ${total_cost:.4f}")

        # Show sample syntheses
        print("\n  Sample cross-domain syntheses:")
        for r in results[-5:]:
            if r["synthesis"]:
                print(f"    {r['week_start'][:10]}: {r['synthesis'][:80]}...")

    config = {
        "name": "cross_domain/v1",
        "description": "YouTube + Conversations + GitHub connections",
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
    parser.add_argument('--limit', type=int, help='Max weeks')
    parser.add_argument('--test', action='store_true', help='Test with 5 weeks')
    args = parser.parse_args()

    if args.test:
        build_cross_domain(limit=5)
    else:
        build_cross_domain(limit=args.limit)
