#!/usr/bin/env python3
"""
Test new pipeline approaches - Round 3
Focus on finding technical problem resolutions with older data.
"""

import os
import json
import duckdb
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path("/Users/mordechai/intellectual_dna/.env"))

FACTS_DIR = Path("/Users/mordechai/intellectual_dna/data/facts")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash-lite"


def call_llm(system: str, user: str, max_tokens: int = 600) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Error: {e}]"


def get_older_days(limit: int = 5):
    """Get days from 2-6 months ago with high message counts (more technical work)."""
    con = duckdb.connect()
    return con.execute(f"""
        SELECT
            CAST(idx.timestamp::TIMESTAMP AS DATE) as date,
            LIST(c.full_content ORDER BY idx.timestamp::TIMESTAMP) as messages,
            COUNT(*) as msg_count
        FROM '{FACTS_DIR}/brain/content.parquet' c
        JOIN '{FACTS_DIR}/brain/index.parquet' idx ON c.event_id = idx.event_id
        WHERE c.event_type = 'message'
          AND idx.subtype = 'user'
          AND c.full_content IS NOT NULL
          AND LENGTH(c.full_content) BETWEEN 30 AND 600
          AND idx.timestamp::TIMESTAMP BETWEEN '2024-06-01' AND '2024-10-01'
        GROUP BY CAST(idx.timestamp::TIMESTAMP AS DATE)
        HAVING COUNT(*) >= 30
        ORDER BY msg_count DESC
        LIMIT {limit}
    """).fetchall()


def get_older_weeks(limit: int = 3):
    """Get weeks from 2-6 months ago."""
    con = duckdb.connect()
    return con.execute(f"""
        SELECT
            DATE_TRUNC('week', idx.timestamp::TIMESTAMP) as week_start,
            LIST(c.full_content ORDER BY idx.timestamp::TIMESTAMP) as messages,
            COUNT(*) as msg_count
        FROM '{FACTS_DIR}/brain/content.parquet' c
        JOIN '{FACTS_DIR}/brain/index.parquet' idx ON c.event_id = idx.event_id
        WHERE c.event_type = 'message'
          AND idx.subtype = 'user'
          AND c.full_content IS NOT NULL
          AND LENGTH(c.full_content) BETWEEN 30 AND 600
          AND idx.timestamp::TIMESTAMP BETWEEN '2024-06-01' AND '2024-10-01'
        GROUP BY DATE_TRUNC('week', idx.timestamp::TIMESTAMP)
        HAVING COUNT(*) >= 50
        ORDER BY msg_count DESC
        LIMIT {limit}
    """).fetchall()


# ============================================================
# ACCOMPLISHMENTS - V3 (proven to work)
# ============================================================
def test_accomplishments(messages: list) -> list:
    system = """Analyze these messages and extract what got DONE.
Write in first person as a daily log.

Look for evidence of:
- Code shipped/deployed/merged
- Bugs fixed
- Features built
- Problems solved
- Things created/generated
- Configurations set up

Format:
✓ [what was done] (category)

Categories: shipped, fixed, built, solved, setup, created

If nothing concrete was completed: "Mostly exploration and discussion today."
Be specific about the actual work, not vague summaries."""

    results = []
    for date, msgs, count in messages:
        sample = msgs[:30]
        user = f"Date: {date}\n\n" + "\n---\n".join(sample)
        response = call_llm(system, user)
        results.append((date, count, response))
    return results


# ============================================================
# EXPERTISE - V3 (proven to work)
# ============================================================
def test_expertise(messages: list) -> list:
    system = """Identify technologies and tools this person is working with confidently.

Look for:
- Technologies mentioned multiple times
- Commands/code being written (not asked about)
- Technical explanations given
- Problem-solving in specific domains

For each technology found:
TECH: [name]
EVIDENCE: [brief quote or description of usage]
CONFIDENCE: high/medium

Focus on: programming languages, databases, frameworks, tools, platforms.
Skip generic things like "file management" or "IDE"."""

    results = []
    for week, msgs, count in messages:
        sample = msgs[:50]
        user = f"Week of {week}:\n\n" + "\n---\n".join(sample)
        response = call_llm(system, user)
        results.append((week, count, response))
    return results


# ============================================================
# PROBLEMS - V4: Broader scope, look for any issue resolution
# ============================================================
def test_problems_v4(messages: list) -> list:
    """Version 4: Any issue that got resolved."""
    system = """Find any issues or blockers that were RESOLVED in these messages.

Look for patterns like:
- "error... fixed by..."
- "wasn't working... now works"
- "the issue was..."
- "figured out..."
- "the problem was..."
- "solved by..."
- Configuration changes that fixed things
- Workarounds that unblocked progress

Format each resolution as:
ISSUE: [what was broken/blocking]
RESOLUTION: [what fixed it]
DOMAIN: [tech/area]

Include both big and small fixes. If truly none found: "No resolutions found." """

    results = []
    for date, msgs, count in messages:
        sample = msgs[:35]
        user = f"Date: {date}\n\n" + "\n---\n".join(sample)
        response = call_llm(system, user)
        results.append((date, count, response))
    return results


# ============================================================
# TECH STACK - V2: With validation
# ============================================================
def test_tech_stack_v2(messages: list) -> list:
    """Version 2: Only tech actually mentioned in messages."""
    system = """List technologies that appear in these messages.

ONLY include if you can find the exact word/name in the messages.
Format: TECH_NAME (category) - "brief quote showing usage"

Categories: language, database, framework, tool, platform, api, library

Do NOT include generic technologies that might be assumed but aren't mentioned.
If very few technologies found, that's fine - be accurate over comprehensive."""

    results = []
    for week, msgs, count in messages:
        sample = msgs[:60]
        user = "\n---\n".join(sample)
        response = call_llm(system, user)
        results.append((week, count, response))
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING PIPELINES - ROUND 3 (older data)")
    print("=" * 60)

    daily = get_older_days(4)
    weekly = get_older_weeks(2)

    print(f"\nTesting on {len(daily)} days (June-Oct 2024), {len(weekly)} weeks\n")

    # Accomplishments
    print("\n" + "=" * 60)
    print("📦 ACCOMPLISHMENTS (high-activity days)")
    print("=" * 60)
    for date, count, result in test_accomplishments(daily[:2]):
        print(f"\n[{date}] ({count} msgs)")
        print(result[:500])

    # Expertise
    print("\n" + "=" * 60)
    print("🎯 EXPERTISE (high-activity weeks)")
    print("=" * 60)
    for week, count, result in test_expertise(weekly[:2]):
        print(f"\n[Week of {week}] ({count} msgs)")
        print(result[:600])

    # Problems V4
    print("\n" + "=" * 60)
    print("🔧 PROBLEMS V4 (broader scope)")
    print("=" * 60)
    for date, count, result in test_problems_v4(daily[:3]):
        print(f"\n[{date}] ({count} msgs)")
        print(result[:500])

    # Tech Stack V2
    print("\n" + "=" * 60)
    print("💻 TECH STACK V2 (validated)")
    print("=" * 60)
    for week, count, result in test_tech_stack_v2(weekly[:2]):
        print(f"\n[Week of {week}] ({count} msgs)")
        print(result[:500])
