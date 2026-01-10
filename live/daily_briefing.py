#!/usr/bin/env python3
"""
Daily Briefing Agent - Proactive Brain Intelligence

Runs at 6am via launchd. Generates insights you didn't ask for:
- Circling themes (mentioned >3x)
- Contradictions with past self
- Stalled projects (mentioned but no commits)
- Forgotten insights (said once, never again)

Output: ~/intellectual_dna/briefings/YYYY-MM-DD.md

Usage:
    python daily_briefing.py              # Generate today's briefing
    python daily_briefing.py --speak      # Also speak the summary
    python daily_briefing.py --days 7     # Analyze last 7 days
"""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import httpx

from config import (
    BASE,
    PARQUET_PATH,
    GITHUB_COMMITS_PARQUET,
    LANCE_PATH,
)

# Output
BRIEFINGS_DIR = BASE / "briefings"
BRIEFINGS_DIR.mkdir(exist_ok=True)

# LLM Config (Gemini 3 Flash via OpenRouter)
# Load from .env file if not in environment
from dotenv import load_dotenv
load_dotenv(BASE / ".env")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL = "google/gemini-3-flash-preview"  # Per CLAUDE.md: always use gemini-3-flash-preview


def get_recent_messages(days: int = 1) -> list[dict]:
    """Get user messages from the last N days."""
    con = duckdb.connect()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    query = f"""
        SELECT
            content,
            conversation_title,
            source,
            created,
            year,
            month
        FROM read_parquet('{PARQUET_PATH}')
        WHERE role = 'user'
        AND CAST(created AS DATE) >= DATE '{cutoff}'
        AND content IS NOT NULL
        AND LENGTH(content) > 20
        ORDER BY created DESC
        LIMIT 500
    """

    results = con.execute(query).fetchall()
    return [
        {
            "content": r[0],
            "title": r[1],
            "source": r[2],
            "created": r[3],
            "year": r[4],
            "month": r[5],
        }
        for r in results
    ]


def get_historical_positions(topic: str, limit: int = 10) -> list[str]:
    """Get past statements about a topic using semantic search."""
    try:
        import lancedb
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        embedding = model.encode(topic).tolist()

        db = lancedb.connect(str(LANCE_PATH))
        tbl = db.open_table("message")

        # Search older messages (exclude last 7 days)
        results = tbl.search(embedding).limit(limit * 2).to_pandas()

        # Filter to older than 7 days
        cutoff = datetime.now() - timedelta(days=7)
        older = []
        for _, row in results.iterrows():
            # Approximate date check using year/month
            msg_date = datetime(row.get('year', 2024), row.get('month', 1), 15)
            if msg_date < cutoff:
                older.append(row.get('content', '')[:500])

        return older[:limit]
    except Exception as e:
        print(f"Warning: Could not get historical positions: {e}", file=sys.stderr)
        return []


def get_github_activity(days: int = 7) -> set[str]:
    """Get projects with recent GitHub commits."""
    if not GITHUB_COMMITS_PARQUET.exists():
        return set()

    con = duckdb.connect()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        results = con.execute(f"""
            SELECT DISTINCT repo_name
            FROM read_parquet('{GITHUB_COMMITS_PARQUET}')
            WHERE CAST(timestamp AS DATE) >= DATE '{cutoff}'
        """).fetchall()
        return {r[0] for r in results}
    except:
        return set()


def extract_themes(messages: list[dict]) -> list[tuple[str, int]]:
    """Extract recurring themes from messages."""
    # Simple keyword extraction - words that appear >3 times
    word_counts = Counter()

    stopwords = {
        'the', 'a', 'an', 'is', 'it', 'to', 'of', 'and', 'in', 'that', 'this',
        'for', 'on', 'with', 'as', 'at', 'by', 'be', 'are', 'was', 'were',
        'i', 'you', 'we', 'me', 'my', 'your', 'can', 'do', 'have', 'has',
        'just', 'not', 'but', 'or', 'if', 'so', 'what', 'how', 'when', 'all',
        'there', 'their', 'from', 'will', 'would', 'could', 'should', 'get',
        'make', 'like', 'want', 'need', 'use', 'using', 'used', 'one', 'also',
        'now', 'think', 'know', 'see', 'look', 'give', 'let', 'yes', 'no',
        # Metadata/structural words
        'user', 'assistant', 'message', 'content', 'data', 'file', 'files',
        'code', 'about', 'here', 'some', 'more', 'then', 'than', 'been',
        'being', 'were', 'other', 'into', 'only', 'over', 'such', 'through',
        'very', 'most', 'after', 'before', 'between', 'each', 'same', 'which',
    }

    for msg in messages:
        content = msg.get('content', '').lower()
        words = content.split()
        for word in words:
            # Clean word
            word = ''.join(c for c in word if c.isalnum())
            if len(word) > 3 and word not in stopwords:
                word_counts[word] += 1

    # Return words mentioned >3 times
    themes = [(word, count) for word, count in word_counts.most_common(20) if count > 3]
    return themes


def extract_project_mentions(messages: list[dict]) -> set[str]:
    """Extract project names mentioned in messages."""
    projects = set()

    # Look for common project indicators
    project_patterns = [
        'intellectual_dna', 'intellectual-dna', 'brain', 'mcp',
        'sparkii', 'portfolio', 'jewtube', 'sefaria',
    ]

    for msg in messages:
        content = msg.get('content', '').lower()
        for pattern in project_patterns:
            if pattern in content:
                projects.add(pattern)

    return projects


def call_llm(prompt: str, max_tokens: int = 1000) -> str:
    """Call LLM via OpenRouter."""
    if not OPENROUTER_API_KEY:
        return "[LLM unavailable - set OPENROUTER_API_KEY]"

    try:
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[LLM error: {e}]"


def find_contradictions(recent_messages: list[dict], themes: list[tuple[str, int]]) -> list[str]:
    """Find potential contradictions between recent and historical statements."""
    contradictions = []

    # Get top themes
    top_themes = [t[0] for t in themes[:5]]

    for theme in top_themes:
        # Get recent statements about this theme
        recent_statements = [
            m['content'][:300] for m in recent_messages
            if theme in m.get('content', '').lower()
        ][:3]

        if not recent_statements:
            continue

        # Get historical statements
        historical = get_historical_positions(theme, limit=5)

        if not historical:
            continue

        # Ask LLM to find contradictions
        prompt = f"""Compare these recent statements about "{theme}" with historical ones.

Recent (last few days):
{chr(10).join(f'- {s}' for s in recent_statements)}

Historical (older):
{chr(10).join(f'- {s}' for s in historical)}

If there's a meaningful contradiction or shift in position, describe it in ONE sentence.
If they're consistent, respond with just "CONSISTENT".
Be specific about what changed."""

        result = call_llm(prompt, max_tokens=150)

        if result and "CONSISTENT" not in result.upper() and "[LLM" not in result:
            contradictions.append(f"**{theme}**: {result}")

    return contradictions


def find_forgotten_insights(months: int = 6) -> list[str]:
    """Find insights mentioned once long ago but not recently."""
    con = duckdb.connect()

    # Get distinctive phrases from 3-6 months ago
    old_start = (datetime.now() - timedelta(days=months*30)).strftime("%Y-%m-%d")
    old_end = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    recent_start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    try:
        # Get old messages
        old_msgs = con.execute(f"""
            SELECT content
            FROM read_parquet('{PARQUET_PATH}')
            WHERE role = 'user'
            AND CAST(created AS DATE) >= DATE '{old_start}'
            AND CAST(created AS DATE) < DATE '{old_end}'
            AND content IS NOT NULL
            AND LENGTH(content) > 50
            LIMIT 200
        """).fetchall()

        # Get recent messages
        recent_msgs = con.execute(f"""
            SELECT content
            FROM read_parquet('{PARQUET_PATH}')
            WHERE role = 'user'
            AND CAST(created AS DATE) >= DATE '{recent_start}'
            AND content IS NOT NULL
            LIMIT 500
        """).fetchall()

        recent_text = ' '.join(r[0].lower() for r in recent_msgs)

        # Find phrases in old that aren't in recent
        forgotten = []
        for (content,) in old_msgs[:50]:
            # Extract potential insight phrases
            if any(marker in content.lower() for marker in [
                'insight', 'realize', 'breakthrough', 'principle', 'key is',
                'the trick', 'important', 'lesson', 'framework'
            ]):
                # Check if core concepts are in recent
                words = set(content.lower().split())
                key_words = [w for w in words if len(w) > 5][:3]

                if key_words and not any(w in recent_text for w in key_words):
                    forgotten.append(content[:200])

        return forgotten[:3]
    except Exception as e:
        print(f"Warning: Could not find forgotten insights: {e}", file=sys.stderr)
        return []


def generate_briefing(
    days: int,
    messages: list[dict],
    themes: list[tuple[str, int]],
    contradictions: list[str],
    stalled: set[str],
    forgotten: list[str],
) -> str:
    """Generate the final briefing markdown."""

    date_str = datetime.now().strftime("%B %d, %Y")

    briefing = f"""# Daily Briefing - {date_str}

*Analyzing last {days} day(s): {len(messages)} messages*

---

## 🔄 Circling Themes

"""

    if themes:
        for theme, count in themes[:7]:
            briefing += f"- **{theme}** ({count}x)\n"
    else:
        briefing += "_No recurring themes detected._\n"

    briefing += """
---

## ⚡ Contradictions

"""

    if contradictions:
        for c in contradictions:
            briefing += f"- {c}\n"
    else:
        briefing += "_No contradictions detected with past positions._\n"

    briefing += """
---

## 🚧 Stalled Projects

*Mentioned but no recent commits*

"""

    if stalled:
        for project in stalled:
            briefing += f"- {project}\n"
    else:
        briefing += "_All mentioned projects have recent activity._\n"

    briefing += """
---

## 💎 Forgotten Insights

*From 3-6 months ago, not mentioned recently*

"""

    if forgotten:
        for insight in forgotten:
            briefing += f"> {insight}...\n\n"
    else:
        briefing += "_No forgotten insights surfaced._\n"

    # Generate summary with LLM
    briefing += """
---

## 📋 Summary

"""

    summary_prompt = f"""Based on this daily briefing data, write a 3-bullet summary for Mordechai.
Be direct, specific, and actionable. No fluff.

Themes: {', '.join(t[0] for t in themes[:5])}
Contradictions: {len(contradictions)} found
Stalled projects: {', '.join(stalled) if stalled else 'none'}
Forgotten insights: {len(forgotten)} found

Write exactly 3 bullets, each 1 sentence. Focus on what matters most."""

    summary = call_llm(summary_prompt, max_tokens=200)
    briefing += summary + "\n"

    briefing += f"""
---

*Generated at {datetime.now().strftime("%H:%M")} by Daily Briefing Agent*
"""

    return briefing


def speak_summary(briefing: str):
    """Speak the summary section using Piper TTS."""
    # Extract summary section
    if "## 📋 Summary" in briefing:
        summary = briefing.split("## 📋 Summary")[1].split("---")[0].strip()

        # Clean for speech
        summary = summary.replace("*", "").replace("#", "").replace(">", "")
        summary = summary[:500]  # Limit length

        try:
            subprocess.run(
                [str(Path.home() / ".local/bin/speak"), summary],
                timeout=30,
                capture_output=True,
            )
        except Exception as e:
            print(f"Warning: Could not speak summary: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Daily Briefing Agent")
    parser.add_argument("--days", type=int, default=1, help="Days to analyze")
    parser.add_argument("--speak", action="store_true", help="Speak the summary")
    args = parser.parse_args()

    print(f"🧠 Daily Briefing Agent", file=sys.stderr)
    print(f"   Analyzing last {args.days} day(s)...", file=sys.stderr)

    # 1. Get recent messages
    messages = get_recent_messages(days=args.days)
    print(f"   Found {len(messages)} messages", file=sys.stderr)

    if not messages:
        print("   No messages found. Exiting.", file=sys.stderr)
        return

    # 2. Extract themes
    themes = extract_themes(messages)
    print(f"   Extracted {len(themes)} themes", file=sys.stderr)

    # 3. Find contradictions
    print("   Checking for contradictions...", file=sys.stderr)
    contradictions = find_contradictions(messages, themes)
    print(f"   Found {len(contradictions)} contradictions", file=sys.stderr)

    # 4. Find stalled projects
    mentioned = extract_project_mentions(messages)
    active = get_github_activity(days=7)
    stalled = mentioned - active
    print(f"   Found {len(stalled)} stalled projects", file=sys.stderr)

    # 5. Find forgotten insights
    print("   Searching for forgotten insights...", file=sys.stderr)
    forgotten = find_forgotten_insights(months=6)
    print(f"   Found {len(forgotten)} forgotten insights", file=sys.stderr)

    # 6. Generate briefing
    print("   Generating briefing...", file=sys.stderr)
    briefing = generate_briefing(
        days=args.days,
        messages=messages,
        themes=themes,
        contradictions=contradictions,
        stalled=stalled,
        forgotten=forgotten,
    )

    # 7. Save
    output_path = BRIEFINGS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.md"
    output_path.write_text(briefing)
    print(f"   Saved to {output_path}", file=sys.stderr)

    # 8. Speak if requested
    if args.speak:
        print("   Speaking summary...", file=sys.stderr)
        speak_summary(briefing)

    # Also print to stdout
    print(briefing)


if __name__ == "__main__":
    main()
