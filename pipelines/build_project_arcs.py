#!/usr/bin/env python3
"""
Build project_arcs interpretation: Track project lifecycle from idea to completion/abandonment.

Uses Gemini 2.5 Flash Lite to identify project patterns - how projects evolve
through idea, build, debug, ship, maintain, or abandon phases.
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
INTERP_DIR = DATA_DIR / "interpretations" / "project_arcs" / "v1"

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
        "X-Title": "Intellectual DNA Project Arcs"
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """Identify projects being worked on and their lifecycle stage.
Project lifecycle stages:
1. Ideation: Brainstorming, exploring possibilities
2. Planning: Designing, architecting
3. Building: Active development, coding
4. Debugging: Fixing issues, troubleshooting
5. Shipping: Deploying, launching
6. Maintaining: Updates, small fixes
7. Abandoned: No longer worked on
8. Pivoting: Changing direction significantly

Write in FIRST PERSON as if reflecting on your own projects.

Output JSON: {"projects": [{"name": "project name or description", "stage": "ideation|planning|building|debugging|shipping|maintaining|abandoned|pivoting", "momentum": "high|medium|low|stalled", "blockers": "what's blocking progress if any", "next_step": "what needs to happen next"}], "project_pattern_note": "observations about how I approach projects"}"""
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


def parse_projects(llm_output: str) -> dict:
    """Parse LLM output for project data."""
    import re
    try:
        match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}
    except:
        return {}


def build_project_arcs(limit: int = None):
    """Build project arc analysis."""
    print("Building project_arcs/v1 interpretation...")
    print(f"  Model: {MODEL}")

    if not OPENROUTER_API_KEY:
        print("  ERROR: OPENROUTER_API_KEY not found")
        return

    con = duckdb.connect()
    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERP_DIR / "weekly.parquet"

    # Analyze weekly for project progression
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

    all_projects = {}  # project_name -> lifecycle data
    results = []
    total_cost = 0.0
    start_time = time.time()

    for i, (week_start, messages, msg_count) in enumerate(to_process):
        sample = messages[:30] if len(messages) > 30 else messages
        sample_text = "\n---\n".join(sample)

        prompt = f"""Week of {week_start}
{msg_count} total messages

Messages:
{sample_text}

What projects was I working on this week? What stage is each project at?"""

        response = call_gemini(prompt)
        parsed = parse_projects(response)

        projects = parsed.get("projects", [])
        pattern_note = parsed.get("project_pattern_note", "")

        # Track project lifecycles
        for project in projects:
            name = project.get("name", "").lower()[:50]
            if name and len(name) > 2:
                if name not in all_projects:
                    all_projects[name] = {
                        "name": project.get("name", name),
                        "stages": [],
                        "first_seen": str(week_start)
                    }
                all_projects[name]["stages"].append({
                    "week": str(week_start),
                    "stage": project.get("stage", "unknown"),
                    "momentum": project.get("momentum", "unknown"),
                    "blockers": project.get("blockers", "")
                })

        results.append({
            "week_start": str(week_start),
            "projects_json": json.dumps(projects),
            "pattern_note": pattern_note,
            "project_count": len(projects),
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
            print(f"  [{i+1}/{len(to_process)}] {week_start} - {len(all_projects)} projects tracked - {rate:.1f} req/min - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if results:
        # Save weekly data
        con.execute("""
            CREATE TABLE project_weekly (
                week_start DATE,
                projects_json VARCHAR,
                pattern_note VARCHAR,
                project_count INTEGER,
                message_count INTEGER,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO project_weekly VALUES (?, ?, ?, ?, ?, ?)",
                       [r["week_start"], r["projects_json"], r["pattern_note"],
                        r["project_count"], r["message_count"], r["processed_at"]])

        con.execute(f"COPY project_weekly TO '{output_path}' (FORMAT PARQUET)")

        # Save project lifecycles
        projects_path = INTERP_DIR / "projects.parquet"
        con.execute("""
            CREATE TABLE project_lifecycles (
                name VARCHAR,
                first_seen DATE,
                weeks_active INTEGER,
                stage_progression VARCHAR,
                final_stage VARCHAR,
                outcome VARCHAR,
                processed_at VARCHAR
            )
        """)

        for name, data in all_projects.items():
            stages = data["stages"]
            final = stages[-1]["stage"] if stages else "unknown"
            # Determine outcome
            if final in ["shipped", "shipping", "maintaining"]:
                outcome = "completed"
            elif final == "abandoned":
                outcome = "abandoned"
            elif final == "pivoting":
                outcome = "pivoted"
            else:
                outcome = "in_progress"

            con.execute("INSERT INTO project_lifecycles VALUES (?, ?, ?, ?, ?, ?, ?)",
                       [data["name"], data["first_seen"], len(stages),
                        json.dumps([s["stage"] for s in stages]), final, outcome,
                        datetime.now().isoformat()])

        con.execute(f"COPY project_lifecycles TO '{projects_path}' (FORMAT PARQUET)")

        print(f"\n  Processed: {len(results)} weeks")
        print(f"  Unique projects tracked: {len(all_projects)}")
        print(f"  Total cost: ${total_cost:.4f}")

        # Show project outcomes
        outcomes = {"completed": 0, "abandoned": 0, "pivoted": 0, "in_progress": 0}
        for name, data in all_projects.items():
            stages = data["stages"]
            final = stages[-1]["stage"] if stages else "unknown"
            if final in ["shipped", "shipping", "maintaining"]:
                outcomes["completed"] += 1
            elif final == "abandoned":
                outcomes["abandoned"] += 1
            elif final == "pivoting":
                outcomes["pivoted"] += 1
            else:
                outcomes["in_progress"] += 1

        print("\n  Project outcomes:")
        for outcome, count in sorted(outcomes.items(), key=lambda x: -x[1]):
            print(f"    {outcome}: {count}")

        # Show longest-running projects
        print("\n  Longest-running projects:")
        sorted_projects = sorted(all_projects.items(), key=lambda x: -len(x[1]["stages"]))
        for name, data in sorted_projects[:10]:
            stages = [s["stage"] for s in data["stages"]]
            arc = " → ".join(stages[-4:]) if len(stages) > 4 else " → ".join(stages)
            print(f"    [{len(stages)}w] {data['name'][:40]}: {arc}")

    config = {
        "name": "project_arcs/v1",
        "description": "Project lifecycle patterns",
        "model": MODEL,
        "voice": "first-person",
        "created": datetime.now().strftime("%Y-%m-%d")
    }
    with open(INTERP_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return len(all_projects)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Max weeks')
    parser.add_argument('--test', action='store_true', help='Test with 5 weeks')
    args = parser.parse_args()

    if args.test:
        build_project_arcs(limit=5)
    else:
        build_project_arcs(limit=args.limit)
