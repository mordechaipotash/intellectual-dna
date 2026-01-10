#!/usr/bin/env python3
"""
Build TOOL_STACK_COMBOS interpretation: Track technology combinations and evolution.
Extracts: tech stack decisions, tool pairings, framework choices, architecture patterns.

Model: Claude Sonnet 4.5 - Best at understanding code/tech relationships
Budget: $5 (~1100 requests)
"""

import os
import json
import re
import time
import duckdb
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from collections import Counter

load_dotenv(Path("/Users/mordechai/intellectual_dna/.env"))

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
DATA_DIR = BASE_DIR / "data"
FACTS_DIR = DATA_DIR / "facts"
INTERP_DIR = DATA_DIR / "interpretations" / "tool_stack_combos" / "v1"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-sonnet-4.5"

# Claude Sonnet 4.5: $3.00/1M input, $15.00/1M output
COST_PER_1M_INPUT = 3.00
COST_PER_1M_OUTPUT = 15.00

REQUESTS_PER_MINUTE = 25  # Claude rate limits
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE

# Tool/tech triggers from 100x deep mining
TECH_TRIGGERS = [
    "lets use", "supabase", "vercel", "duckdb", "parquet",
    "next", "nextjs", "react", "claude", "mcp", "deploy",
    "typescript", "python", "fastapi", "openrouter", "ollama",
    "embedding", "vector", "postgres", "prisma", "drizzle",
    "tailwind", "shadcn", "cursor", "vscode", "docker"
]


def call_claude(prompt: str, max_tokens: int = 600) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Intellectual DNA Tool Stacks"
    }
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """Extract technology stack decisions and tool combinations from these messages.

Look for:
- Stack decisions: "let's use X with Y" or "X + Y"
- Tool pairings: technologies used together intentionally
- Framework choices: selecting a framework over alternatives
- Architecture patterns: backend/frontend/data layer decisions
- Migration signals: moving from one tech to another
- Integration patterns: connecting multiple services

For each combination found:
STACK: [list of technologies, e.g., "DuckDB + Parquet + Python"]
DECISION: [what problem this stack solves]
LAYER: [frontend|backend|data|infra|ai|fullstack]
CONFIDENCE: [explicit|implicit|experimental]

Only include ACTUAL technology decisions being made, not general mentions.
If no stack decisions found: "No tool stack signals this period."
Focus on the WHY behind combinations - what makes them work together."""
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=45)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Error: {str(e)[:50]}]"


def parse_stack_patterns(text: str) -> list:
    """Parse tool stack patterns from LLM output."""
    patterns = []
    lines = text.split('\n')

    current = {}
    for line in lines:
        line = line.strip()
        if line.startswith('STACK:'):
            if current and 'stack' in current:
                patterns.append(current)
            current = {'stack': line[6:].strip()}
        elif line.startswith('DECISION:') and current:
            current['decision'] = line[9:].strip()
        elif line.startswith('LAYER:') and current:
            current['layer'] = line[6:].strip().lower()
        elif line.startswith('CONFIDENCE:') and current:
            current['confidence'] = line[11:].strip().lower()

    if current and 'stack' in current:
        patterns.append(current)

    return patterns


def has_tech_triggers(messages: list) -> bool:
    """Check if messages contain tech trigger phrases."""
    text = ' '.join(messages).lower()
    return sum(1 for trigger in TECH_TRIGGERS if trigger in text) >= 2  # At least 2 tech mentions


def extract_tech_mentions(messages: list) -> list:
    """Extract all tech mentions from messages."""
    text = ' '.join(messages).lower()
    return [t for t in TECH_TRIGGERS if t in text]


def build_tool_stack_combos(limit: int = None, budget: float = 5.0):
    """Build tool stack combinations interpretation."""
    print("Building tool_stack_combos/v1 interpretation...")
    print(f"  Model: {MODEL}")
    print(f"  Budget: ${budget:.2f}")

    if not OPENROUTER_API_KEY:
        print("  ERROR: OPENROUTER_API_KEY not found")
        return

    con = duckdb.connect()
    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERP_DIR / "weekly.parquet"

    # Check existing
    processed_weeks = set()
    if output_path.exists():
        existing = con.execute(f"SELECT week_start FROM '{output_path}'").fetchall()
        processed_weeks = {str(row[0]) for row in existing}
        print(f"  Already processed: {len(processed_weeks)} weeks")

    # Get weekly messages
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
          AND LENGTH(c.full_content) BETWEEN 30 AND 1200
        GROUP BY DATE_TRUNC('week', idx.timestamp::TIMESTAMP)
        HAVING COUNT(*) >= 15
        ORDER BY week_start
    """).fetchall()

    # Filter to weeks with tech triggers and not yet processed
    to_process = [(w, m, c) for w, m, c in weekly_messages
                  if str(w) not in processed_weeks and has_tech_triggers(m)]

    if limit:
        to_process = to_process[:limit]

    print(f"  Weeks with tech mentions: {len(to_process)}")

    if not to_process:
        print("  Nothing to process!")
        return

    results = []
    total_cost = 0.0
    total_stacks = 0
    start_time = time.time()

    for i, (week_start, messages, msg_count) in enumerate(to_process):
        # Check budget
        if total_cost >= budget:
            print(f"  Budget exhausted at ${total_cost:.4f}")
            break

        # Sample messages with tech mentions preferentially
        tech_msgs = [m for m in messages if any(t in m.lower() for t in TECH_TRIGGERS)]
        other_msgs = [m for m in messages if m not in tech_msgs]
        sample = tech_msgs[:35] + other_msgs[:15]

        # Add context about tech mentions found
        tech_found = extract_tech_mentions(messages)
        prompt = f"Week of {week_start}\nTechnologies mentioned this week: {', '.join(set(tech_found))}\n\n" + "\n---\n".join(sample[:50])

        response = call_claude(prompt)
        stacks = parse_stack_patterns(response)
        total_stacks += len(stacks)

        # Extract layers for summary
        layers = list(set(s.get('layer', 'unknown') for s in stacks if s.get('layer')))

        # Extract unique techs mentioned in stacks
        all_techs = []
        for s in stacks:
            stack_str = s.get('stack', '')
            all_techs.extend([t.strip().lower() for t in re.split(r'[+,\s]+', stack_str) if len(t.strip()) > 1])

        results.append({
            "week_start": str(week_start),
            "stacks_json": json.dumps(stacks),
            "stack_count": len(stacks),
            "layers": json.dumps(layers),
            "techs_used": json.dumps(list(set(all_techs))),
            "raw_response": response[:600],
            "message_count": msg_count,
            "processed_at": datetime.now().isoformat()
        })

        # Calculate cost
        input_tokens = len(prompt) / 4
        output_tokens = len(response) / 4
        cost = (input_tokens * COST_PER_1M_INPUT / 1_000_000) + (output_tokens * COST_PER_1M_OUTPUT / 1_000_000)
        total_cost += cost

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            layers_str = ', '.join(layers[:3]) if layers else 'none'
            print(f"  [{i+1}/{len(to_process)}] {week_start} - {len(stacks)} stacks ({layers_str}) - {rate:.1f} req/min - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if results:
        con.execute("""
            CREATE TABLE tool_stack_combos (
                week_start DATE,
                stacks_json VARCHAR,
                stack_count INTEGER,
                layers VARCHAR,
                techs_used VARCHAR,
                raw_response VARCHAR,
                message_count INTEGER,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO tool_stack_combos VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       [r["week_start"], r["stacks_json"], r["stack_count"],
                        r["layers"], r["techs_used"], r["raw_response"],
                        r["message_count"], r["processed_at"]])

        if output_path.exists():
            con.execute(f"INSERT INTO tool_stack_combos SELECT * FROM '{output_path}'")

        con.execute(f"COPY tool_stack_combos TO '{output_path}' (FORMAT PARQUET)")

        print(f"\n  Processed: {len(results)} weeks")
        print(f"  Total stack decisions: {total_stacks}")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Remaining budget: ${budget - total_cost:.4f}")

        # Show layer frequency
        all_layers = []
        for r in results:
            all_layers.extend(json.loads(r["layers"]))
        layer_dist = Counter(all_layers).most_common()
        if layer_dist:
            print(f"\n  Layer distribution:")
            for layer, count in layer_dist:
                print(f"    {layer}: {count} weeks")

        # Show top technologies
        all_techs = []
        for r in results:
            all_techs.extend(json.loads(r["techs_used"]))
        top_techs = Counter(all_techs).most_common(15)
        if top_techs:
            print(f"\n  Top technologies:")
            for tech, count in top_techs:
                print(f"    {tech}: {count} mentions")

        # Show sample stacks
        print(f"\n  Sample stack decisions:")
        for r in results[-5:]:
            stack_list = json.loads(r["stacks_json"])
            if stack_list:
                first = stack_list[0]
                stack = first.get('stack', '?')[:50]
                decision = first.get('decision', '?')[:30]
                print(f"    {r['week_start'][:10]}: {stack} → {decision}...")

    config = {
        "name": "tool_stack_combos/v1",
        "description": "Technology stack decisions and combinations",
        "model": MODEL,
        "triggers": TECH_TRIGGERS,
        "voice": "first-person",
        "created": datetime.now().strftime("%Y-%m-%d")
    }
    with open(INTERP_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return len(results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Max weeks')
    parser.add_argument('--budget', type=float, default=5.0, help='Max budget in USD')
    parser.add_argument('--test', action='store_true', help='Test with 5 weeks')
    args = parser.parse_args()

    if args.test:
        build_tool_stack_combos(limit=5, budget=0.50)
    else:
        build_tool_stack_combos(limit=args.limit, budget=args.budget)
