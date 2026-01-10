#!/usr/bin/env python3
"""
TOOL_STACKS v2 - MAXED OUT Gemini 3 Flash for Tech Analysis

Optimizations:
1. JSON Schema mode - guaranteed structured tech stack output
2. Monthly batches - see full stack evolution
3. 200+ messages per batch - complete context
4. Tech relationship extraction - what works with what
"""

import os
import json
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
INTERP_DIR = DATA_DIR / "interpretations" / "tool_stacks" / "v2"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3-flash-preview"

COST_PER_1M_INPUT = 0.50
COST_PER_1M_OUTPUT = 3.00

TOOL_STACK_SCHEMA = {
    "type": "object",
    "properties": {
        "stacks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "technologies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of technologies used together"
                    },
                    "purpose": {"type": "string", "description": "What this stack accomplishes"},
                    "layer": {
                        "type": "string",
                        "enum": ["frontend", "backend", "data", "infra", "ai", "fullstack", "devtools"]
                    },
                    "decision_quote": {"type": "string", "description": "Quote showing the decision"},
                    "confidence": {
                        "type": "string",
                        "enum": ["exploring", "trying", "using", "committed"]
                    }
                },
                "required": ["technologies", "purpose", "layer"]
            }
        },
        "tech_relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tech_a": {"type": "string"},
                    "tech_b": {"type": "string"},
                    "relationship": {
                        "type": "string",
                        "enum": ["replaces", "complements", "integrates_with", "migrating_to"]
                    }
                },
                "required": ["tech_a", "tech_b", "relationship"]
            }
        },
        "dominant_stack": {
            "type": "string",
            "description": "The main technology combination this month"
        },
        "new_adoptions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Technologies being tried for the first time"
        },
        "abandonments": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Technologies being dropped or replaced"
        }
    },
    "required": ["stacks", "dominant_stack"]
}

TECH_TRIGGERS = [
    "lets use", "using", "supabase", "vercel", "duckdb", "parquet",
    "next", "nextjs", "react", "claude", "mcp", "deploy",
    "typescript", "python", "fastapi", "openrouter", "ollama",
    "embedding", "vector", "postgres", "prisma", "drizzle",
    "tailwind", "shadcn", "cursor", "vscode", "docker",
    "gemini", "grok", "openai", "anthropic", "langchain"
]


def call_gemini_structured(prompt: str, max_tokens: int = 2000) -> dict:
    """Call Gemini with JSON Schema mode."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Intellectual DNA Tech Stacks v2"
    }
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are analyzing conversation messages for technology stack decisions.

Extract:
1. STACKS: Groups of technologies used together (e.g., "DuckDB + Parquet + Python")
2. RELATIONSHIPS: How technologies relate (replaces, complements, integrates)
3. ADOPTIONS: New technologies being tried
4. ABANDONMENTS: Technologies being dropped

Focus on DECISIONS - not just mentions. Look for:
- "let's use X with Y"
- "switching from X to Y"
- "X works great with Y"
- "deploying on X"
- Architecture choices

Be thorough - a month of development has many stack decisions."""
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "tech_stacks",
                "strict": True,
                "schema": TOOL_STACK_SCHEMA
            }
        }
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except json.JSONDecodeError as e:
        return {"stacks": [], "dominant_stack": "", "error": f"JSON: {str(e)[:50]}"}
    except Exception as e:
        return {"stacks": [], "dominant_stack": "", "error": str(e)[:50]}


def build_tool_stacks_v2(limit: int = None, budget: float = 5.0):
    """Build tool stacks with MAXED OUT Gemini 3 Flash."""
    print("="*70)
    print("🔧 TOOL_STACKS v2 - MAXED OUT GEMINI 3 FLASH")
    print("="*70)
    print(f"  Model: {MODEL}")
    print(f"  Features: JSON Schema, tech relationships, monthly batches")
    print(f"  Budget: ${budget:.2f}")
    print("="*70)

    if not OPENROUTER_API_KEY:
        print("  ERROR: OPENROUTER_API_KEY not found")
        return

    con = duckdb.connect()
    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERP_DIR / "monthly.parquet"

    # Get MONTHLY messages
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
          AND LENGTH(c.full_content) BETWEEN 20 AND 1500
        GROUP BY DATE_TRUNC('month', idx.timestamp::TIMESTAMP)
        HAVING COUNT(*) >= 50
        ORDER BY month_start
    """).fetchall()

    # Check existing
    processed_months = set()
    if output_path.exists():
        existing = con.execute(f"SELECT month_start FROM '{output_path}'").fetchall()
        processed_months = {str(row[0]) for row in existing}
        print(f"  Already processed: {len(processed_months)} months")

    # Filter to months with tech mentions
    def has_tech_mentions(msgs):
        text = ' '.join(msgs).lower()
        return sum(1 for t in TECH_TRIGGERS if t in text) >= 3

    to_process = [(m, msgs, c) for m, msgs, c in monthly_messages
                  if str(m) not in processed_months and has_tech_mentions(msgs)]

    if limit:
        to_process = to_process[:limit]

    print(f"  Months with tech mentions: {len(to_process)}")

    if not to_process:
        print("  Nothing to process!")
        return

    results = []
    total_cost = 0.0
    total_stacks = 0
    start_time = time.time()

    for i, (month_start, messages, msg_count) in enumerate(to_process):
        if total_cost >= budget:
            print(f"\n  💰 Budget exhausted at ${total_cost:.4f}")
            break

        # Prioritize messages with tech mentions
        tech_msgs = [m for m in messages if any(t in m.lower() for t in TECH_TRIGGERS)]
        other_msgs = [m for m in messages if m not in tech_msgs]

        sample = tech_msgs[:150] + other_msgs[:50]

        # List detected technologies
        text = ' '.join(messages).lower()
        detected_techs = [t for t in TECH_TRIGGERS if t in text]

        prompt = f"""MONTH: {month_start.strftime('%B %Y')}
TOTAL MESSAGES: {msg_count}
DETECTED TECHNOLOGIES: {', '.join(set(detected_techs))}

Find ALL technology stack decisions in these messages.
Look for: architecture choices, tool combinations, migrations, new adoptions.

MESSAGES:
{"="*50}
""" + "\n---\n".join(sample)

        result = call_gemini_structured(prompt)
        stacks = result.get("stacks", [])
        total_stacks += len(stacks)

        # Calculate cost
        input_chars = len(prompt)
        output_chars = len(json.dumps(result))
        input_tokens = input_chars / 4
        output_tokens = output_chars / 4
        cost = (input_tokens * COST_PER_1M_INPUT / 1_000_000) + (output_tokens * COST_PER_1M_OUTPUT / 1_000_000)
        total_cost += cost

        # Extract all technologies mentioned
        all_techs = []
        for s in stacks:
            all_techs.extend(s.get("technologies", []))

        results.append({
            "month_start": str(month_start),
            "stacks_json": json.dumps(stacks),
            "stack_count": len(stacks),
            "relationships_json": json.dumps(result.get("tech_relationships", [])),
            "dominant_stack": result.get("dominant_stack", ""),
            "new_adoptions": json.dumps(result.get("new_adoptions", [])),
            "abandonments": json.dumps(result.get("abandonments", [])),
            "all_techs": json.dumps(list(set(all_techs))),
            "message_count": msg_count,
            "processed_at": datetime.now().isoformat()
        })

        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
        dominant = result.get("dominant_stack", "")[:30]

        print(f"  [{i+1}/{len(to_process)}] {month_start.strftime('%Y-%m')} | {len(stacks)} stacks | ${cost:.4f} | {rate:.1f}/min")
        if dominant:
            print(f"       Dominant: {dominant}")

        time.sleep(2.5)

    if results:
        con.execute("""
            CREATE TABLE tool_stacks_v2 (
                month_start DATE,
                stacks_json VARCHAR,
                stack_count INTEGER,
                relationships_json VARCHAR,
                dominant_stack VARCHAR,
                new_adoptions VARCHAR,
                abandonments VARCHAR,
                all_techs VARCHAR,
                message_count INTEGER,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO tool_stacks_v2 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       [r["month_start"], r["stacks_json"], r["stack_count"],
                        r["relationships_json"], r["dominant_stack"], r["new_adoptions"],
                        r["abandonments"], r["all_techs"], r["message_count"], r["processed_at"]])

        if output_path.exists():
            con.execute(f"INSERT INTO tool_stacks_v2 SELECT * FROM '{output_path}'")

        con.execute(f"COPY tool_stacks_v2 TO '{output_path}' (FORMAT PARQUET)")

        # Summary
        print("\n" + "="*70)
        print("📊 EXTRACTION COMPLETE")
        print("="*70)
        print(f"  Months processed: {len(results)}")
        print(f"  Total stack decisions: {total_stacks}")
        print(f"  Total cost: ${total_cost:.4f}")

        # Technology frequency
        all_techs = []
        for r in results:
            all_techs.extend(json.loads(r["all_techs"]))
        tech_freq = Counter(all_techs).most_common(15)
        print(f"\n  🔧 Top Technologies:")
        for tech, count in tech_freq:
            print(f"     {tech}: {count} months")

        # Dominant stacks over time
        print(f"\n  📈 Stack Evolution:")
        for r in results[-12:]:
            dominant = r["dominant_stack"][:40] if r["dominant_stack"] else "varied"
            print(f"     {r['month_start'][:7]}: {dominant}")

        # New adoptions
        all_adoptions = []
        for r in results:
            all_adoptions.extend(json.loads(r["new_adoptions"]))
        if all_adoptions:
            adoption_freq = Counter(all_adoptions).most_common(10)
            print(f"\n  🆕 Technologies Adopted:")
            for tech, count in adoption_freq:
                print(f"     {tech}: {count} times")

    config = {
        "name": "tool_stacks/v2",
        "description": "MAXED OUT tech stack extraction with Gemini 3",
        "model": MODEL,
        "features": ["json_schema", "tech_relationships", "monthly", "1M_context"],
        "schema": TOOL_STACK_SCHEMA,
        "created": datetime.now().strftime("%Y-%m-%d")
    }
    with open(INTERP_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return len(results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Max months')
    parser.add_argument('--budget', type=float, default=5.0, help='Budget USD')
    parser.add_argument('--test', action='store_true', help='Test 3 months')
    args = parser.parse_args()

    if args.test:
        build_tool_stacks_v2(limit=3, budget=1.0)
    else:
        build_tool_stacks_v2(limit=args.limit, budget=args.budget)
