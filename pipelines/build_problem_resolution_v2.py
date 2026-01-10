#!/usr/bin/env python3
"""
PROBLEM_RESOLUTION v2 - MAXED OUT Grok 3 with Think Mode

Optimizations:
1. Grok 3 Think mode - extended reasoning for root cause analysis
2. JSON structured output
3. Monthly batches - full debugging journey context
4. 1M token context - entire month's problem-solving in one shot
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
INTERP_DIR = DATA_DIR / "interpretations" / "problem_resolution" / "v2"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Use Grok 3 (full) for maximum reasoning power
MODEL = "x-ai/grok-3-beta"

# Grok 3: $4.00/1M input, $20.00/1M output (but worth it for reasoning)
COST_PER_1M_INPUT = 4.00
COST_PER_1M_OUTPUT = 20.00

RESOLUTION_TRIGGERS = [
    "fix", "bug", "not working", "error", "the issue was",
    "turns out", "it worked", "working now", "finally",
    "solved", "figured out", "the problem was", "root cause",
    "aha", "that's why", "realized", "debugging", "traced",
    "found it", "got it working", "the culprit", "mystery solved"
]


def call_grok_think(prompt: str, max_tokens: int = 2500) -> dict:
    """Call Grok 3 with extended reasoning for deep problem analysis."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Intellectual DNA Problem Resolution v2"
    }

    system_prompt = """You are analyzing conversation messages for COMPLETE problem→resolution chains.

Your task is to find debugging journeys where:
1. A problem was clearly identified (symptom)
2. Investigation happened (what was tried)
3. Root cause was discovered (the actual issue)
4. A fix was applied (resolution)
5. Success was confirmed (it worked)

For each chain, extract:
- The EXACT "aha moment" quote if present
- The debugging approach used
- How long it likely took to resolve
- What category of problem it was

Be THOROUGH. A single month may have many debugging sessions.
Focus on COMPLETE chains where you can trace symptom→cause→fix→confirmation.

Return valid JSON with this structure:
{
  "resolution_chains": [
    {
      "symptom": "what appeared broken",
      "investigation": "what was tried/examined",
      "root_cause": "the actual underlying issue",
      "resolution": "what fixed it",
      "aha_quote": "exact quote of realization moment",
      "domain": "code|config|data|logic|integration|performance|dependency|environment",
      "difficulty": "trivial|moderate|deep|nightmare",
      "time_to_resolve": "minutes|hours|day|days|week"
    }
  ],
  "debugging_patterns": ["list of recurring debugging approaches used"],
  "hardest_problem": "brief description of the most difficult issue this month",
  "total_problems_solved": 0
}"""

    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        # Request JSON output
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=90)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except json.JSONDecodeError as e:
        return {"resolution_chains": [], "error": f"JSON: {str(e)[:50]}"}
    except Exception as e:
        return {"resolution_chains": [], "error": str(e)[:50]}


def build_problem_resolution_v2(limit: int = None, budget: float = 5.0):
    """Build problem resolution with MAXED OUT Grok 3."""
    print("="*70)
    print("🔍 PROBLEM_RESOLUTION v2 - MAXED OUT GROK 3")
    print("="*70)
    print(f"  Model: {MODEL}")
    print(f"  Features: Think mode, JSON output, 1M context")
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
          AND LENGTH(c.full_content) BETWEEN 20 AND 2000
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

    # Filter to months with resolution signals
    def has_resolution_signals(msgs):
        text = ' '.join(msgs).lower()
        problem = any(w in text for w in ['bug', 'error', 'not working', 'issue', 'broken', 'fail'])
        solution = any(w in text for w in ['fixed', 'solved', 'working', 'it worked', 'turns out'])
        return problem and solution

    to_process = [(m, msgs, c) for m, msgs, c in monthly_messages
                  if str(m) not in processed_months and has_resolution_signals(msgs)]

    if limit:
        to_process = to_process[:limit]

    print(f"  Months with resolution signals: {len(to_process)}")

    if not to_process:
        print("  Nothing to process!")
        return

    results = []
    total_cost = 0.0
    total_chains = 0
    start_time = time.time()

    for i, (month_start, messages, msg_count) in enumerate(to_process):
        if total_cost >= budget:
            print(f"\n  💰 Budget exhausted at ${total_cost:.4f}")
            break

        # Prioritize messages with resolution triggers
        resolution_msgs = [m for m in messages if any(t in m.lower() for t in RESOLUTION_TRIGGERS)]
        other_msgs = [m for m in messages if m not in resolution_msgs]

        # Take up to 120 resolution messages + 30 context
        sample = resolution_msgs[:120] + other_msgs[:30]

        prompt = f"""MONTH: {month_start.strftime('%B %Y')}
TOTAL MESSAGES: {msg_count}
MESSAGES WITH PROBLEM/RESOLUTION SIGNALS: {len(resolution_msgs)}

Find ALL complete debugging journeys in these messages.
A complete journey has: symptom → investigation → root cause → fix → confirmation.

MESSAGES:
{"="*50}
""" + "\n---\n".join(sample)

        result = call_grok_think(prompt)
        chains = result.get("resolution_chains", [])
        total_chains += len(chains)

        # Calculate cost
        input_chars = len(prompt)
        output_chars = len(json.dumps(result))
        input_tokens = input_chars / 4
        output_tokens = output_chars / 4
        cost = (input_tokens * COST_PER_1M_INPUT / 1_000_000) + (output_tokens * COST_PER_1M_OUTPUT / 1_000_000)
        total_cost += cost

        # Extract domains and difficulties
        domains = [c.get("domain", "unknown") for c in chains]
        difficulties = [c.get("difficulty", "unknown") for c in chains]
        aha_quotes = [c.get("aha_quote", "") for c in chains if c.get("aha_quote")]

        results.append({
            "month_start": str(month_start),
            "chains_json": json.dumps(chains),
            "chain_count": len(chains),
            "domains": json.dumps(list(set(domains))),
            "difficulties": json.dumps(list(set(difficulties))),
            "debugging_patterns": json.dumps(result.get("debugging_patterns", [])),
            "hardest_problem": result.get("hardest_problem", ""),
            "aha_quotes": json.dumps(aha_quotes[:5]),
            "message_count": msg_count,
            "processed_at": datetime.now().isoformat()
        })

        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
        hardest = result.get("hardest_problem", "")[:40]

        print(f"  [{i+1}/{len(to_process)}] {month_start.strftime('%Y-%m')} | {len(chains)} chains | ${cost:.4f} | {rate:.1f}/min")
        if hardest:
            print(f"       Hardest: {hardest}...")

        time.sleep(3)  # Rate limit for Grok

    if results:
        con.execute("""
            CREATE TABLE problem_resolution_v2 (
                month_start DATE,
                chains_json VARCHAR,
                chain_count INTEGER,
                domains VARCHAR,
                difficulties VARCHAR,
                debugging_patterns VARCHAR,
                hardest_problem VARCHAR,
                aha_quotes VARCHAR,
                message_count INTEGER,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO problem_resolution_v2 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       [r["month_start"], r["chains_json"], r["chain_count"],
                        r["domains"], r["difficulties"], r["debugging_patterns"],
                        r["hardest_problem"], r["aha_quotes"], r["message_count"], r["processed_at"]])

        if output_path.exists():
            con.execute(f"INSERT INTO problem_resolution_v2 SELECT * FROM '{output_path}'")

        con.execute(f"COPY problem_resolution_v2 TO '{output_path}' (FORMAT PARQUET)")

        # Summary
        print("\n" + "="*70)
        print("📊 EXTRACTION COMPLETE")
        print("="*70)
        print(f"  Months processed: {len(results)}")
        print(f"  Total resolution chains: {total_chains}")
        print(f"  Total cost: ${total_cost:.4f}")

        # Domain distribution
        all_domains = []
        for r in results:
            all_domains.extend(json.loads(r["domains"]))
        domain_dist = Counter(all_domains).most_common()
        print(f"\n  🔧 Problem Domains:")
        for domain, count in domain_dist:
            print(f"     {domain}: {count}")

        # Difficulty distribution
        all_diff = []
        for r in results:
            for c in json.loads(r["chains_json"]):
                if c.get("difficulty"):
                    all_diff.append(c["difficulty"])
        diff_dist = Counter(all_diff).most_common()
        print(f"\n  📈 Difficulty Distribution:")
        for diff, count in diff_dist:
            pct = count / len(all_diff) * 100 if all_diff else 0
            print(f"     {diff}: {count} ({pct:.0f}%)")

        # Best aha moments
        print(f"\n  💡 Sample 'Aha' Moments:")
        for r in results[-6:]:
            ahas = json.loads(r["aha_quotes"])
            if ahas:
                print(f"     {r['month_start'][:7]}: \"{ahas[0][:60]}...\"")

    config = {
        "name": "problem_resolution/v2",
        "description": "MAXED OUT problem resolution with Grok 3 Think mode",
        "model": MODEL,
        "features": ["grok_think", "json_output", "1M_context", "monthly"],
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
        build_problem_resolution_v2(limit=3, budget=1.0)
    else:
        build_problem_resolution_v2(limit=args.limit, budget=args.budget)
