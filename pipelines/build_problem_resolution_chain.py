#!/usr/bin/env python3
"""
Build PROBLEM_RESOLUTION_CHAIN interpretation: Track issue→solution patterns.
Extracts: debugging journeys, root cause discoveries, "aha" moments, fix patterns.

Model: Grok 3 Mini - Excellent at reasoning and problem-solving analysis
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
INTERP_DIR = DATA_DIR / "interpretations" / "problem_resolution_chain" / "v1"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "x-ai/grok-3-mini-beta"

# Grok 3 Mini: $3.00/1M input, $15.00/1M output
COST_PER_1M_INPUT = 3.00
COST_PER_1M_OUTPUT = 15.00

REQUESTS_PER_MINUTE = 25
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE

# Problem resolution triggers from 100x deep mining
RESOLUTION_TRIGGERS = [
    "fix", "bug", "not working", "error", "the issue was",
    "turns out", "it worked", "working now", "finally",
    "solved", "figured out", "the problem was", "root cause",
    "aha", "that's why", "realized", "debugging", "traced",
    "found it", "got it working", "the culprit", "mystery solved"
]


def call_grok(prompt: str, max_tokens: int = 700) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Intellectual DNA Problem Resolution"
    }
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """Analyze these messages for PROBLEM→RESOLUTION chains.

Look for complete debugging/problem-solving journeys:
- Initial symptom: what appeared broken or wrong
- Investigation: what was tried or examined
- Root cause: the underlying issue discovered
- Resolution: how it was fixed
- "Aha" moment: the breakthrough insight

For each resolution chain found:
SYMPTOM: [what appeared wrong]
ROOT_CAUSE: [actual underlying issue]
RESOLUTION: [what fixed it]
AHA_MOMENT: [the key insight, if present]
DOMAIN: [code|config|data|logic|integration|performance|ui]
DIFFICULTY: [trivial|moderate|deep|nightmare]

Focus on COMPLETE resolution chains where:
1. Problem was clearly identified
2. Cause was discovered
3. Fix was applied
4. Success was confirmed

Skip ongoing/unresolved issues.
If no complete resolutions found: "No complete resolution chains this period."
Quote the actual language used when describing the aha moment."""
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


def parse_resolution_chains(text: str) -> list:
    """Parse problem resolution chains from LLM output."""
    chains = []
    lines = text.split('\n')

    current = {}
    for line in lines:
        line = line.strip()
        if line.startswith('SYMPTOM:'):
            if current and 'symptom' in current:
                chains.append(current)
            current = {'symptom': line[8:].strip()}
        elif line.startswith('ROOT_CAUSE:') and current:
            current['root_cause'] = line[11:].strip()
        elif line.startswith('RESOLUTION:') and current:
            current['resolution'] = line[11:].strip()
        elif line.startswith('AHA_MOMENT:') and current:
            current['aha_moment'] = line[11:].strip()
        elif line.startswith('DOMAIN:') and current:
            current['domain'] = line[7:].strip().lower()
        elif line.startswith('DIFFICULTY:') and current:
            current['difficulty'] = line[11:].strip().lower()

    if current and 'symptom' in current and 'resolution' in current:
        chains.append(current)

    return chains


def has_resolution_triggers(messages: list) -> bool:
    """Check if messages contain resolution trigger phrases."""
    text = ' '.join(messages).lower()
    # Need both problem indicators AND resolution indicators
    problem_words = ['bug', 'error', 'not working', 'issue', 'problem', 'broken', 'fail']
    resolution_words = ['fixed', 'solved', 'working', 'it worked', 'turns out', 'figured out']

    has_problem = any(w in text for w in problem_words)
    has_resolution = any(w in text for w in resolution_words)

    return has_problem and has_resolution


def extract_resolution_signals(messages: list) -> dict:
    """Extract problem/resolution signal counts."""
    text = ' '.join(messages).lower()
    signals = {}
    for trigger in RESOLUTION_TRIGGERS:
        count = text.count(trigger)
        if count > 0:
            signals[trigger] = count
    return signals


def build_problem_resolution_chain(limit: int = None, budget: float = 5.0):
    """Build problem resolution chain interpretation."""
    print("Building problem_resolution_chain/v1 interpretation...")
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
          AND LENGTH(c.full_content) BETWEEN 30 AND 1500
        GROUP BY DATE_TRUNC('week', idx.timestamp::TIMESTAMP)
        HAVING COUNT(*) >= 15
        ORDER BY week_start
    """).fetchall()

    # Filter to weeks with resolution signals and not yet processed
    to_process = [(w, m, c) for w, m, c in weekly_messages
                  if str(w) not in processed_weeks and has_resolution_triggers(m)]

    if limit:
        to_process = to_process[:limit]

    print(f"  Weeks with resolution signals: {len(to_process)}")

    if not to_process:
        print("  Nothing to process!")
        return

    results = []
    total_cost = 0.0
    total_chains = 0
    start_time = time.time()

    for i, (week_start, messages, msg_count) in enumerate(to_process):
        # Check budget
        if total_cost >= budget:
            print(f"  Budget exhausted at ${total_cost:.4f}")
            break

        # Sample messages with resolution triggers preferentially
        resolution_msgs = [m for m in messages if any(t in m.lower() for t in RESOLUTION_TRIGGERS)]
        other_msgs = [m for m in messages if m not in resolution_msgs]
        sample = resolution_msgs[:40] + other_msgs[:10]

        # Add context about signals found
        signals = extract_resolution_signals(messages)
        top_signals = sorted(signals.items(), key=lambda x: -x[1])[:5]
        signal_str = ', '.join(f'"{k}"({v})' for k, v in top_signals)

        prompt = f"Week of {week_start}\nResolution signals detected: {signal_str}\n\n" + "\n---\n".join(sample[:50])

        response = call_grok(prompt)
        chains = parse_resolution_chains(response)
        total_chains += len(chains)

        # Extract domains for summary
        domains = list(set(c.get('domain', 'unknown') for c in chains if c.get('domain')))
        difficulties = list(set(c.get('difficulty', 'unknown') for c in chains if c.get('difficulty')))

        # Extract aha moments
        aha_moments = [c.get('aha_moment', '') for c in chains if c.get('aha_moment')]

        results.append({
            "week_start": str(week_start),
            "chains_json": json.dumps(chains),
            "chain_count": len(chains),
            "domains": json.dumps(domains),
            "difficulties": json.dumps(difficulties),
            "aha_moments": json.dumps(aha_moments[:3]),  # Top 3
            "raw_response": response[:700],
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
            domains_str = ', '.join(domains[:3]) if domains else 'none'
            print(f"  [{i+1}/{len(to_process)}] {week_start} - {len(chains)} chains ({domains_str}) - {rate:.1f} req/min - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if results:
        con.execute("""
            CREATE TABLE problem_resolution_chain (
                week_start DATE,
                chains_json VARCHAR,
                chain_count INTEGER,
                domains VARCHAR,
                difficulties VARCHAR,
                aha_moments VARCHAR,
                raw_response VARCHAR,
                message_count INTEGER,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO problem_resolution_chain VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       [r["week_start"], r["chains_json"], r["chain_count"],
                        r["domains"], r["difficulties"], r["aha_moments"],
                        r["raw_response"], r["message_count"], r["processed_at"]])

        if output_path.exists():
            con.execute(f"INSERT INTO problem_resolution_chain SELECT * FROM '{output_path}'")

        con.execute(f"COPY problem_resolution_chain TO '{output_path}' (FORMAT PARQUET)")

        print(f"\n  Processed: {len(results)} weeks")
        print(f"  Total resolution chains: {total_chains}")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Remaining budget: ${budget - total_cost:.4f}")

        # Show domain frequency
        all_domains = []
        for r in results:
            all_domains.extend(json.loads(r["domains"]))
        domain_dist = Counter(all_domains).most_common()
        if domain_dist:
            print(f"\n  Problem domains:")
            for domain, count in domain_dist:
                print(f"    {domain}: {count} weeks")

        # Show difficulty distribution
        all_difficulties = []
        for r in results:
            all_difficulties.extend(json.loads(r["difficulties"]))
        diff_dist = Counter(all_difficulties).most_common()
        if diff_dist:
            print(f"\n  Difficulty distribution:")
            for diff, count in diff_dist:
                print(f"    {diff}: {count}")

        # Show sample aha moments
        print(f"\n  Sample \"Aha\" moments:")
        for r in results[-8:]:
            aha_list = json.loads(r["aha_moments"])
            if aha_list:
                aha = aha_list[0][:60]
                print(f"    {r['week_start'][:10]}: \"{aha}...\"")

        # Show sample resolution chains
        print(f"\n  Sample resolution chains:")
        for r in results[-5:]:
            chain_list = json.loads(r["chains_json"])
            if chain_list:
                first = chain_list[0]
                symptom = first.get('symptom', '?')[:30]
                resolution = first.get('resolution', '?')[:30]
                print(f"    {r['week_start'][:10]}: {symptom}... → {resolution}...")

    config = {
        "name": "problem_resolution_chain/v1",
        "description": "Problem→Resolution debugging journeys with aha moments",
        "model": MODEL,
        "triggers": RESOLUTION_TRIGGERS,
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
        build_problem_resolution_chain(limit=5, budget=0.50)
    else:
        build_problem_resolution_chain(limit=args.limit, budget=args.budget)
