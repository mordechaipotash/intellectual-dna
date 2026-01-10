#!/usr/bin/env python3
"""
Build signature_phrases interpretation: Extract unique expressions and verbal fingerprints.

Uses Gemini 2.5 Flash Lite to identify recurring phrases, unique expressions,
and verbal patterns that define Mordechai's communication style.
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
INTERP_DIR = DATA_DIR / "interpretations" / "signature_phrases" / "v1"

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
        "X-Title": "Intellectual DNA Signature Phrases"
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """Identify signature phrases and unique expressions in these messages.
Look for:
1. Recurring phrases (used multiple times)
2. Unique metaphors or framings
3. Verbal tics or patterns
4. Coined terms or personal vocabulary
5. Characteristic sentence structures

Write in FIRST PERSON as if you are the speaker reflecting on your own patterns.

Output JSON array: [{"phrase": "...", "category": "metaphor|coined_term|verbal_tic|framing|catchphrase", "frequency": "high|medium|low", "meaning": "what I mean when I say this", "example_context": "brief context"}]

Return 3-7 phrases per batch. Focus on distinctive patterns, not common expressions."""
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


def parse_phrases(llm_output: str) -> list:
    """Parse LLM output for phrase extraction."""
    import re
    try:
        match = re.search(r'\[.*\]', llm_output, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []
    except:
        return []


def build_signature_phrases(limit: int = None):
    """Build signature phrase extraction."""
    print("Building signature_phrases/v1 interpretation...")
    print(f"  Model: {MODEL}")

    if not OPENROUTER_API_KEY:
        print("  ERROR: OPENROUTER_API_KEY not found")
        return

    con = duckdb.connect()
    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERP_DIR / "phrases.parquet"

    # Get weekly message batches for phrase analysis
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
          AND LENGTH(c.full_content) BETWEEN 20 AND 500
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

    all_phrases = {}  # phrase -> {data}
    total_cost = 0.0
    start_time = time.time()

    for i, (week_start, messages, msg_count) in enumerate(to_process):
        # Sample messages from the week
        sample = messages[:30] if len(messages) > 30 else messages
        sample_text = "\n---\n".join(sample)

        prompt = f"""Week of {week_start}
{msg_count} total messages

Sample messages:
{sample_text}

Identify my signature phrases and unique expressions from this week."""

        response = call_gemini(prompt)
        parsed = parse_phrases(response)

        for phrase_data in parsed:
            phrase = phrase_data.get("phrase", "")
            if phrase and len(phrase) > 3:
                if phrase in all_phrases:
                    all_phrases[phrase]["occurrences"] += 1
                    all_phrases[phrase]["weeks"].append(str(week_start))
                else:
                    all_phrases[phrase] = {
                        "phrase": phrase,
                        "category": phrase_data.get("category", "unknown"),
                        "frequency": phrase_data.get("frequency", "low"),
                        "meaning": phrase_data.get("meaning", ""),
                        "example_context": phrase_data.get("example_context", ""),
                        "occurrences": 1,
                        "weeks": [str(week_start)],
                        "first_seen": str(week_start)
                    }

        input_tokens = len(prompt) / 4
        output_tokens = len(response) / 4
        cost = (input_tokens * 0.075 / 1_000_000) + (output_tokens * 0.30 / 1_000_000)
        total_cost += cost

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{i+1}/{len(to_process)}] {week_start} - {len(all_phrases)} phrases - {rate:.1f} req/min - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if all_phrases:
        # Convert to list and sort by occurrences
        results = sorted(all_phrases.values(), key=lambda x: -x["occurrences"])

        con.execute("""
            CREATE TABLE signature_phrases (
                phrase VARCHAR,
                category VARCHAR,
                frequency VARCHAR,
                meaning VARCHAR,
                example_context VARCHAR,
                occurrences INTEGER,
                weeks VARCHAR,
                first_seen DATE,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO signature_phrases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       [r["phrase"], r["category"], r["frequency"], r["meaning"],
                        r["example_context"], r["occurrences"], json.dumps(r["weeks"]),
                        r["first_seen"], datetime.now().isoformat()])

        con.execute(f"COPY signature_phrases TO '{output_path}' (FORMAT PARQUET)")

        print(f"\n  Unique phrases found: {len(results)}")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Wrote to: {output_path}")

        # Show top phrases
        print("\n  Top 10 signature phrases:")
        for r in results[:10]:
            print(f"    [{r['occurrences']}x] \"{r['phrase']}\" ({r['category']})")

    config = {
        "name": "signature_phrases/v1",
        "description": "Unique expressions and verbal fingerprints",
        "model": MODEL,
        "voice": "first-person",
        "created": datetime.now().strftime("%Y-%m-%d")
    }
    with open(INTERP_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return len(all_phrases)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Max weeks')
    parser.add_argument('--test', action='store_true', help='Test with 5 weeks')
    args = parser.parse_args()

    if args.test:
        build_signature_phrases(limit=5)
    else:
        build_signature_phrases(limit=args.limit)
