#!/usr/bin/env python3
"""
Build phrase_context interpretation: Add meaning context to signature phrases.

Uses Gemini 2.5 Flash Lite to analyze the context and meaning
of extracted signature phrases.
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
INTERP_DIR = DATA_DIR / "interpretations" / "phrase_context" / "v1"
PHRASES_PATH = DATA_DIR / "interpretations" / "phrases" / "v1" / "phrases.parquet"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash-lite"

REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE


def call_gemini(prompt: str, max_tokens: int = 300) -> str:
    """Call Gemini via OpenRouter API."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Intellectual DNA Phrase Context"
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """Analyze this recurring phrase from someone's conversations to understand its meaning and usage.

Provide:
1. Likely meaning/function of this phrase
2. Context category: thinking-aloud, emphasis, filler, technical, emotional, instruction, question
3. Whether it's a verbal tic (habitual) or meaningful (carries information)
4. One sentence describing how this phrase reveals the speaker's style

Output JSON: {"meaning": "...", "category": "...", "type": "habitual|meaningful", "style_insight": "..."}"""
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


def parse_context(llm_output: str) -> dict:
    """Parse LLM output for phrase context."""
    try:
        import re
        match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}
    except:
        return {}


def get_phrase_examples(con, phrase: str, limit: int = 5) -> list:
    """Get example usages of a phrase from conversations."""
    try:
        results = con.execute(f"""
            SELECT LEFT(full_content, 300) as excerpt
            FROM '{FACTS_DIR}/brain/content.parquet'
            WHERE LOWER(full_content) LIKE '%{phrase.replace("'", "''")}%'
              AND event_type = 'message'
              AND LENGTH(full_content) < 500
            LIMIT {limit}
        """).fetchall()
        return [r[0] for r in results]
    except:
        return []


def build_phrase_context(limit: int = None):
    """Add context to signature phrases."""
    print("Building phrase_context/v1 interpretation...")
    print(f"  Model: {MODEL}")

    if not PHRASES_PATH.exists():
        print(f"  ERROR: {PHRASES_PATH} not found - run phrases/v1 first")
        return

    if not OPENROUTER_API_KEY:
        print("  ERROR: OPENROUTER_API_KEY not found")
        return

    con = duckdb.connect()
    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERP_DIR / "contexts.parquet"

    # Check existing
    processed_phrases = set()
    if output_path.exists():
        existing = con.execute(f"SELECT phrase FROM '{output_path}'").fetchall()
        processed_phrases = {row[0] for row in existing}
        print(f"  Already processed: {len(processed_phrases)} phrases")

    # Get top phrases (trigrams and fourgrams are most meaningful)
    phrases = con.execute(f"""
        SELECT phrase, count, ngram_size
        FROM '{PHRASES_PATH}'
        WHERE ngram_size >= 3
          AND count >= 10
        ORDER BY count DESC
        LIMIT 200
    """).fetchall()

    to_process = [(p, c, n) for p, c, n in phrases if p not in processed_phrases]

    if limit:
        to_process = to_process[:limit]

    print(f"  Phrases to process: {len(to_process)}")

    if not to_process:
        print("  Nothing to process!")
        return

    results = []
    total_cost = 0.0
    start_time = time.time()

    for i, (phrase, count, ngram_size) in enumerate(to_process):
        # Get example usages
        examples = get_phrase_examples(con, phrase, limit=3)
        example_text = "\n".join([f'- "{ex}"' for ex in examples]) if examples else "No examples available"

        prompt = f"""Phrase: "{phrase}"
Times used: {count}

Example usages:
{example_text}

What does this phrase reveal about the speaker?"""

        response = call_gemini(prompt)
        parsed = parse_context(response)

        results.append({
            "phrase": phrase,
            "count": count,
            "ngram_size": ngram_size,
            "meaning": parsed.get("meaning", ""),
            "category": parsed.get("category", "unknown"),
            "phrase_type": parsed.get("type", "unknown"),
            "style_insight": parsed.get("style_insight", ""),
            "processed_at": datetime.now().isoformat()
        })

        input_tokens = len(prompt) / 4
        output_tokens = len(response) / 4
        cost = (input_tokens * 0.075 / 1_000_000) + (output_tokens * 0.30 / 1_000_000)
        total_cost += cost

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{i+1}/{len(to_process)}] \"{phrase}\" - {rate:.1f} req/min - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if results:
        con.execute("""
            CREATE TABLE phrase_contexts (
                phrase VARCHAR,
                count INTEGER,
                ngram_size INTEGER,
                meaning VARCHAR,
                category VARCHAR,
                phrase_type VARCHAR,
                style_insight VARCHAR,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO phrase_contexts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       [r["phrase"], r["count"], r["ngram_size"], r["meaning"],
                        r["category"], r["phrase_type"], r["style_insight"], r["processed_at"]])

        if output_path.exists():
            con.execute(f"INSERT INTO phrase_contexts SELECT * FROM '{output_path}'")

        con.execute(f"COPY phrase_contexts TO '{output_path}' (FORMAT PARQUET)")

        print(f"\n  Processed: {len(results)} phrases")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Wrote to: {output_path}")

        # Show sample insights
        print("\n  Sample insights:")
        for r in results[:5]:
            print(f"    \"{r['phrase']}\": {r['style_insight'][:80]}...")

    config = {
        "name": "phrase_context/v1",
        "description": "LLM-analyzed phrase meanings and style insights",
        "model": MODEL,
        "depends_on": ["phrases/v1"],
        "created": datetime.now().strftime("%Y-%m-%d")
    }
    with open(INTERP_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return len(results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Max phrases')
    parser.add_argument('--test', action='store_true', help='Test with 10 phrases')
    args = parser.parse_args()

    if args.test:
        build_phrase_context(limit=10)
    else:
        build_phrase_context(limit=args.limit)
