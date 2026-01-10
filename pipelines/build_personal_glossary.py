#!/usr/bin/env python3
"""
Build personal_glossary interpretation: Define personal terms and concepts.

Uses Gemini 2.5 Flash Lite to identify and define terms/concepts
that have personal meanings in Mordechai's vocabulary.
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
INTERP_DIR = DATA_DIR / "interpretations" / "glossary" / "v1"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash-lite"

REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE

# Seed terms to look for (from SEED principles and common patterns)
SEED_TERMS = [
    "bottleneck", "compression", "inversion", "agency", "seeds",
    "translation", "temporal", "cognitive architecture", "monotropic",
    "prosthetic", "intellectual dna", "focus tunnel", "flow state",
    "leverage point", "feedback loop", "mental model", "framework"
]


def call_gemini(prompt: str, max_tokens: int = 400) -> str:
    """Call Gemini via OpenRouter API."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Intellectual DNA Personal Glossary"
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """Analyze how this person uses a specific term/concept based on their messages.

Provide:
1. Their personal definition (how THEY use it, not dictionary definition)
2. Related terms they associate with it
3. Whether it's: technical, philosophical, personal-shorthand, borrowed-concept
4. Confidence level: high (clear pattern), medium (some evidence), low (sparse data)

Output JSON: {"definition": "...", "related_terms": ["...", "..."], "type": "...", "confidence": "...", "example_usage": "brief quote showing usage"}"""
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


def parse_glossary(llm_output: str) -> dict:
    """Parse LLM output for glossary entry."""
    try:
        import re
        match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}
    except:
        return {}


def find_candidate_terms(con) -> list:
    """Find candidate terms that might have personal meanings."""
    # Look for capitalized concepts, repeated unique terms
    results = con.execute(f"""
        WITH word_counts AS (
            SELECT
                LOWER(word) as term,
                COUNT(*) as uses
            FROM (
                SELECT UNNEST(STRING_SPLIT(full_content, ' ')) as word
                FROM '{FACTS_DIR}/brain/content.parquet'
                WHERE event_type = 'message'
                  AND LENGTH(full_content) < 500
            )
            WHERE LENGTH(word) >= 5
              AND word NOT LIKE '%/%'
              AND word NOT LIKE '%@%'
              AND word NOT LIKE '%:%'
            GROUP BY LOWER(word)
            HAVING COUNT(*) >= 20
        )
        SELECT term, uses
        FROM word_counts
        ORDER BY uses DESC
        LIMIT 100
    """).fetchall()

    # Combine with seed terms
    found_terms = [(t, c) for t, c in results]
    for term in SEED_TERMS:
        if term not in [t for t, c in found_terms]:
            found_terms.append((term, 0))

    return found_terms


def get_term_examples(con, term: str, limit: int = 8) -> list:
    """Get example usages of a term."""
    try:
        results = con.execute(f"""
            SELECT LEFT(full_content, 400) as excerpt
            FROM '{FACTS_DIR}/brain/content.parquet'
            WHERE LOWER(full_content) LIKE '%{term.replace("'", "''")}%'
              AND event_type = 'message'
              AND LENGTH(full_content) BETWEEN 50 AND 800
            LIMIT {limit}
        """).fetchall()
        return [r[0] for r in results]
    except:
        return []


def build_personal_glossary(limit: int = None):
    """Build personal glossary from conversation patterns."""
    print("Building glossary/v1 interpretation...")
    print(f"  Model: {MODEL}")

    if not OPENROUTER_API_KEY:
        print("  ERROR: OPENROUTER_API_KEY not found")
        return

    con = duckdb.connect()
    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERP_DIR / "terms.parquet"

    # Check existing
    processed_terms = set()
    if output_path.exists():
        existing = con.execute(f"SELECT term FROM '{output_path}'").fetchall()
        processed_terms = {row[0] for row in existing}
        print(f"  Already processed: {len(processed_terms)} terms")

    # Get candidate terms
    candidates = find_candidate_terms(con)
    to_process = [(t, c) for t, c in candidates if t not in processed_terms]

    # Prioritize seed terms
    seed_set = set(SEED_TERMS)
    to_process.sort(key=lambda x: (x[0] not in seed_set, -x[1]))

    if limit:
        to_process = to_process[:limit]

    print(f"  Terms to process: {len(to_process)}")

    if not to_process:
        print("  Nothing to process!")
        return

    results = []
    total_cost = 0.0
    start_time = time.time()

    for i, (term, count) in enumerate(to_process):
        examples = get_term_examples(con, term, limit=6)
        if len(examples) < 2:
            continue  # Skip terms with too few examples

        example_text = "\n---\n".join(examples)
        prompt = f"""Term: "{term}"
Approximate uses: {count}

Example messages containing this term:
{example_text}

How does this person define/use "{term}"?"""

        response = call_gemini(prompt)
        parsed = parse_glossary(response)

        if parsed.get("definition"):
            results.append({
                "term": term,
                "use_count": count,
                "definition": parsed.get("definition", ""),
                "related_terms": json.dumps(parsed.get("related_terms", [])),
                "term_type": parsed.get("type", "unknown"),
                "confidence": parsed.get("confidence", "unknown"),
                "example_usage": parsed.get("example_usage", ""),
                "processed_at": datetime.now().isoformat()
            })

        input_tokens = len(prompt) / 4
        output_tokens = len(response) / 4
        cost = (input_tokens * 0.075 / 1_000_000) + (output_tokens * 0.30 / 1_000_000)
        total_cost += cost

        if (i + 1) % 15 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{i+1}/{len(to_process)}] \"{term}\" - {rate:.1f} req/min - ${total_cost:.4f}")

        time.sleep(REQUEST_DELAY)

    if results:
        con.execute("""
            CREATE TABLE glossary (
                term VARCHAR,
                use_count INTEGER,
                definition VARCHAR,
                related_terms VARCHAR,
                term_type VARCHAR,
                confidence VARCHAR,
                example_usage VARCHAR,
                processed_at VARCHAR
            )
        """)

        for r in results:
            con.execute("INSERT INTO glossary VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       [r["term"], r["use_count"], r["definition"], r["related_terms"],
                        r["term_type"], r["confidence"], r["example_usage"], r["processed_at"]])

        if output_path.exists():
            con.execute(f"INSERT INTO glossary SELECT * FROM '{output_path}'")

        con.execute(f"COPY glossary TO '{output_path}' (FORMAT PARQUET)")

        print(f"\n  Processed: {len(results)} terms with definitions")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Wrote to: {output_path}")

        # Show sample definitions
        print("\n  Sample definitions:")
        for r in [x for x in results if x["confidence"] == "high"][:3]:
            print(f"    {r['term']}: {r['definition'][:80]}...")

    config = {
        "name": "glossary/v1",
        "description": "Personal term definitions from usage patterns",
        "model": MODEL,
        "seed_terms": SEED_TERMS,
        "created": datetime.now().strftime("%Y-%m-%d")
    }
    with open(INTERP_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return len(results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Max terms')
    parser.add_argument('--test', action='store_true', help='Test with 10 terms')
    args = parser.parse_args()

    if args.test:
        build_personal_glossary(limit=10)
    else:
        build_personal_glossary(limit=args.limit)
