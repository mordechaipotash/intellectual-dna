#!/usr/bin/env python3
"""
Enrich signature phrases with meaning and style insights using OpenRouter
Run after build_signature_phrases_v2.py
"""

import os
import json
import requests
import duckdb
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import pyarrow as pa
import pyarrow.parquet as pq

load_dotenv(Path("/Users/mordechai/intellectual_dna/.env"))

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
PHRASES_PATH = BASE_DIR / "data/interpretations/signature_phrases/v2/phrases.parquet"
OUTPUT_PATH = BASE_DIR / "data/interpretations/signature_phrases/v2/enriched.parquet"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3-flash-preview"  # Always use Gemini 3 Flash

# JSON Schema for structured output
PHRASE_SCHEMA = {
    "type": "object",
    "properties": {
        "meaning": {
            "type": "string",
            "description": "What this phrase typically means when the user says it (1-2 sentences)"
        },
        "style_insight": {
            "type": "string",
            "description": "What frequent use of this phrase reveals about their communication style (1-2 sentences)"
        }
    },
    "required": ["meaning", "style_insight"]
}


def call_openrouter(phrase: str, count: int, category: str) -> dict:
    """Call OpenRouter API to enrich a phrase."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Signature Phrases Enrichment"
    }

    prompt = f"""Analyze this recurring phrase from a user's AI conversations:

Phrase: "{phrase}"
Frequency: {count} times
Category: {category}

What does this phrase mean when they use it, and what does it reveal about their style?"""

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You analyze recurring phrases from conversation history. Be specific and insightful. Focus on what the phrase reveals about communication patterns."
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.3,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "phrase_analysis",
                "strict": True,
                "schema": PHRASE_SCHEMA
            }
        }
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        return {"meaning": None, "style_insight": None, "error": str(e)[:50]}


def run_enrichment(top_n: int = 50):
    """Enrich top N phrases."""
    print("=" * 60)
    print("SIGNATURE PHRASES ENRICHMENT (OpenRouter)")
    print("=" * 60)

    if not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY not found in .env")
        return

    con = duckdb.connect()

    # Load phrases (exclude 'general' category which has noise)
    print("\n📥 Loading phrases...")
    results = con.execute(f"""
        SELECT phrase, count, category, ngram_size, first_word
        FROM '{PHRASES_PATH}'
        WHERE category <> 'general'
        ORDER BY count DESC
        LIMIT {top_n}
    """).fetchall()

    print(f"   Loaded {len(results)} phrases for enrichment")

    # Enrich each phrase
    print("\n🤖 Enriching with Gemini 2.0 Flash...")
    enriched = []

    for i, (phrase, count, category, ngram_size, first_word) in enumerate(results):
        print(f"  [{i+1}/{len(results)}] \"{phrase}\" ({count}x)...", end=" ")

        result = call_openrouter(phrase, count, category)

        enriched.append({
            'phrase': phrase,
            'count': count,
            'category': category,
            'ngram_size': ngram_size,
            'first_word': first_word,
            'meaning': result.get('meaning'),
            'style_insight': result.get('style_insight'),
        })

        if result.get('meaning'):
            print("✓")
        else:
            print(f"✗ {result.get('error', 'unknown')}")

    # Count successes
    with_meaning = sum(1 for p in enriched if p.get('meaning'))
    print(f"\n✅ Enriched {with_meaning}/{len(enriched)} phrases")

    # Save enriched data
    table = pa.table({
        'phrase': [p['phrase'] for p in enriched],
        'count': [p['count'] for p in enriched],
        'category': [p['category'] for p in enriched],
        'ngram_size': [p['ngram_size'] for p in enriched],
        'first_word': [p['first_word'] for p in enriched],
        'meaning': [p.get('meaning') for p in enriched],
        'style_insight': [p.get('style_insight') for p in enriched],
        'processed_at': [datetime.now().isoformat() for _ in enriched],
    })

    pq.write_table(table, OUTPUT_PATH)
    print(f"\n💾 Saved to: {OUTPUT_PATH}")

    # Show sample
    print("\n" + "=" * 60)
    print("SAMPLE ENRICHED PHRASES")
    print("=" * 60)
    for p in enriched[:8]:
        print(f"\n\"{p['phrase']}\" ({p['count']}x, {p['category']})")
        if p.get('meaning'):
            print(f"  📝 {p['meaning']}")
        if p.get('style_insight'):
            print(f"  💡 {p['style_insight']}")


if __name__ == "__main__":
    import sys
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_enrichment(top_n)
