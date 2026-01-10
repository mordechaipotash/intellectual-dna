#!/usr/bin/env python3
"""Direction 2: Test Gemini classification on unclassified docs."""

import os
import json
import requests
import duckdb
from pathlib import Path

# Config
MODEL = "google/gemini-3-flash-preview"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
PARQUET_PATH = Path.home() / "intellectual_dna/data/facts/markdown_files_v3.parquet"

# Categories that Direction 1 didn't catch
CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": [
                "JEWISH_PHILOSOPHY",      # Mussar, hashkafa, machshava
                "MISHNAH_COMMENTARY",     # Bartenura, Rambam on Mishnah
                "YOUTUBE_TORAH",          # Torah video transcripts
                "EXPENSE_TRACKING",       # Financial records
                "PERSONAL_REFLECTION",    # Journal, thoughts
                "TECHNICAL_DOC",          # Code, APIs, dev
                "UNKNOWN"                 # Can't determine
            ]
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
        },
        "reasoning": {
            "type": "string"
        }
    },
    "required": ["category", "confidence", "reasoning"]
}

def classify_doc(filename: str, content_preview: str) -> dict:
    """Classify a single document using Gemini."""
    prompt = f"""Classify this document into one category.

Filename: {filename}
Content (first 500 chars):
{content_preview[:500]}

Categories:
- JEWISH_PHILOSOPHY: Mussar texts, hashkafa, ethical works (Nefesh HaChayim, Eight Chapters, etc.)
- MISHNAH_COMMENTARY: Commentary on Mishnah (Bartenura, Tosafot Yom Tov, etc.)
- YOUTUBE_TORAH: Torah video transcripts, Q&A sessions with rabbis
- EXPENSE_TRACKING: Financial records, receipts, billing
- PERSONAL_REFLECTION: Personal journals, thoughts, reflections
- TECHNICAL_DOC: Code documentation, APIs, development
- UNKNOWN: Can't determine with confidence

Respond with JSON only."""

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "classification",
                    "schema": CLASSIFICATION_SCHEMA
                }
            },
            "temperature": 0.1
        },
        timeout=30
    )

    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return json.loads(content)
    else:
        return {"error": response.text}

def main():
    # Sample 10 unclassified docs
    con = duckdb.connect()
    query = f"""
    SELECT filename, LEFT(content, 500) as preview
    FROM '{PARQUET_PATH}'
    WHERE NOT (
        first_line LIKE 'conversation: %' OR first_line LIKE 'job_title: %' OR
        first_line LIKE 'Brand: %' OR first_line LIKE 'Daf: %' OR
        domain = 'claude_code' OR domain = 'Mordechai Dev 2025'
    )
    ORDER BY RANDOM()
    LIMIT 10
    """

    docs = con.execute(query).fetchall()

    print(f"Testing Gemini classification on {len(docs)} docs...\n")

    results = []
    for filename, preview in docs:
        print(f"Classifying: {filename[:50]}...")
        result = classify_doc(filename, preview)
        result["filename"] = filename
        results.append(result)

        if "error" not in result:
            print(f"  → {result['category']} ({result['confidence']:.0%})")
        else:
            print(f"  → ERROR: {result['error'][:50]}")

    # Summary
    print("\n" + "="*50)
    print("DIRECTION 2 RESULTS:")
    categories = {}
    for r in results:
        cat = r.get("category", "ERROR")
        categories[cat] = categories.get(cat, 0) + 1

    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

if __name__ == "__main__":
    main()
