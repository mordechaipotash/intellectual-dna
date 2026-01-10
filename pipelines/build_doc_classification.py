#!/usr/bin/env python3
"""
Build document classification layer for markdown harvest.
Hybrid approach: SQL rules (92%) + Gemini batch (8%).

Usage:
    python -m pipelines.build_doc_classification          # Dry run
    python -m pipelines.build_doc_classification --apply  # Apply to parquet
    python -m pipelines.build_doc_classification --gemini # Run Gemini on unclassified
"""

import os
import json
import argparse
import requests
import duckdb
from pathlib import Path
from datetime import datetime

# Config
BASE_DIR = Path.home() / "intellectual_dna"
PARQUET_PATH = BASE_DIR / "data/facts/markdown_files_v3.parquet"
OUTPUT_PATH = BASE_DIR / "data/facts/markdown_files_v4.parquet"
GEMINI_CACHE = BASE_DIR / "data/interpretations/doc_classification/gemini_cache.json"

MODEL = "google/gemini-3-flash-preview"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# The hybrid classifier SQL
HYBRID_CLASSIFIER_SQL = """
CASE
  -- DIRECTION 1: Rule-based patterns (first priority)

  -- ChatGPT exports
  WHEN first_line LIKE 'conversation: %' THEN 'CHATGPT_EXPORT'
  -- WOTC Job data
  WHEN first_line LIKE 'job_title: %' THEN 'WOTC_JOB_DATA'
  -- Donor CRM
  WHEN first_line LIKE 'Select: DONOR%' THEN 'DONOR_CRM'
  -- Workspaces
  WHEN domain = 'claude_code' THEN 'CLAUDE_CODE'
  WHEN domain = 'Mordechai Dev 2025' THEN 'DEV_WORKSPACE'
  -- Optical lens
  WHEN first_line LIKE 'Brand: %' THEN 'OPTICAL_LENS'
  -- Talmud patterns
  WHEN first_line LIKE 'Daf: %' OR first_line LIKE ':  %a' OR first_line LIKE ':  %b' THEN 'TALMUD_DAF'
  WHEN first_line LIKE 'Makkot: %' OR first_line LIKE 'Berakhot: %' THEN 'TALMUD_HEBREW'
  -- Jewish texts
  WHEN first_line LIKE 'Author: Living Lchaim%' THEN 'TORAH_VIDEO'
  WHEN first_line LIKE 'Shulchan Arukh%' THEN 'HALACHA'
  WHEN first_line LIKE 'Sefer: %' THEN 'JEWISH_SEFER'
  WHEN first_line LIKE 'Ikar Tosafot%' THEN 'TALMUD_COMMENTARY'
  WHEN first_line LIKE 'זכות: %' THEN 'JEWISH_TRACKER'
  -- Project docs
  WHEN first_line LIKE 'Deployed: %' OR first_line LIKE 'Depoyed: %' THEN 'PROJECT_DOC'
  WHEN first_line LIKE 'merge: %' THEN 'MERGE_DOC'
  -- CRM
  WHEN first_line LIKE 'Linkedin: %' OR first_line LIKE 'Rank: %' THEN 'CRM_CONTACT'
  WHEN first_line LIKE 'column %: %' THEN 'DATABASE_EXPORT'
  WHEN first_line LIKE 'Visibility: %' THEN 'NOTION_METADATA'
  WHEN first_line LIKE 'Complete: %' THEN 'TASK_TRACKER'
  -- Content patterns (Direction 1)
  WHEN LOWER(content) LIKE '%gemara%' AND LOWER(content) LIKE '%rashi%' THEN 'TALMUD_STUDY'
  WHEN LOWER(content) LIKE '%r&d tax%' OR LOWER(content) LIKE '%wotc%' THEN 'TAX_CREDIT'
  WHEN LOWER(content) LIKE '%sparkii%' THEN 'SPARKII_PROJECT'
  WHEN LOWER(content) LIKE '%progressive lens%' THEN 'OPTICAL_LENS'
  WHEN LOWER(content) LIKE '%asperger%' OR LOWER(content) LIKE '%monotropi%' THEN 'NEURODIVERGENCE'
  WHEN LOWER(content) LIKE '%embedding%' AND LOWER(content) LIKE '%vector%' THEN 'ML_EMBEDDINGS'
  WHEN LOWER(content) LIKE '%endpoint%' AND LOWER(content) LIKE '%request%' AND LOWER(content) LIKE '%response%' THEN 'API_DOC'

  -- DIRECTION 3: Content fingerprinting (fallback)
  WHEN LOWER(content) LIKE '%video id:%' OR LOWER(content) LIKE '%video title:%'
       OR LOWER(content) LIKE '%youtube.com/watch%' THEN 'VIDEO_TRANSCRIPT'
  WHEN LOWER(content) LIKE '%ILS%' AND (LOWER(content) LIKE '%gross%' OR LOWER(content) LIKE '%net%') THEN 'FINANCIAL_RECORD'
  WHEN LOWER(content) LIKE '%USD%' AND (LOWER(content) LIKE '%gross%' OR LOWER(content) LIKE '%net%') THEN 'FINANCIAL_RECORD'
  WHEN LOWER(filename) LIKE '%nefesh hachayi%' OR LOWER(filename) LIKE '%eight chapters%' THEN 'JEWISH_PHILOSOPHY'
  WHEN LOWER(filename) LIKE '%bartenura%' THEN 'MISHNAH_COMMENTARY'
  WHEN LOWER(content) LIKE '%def %(%' OR LOWER(content) LIKE '%import %from%' THEN 'CODE_DOC'
  WHEN LOWER(filename) LIKE '%bituach%leumi%' THEN 'GOVERNMENT_DOC'

  ELSE 'NEEDS_GEMINI'
END
"""

GEMINI_CATEGORIES = [
    "JEWISH_PHILOSOPHY",      # Mussar, hashkafa, machshava
    "MISHNAH_COMMENTARY",     # Bartenura, Rambam on Mishnah
    "YOUTUBE_TORAH",          # Torah video transcripts
    "EXPENSE_TRACKING",       # Financial records
    "PERSONAL_REFLECTION",    # Journal, thoughts
    "TECHNICAL_DOC",          # Code, APIs, dev
    "GENERAL_NOTES",          # Misc notes
    "UNKNOWN"                 # Can't determine
]

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": GEMINI_CATEGORIES
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
        }
    },
    "required": ["category", "confidence"]
}


def get_classification_stats(con) -> dict:
    """Get current classification statistics."""
    query = f"""
    SELECT
      {HYBRID_CLASSIFIER_SQL} as category,
      COUNT(*) as docs,
      SUM(word_count) as words
    FROM '{PARQUET_PATH}'
    GROUP BY category
    ORDER BY words DESC
    """
    result = con.execute(query).fetchall()
    return {row[0]: {"docs": row[1], "words": row[2]} for row in result}


def get_unclassified_docs(con, limit: int = 100) -> list:
    """Get docs that need Gemini classification."""
    query = f"""
    SELECT filename, LEFT(content, 500) as preview
    FROM '{PARQUET_PATH}'
    WHERE {HYBRID_CLASSIFIER_SQL} = 'NEEDS_GEMINI'
    ORDER BY word_count DESC
    LIMIT {limit}
    """
    return con.execute(query).fetchall()


def classify_with_gemini(filename: str, preview: str) -> dict:
    """Classify a single document using Gemini."""
    if not OPENROUTER_API_KEY:
        return {"error": "OPENROUTER_API_KEY not set"}

    prompt = f"""Classify this document into exactly one category.

Filename: {filename}
Content (first 500 chars):
{preview}

Categories:
- JEWISH_PHILOSOPHY: Mussar texts, hashkafa, ethical works (Nefesh HaChayim, Eight Chapters)
- MISHNAH_COMMENTARY: Commentary on Mishnah (Bartenura, Tosafot Yom Tov)
- YOUTUBE_TORAH: Torah video transcripts, Q&A sessions with rabbis
- EXPENSE_TRACKING: Financial records, receipts, billing
- PERSONAL_REFLECTION: Personal journals, thoughts, reflections
- TECHNICAL_DOC: Code documentation, APIs, development
- GENERAL_NOTES: General notes that don't fit other categories
- UNKNOWN: Can't determine with confidence

Respond with JSON only."""

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://intellectual-dna.local"
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
            return {"error": response.text[:100]}
    except Exception as e:
        return {"error": str(e)[:100]}


def run_gemini_batch(limit: int = 100):
    """Run Gemini classification on unclassified docs."""
    con = duckdb.connect()
    docs = get_unclassified_docs(con, limit)

    print(f"Classifying {len(docs)} docs with Gemini...")

    # Load cache
    cache = {}
    if GEMINI_CACHE.exists():
        cache = json.loads(GEMINI_CACHE.read_text())

    results = []
    for i, (filename, preview) in enumerate(docs):
        # Check cache
        if filename in cache:
            results.append({"filename": filename, **cache[filename]})
            print(f"  [{i+1}/{len(docs)}] {filename[:40]}... (cached)")
            continue

        # Classify
        result = classify_with_gemini(filename, preview)
        result["filename"] = filename
        results.append(result)

        # Cache result
        if "error" not in result:
            cache[filename] = {"category": result["category"], "confidence": result["confidence"]}
            print(f"  [{i+1}/{len(docs)}] {filename[:40]}... → {result['category']} ({result['confidence']:.0%})")
        else:
            print(f"  [{i+1}/{len(docs)}] {filename[:40]}... → ERROR")

    # Save cache
    GEMINI_CACHE.parent.mkdir(parents=True, exist_ok=True)
    GEMINI_CACHE.write_text(json.dumps(cache, indent=2))

    # Summary
    categories = {}
    for r in results:
        cat = r.get("category", "ERROR")
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\nGemini Results ({len(docs)} docs):")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    return results


def apply_classification():
    """Apply hybrid classification to parquet and save v4."""
    con = duckdb.connect()

    print(f"Reading from: {PARQUET_PATH}")
    print(f"Writing to: {OUTPUT_PATH}")

    # Load Gemini cache
    gemini_map = {}
    if GEMINI_CACHE.exists():
        cache = json.loads(GEMINI_CACHE.read_text())
        gemini_map = {k: v["category"] for k, v in cache.items()}
        print(f"Loaded {len(gemini_map)} Gemini classifications from cache")

    # Build CASE statement for Gemini overrides
    gemini_cases = ""
    if gemini_map:
        for filename, category in gemini_map.items():
            safe_filename = filename.replace("'", "''")
            gemini_cases += f"WHEN filename = '{safe_filename}' THEN '{category}'\n"

    # Combined classifier: Gemini cache first, then hybrid rules
    combined_classifier = f"""
    CASE
      {gemini_cases}
      {HYBRID_CLASSIFIER_SQL.replace('CASE', '').replace('END', '')}
    END
    """

    query = f"""
    COPY (
      SELECT
        *,
        {combined_classifier} as classified_category
      FROM '{PARQUET_PATH}'
    ) TO '{OUTPUT_PATH}' (FORMAT PARQUET)
    """

    con.execute(query)

    # Verify
    stats_query = f"""
    SELECT classified_category, COUNT(*), SUM(word_count)
    FROM '{OUTPUT_PATH}'
    GROUP BY classified_category
    ORDER BY SUM(word_count) DESC
    """
    stats = con.execute(stats_query).fetchall()

    print(f"\nClassification complete! Saved to {OUTPUT_PATH}")
    print(f"\nFinal categories ({len(stats)} total):")
    for cat, docs, words in stats:
        pct = words * 100 / sum(s[2] for s in stats)
        print(f"  {cat}: {docs:,} docs, {words:,} words ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Build document classification")
    parser.add_argument("--apply", action="store_true", help="Apply classification to parquet")
    parser.add_argument("--gemini", type=int, default=0, help="Run Gemini on N unclassified docs")
    args = parser.parse_args()

    con = duckdb.connect()

    if args.gemini > 0:
        run_gemini_batch(args.gemini)
    elif args.apply:
        apply_classification()
    else:
        # Dry run - show stats
        print("Classification Statistics (dry run):\n")
        stats = get_classification_stats(con)
        total_docs = sum(s["docs"] for s in stats.values())
        total_words = sum(s["words"] for s in stats.values())

        for cat, data in sorted(stats.items(), key=lambda x: -x[1]["words"]):
            pct = data["words"] * 100 / total_words
            print(f"  {cat}: {data['docs']:,} docs, {data['words']:,} words ({pct:.1f}%)")

        needs_gemini = stats.get("NEEDS_GEMINI", {"docs": 0, "words": 0})
        print(f"\n→ {needs_gemini['docs']:,} docs need Gemini ({needs_gemini['words'] * 100 / total_words:.1f}% of words)")
        print(f"\nRun with --gemini 100 to classify 100 docs with Gemini")
        print(f"Run with --apply to save classified parquet")


if __name__ == "__main__":
    main()
