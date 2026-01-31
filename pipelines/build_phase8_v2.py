#!/usr/bin/env python3
"""
Phase 8: V2 Interpretation Upgrades
Enhance V1-only layers with LLM analysis using Gemini 2.5 Flash Lite.

Layers to upgrade:
- daily_accomplishments: Add impact scoring, category, SEED alignment
- conversation_titles: Add sentiment, energy, topic extraction
- insights: Add confidence scoring, validation
- mood: Add productivity correlation
- glossary: Add usage frequency enrichment
"""

import os
import json
import time
import duckdb
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import threading

# Load environment variables
load_dotenv(Path("/Users/mordechai/intellectual_dna/.env"))

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
DATA_DIR = BASE_DIR / "data"
INTERP_DIR = DATA_DIR / "interpretations"

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash-lite"

# Rate limiting
REQUESTS_PER_MINUTE = 60
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE
request_lock = threading.Lock()
last_request_time = 0


def ensure_dir(path: Path):
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
    """Call Gemini via OpenRouter API with rate limiting."""
    global last_request_time

    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    # Rate limiting
    with request_lock:
        elapsed = time.time() - last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        last_request_time = time.time()

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Intellectual DNA Phase 8"
    }

    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Error: {str(e)[:50]}]"


def upgrade_accomplishments(limit: int = None):
    """
    Upgrade daily_accomplishments with impact scoring and categorization.
    """
    print("\n=== Upgrading Daily Accomplishments ===")

    v1_path = INTERP_DIR / "daily_accomplishments" / "v1" / "daily.parquet"
    out_dir = INTERP_DIR / "daily_accomplishments" / "v2"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Get V1 data
    query = f"SELECT date, accomplishments_json, accomplishment_count FROM '{v1_path}' ORDER BY date"
    if limit:
        query += f" LIMIT {limit}"

    rows = con.execute(query).fetchall()
    print(f"  Processing {len(rows)} days...")

    system_prompt = """Analyze these accomplishments and return JSON:
{"impact_score": 1-10, "primary_category": "coding|research|learning|writing|planning|debugging|other", "seed_alignment": "bottleneck|compression|agency|seeds|inversion|translation|temporal|none"}
Only return the JSON, nothing else."""

    results = []
    for i, (date, accomplishments_json, count) in enumerate(rows):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(rows)}")

        try:
            accomplishments = json.loads(accomplishments_json) if accomplishments_json else []
            if not accomplishments:
                results.append((date, 0, "none", "none", accomplishments_json, count))
                continue

            # Take first 5 accomplishments for analysis
            sample = accomplishments[:5]
            prompt = f"Accomplishments for {date}:\n" + "\n".join(f"- {a}" for a in sample)

            response = call_llm(system_prompt, prompt, max_tokens=100)

            try:
                parsed = json.loads(response)
                impact = parsed.get("impact_score", 5)
                category = parsed.get("primary_category", "other")
                seed = parsed.get("seed_alignment", "none")
            except:
                impact, category, seed = 5, "other", "none"

            results.append((date, impact, category, seed, accomplishments_json, count))
        except Exception as e:
            results.append((date, 0, "error", "none", accomplishments_json, count))

    # Write results using DuckDB
    con.execute("DROP TABLE IF EXISTS accomplishments_v2")
    con.execute("""
        CREATE TABLE accomplishments_v2 (
            date VARCHAR, impact_score INTEGER, primary_category VARCHAR,
            seed_alignment VARCHAR, accomplishments_preview VARCHAR, count INTEGER
        )
    """)
    for r in results:
        date = str(r[0]).replace("'", "''")
        category = str(r[2]).replace("'", "''") if r[2] else "other"
        seed = str(r[3]).replace("'", "''") if r[3] else "none"
        preview = str(r[4])[:100].replace("'", "''") if r[4] else ""
        con.execute(f"""
            INSERT INTO accomplishments_v2 VALUES (
                '{date}', {r[1]}, '{category}', '{seed}', '{preview}', {r[5]}
            )
        """)
    con.execute(f"COPY accomplishments_v2 TO '{out_dir}/daily.parquet' (FORMAT PARQUET)")

    # Save config
    config = {"model": MODEL, "processed_at": datetime.now().isoformat(), "rows": len(results)}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"  Upgraded {len(results)} days")
    return len(results)


def upgrade_titles(limit: int = None):
    """
    Upgrade conversation_titles with sentiment and topic extraction.
    """
    print("\n=== Upgrading Conversation Titles ===")

    v1_path = INTERP_DIR / "conversation_titles" / "v1" / "titles.parquet"
    out_dir = INTERP_DIR / "conversation_titles" / "v2"
    ensure_dir(out_dir)

    con = duckdb.connect()

    query = f"SELECT conversation_id, title, message_count FROM '{v1_path}' ORDER BY start_time DESC"
    if limit:
        query += f" LIMIT {limit}"

    rows = con.execute(query).fetchall()
    print(f"  Processing {len(rows)} titles...")

    system_prompt = """Analyze this conversation title and return JSON:
{"sentiment": "positive|neutral|negative|frustrated", "energy": "high|medium|low", "topic": "brief topic in 2-3 words"}
Only return the JSON."""

    results = []
    for i, (conv_id, title, msg_count) in enumerate(rows):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(rows)}")

        if not title or len(title) < 3:
            results.append((conv_id, title, msg_count, "neutral", "low", "unknown"))
            continue

        response = call_llm(system_prompt, f"Title: {title}", max_tokens=80)

        try:
            parsed = json.loads(response)
            sentiment = parsed.get("sentiment", "neutral")
            energy = parsed.get("energy", "medium")
            topic = parsed.get("topic", "unknown")[:30]
        except:
            sentiment, energy, topic = "neutral", "medium", "unknown"

        results.append((conv_id, title, msg_count, sentiment, energy, topic))

    # Write to parquet using DuckDB
    con.execute("DROP TABLE IF EXISTS titles_v2")
    con.execute("""
        CREATE TABLE titles_v2 (
            conversation_id VARCHAR, title VARCHAR, message_count INTEGER,
            sentiment VARCHAR, energy VARCHAR, primary_topic VARCHAR
        )
    """)
    for r in results:
        conv_id = str(r[0]).replace("'", "''")
        title_escaped = str(r[1])[:200].replace("'", "''") if r[1] else ""
        sentiment = str(r[3]).replace("'", "''") if r[3] else "neutral"
        energy = str(r[4]).replace("'", "''") if r[4] else "medium"
        topic_escaped = str(r[5])[:50].replace("'", "''") if r[5] else ""
        con.execute(f"""
            INSERT INTO titles_v2 VALUES (
                '{conv_id}', '{title_escaped}', {r[2] or 0},
                '{sentiment}', '{energy}', '{topic_escaped}'
            )
        """)
    con.execute(f"COPY titles_v2 TO '{out_dir}/titles.parquet' (FORMAT PARQUET)")

    config = {"model": MODEL, "processed_at": datetime.now().isoformat(), "rows": len(results)}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"  Upgraded {len(results)} titles")
    return len(results)


def upgrade_insights(limit: int = None):
    """
    Upgrade insights with confidence scoring.
    """
    print("\n=== Upgrading Insights ===")

    v1_path = INTERP_DIR / "insights" / "v1" / "insights.parquet"
    out_dir = INTERP_DIR / "insights" / "v2"
    ensure_dir(out_dir)

    con = duckdb.connect()

    query = f"SELECT date, insight, category, significance FROM '{v1_path}' ORDER BY date"
    if limit:
        query += f" LIMIT {limit}"

    rows = con.execute(query).fetchall()
    print(f"  Processing {len(rows)} insights...")

    system_prompt = """Rate this insight and return JSON:
{"confidence": 0.0-1.0 (how certain/validated), "actionable": true/false, "depth": "surface|medium|deep"}
Only return the JSON."""

    results = []
    for i, (date, insight, category, significance) in enumerate(rows):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(rows)}")

        if not insight or len(insight) < 10:
            results.append((date, insight, category, significance, 0.5, False, "surface"))
            continue

        response = call_llm(system_prompt, f"Insight: {insight[:200]}", max_tokens=60)

        try:
            parsed = json.loads(response)
            confidence = min(1.0, max(0.0, float(parsed.get("confidence", 0.5))))
            actionable = parsed.get("actionable", False)
            depth = parsed.get("depth", "surface")
        except:
            confidence, actionable, depth = 0.5, False, "surface"

        results.append((date, insight, category, significance, confidence, actionable, depth))

    # Write to parquet using DuckDB
    con.execute("DROP TABLE IF EXISTS insights_v2")
    con.execute("""
        CREATE TABLE insights_v2 (
            date VARCHAR, insight VARCHAR, category VARCHAR,
            significance VARCHAR, confidence DOUBLE, actionable BOOLEAN, depth VARCHAR
        )
    """)
    for r in results:
        date = str(r[0]).replace("'", "''")
        insight_escaped = str(r[1])[:500].replace("'", "''") if r[1] else ""
        cat_escaped = str(r[2])[:50].replace("'", "''") if r[2] else ""
        sig_escaped = str(r[3])[:50].replace("'", "''") if r[3] else ""
        depth = str(r[6]).replace("'", "''") if r[6] else "surface"
        con.execute(f"""
            INSERT INTO insights_v2 VALUES (
                '{date}', '{insight_escaped}', '{cat_escaped}',
                '{sig_escaped}', {r[4]}, {r[5]}, '{depth}'
            )
        """)
    con.execute(f"COPY insights_v2 TO '{out_dir}/insights.parquet' (FORMAT PARQUET)")

    config = {"model": MODEL, "processed_at": datetime.now().isoformat(), "rows": len(results)}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"  Upgraded {len(results)} insights")
    return len(results)


def upgrade_mood(limit: int = None):
    """
    Upgrade mood with productivity correlation hints.
    """
    print("\n=== Upgrading Mood ===")

    v1_path = INTERP_DIR / "mood" / "v1" / "daily.parquet"
    prod_path = INTERP_DIR / "productivity_matrix" / "v1" / "daily.parquet"
    out_dir = INTERP_DIR / "mood" / "v2"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Join mood with productivity for correlation analysis
    query = f"""
        SELECT
            m.date, m.mood, m.energy, m.cognitive_state, m.stress, m.explanation,
            COALESCE(p.productivity_score, 0) as productivity_score,
            COALESCE(p.accomplishments, 0) as accomplishments
        FROM '{v1_path}' m
        LEFT JOIN '{prod_path}' p ON DATE_TRUNC('day', m.date) = p.date
        ORDER BY m.date
    """
    if limit:
        query += f" LIMIT {limit}"

    rows = con.execute(query).fetchall()
    print(f"  Processing {len(rows)} mood records...")

    system_prompt = """Given mood and productivity data, return JSON:
{"productivity_prediction": "high|medium|low", "energy_trend": "rising|stable|falling", "burnout_risk": "low|medium|high"}
Only return the JSON."""

    results = []
    for i, (date, mood, energy, cognitive, stress, explanation, prod_score, accomplishments) in enumerate(rows):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(rows)}")

        prompt = f"Mood: {mood}, Energy: {energy}, Stress: {stress}, Productivity: {prod_score}, Accomplishments: {accomplishments}"
        response = call_llm(system_prompt, prompt, max_tokens=60)

        try:
            parsed = json.loads(response)
            pred = parsed.get("productivity_prediction", "medium")
            trend = parsed.get("energy_trend", "stable")
            burnout = parsed.get("burnout_risk", "low")
        except:
            pred, trend, burnout = "medium", "stable", "low"

        results.append((date, mood, energy, cognitive, stress, prod_score, pred, trend, burnout))

    # Write to parquet using DuckDB
    con.execute("DROP TABLE IF EXISTS mood_v2")
    con.execute("""
        CREATE TABLE mood_v2 (
            date VARCHAR, mood VARCHAR, energy VARCHAR, cognitive_state VARCHAR,
            stress VARCHAR, productivity_score DOUBLE, productivity_prediction VARCHAR,
            energy_trend VARCHAR, burnout_risk VARCHAR
        )
    """)
    for r in results:
        date = str(r[0]).replace("'", "''")
        mood_escaped = str(r[1])[:50].replace("'", "''") if r[1] else ""
        energy_escaped = str(r[2])[:50].replace("'", "''") if r[2] else ""
        cog_escaped = str(r[3])[:50].replace("'", "''") if r[3] else ""
        stress_escaped = str(r[4])[:50].replace("'", "''") if r[4] else ""
        pred = str(r[6]).replace("'", "''") if r[6] else "medium"
        trend = str(r[7]).replace("'", "''") if r[7] else "stable"
        burnout = str(r[8]).replace("'", "''") if r[8] else "low"
        con.execute(f"""
            INSERT INTO mood_v2 VALUES (
                '{date}', '{mood_escaped}', '{energy_escaped}', '{cog_escaped}',
                '{stress_escaped}', {r[5] or 0}, '{pred}', '{trend}', '{burnout}'
            )
        """)
    con.execute(f"COPY mood_v2 TO '{out_dir}/daily.parquet' (FORMAT PARQUET)")

    config = {"model": MODEL, "processed_at": datetime.now().isoformat(), "rows": len(results)}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"  Upgraded {len(results)} mood records")
    return len(results)


def upgrade_glossary():
    """
    Upgrade glossary with usage frequency from conversations.
    """
    print("\n=== Upgrading Glossary ===")

    v1_path = INTERP_DIR / "glossary" / "v1" / "terms.parquet"
    conv_path = DATA_DIR / "all_conversations.parquet"
    out_dir = INTERP_DIR / "glossary" / "v2"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Get terms and count usage in conversations
    terms = con.execute(f"SELECT term, definition, term_type, example_usage FROM '{v1_path}'").fetchall()
    print(f"  Processing {len(terms)} terms...")

    results = []
    for term, definition, term_type, example in terms:
        # Count usage in conversations
        try:
            count = con.execute(f"""
                SELECT COUNT(*) FROM '{conv_path}'
                WHERE LOWER(content) LIKE '%{term.lower()}%'
            """).fetchone()[0]
        except:
            count = 0

        # Find first and last usage
        try:
            first_last = con.execute(f"""
                SELECT MIN(created), MAX(created) FROM '{conv_path}'
                WHERE LOWER(content) LIKE '%{term.lower()}%'
            """).fetchone()
            first_use = str(first_last[0])[:10] if first_last[0] else None
            last_use = str(first_last[1])[:10] if first_last[1] else None
        except:
            first_use, last_use = None, None

        results.append((term, definition, term_type, example, count, first_use, last_use))

    # Write to parquet using DuckDB
    con.execute("DROP TABLE IF EXISTS glossary_v2")
    con.execute("""
        CREATE TABLE glossary_v2 (
            term VARCHAR, definition VARCHAR, term_type VARCHAR,
            example_usage VARCHAR, conversation_mentions INTEGER,
            first_used VARCHAR, last_used VARCHAR
        )
    """)
    for r in results:
        con.execute(f"""
            INSERT INTO glossary_v2 VALUES (
                '{r[0].replace("'", "''")}',
                '{str(r[1])[:500].replace("'", "''") if r[1] else ''}',
                '{r[2] or ''}',
                '{str(r[3])[:200].replace("'", "''") if r[3] else ''}',
                {r[4]},
                {f"'{r[5]}'" if r[5] else 'NULL'},
                {f"'{r[6]}'" if r[6] else 'NULL'}
            )
        """)
    con.execute(f"COPY glossary_v2 TO '{out_dir}/terms.parquet' (FORMAT PARQUET)")

    config = {"model": "sql-enrichment", "processed_at": datetime.now().isoformat(), "rows": len(results)}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"  Upgraded {len(results)} terms")
    return len(results)


def build_all(limit: int = None, skip_completed: bool = True):
    """Build all V2 upgrades."""
    print(f"Phase 8: V2 Interpretation Upgrades")
    print(f"Model: {MODEL}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not found. Set it in .env file.")
        return

    # Check what's already done (by file size - placeholder files are tiny)
    import os
    def is_complete(path, min_size=5000):
        try:
            return os.path.getsize(path) > min_size
        except:
            return False

    # Run upgrades (skip if already complete with real data)
    if not skip_completed or not is_complete(INTERP_DIR / "glossary/v2/terms.parquet", 10000):
        upgrade_glossary()
    else:
        print("\n=== Skipping Glossary (already complete) ===")

    if not skip_completed or not is_complete(INTERP_DIR / "daily_accomplishments/v2/daily.parquet", 20000):
        upgrade_accomplishments(limit)
    else:
        print("\n=== Skipping Accomplishments (already complete) ===")

    if not skip_completed or not is_complete(INTERP_DIR / "conversation_titles/v2/titles.parquet", 20000):
        upgrade_titles(limit)
    else:
        print("\n=== Skipping Titles (already complete) ===")

    if not skip_completed or not is_complete(INTERP_DIR / "insights/v2/insights.parquet", 10000):
        upgrade_insights(limit)
    else:
        print("\n=== Skipping Insights (already complete) ===")

    if not skip_completed or not is_complete(INTERP_DIR / "mood/v2/daily.parquet", 10000):
        upgrade_mood(limit)
    else:
        print("\n=== Skipping Mood (already complete) ===")

    print("\n✅ Phase 8 V2 upgrades complete!")


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    build_all(limit)
