#!/usr/bin/env python3
"""
Master orchestrator: Run ALL Gemini interpretation pipelines.

Execution Plan:
===============
PHASE 1 - Independent pipelines (can run now):
  1. conversation_titles  - ~5000 convos @ 30/min = ~3 hrs, ~$0.15
  2. question_extraction  - ~600 days @ 30/min = ~20 min, ~$0.05
  3. insight_mining       - ~400 days @ 30/min = ~13 min, ~$0.05
  4. mood_patterns        - ~900 days @ 30/min = ~30 min, ~$0.05
  5. personal_glossary    - ~100 terms @ 30/min = ~3 min, ~$0.02
  6. phrase_context       - ~100 phrases @ 30/min = ~3 min, ~$0.01

PHASE 2 - Depends on focus/v2 (wait for completion):
  7. weekly_summaries     - ~130 weeks @ 30/min = ~4 min, ~$0.01
  8. youtube_link         - ~100 weeks @ 30/min = ~3 min, ~$0.10

Total estimated cost: ~$0.44
Total estimated time: ~4 hours (mostly conversation_titles)
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
PYTHON = BASE_DIR / "mordelab" / "02-monotropic-prosthetic" / "mcp-env" / "bin" / "python"
FOCUS_V2_PATH = BASE_DIR / "data" / "interpretations" / "focus" / "v2" / "daily.parquet"

# Pipeline definitions with dependencies and estimates
PIPELINES = {
    # Phase 1: Independent
    "conversation_titles": {
        "script": "build_conversation_titles.py",
        "depends_on": None,
        "est_items": 5000,
        "est_cost": 0.15,
        "priority": 1
    },
    "question_extraction": {
        "script": "build_question_extraction.py",
        "depends_on": None,
        "est_items": 600,
        "est_cost": 0.05,
        "priority": 2
    },
    "insight_mining": {
        "script": "build_insight_mining.py",
        "depends_on": None,
        "est_items": 400,
        "est_cost": 0.05,
        "priority": 3
    },
    "mood_patterns": {
        "script": "build_mood_patterns.py",
        "depends_on": None,
        "est_items": 900,
        "est_cost": 0.05,
        "priority": 4
    },
    "personal_glossary": {
        "script": "build_personal_glossary.py",
        "depends_on": None,
        "est_items": 100,
        "est_cost": 0.02,
        "priority": 5
    },
    "phrase_context": {
        "script": "build_phrase_context.py",
        "depends_on": "phrases/v1",
        "est_items": 100,
        "est_cost": 0.01,
        "priority": 6
    },
    # Phase 2: Depends on focus/v2
    "weekly_summaries": {
        "script": "build_weekly_summaries.py",
        "depends_on": "focus/v2",
        "est_items": 130,
        "est_cost": 0.01,
        "priority": 7
    },
    "youtube_link": {
        "script": "build_youtube_link.py",
        "depends_on": "focus/v2",
        "est_items": 100,
        "est_cost": 0.10,
        "priority": 8
    }
}


def check_focus_v2_ready():
    """Check if focus/v2 has completed (934 days)."""
    import duckdb
    if not FOCUS_V2_PATH.exists():
        return False, 0
    con = duckdb.connect()
    count = con.execute(f"SELECT COUNT(*) FROM '{FOCUS_V2_PATH}'").fetchone()[0]
    return count >= 900, count  # Consider ready at 900+


def run_pipeline(name: str, config: dict) -> tuple:
    """Run a single pipeline. Returns (success, duration, output)."""
    script = BASE_DIR / "pipelines" / config["script"]

    print(f"\n{'='*60}")
    print(f"🚀 STARTING: {name}")
    print(f"   Script: {config['script']}")
    print(f"   Est. items: {config['est_items']}")
    print(f"   Est. cost: ${config['est_cost']:.2f}")
    print(f"{'='*60}")

    start = time.time()

    try:
        result = subprocess.run(
            [str(PYTHON), str(script)],
            capture_output=True,
            text=True,
            timeout=14400  # 4 hour timeout
        )

        duration = time.time() - start
        success = result.returncode == 0
        output = result.stdout + result.stderr

        if success:
            print(f"✅ COMPLETED: {name} in {duration/60:.1f} min")
        else:
            print(f"❌ FAILED: {name}")
            print(f"   Error: {result.stderr[:500]}")

        return success, duration, output

    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {name} exceeded 4 hours")
        return False, 14400, "Timeout"
    except Exception as e:
        print(f"❌ ERROR: {name} - {str(e)}")
        return False, 0, str(e)


def run_all():
    """Run all pipelines in optimal order."""
    print("="*70)
    print("🧠 INTELLECTUAL DNA - FULL GEMINI PIPELINE EXECUTION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total estimated cost: $0.44")
    print(f"Total estimated time: ~4 hours")
    print("="*70)

    results = {}
    total_cost = 0.0
    total_time = 0.0

    # Sort by priority
    sorted_pipelines = sorted(PIPELINES.items(), key=lambda x: x[1]["priority"])

    # Phase 1: Run independent pipelines
    print("\n📦 PHASE 1: Independent Pipelines")
    print("-"*40)

    for name, config in sorted_pipelines:
        if config["depends_on"] == "focus/v2":
            continue  # Skip focus/v2 dependent pipelines

        # Check other dependencies
        if config["depends_on"] and config["depends_on"] != "focus/v2":
            dep_path = BASE_DIR / "data" / "interpretations" / config["depends_on"].replace("/", "/")
            # Just continue, phrases/v1 should already exist

        success, duration, output = run_pipeline(name, config)
        results[name] = {"success": success, "duration": duration}
        total_time += duration
        if success:
            total_cost += config["est_cost"]

    # Phase 2: Wait for focus/v2 and run dependent pipelines
    print("\n📦 PHASE 2: Focus/v2 Dependent Pipelines")
    print("-"*40)

    # Check focus/v2 status
    ready, count = check_focus_v2_ready()
    if not ready:
        print(f"⏳ Focus/v2 has {count} days, waiting for completion...")
        while not ready:
            time.sleep(60)  # Check every minute
            ready, count = check_focus_v2_ready()
            print(f"   Focus/v2: {count}/934 days...")

    print(f"✅ Focus/v2 ready with {count} days")

    for name, config in sorted_pipelines:
        if config["depends_on"] != "focus/v2":
            continue

        success, duration, output = run_pipeline(name, config)
        results[name] = {"success": success, "duration": duration}
        total_time += duration
        if success:
            total_cost += config["est_cost"]

    # Final summary
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)

    succeeded = sum(1 for r in results.values() if r["success"])
    print(f"Pipelines: {succeeded}/{len(results)} succeeded")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"Estimated cost: ${total_cost:.2f}")

    print("\nResults by pipeline:")
    for name, result in results.items():
        status = "✅" if result["success"] else "❌"
        print(f"  {status} {name}: {result['duration']/60:.1f} min")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == '__main__':
    run_all()
