#!/usr/bin/env python3
"""
Phase 8: Run the 3 new high-impact pipelines.

Proven approaches from iterative testing:
- daily_accomplishments: What got DONE each day (✓ format)
- weekly_expertise: Technologies used confidently (TECH/EVIDENCE/CONFIDENCE)
- problem_resolutions: Issues that got RESOLVED (ISSUE/RESOLUTION/DOMAIN)

All use Gemini Flash Lite for cost efficiency.
Estimated cost: ~$0.10-0.20 total
Estimated time: ~20-40 minutes
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
PYTHON = BASE_DIR / "mordelab" / "02-monotropic-prosthetic" / "mcp-env" / "bin" / "python"

# Pipeline definitions - all use Flash Lite
PIPELINES = {
    "daily_accomplishments": {
        "script": "build_daily_accomplishments.py",
        "description": "What got DONE each day",
        "granularity": "daily",
        "priority": 1
    },
    "weekly_expertise": {
        "script": "build_weekly_expertise.py",
        "description": "Technologies used confidently",
        "granularity": "weekly",
        "priority": 2
    },
    "problem_resolutions": {
        "script": "build_problem_resolutions.py",
        "description": "Issues that got resolved",
        "granularity": "weekly",
        "priority": 3
    }
}


def run_pipeline(name: str, config: dict) -> tuple:
    """Run a single pipeline. Returns (success, duration, output)."""
    script = BASE_DIR / "pipelines" / config["script"]

    print(f"\n{'='*60}")
    print(f"🚀 STARTING: {name}")
    print(f"   Description: {config['description']}")
    print(f"   Granularity: {config['granularity']}")
    print(f"{'='*60}")

    start = time.time()

    try:
        result = subprocess.run(
            [str(PYTHON), "-u", str(script)],
            timeout=3600  # 1 hour timeout
        )

        duration = time.time() - start
        success = result.returncode == 0

        if success:
            print(f"✅ COMPLETED: {name} in {duration/60:.1f} min")
        else:
            print(f"❌ FAILED: {name}")

        return success, duration, ""

    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {name} exceeded 1 hour")
        return False, 3600, "Timeout"
    except Exception as e:
        print(f"❌ ERROR: {name} - {str(e)}")
        return False, 0, str(e)


def run_all():
    """Run all Phase 8 pipelines."""
    print("="*70)
    print("🧬 INTELLECTUAL DNA - PHASE 8: HIGH-IMPACT PIPELINES")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Pipelines: 3 (all Flash Lite)")
    print("="*70)

    results = {}
    total_time = 0.0

    # Sort by priority
    sorted_pipelines = sorted(PIPELINES.items(), key=lambda x: x[1]["priority"])

    for name, config in sorted_pipelines:
        success, duration, output = run_pipeline(name, config)
        results[name] = {"success": success, "duration": duration}
        total_time += duration

    # Final summary
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)

    succeeded = sum(1 for r in results.values() if r["success"])
    print(f"Pipelines: {succeeded}/{len(results)} succeeded")
    print(f"Total time: {total_time/60:.1f} minutes")

    print("\nResults by pipeline:")
    for name, result in results.items():
        status = "✅" if result["success"] else "❌"
        desc = PIPELINES[name]["description"]
        print(f"  {status} {name}: {result['duration']/60:.1f} min - {desc}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == '__main__':
    run_all()
