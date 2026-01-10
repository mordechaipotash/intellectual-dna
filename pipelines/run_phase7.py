#!/usr/bin/env python3
"""
Phase 7: Run all new interpretation pipelines.

9 new pipelines in two tiers:
- Tier 1-2 (Flash Lite): signature_phrases, decision_patterns, tool_preferences,
                         learning_trajectories, project_arcs, collaboration_patterns
- Tier 3 (Flash 3): monthly_themes, intellectual_evolution, cross_domain

Estimated cost: ~$2-3 total
Estimated time: ~1-2 hours
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
PYTHON = BASE_DIR / "mordelab" / "02-monotropic-prosthetic" / "mcp-env" / "bin" / "python"

# Pipeline definitions
PIPELINES = {
    # Tier 1: Extraction (Flash Lite)
    "signature_phrases": {
        "script": "build_signature_phrases.py",
        "tier": "extraction",
        "model": "flash-lite",
        "priority": 1
    },
    "decision_patterns": {
        "script": "build_decision_patterns.py",
        "tier": "extraction",
        "model": "flash-lite",
        "priority": 2
    },
    "tool_preferences": {
        "script": "build_tool_preferences.py",
        "tier": "extraction",
        "model": "flash-lite",
        "priority": 3
    },
    # Tier 2: Analysis (Flash Lite)
    "learning_trajectories": {
        "script": "build_learning_trajectories.py",
        "tier": "analysis",
        "model": "flash-lite",
        "priority": 4
    },
    "project_arcs": {
        "script": "build_project_arcs.py",
        "tier": "analysis",
        "model": "flash-lite",
        "priority": 5
    },
    "collaboration_patterns": {
        "script": "build_collaboration_patterns.py",
        "tier": "analysis",
        "model": "flash-lite",
        "priority": 6
    },
    # Tier 3: Synthesis (Flash 3)
    "monthly_themes": {
        "script": "build_monthly_themes.py",
        "tier": "synthesis",
        "model": "flash-3",
        "priority": 7
    },
    "intellectual_evolution": {
        "script": "build_intellectual_evolution.py",
        "tier": "synthesis",
        "model": "flash-3",
        "priority": 8
    },
    "cross_domain": {
        "script": "build_cross_domain.py",
        "tier": "synthesis",
        "model": "flash-3",
        "priority": 9
    }
}


def run_pipeline(name: str, config: dict) -> tuple:
    """Run a single pipeline. Returns (success, duration, output)."""
    script = BASE_DIR / "pipelines" / config["script"]

    print(f"\n{'='*60}")
    print(f"🚀 STARTING: {name}")
    print(f"   Tier: {config['tier']}")
    print(f"   Model: {config['model']}")
    print(f"{'='*60}")

    start = time.time()

    try:
        # Stream output directly instead of capturing
        result = subprocess.run(
            [str(PYTHON), "-u", str(script)],
            timeout=7200  # 2 hour timeout per pipeline
        )

        duration = time.time() - start
        success = result.returncode == 0
        output = ""

        if success:
            print(f"✅ COMPLETED: {name} in {duration/60:.1f} min")
        else:
            print(f"❌ FAILED: {name}")
            print(f"   Error: {result.stderr[:500]}")

        return success, duration, output

    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {name} exceeded 2 hours")
        return False, 7200, "Timeout"
    except Exception as e:
        print(f"❌ ERROR: {name} - {str(e)}")
        return False, 0, str(e)


def run_all():
    """Run all pipelines in order."""
    print("="*70)
    print("🧬 INTELLECTUAL DNA - PHASE 7: NEW INTERPRETATIONS")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Pipelines: 9 total (6 Flash Lite, 3 Flash 3)")
    print("="*70)

    results = {}
    total_time = 0.0

    # Sort by priority
    sorted_pipelines = sorted(PIPELINES.items(), key=lambda x: x[1]["priority"])

    # Run Tier 1-2 (Flash Lite) first
    print("\n📦 TIER 1-2: Extraction & Analysis (Flash Lite)")
    print("-"*40)

    for name, config in sorted_pipelines:
        if config["model"] == "flash-lite":
            success, duration, output = run_pipeline(name, config)
            results[name] = {"success": success, "duration": duration}
            total_time += duration

    # Run Tier 3 (Flash 3)
    print("\n📦 TIER 3: Synthesis (Flash 3)")
    print("-"*40)

    for name, config in sorted_pipelines:
        if config["model"] == "flash-3":
            success, duration, output = run_pipeline(name, config)
            results[name] = {"success": success, "duration": duration}
            total_time += duration

    # Final summary
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)

    succeeded = sum(1 for r in results.values() if r["success"])
    print(f"Pipelines: {succeeded}/{len(results)} succeeded")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")

    print("\nResults by pipeline:")
    for name, result in results.items():
        status = "✅" if result["success"] else "❌"
        model = PIPELINES[name]["model"]
        print(f"  {status} {name} ({model}): {result['duration']/60:.1f} min")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == '__main__':
    run_all()
