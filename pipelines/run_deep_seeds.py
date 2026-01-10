#!/usr/bin/env python3
"""
Run Deep Seed Pipelines - 3 practical extraction patterns from 100x mining.

Models matched to task:
- MVP_VELOCITY: Gemini 3 Flash (fast pattern extraction)
- TOOL_STACK_COMBOS: Claude Sonnet 4.5 (best tech understanding)
- PROBLEM_RESOLUTION_CHAIN: Grok 3 Mini (reasoning/problem-solving)

Total Budget: $15 ($5 per seed)
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
PYTHON = BASE_DIR / "mordelab" / "02-monotropic-prosthetic" / "mcp-env" / "bin" / "python"

PIPELINES = {
    "mvp_velocity": {
        "script": "build_mvp_velocity.py",
        "model": "Gemini 3 Flash",
        "description": "Rapid prototyping patterns",
        "budget": 5.0
    },
    "tool_stack_combos": {
        "script": "build_tool_stack_combos.py",
        "model": "Claude Sonnet 4.5",
        "description": "Technology stack decisions",
        "budget": 5.0
    },
    "problem_resolution_chain": {
        "script": "build_problem_resolution_chain.py",
        "model": "Grok 3 Mini",
        "description": "Problem→Resolution journeys",
        "budget": 5.0
    }
}


def run_pipeline(name: str, config: dict, test_mode: bool = False) -> tuple:
    """Run a single pipeline. Returns (success, duration)."""
    script = BASE_DIR / "pipelines" / config["script"]

    print(f"\n{'='*60}")
    print(f"🧬 DEEP SEED: {name}")
    print(f"   Model: {config['model']}")
    print(f"   Description: {config['description']}")
    print(f"   Budget: ${config['budget']:.2f}")
    print(f"{'='*60}")

    start = time.time()
    args = [str(PYTHON), "-u", str(script)]

    if test_mode:
        args.extend(["--test"])
    else:
        args.extend(["--budget", str(config["budget"])])

    try:
        result = subprocess.run(args, timeout=7200)  # 2 hour timeout
        duration = time.time() - start
        success = result.returncode == 0

        if success:
            print(f"✅ COMPLETED: {name} in {duration/60:.1f} min")
        else:
            print(f"❌ FAILED: {name}")

        return success, duration

    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {name} exceeded 2 hours")
        return False, 7200
    except Exception as e:
        print(f"❌ ERROR: {name} - {str(e)}")
        return False, 0


def run_all(test_mode: bool = False, pipelines: list = None):
    """Run deep seed pipelines."""
    print("="*70)
    print("🧬 INTELLECTUAL DNA - DEEP SEED EXTRACTION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'TEST (5 weeks each)' if test_mode else 'FULL ($5 budget each)'}")
    print(f"Total budget: $15.00")
    print("="*70)

    to_run = pipelines or list(PIPELINES.keys())

    results = {}
    total_time = 0.0

    for name in to_run:
        if name not in PIPELINES:
            print(f"⚠️  Unknown pipeline: {name}")
            continue

        config = PIPELINES[name]
        success, duration = run_pipeline(name, config, test_mode)
        results[name] = {"success": success, "duration": duration}
        total_time += duration

    # Final summary
    print("\n" + "="*70)
    print("📊 DEEP SEED EXTRACTION SUMMARY")
    print("="*70)

    succeeded = sum(1 for r in results.values() if r["success"])
    print(f"Pipelines: {succeeded}/{len(results)} succeeded")
    print(f"Total time: {total_time/60:.1f} minutes")

    print("\nResults by pipeline:")
    for name, result in results.items():
        status = "✅" if result["success"] else "❌"
        config = PIPELINES[name]
        print(f"  {status} {name}")
        print(f"     Model: {config['model']}")
        print(f"     Time: {result['duration']/60:.1f} min")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run Deep Seed extraction pipelines")
    parser.add_argument('--test', action='store_true', help='Test mode (5 weeks each)')
    parser.add_argument('--pipeline', choices=list(PIPELINES.keys()),
                       help='Run specific pipeline only')
    args = parser.parse_args()

    if args.pipeline:
        run_all(test_mode=args.test, pipelines=[args.pipeline])
    else:
        run_all(test_mode=args.test)
