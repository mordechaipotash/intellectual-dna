#!/usr/bin/env python3
"""
Unified rebuild pipeline for intellectual_dna data layers.

2026 Architecture:
- facts/       → Immutable truth (parquet)
- vectors/     → Embeddings (LanceDB, 440MB)
- derived/     → Interpretations (v1/v2, rebuilable)

Usage:
    python pipelines/rebuild.py              # Show status
    python pipelines/rebuild.py all          # Rebuild everything
    python pipelines/rebuild.py facts        # Rebuild facts layer only
    python pipelines/rebuild.py vectors      # Sync to LanceDB
    python pipelines/rebuild.py derived      # Rebuild all interpretations
    python pipelines/rebuild.py derived-v2   # Rebuild only v2 (LLM-based)
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
PIPELINES_DIR = BASE_DIR / "pipelines"
PYTHON = BASE_DIR / "mordelab/02-monotropic-prosthetic/mcp-env/bin/python"

# Pipeline categories
FACTS_SCRIPTS = {
    'spend': PIPELINES_DIR / 'build_facts_spend.py',
    'brain-layers': PIPELINES_DIR / 'build_brain_layers.py',
    'temporal': PIPELINES_DIR / 'build_temporal_dim.py',
}

VECTORS_SCRIPTS = {
    'lancedb-migrate': PIPELINES_DIR / 'migrate_to_lancedb.py',
    'sync': BASE_DIR / 'live/sync.py',
}

# V2 interpretations (LLM-powered, ~$0.10-0.50 each)
V2_SCRIPTS = {
    'focus-v2': PIPELINES_DIR / 'build_focus_v2.py',
    'mvp-velocity-v2': PIPELINES_DIR / 'build_mvp_velocity_v2.py',
    'tool-stacks-v2': PIPELINES_DIR / 'build_tool_stacks_v2.py',
    'problem-resolution-v2': PIPELINES_DIR / 'build_problem_resolution_v2.py',
    'monthly-themes-v2': PIPELINES_DIR / 'build_monthly_themes_v2.py',
    'questions-v2': PIPELINES_DIR / 'build_questions_v2.py',
    'signature-phrases-v2': PIPELINES_DIR / 'build_signature_phrases_v2.py',
    'intellectual-evolution-v2': PIPELINES_DIR / 'build_intellectual_evolution_v2.py',
    'weekly-summaries': PIPELINES_DIR / 'build_weekly_summaries.py',
}

# V1 interpretations (rule-based, fast)
V1_SCRIPTS = {
    'focus-v1': PIPELINES_DIR / 'build_focus_v1.py',
    'mood-patterns': PIPELINES_DIR / 'build_mood_patterns.py',
    'daily-accomplishments': PIPELINES_DIR / 'build_daily_accomplishments.py',
    'phrases-v1': PIPELINES_DIR / 'build_phrases_v1.py',
    'project-arcs': PIPELINES_DIR / 'build_project_arcs.py',
    'personal-glossary': PIPELINES_DIR / 'build_personal_glossary.py',
}

# All scripts combined
ALL_SCRIPTS = {**FACTS_SCRIPTS, **V1_SCRIPTS, **V2_SCRIPTS}


def run_script(name: str, script_path: Path) -> bool:
    """Run a pipeline script and return success status."""
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"  Script:  {script_path.name}")
    print(f"  Time:    {datetime.now().strftime('%H:%M:%S')}")
    print('='*60)

    if not script_path.exists():
        print(f"  ERROR: Script not found: {script_path}")
        return False

    try:
        result = subprocess.run(
            [str(PYTHON), str(script_path)],
            cwd=str(BASE_DIR),
            capture_output=False
        )
        return result.returncode == 0
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def show_status():
    """Show current status of all data layers."""
    print("\n" + "="*60)
    print("  INTELLECTUAL DNA - Data Layer Status")
    print("="*60)

    # Check facts/spend
    spend_dir = BASE_DIR / "data/facts/spend"
    if spend_dir.exists():
        files = list(spend_dir.glob("*.parquet"))
        print(f"\n📊 facts/spend/: {len(files)} parquets")
        for f in sorted(files):
            size = f.stat().st_size / 1024
            print(f"   - {f.name}: {size:.1f} KB")
    else:
        print("\n📊 facts/spend/: NOT BUILT")

    # Check brain layers
    brain_dir = BASE_DIR / "data/facts/brain"
    if brain_dir.exists():
        print(f"\n🧠 facts/brain/:")
        for layer in ['index.parquet', 'summary.parquet', 'content.parquet']:
            path = brain_dir / layer
            if path.exists():
                size = path.stat().st_size / (1024*1024)
                print(f"   - {layer}: {size:.1f} MB")
            else:
                print(f"   - {layer}: NOT FOUND")
        deep = brain_dir / "deep"
        if deep.exists():
            links = list(deep.glob("*"))
            print(f"   - deep/: {len(links)} sources linked")
    else:
        print("\n🧠 facts/brain/: NOT BUILT")

    # Check temporal
    temporal_path = BASE_DIR / "data/facts/temporal_dim.parquet"
    if temporal_path.exists():
        size = temporal_path.stat().st_size / 1024
        print(f"\n📅 temporal_dim.parquet: {size:.1f} KB")
    else:
        print("\n📅 temporal_dim.parquet: NOT BUILT")

    # Check interpretations
    interp_dir = BASE_DIR / "data/interpretations"
    if interp_dir.exists():
        print(f"\n💡 interpretations/:")
        for version_dir in sorted(interp_dir.rglob("*/v*")):
            if version_dir.is_dir():
                parquets = list(version_dir.glob("*.parquet"))
                name = f"{version_dir.parent.name}/{version_dir.name}"
                if parquets:
                    print(f"   - {name}: {len(parquets)} parquet(s)")
                else:
                    print(f"   - {name}: (no parquets)")
    else:
        print("\n💡 interpretations/: NOT BUILT")

    # Check seed
    seed_path = BASE_DIR / "data/seed/principles.json"
    if seed_path.exists():
        if seed_path.is_symlink():
            target = seed_path.resolve()
            print(f"\n🌱 seed/principles.json → {target.name}")
        else:
            print(f"\n🌱 seed/principles.json: exists")
    else:
        print("\n🌱 seed/: NOT BUILT")

    print("\n" + "="*60)


def rebuild_category(name: str, scripts: dict) -> bool:
    """Rebuild all scripts in a category."""
    print(f"\n🔄 REBUILDING: {name.upper()}")
    print("="*60)

    success = True
    for script_name, script_path in scripts.items():
        if script_path.exists():
            if not run_script(script_name, script_path):
                print(f"\n⚠️ {script_name} failed, continuing...")
                success = False
        else:
            print(f"\n⏭️ Skipping {script_name} (not found)")

    return success


def rebuild_all():
    """Rebuild all layers in correct order: facts → vectors → derived."""
    print("\n🔄 REBUILDING ALL DATA LAYERS")
    print("="*60)

    # 1. Facts first (immutable base)
    rebuild_category("facts", FACTS_SCRIPTS)

    # 2. Vectors (embeddings)
    print("\n📊 Vectors: Run 'python live/sync.py --all' for full embed")

    # 3. Derived interpretations
    rebuild_category("derived-v1", V1_SCRIPTS)
    rebuild_category("derived-v2", V2_SCRIPTS)

    print("\n" + "="*60)
    print("  ✅ REBUILD COMPLETE")
    print("="*60)


def main():
    if len(sys.argv) < 2:
        show_status()
        print("\nUsage:")
        print("  python pipelines/rebuild.py status       # Show status")
        print("  python pipelines/rebuild.py all          # Rebuild everything")
        print("  python pipelines/rebuild.py facts        # Rebuild facts only")
        print("  python pipelines/rebuild.py derived      # Rebuild all interpretations")
        print("  python pipelines/rebuild.py derived-v1   # Rebuild v1 (fast)")
        print("  python pipelines/rebuild.py derived-v2   # Rebuild v2 (LLM, $$$)")
        print("  python pipelines/rebuild.py <script>     # Run specific script")
        return

    cmd = sys.argv[1].lower()

    if cmd == 'status':
        show_status()
    elif cmd == 'all':
        rebuild_all()
    elif cmd == 'facts':
        rebuild_category("facts", FACTS_SCRIPTS)
    elif cmd == 'vectors':
        rebuild_category("vectors", VECTORS_SCRIPTS)
    elif cmd == 'derived':
        rebuild_category("derived-v1", V1_SCRIPTS)
        rebuild_category("derived-v2", V2_SCRIPTS)
    elif cmd == 'derived-v1':
        rebuild_category("derived-v1", V1_SCRIPTS)
    elif cmd == 'derived-v2':
        rebuild_category("derived-v2", V2_SCRIPTS)
    elif cmd in ALL_SCRIPTS:
        run_script(cmd, ALL_SCRIPTS[cmd])
    else:
        print(f"Unknown command: {cmd}")
        print("Available categories: all, facts, vectors, derived, derived-v1, derived-v2")
        print(f"Available scripts: {', '.join(ALL_SCRIPTS.keys())}")
        sys.exit(1)


if __name__ == '__main__':
    main()
