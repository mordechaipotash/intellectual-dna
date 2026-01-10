#!/usr/bin/env python3
"""
Phase 6 progress dashboard.
Usage: python pipelines/status_phase6.py
"""

from pathlib import Path

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
DATA_DIR = BASE_DIR / "data"
MCP_SERVER = BASE_DIR / "mordelab/02-monotropic-prosthetic/mcp_brain_server.py"

def check_mcp_tools():
    """Check which Phase 6 MCP tools exist."""
    tools = {
        'query_focus': False,
        'query_spend': False,
        'query_timeline': False,
    }

    if MCP_SERVER.exists():
        content = MCP_SERVER.read_text()
        for tool in tools:
            if f"def {tool}" in content:
                tools[tool] = True

    return tools

def check_interpretations():
    """Check interpretation versions."""
    interp_dir = DATA_DIR / "interpretations"
    versions = {}

    # Focus versions
    focus_v1 = interp_dir / "focus/v1/daily.parquet"
    focus_v2 = interp_dir / "focus/v2/daily.parquet"
    versions['focus/v1'] = focus_v1.exists()
    versions['focus/v2'] = focus_v2.exists()

    # Phrases
    phrases_v1 = interp_dir / "phrases/v1/phrases.parquet"
    versions['phrases/v1'] = phrases_v1.exists()

    return versions

def check_gemini():
    """Check if Gemini client exists."""
    client_path = BASE_DIR / "pipelines/utils/gemini_client.py"
    return client_path.exists()

def main():
    print("\n" + "="*50)
    print("  PHASE 6 PROGRESS DASHBOARD")
    print("="*50)

    # MCP Tools (6A)
    print("\n📡 6A: MCP TOOLS")
    tools = check_mcp_tools()
    for tool, exists in tools.items():
        status = "✅" if exists else "⬜"
        print(f"   {status} {tool}()")
    done = sum(tools.values())
    print(f"   Progress: {done}/3 tools")

    # Phrases (6E)
    print("\n🗣️  6E: PHRASE EXTRACTION")
    interps = check_interpretations()
    status = "✅" if interps.get('phrases/v1') else "⬜"
    print(f"   {status} phrases/v1")

    # Focus v2 (6C)
    print("\n🧠 6C: FOCUS/V2 (LLM)")
    gemini = check_gemini()
    print(f"   {'✅' if gemini else '⬜'} Gemini client")
    print(f"   {'✅' if interps.get('focus/v2') else '⬜'} focus/v2 parquet")

    # Overall
    print("\n" + "-"*50)
    total_tasks = 6  # 3 tools + phrases + gemini + focus_v2
    done_tasks = sum(tools.values()) + (1 if interps.get('phrases/v1') else 0) + (1 if gemini else 0) + (1 if interps.get('focus/v2') else 0)
    pct = int(100 * done_tasks / total_tasks)
    bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
    print(f"  OVERALL: {bar} {pct}% ({done_tasks}/{total_tasks})")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
