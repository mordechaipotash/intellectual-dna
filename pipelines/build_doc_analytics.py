#!/usr/bin/env python3
"""
INTELLECTUAL ANALYTICS PIPELINE
Catch up on 3 years of output using Gemini 3 Flash.

Analyzes:
- 137 distilled markdown docs (998K words)
- 99 GitHub READMEs (349KB)

Outputs:
- INVENTORY.md - Every doc explained
- TIMELINE.md - Chronological evolution
- DOMAINS/*.md - Deep dives per domain
- PROJECT_STATUS.md - GitHub repo states
- SYNTHESIS.md - Master catch-up doc

Usage:
    python -m pipelines.build_doc_analytics              # Full run
    python -m pipelines.build_doc_analytics --phase 1    # Just inventory
    python -m pipelines.build_doc_analytics --dry-run    # Preview only
"""

import json
import os
import re
import time
from pathlib import Path
from datetime import datetime

import duckdb
import httpx

# Config
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "obsidian_distilled" / "meta"
CACHE_FILE = OUTPUT_DIR / "analytics_cache.json"

MODEL = "google/gemini-3-flash-preview"
API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Gemini 3 Flash pricing
INPUT_COST_PER_M = 0.50
OUTPUT_COST_PER_M = 3.00

# Schema for document analysis
DOC_ANALYTICS_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["framework", "project", "insight", "exploration", "operational", "reference"]
        },
        "era": {
            "type": "string",
            "description": "When created: 2023-early, 2023-late, 2024-early, 2024-late, 2025-H1, 2025-H2, 2026, or unknown"
        },
        "maturity": {
            "type": "string",
            "enum": ["seed", "growing", "mature", "archived"]
        },
        "domains": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Domains: Tax, SHELET, Neurodivergence, Agency, Jewish, Israel, Startup, Conductor, AI, Compression, Bottleneck, Prosthetic"
        },
        "one_sentence": {
            "type": "string",
            "description": "What this doc IS in plain English (max 150 chars)"
        },
        "why_it_matters": {
            "type": "string",
            "description": "Why past-Mordechai created this (max 200 chars)"
        },
        "key_insight": {
            "type": "string",
            "description": "The single best insight from this doc"
        },
        "best_quote": {
            "type": "string",
            "description": "Best direct quote from Mordechai's words"
        },
        "status": {
            "type": "string",
            "enum": ["active", "dormant", "completed", "abandoned", "evergreen"]
        },
        "connections": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Related concepts or themes"
        }
    },
    "required": ["type", "maturity", "domains", "one_sentence", "status"]
}


def call_gemini(prompt: str, schema: dict = None, max_tokens: int = 1000) -> dict:
    """Call Gemini 3 Flash with optional JSON schema."""
    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")

    messages = [{"role": "user", "content": prompt}]

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    if schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "doc_analysis",
                "strict": True,
                "schema": schema
            }
        }

    # Add thinking for better quality
    payload["extra_body"] = {"thinking": {"type": "enabled", "budget_tokens": 500}}

    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "X-Title": "Intellectual DNA Analytics"
        },
        json=payload,
        timeout=60.0
    )

    if response.status_code != 200:
        print(f"API error: {response.status_code} - {response.text[:200]}")
        return None

    data = response.json()
    content = data["choices"][0]["message"]["content"]

    # Track usage
    usage = data.get("usage", {})

    if schema:
        try:
            return {"result": json.loads(content), "usage": usage}
        except json.JSONDecodeError:
            print(f"JSON parse error: {content[:200]}")
            return None

    return {"result": content, "usage": usage}


def load_markdown_docs() -> list[dict]:
    """Load all distilled markdown docs."""
    parquet_path = DATA_DIR / "distilled" / "mordechai_ip_gems.parquet"

    if not parquet_path.exists():
        print(f"❌ Not found: {parquet_path}")
        return []

    con = duckdb.connect()
    docs = con.execute(f"""
        SELECT
            filename,
            ip_type,
            depth_score,
            energy,
            word_count,
            content
        FROM '{parquet_path}'
        ORDER BY depth_score DESC
    """).fetchall()

    return [
        {
            "source": "markdown",
            "filename": d[0],
            "ip_type": d[1],
            "depth_score": d[2],
            "energy": d[3],
            "word_count": d[4],
            "content": d[5][:8000] if d[5] else ""  # Truncate for API
        }
        for d in docs
    ]


def load_github_readmes() -> list[dict]:
    """Load all GitHub READMEs."""
    parquet_path = DATA_DIR / "github_readmes.parquet"

    if not parquet_path.exists():
        print(f"❌ Not found: {parquet_path}")
        return []

    con = duckdb.connect()
    docs = con.execute(f"""
        SELECT
            repo_name,
            filename,
            content
        FROM '{parquet_path}'
        WHERE content IS NOT NULL AND LENGTH(content) > 100
    """).fetchall()

    return [
        {
            "source": "github",
            "filename": f"{d[0]}/{d[1]}",
            "repo_name": d[0],
            "content": d[2][:6000] if d[2] else ""  # Truncate for API
        }
        for d in docs
    ]


def analyze_doc(doc: dict, cache: dict) -> dict:
    """Analyze a single document with Gemini."""
    cache_key = doc["filename"]

    if cache_key in cache:
        return cache[cache_key]

    # Build prompt
    if doc["source"] == "markdown":
        prompt = f"""Analyze this intellectual document from Mordechai's 3-year archive.

FILENAME: {doc['filename']}
IP_TYPE: {doc.get('ip_type', 'unknown')}
DEPTH_SCORE: {doc.get('depth_score', 0)}
ENERGY: {doc.get('energy', 'unknown')}
WORD_COUNT: {doc.get('word_count', 0)}

CONTENT:
{doc['content']}

---
Analyze this document. What is it? Why did Mordechai create it? What's the key insight?
Extract the best direct quote from his words."""
    else:
        prompt = f"""Analyze this GitHub project README from Mordechai's repos.

REPO: {doc.get('repo_name', 'unknown')}
FILENAME: {doc['filename']}

CONTENT:
{doc['content']}

---
Analyze this project. What does it do? What's its current status (active/dormant/completed/abandoned)?
What domains does it relate to? What's the key insight about why this was built?"""

    result = call_gemini(prompt, schema=DOC_ANALYTICS_SCHEMA)

    if result and result.get("result"):
        analysis = result["result"]
        analysis["_usage"] = result.get("usage", {})
        cache[cache_key] = analysis
        return analysis

    return None


def run_phase1_inventory(docs: list[dict], dry_run: bool = False) -> dict:
    """Phase 1: Build inventory of all docs."""
    print(f"\n📦 PHASE 1: INVENTORY ({len(docs)} docs)")
    print("=" * 50)

    # Load cache
    cache = {}
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())
        print(f"  Loaded {len(cache)} cached analyses")

    results = []
    total_input_tokens = 0
    total_output_tokens = 0

    for i, doc in enumerate(docs):
        if doc["filename"] in cache:
            results.append({"doc": doc, "analysis": cache[doc["filename"]]})
            continue

        if dry_run:
            print(f"  [{i+1}/{len(docs)}] Would analyze: {doc['filename'][:50]}")
            continue

        print(f"  [{i+1}/{len(docs)}] Analyzing: {doc['filename'][:50]}...", end=" ", flush=True)

        analysis = analyze_doc(doc, cache)

        if analysis:
            results.append({"doc": doc, "analysis": analysis})
            usage = analysis.get("_usage", {})
            total_input_tokens += usage.get("prompt_tokens", 0)
            total_output_tokens += usage.get("completion_tokens", 0)
            print(f"✓ {analysis.get('type', '?')}")

            # Save cache periodically
            if i % 10 == 0:
                CACHE_FILE.write_text(json.dumps(cache, indent=2, default=str))
        else:
            print("✗")

        time.sleep(0.3)  # Rate limit

    # Final cache save
    CACHE_FILE.write_text(json.dumps(cache, indent=2, default=str))

    # Cost calculation
    input_cost = (total_input_tokens / 1_000_000) * INPUT_COST_PER_M
    output_cost = (total_output_tokens / 1_000_000) * OUTPUT_COST_PER_M
    total_cost = input_cost + output_cost

    print(f"\n  📊 Tokens: {total_input_tokens:,} in / {total_output_tokens:,} out")
    print(f"  💰 Cost: ${total_cost:.4f}")

    return {"results": results, "cache": cache, "cost": total_cost}


def generate_inventory_md(cache: dict) -> str:
    """Generate INVENTORY.md from cache."""
    lines = [
        "# Document Inventory",
        f"*{len(cache)} documents analyzed | Generated {datetime.now().strftime('%Y-%m-%d')}*\n",
        "---\n",
    ]

    # Group by type
    by_type = {}
    for filename, analysis in cache.items():
        doc_type = analysis.get("type", "unknown")
        if doc_type not in by_type:
            by_type[doc_type] = []
        by_type[doc_type].append((filename, analysis))

    for doc_type in ["framework", "project", "insight", "exploration", "operational", "reference", "unknown"]:
        if doc_type not in by_type:
            continue

        docs = by_type[doc_type]
        lines.append(f"\n## {doc_type.upper()} ({len(docs)})\n")

        for filename, analysis in sorted(docs, key=lambda x: x[0]):
            one_sentence = analysis.get("one_sentence", "No description")
            status = analysis.get("status", "unknown")
            domains = ", ".join(analysis.get("domains", [])[:3])

            lines.append(f"### {filename}")
            lines.append(f"**{one_sentence}**\n")
            lines.append(f"- Status: `{status}` | Domains: {domains}")

            if analysis.get("key_insight"):
                lines.append(f"- 💡 {analysis['key_insight']}")
            if analysis.get("best_quote"):
                lines.append(f"- > \"{analysis['best_quote']}\"")

            lines.append("")

    return "\n".join(lines)


def generate_timeline_md(cache: dict) -> str:
    """Generate TIMELINE.md showing chronological evolution."""
    lines = [
        "# Timeline: 3-Year Evolution",
        f"*Generated {datetime.now().strftime('%Y-%m-%d')}*\n",
        "---\n",
    ]

    # Group by era
    by_era = {}
    for filename, analysis in cache.items():
        era = analysis.get("era", "unknown")
        if era not in by_era:
            by_era[era] = []
        by_era[era].append((filename, analysis))

    era_order = ["2023-early", "2023-late", "2024-early", "2024-late", "2025-H1", "2025-H2", "2026", "unknown"]

    for era in era_order:
        if era not in by_era:
            continue

        docs = by_era[era]
        lines.append(f"\n## {era} ({len(docs)} docs)\n")

        # Group by maturity within era
        mature = [d for d in docs if d[1].get("maturity") == "mature"]
        growing = [d for d in docs if d[1].get("maturity") == "growing"]
        seeds = [d for d in docs if d[1].get("maturity") == "seed"]
        archived = [d for d in docs if d[1].get("maturity") == "archived"]

        if mature:
            lines.append(f"**Mature ({len(mature)}):** " + ", ".join([d[0][:30] for d in mature[:5]]))
        if growing:
            lines.append(f"**Growing ({len(growing)}):** " + ", ".join([d[0][:30] for d in growing[:5]]))
        if seeds:
            lines.append(f"**Seeds ({len(seeds)}):** " + ", ".join([d[0][:30] for d in seeds[:5]]))
        if archived:
            lines.append(f"**Archived ({len(archived)}):** " + ", ".join([d[0][:30] for d in archived[:5]]))

        lines.append("")

    return "\n".join(lines)


def generate_domain_md(cache: dict, domain: str) -> str:
    """Generate a domain deep-dive markdown."""
    # Find all docs in this domain
    domain_docs = []
    for filename, analysis in cache.items():
        domains = analysis.get("domains", [])
        if any(domain.lower() in d.lower() for d in domains):
            domain_docs.append((filename, analysis))

    lines = [
        f"# Domain: {domain}",
        f"*{len(domain_docs)} documents | Generated {datetime.now().strftime('%Y-%m-%d')}*\n",
        "---\n",
    ]

    # Key insights
    lines.append("## Key Insights\n")
    for filename, analysis in domain_docs[:10]:
        if analysis.get("key_insight"):
            lines.append(f"- 💡 {analysis['key_insight']}")

    # Best quotes
    lines.append("\n## Best Quotes\n")
    for filename, analysis in domain_docs[:10]:
        if analysis.get("best_quote"):
            lines.append(f"> \"{analysis['best_quote']}\"")
            lines.append(f"*— {filename}*\n")

    # Document list
    lines.append("\n## Documents\n")
    for filename, analysis in sorted(domain_docs, key=lambda x: x[1].get("maturity", "z")):
        status = analysis.get("status", "?")
        one_sentence = analysis.get("one_sentence", "")[:80]
        lines.append(f"- **{filename}** [{status}]: {one_sentence}")

    return "\n".join(lines)


def generate_synthesis_md(cache: dict) -> str:
    """Generate the master SYNTHESIS.md catch-up document."""
    lines = [
        "# 🧠 SYNTHESIS: Your 3-Year Intellectual Journey",
        f"*{len(cache)} documents analyzed | Generated {datetime.now().strftime('%Y-%m-%d')}*\n",
        "---\n",
    ]

    # Stats
    by_type = {}
    by_status = {}
    by_domain = {}
    all_insights = []
    all_quotes = []

    for filename, analysis in cache.items():
        # Count by type
        t = analysis.get("type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

        # Count by status
        s = analysis.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1

        # Count by domain
        for d in analysis.get("domains", []):
            by_domain[d] = by_domain.get(d, 0) + 1

        # Collect insights
        if analysis.get("key_insight"):
            all_insights.append((filename, analysis["key_insight"]))
        if analysis.get("best_quote"):
            all_quotes.append((filename, analysis["best_quote"]))

    lines.append("## 📊 Overview\n")
    lines.append(f"**Total Documents:** {len(cache)}\n")

    lines.append("### By Type")
    for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
        lines.append(f"- {t}: {count}")

    lines.append("\n### By Status")
    for s, count in sorted(by_status.items(), key=lambda x: -x[1]):
        emoji = {"active": "🟢", "evergreen": "🌲", "completed": "✅", "dormant": "😴", "abandoned": "❌"}.get(s, "❓")
        lines.append(f"- {emoji} {s}: {count}")

    lines.append("\n### Top Domains")
    for d, count in sorted(by_domain.items(), key=lambda x: -x[1])[:10]:
        lines.append(f"- {d}: {count} docs")

    lines.append("\n---\n")
    lines.append("## 💎 Crown Jewel Insights\n")
    for filename, insight in all_insights[:15]:
        lines.append(f"- 💡 {insight}")
        lines.append(f"  *({filename})*\n")

    lines.append("\n---\n")
    lines.append("## 🗣️ Best Quotes (Your Words)\n")
    for filename, quote in all_quotes[:10]:
        lines.append(f"> \"{quote}\"")
        lines.append(f"*— {filename}*\n")

    lines.append("\n---\n")
    lines.append("## 🎯 What To Focus On\n")

    active = [(f, a) for f, a in cache.items() if a.get("status") == "active"]
    dormant = [(f, a) for f, a in cache.items() if a.get("status") == "dormant"]

    lines.append(f"### Active Projects ({len(active)})")
    for filename, analysis in active[:10]:
        lines.append(f"- **{filename}**: {analysis.get('one_sentence', '')[:60]}")

    lines.append(f"\n### Dormant (Could Revive) ({len(dormant)})")
    for filename, analysis in dormant[:10]:
        lines.append(f"- {filename}: {analysis.get('one_sentence', '')[:60]}")

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Intellectual Analytics Pipeline")
    parser.add_argument("--phase", type=int, help="Run specific phase (1-5)")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--generate", action="store_true", help="Generate outputs from cache")
    args = parser.parse_args()

    print("🧠 INTELLECTUAL ANALYTICS PIPELINE")
    print("=" * 50)

    # Ensure output dir exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "domains").mkdir(exist_ok=True)

    if args.generate:
        # Generate outputs from existing cache
        if not CACHE_FILE.exists():
            print("❌ No cache file found. Run analysis first.")
            return

        cache = json.loads(CACHE_FILE.read_text())
        print(f"📂 Loaded {len(cache)} cached analyses")

        # Generate all outputs
        print("\n📝 Generating INVENTORY.md...")
        (OUTPUT_DIR / "INVENTORY.md").write_text(generate_inventory_md(cache))

        print("📝 Generating TIMELINE.md...")
        (OUTPUT_DIR / "TIMELINE.md").write_text(generate_timeline_md(cache))

        print("📝 Generating domain deep-dives...")
        for domain in ["Tax", "SHELET", "Neurodivergence", "Agency", "Jewish", "Israel", "AI", "Compression"]:
            (OUTPUT_DIR / "domains" / f"{domain.upper()}.md").write_text(generate_domain_md(cache, domain))

        print("📝 Generating SYNTHESIS.md...")
        (OUTPUT_DIR / "SYNTHESIS.md").write_text(generate_synthesis_md(cache))

        print("\n✅ All outputs generated!")
        return

    # Load all docs
    print("\n📂 Loading documents...")
    markdown_docs = load_markdown_docs()
    github_docs = load_github_readmes()
    all_docs = markdown_docs + github_docs

    print(f"  Markdown docs: {len(markdown_docs)}")
    print(f"  GitHub READMEs: {len(github_docs)}")
    print(f"  Total: {len(all_docs)}")

    # Run phases
    if args.phase == 1 or args.phase is None:
        result = run_phase1_inventory(all_docs, dry_run=args.dry_run)

        if not args.dry_run and result["cache"]:
            # Generate outputs
            print("\n📝 Generating outputs...")
            (OUTPUT_DIR / "INVENTORY.md").write_text(generate_inventory_md(result["cache"]))
            (OUTPUT_DIR / "TIMELINE.md").write_text(generate_timeline_md(result["cache"]))
            (OUTPUT_DIR / "SYNTHESIS.md").write_text(generate_synthesis_md(result["cache"]))

            # Generate domain deep-dives
            for domain in ["Tax", "SHELET", "Neurodivergence", "Agency", "Jewish", "Israel", "AI", "Compression"]:
                (OUTPUT_DIR / "domains" / f"{domain.upper()}.md").write_text(generate_domain_md(result["cache"], domain))

            print("\n✅ Done! Check obsidian_distilled/meta/ for outputs.")


if __name__ == "__main__":
    main()
