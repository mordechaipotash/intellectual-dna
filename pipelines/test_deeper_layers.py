#!/usr/bin/env python3
"""
Prototype deeper cognitive layers:
1. Belief extraction - What do you believe about X?
2. SEED grounding - Instances of principles in action
3. Reasoning patterns - How you argue/think
4. Contradiction detection - Where views conflict
5. Compression formulas - Mental shortcuts created
"""

import os
import json
import time
import duckdb
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(Path("/Users/mordechai/intellectual_dna/.env"))

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
DATA_DIR = BASE_DIR / "data"
FACTS_DIR = DATA_DIR / "facts"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash-lite"

def call_llm(system: str, user: str, max_tokens: int = 600) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Error: {str(e)[:50]}]"


def get_sample_messages(month: str, limit: int = 40) -> list:
    """Get sample messages from a specific month."""
    con = duckdb.connect()
    rows = con.execute(f"""
        SELECT c.full_content
        FROM '{FACTS_DIR}/brain/content.parquet' c
        JOIN '{FACTS_DIR}/brain/index.parquet' idx ON c.event_id = idx.event_id
        WHERE c.event_type = 'message'
          AND idx.subtype = 'user'
          AND c.full_content IS NOT NULL
          AND LENGTH(c.full_content) BETWEEN 50 AND 800
          AND STRFTIME(idx.timestamp::TIMESTAMP, '%Y-%m') = '{month}'
        ORDER BY idx.timestamp::TIMESTAMP
        LIMIT {limit}
    """).fetchall()
    return [r[0] for r in rows]


def test_belief_extraction():
    """Test extracting beliefs from messages."""
    print("\n" + "="*60)
    print("🧠 TESTING: BELIEF EXTRACTION")
    print("="*60)

    # Get messages from a rich technical month
    messages = get_sample_messages("2024-08", 50)
    sample = "\n---\n".join(messages[:30])

    approaches = {
        "v1_direct": """Extract beliefs from these messages.
For each belief found:
BELIEF: [what the person believes]
EVIDENCE: [quote or paraphrase showing this]
STRENGTH: strong/moderate/tentative""",

        "v2_implicit": """Find IMPLICIT beliefs - things assumed true but not stated directly.
Look for:
- Assumptions behind questions
- Values revealed by choices
- Worldview glimpses

For each:
IMPLICIT BELIEF: [unstated assumption]
REVEALED BY: [what shows this]""",

        "v3_positions": """Identify POSITIONS this person holds on topics.
A position is a stance, not just a fact.

For each position:
TOPIC: [what domain]
POSITION: [their stance]
INDICATOR: [evidence from messages]"""
    }

    results = {}
    for name, system in approaches.items():
        print(f"\n  Testing {name}...")
        response = call_llm(system, f"Messages:\n{sample}")

        # Count extractions
        lines = [l for l in response.split('\n') if l.strip() and ':' in l]
        results[name] = {
            "response": response[:500],
            "extraction_count": len([l for l in lines if l.startswith(('BELIEF:', 'IMPLICIT', 'TOPIC:', 'POSITION:'))]),
            "raw_lines": len(lines)
        }
        print(f"    Extractions: {results[name]['extraction_count']}")
        time.sleep(2)

    return results


def test_seed_grounding():
    """Test finding SEED principles in action."""
    print("\n" + "="*60)
    print("🌱 TESTING: SEED PRINCIPLE GROUNDING")
    print("="*60)

    # SEED principles
    principles = {
        "inversion": "Flipping problems upside down, asking the opposite question",
        "compression": "Distilling complex ideas into simple formulas or phrases",
        "bottleneck": "Identifying the single constraint that limits everything else",
        "agency": "Taking ownership, not waiting for permission or conditions"
    }

    messages = get_sample_messages("2024-07", 50)
    sample = "\n---\n".join(messages[:30])

    approaches = {
        "v1_labeled": f"""Find examples of these thinking patterns in the messages:

INVERSION: {principles['inversion']}
COMPRESSION: {principles['compression']}
BOTTLENECK: {principles['bottleneck']}
AGENCY: {principles['agency']}

For each found:
PRINCIPLE: [which one]
INSTANCE: [quote or description]
CONTEXT: [what they were doing]""",

        "v2_unlabeled": """Find examples of these REASONING MOVES in the messages:
- Flipping a problem to see it differently
- Compressing something complex into a simple rule
- Finding the ONE thing that matters most
- Taking action without waiting

For each:
MOVE: [what type]
EXAMPLE: [from messages]""",

        "v3_trigger": """Look for these TRIGGER PHRASES that signal meta-cognition:
- "the real question is..."
- "what if we flip..."
- "it all comes down to..."
- "instead of waiting..."
- "the bottleneck is..."
- "a simpler way to say this..."

For each found:
TRIGGER: [phrase or pattern]
CONTEXT: [what follows]
PRINCIPLE: [inversion/compression/bottleneck/agency/other]"""
    }

    results = {}
    for name, system in approaches.items():
        print(f"\n  Testing {name}...")
        response = call_llm(system, f"Messages:\n{sample}")

        lines = [l for l in response.split('\n') if l.strip() and ':' in l]
        results[name] = {
            "response": response[:500],
            "extraction_count": len([l for l in lines if l.startswith(('PRINCIPLE:', 'MOVE:', 'TRIGGER:', 'INSTANCE:', 'EXAMPLE:'))]),
            "raw_lines": len(lines)
        }
        print(f"    Extractions: {results[name]['extraction_count']}")
        time.sleep(2)

    return results


def test_reasoning_patterns():
    """Test extracting reasoning/argumentation patterns."""
    print("\n" + "="*60)
    print("🔗 TESTING: REASONING PATTERNS")
    print("="*60)

    messages = get_sample_messages("2024-06", 50)
    sample = "\n---\n".join(messages[:30])

    approaches = {
        "v1_logic": """Identify REASONING PATTERNS in how this person thinks.
Look for:
- How they justify conclusions
- What counts as evidence for them
- How they handle uncertainty

For each pattern:
PATTERN: [name it]
EXAMPLE: [from messages]
FREQUENCY: common/occasional""",

        "v2_moves": """Find ARGUMENTATIVE MOVES - how this person builds or challenges ideas.
Types to look for:
- Analogies ("X is like Y")
- First principles ("fundamentally...")
- Counterexamples ("but what about...")
- Reframes ("another way to see this...")

For each:
MOVE TYPE: [which]
INSTANCE: [quote/paraphrase]""",

        "v3_heuristics": """Extract DECISION HEURISTICS - mental shortcuts this person uses.
Look for:
- Rules of thumb
- Default preferences
- Quick filters

For each:
HEURISTIC: [the rule]
DOMAIN: [where applied]
EVIDENCE: [from messages]"""
    }

    results = {}
    for name, system in approaches.items():
        print(f"\n  Testing {name}...")
        response = call_llm(system, f"Messages:\n{sample}")

        lines = [l for l in response.split('\n') if l.strip() and ':' in l]
        results[name] = {
            "response": response[:500],
            "extraction_count": len([l for l in lines if l.startswith(('PATTERN:', 'MOVE', 'HEURISTIC:', 'INSTANCE:', 'EXAMPLE:'))]),
            "raw_lines": len(lines)
        }
        print(f"    Extractions: {results[name]['extraction_count']}")
        time.sleep(2)

    return results


def test_contradiction_detection():
    """Test finding contradictions across time periods."""
    print("\n" + "="*60)
    print("⚡ TESTING: CONTRADICTION DETECTION")
    print("="*60)

    # Get messages from two different periods
    early = get_sample_messages("2023-06", 30)
    late = get_sample_messages("2024-10", 30)

    early_sample = "\n---\n".join(early[:20])
    late_sample = "\n---\n".join(late[:20])

    approaches = {
        "v1_compare": """Compare these two time periods and find CHANGES or CONTRADICTIONS.

EARLIER (2023):
{early}

LATER (2024):
{later}

For each change:
TOPIC: [what area]
EARLIER VIEW: [position then]
LATER VIEW: [position now]
TYPE: evolution/contradiction/abandonment""",

        "v2_tension": """Find TENSIONS between earlier and later thinking.
Look for:
- Changed priorities
- Reversed positions
- Abandoned interests
- New frameworks

For each:
TENSION: [describe it]
EVIDENCE EARLY: [quote/paraphrase]
EVIDENCE LATE: [quote/paraphrase]""",

        "v3_drift": """Detect BELIEF DRIFT - subtle shifts in position over time.

EARLIER (2023):
{early}

LATER (2024):
{later}

For each drift:
DOMAIN: [topic area]
DIRECTION: [how it shifted]
EARLY SIGNAL: [evidence]
LATE SIGNAL: [evidence]"""
    }

    results = {}
    for name, system in approaches.items():
        print(f"\n  Testing {name}...")
        prompt = system.format(early=early_sample, later=late_sample)
        response = call_llm(prompt, "Analyze the comparison above.")

        lines = [l for l in response.split('\n') if l.strip() and ':' in l]
        results[name] = {
            "response": response[:500],
            "extraction_count": len([l for l in lines if l.startswith(('TOPIC:', 'TENSION:', 'DOMAIN:', 'DRIFT:', 'TYPE:', 'DIRECTION:'))]),
            "raw_lines": len(lines)
        }
        print(f"    Extractions: {results[name]['extraction_count']}")
        time.sleep(2)

    return results


def test_compression_formulas():
    """Test extracting compression formulas / mental shortcuts."""
    print("\n" + "="*60)
    print("📦 TESTING: COMPRESSION FORMULAS")
    print("="*60)

    messages = get_sample_messages("2024-09", 50)
    sample = "\n---\n".join(messages[:30])

    approaches = {
        "v1_formulas": """Find COMPRESSION FORMULAS - where complex ideas get distilled.
Look for phrases like:
- "X is just Y"
- "The key is..."
- "It all comes down to..."
- "The pattern is..."

For each:
FORMULA: [the compressed insight]
FULL CONTEXT: [what it compresses]""",

        "v2_rules": """Extract PERSONAL RULES - guidelines this person has created.
Look for:
- "I always..."
- "Never do X without Y"
- "The rule is..."
- "My approach is..."

For each:
RULE: [the principle]
DOMAIN: [where it applies]""",

        "v3_mantras": """Find WORKING MANTRAS - repeated phrases that guide action.
These are condensed wisdom used during work.

For each:
MANTRA: [the phrase]
MEANING: [what it means in practice]
FREQUENCY: [if mentioned multiple times]"""
    }

    results = {}
    for name, system in approaches.items():
        print(f"\n  Testing {name}...")
        response = call_llm(system, f"Messages:\n{sample}")

        lines = [l for l in response.split('\n') if l.strip() and ':' in l]
        results[name] = {
            "response": response[:500],
            "extraction_count": len([l for l in lines if l.startswith(('FORMULA:', 'RULE:', 'MANTRA:', 'FULL', 'DOMAIN:', 'MEANING:'))]),
            "raw_lines": len(lines)
        }
        print(f"    Extractions: {results[name]['extraction_count']}")
        time.sleep(2)

    return results


def run_all_tests():
    """Run all prototype tests and summarize."""
    print("="*70)
    print("🔬 DEEPER LAYER PROTOTYPES - TESTING")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")

    all_results = {}

    all_results["belief"] = test_belief_extraction()
    all_results["seed"] = test_seed_grounding()
    all_results["reasoning"] = test_reasoning_patterns()
    all_results["contradiction"] = test_contradiction_detection()
    all_results["compression"] = test_compression_formulas()

    # Summary
    print("\n" + "="*70)
    print("📊 RESULTS SUMMARY")
    print("="*70)

    for layer, approaches in all_results.items():
        print(f"\n{layer.upper()}:")
        best = max(approaches.items(), key=lambda x: x[1]['extraction_count'])
        for name, data in approaches.items():
            marker = "👑" if name == best[0] else "  "
            print(f"  {marker} {name}: {data['extraction_count']} extractions ({data['raw_lines']} lines)")

    # Show best approach samples
    print("\n" + "="*70)
    print("🏆 BEST APPROACH SAMPLES")
    print("="*70)

    for layer, approaches in all_results.items():
        best_name, best_data = max(approaches.items(), key=lambda x: x[1]['extraction_count'])
        print(f"\n{layer.upper()} - {best_name}:")
        print("-"*40)
        print(best_data['response'][:400])
        print("...")

    return all_results


if __name__ == '__main__':
    run_all_tests()
