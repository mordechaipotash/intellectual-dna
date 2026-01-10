#!/usr/bin/env python3
"""
Compare Gemini 2.5 Flash Lite vs Gemini 3 Flash Preview on 30 conversations.
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
FACTS_DIR = BASE_DIR / "data" / "facts"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "lite": {
        "name": "google/gemini-2.5-flash-lite",
        "input_cost": 0.075,  # per 1M tokens
        "output_cost": 0.30
    },
    "preview": {
        "name": "google/gemini-3-flash-preview",
        "input_cost": 0.50,
        "output_cost": 3.00
    }
}


def call_model(model_key: str, prompt: str, max_tokens: int = 300) -> tuple:
    """Call model and return (response, input_tokens, output_tokens)."""
    model = MODELS[model_key]

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-dna.local",
        "X-Title": "Gemini Model Comparison"
    }

    data = {
        "model": model["name"],
        "messages": [
            {
                "role": "system",
                "content": "You are analyzing conversation snippets to identify the main focus topics. Be concise and specific. Output a 1-2 sentence summary of what the person was thinking about."
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()

        content = result["choices"][0]["message"]["content"].strip()
        usage = result.get("usage", {})
        input_tokens = usage.get("prompt_tokens", len(prompt) // 4)
        output_tokens = usage.get("completion_tokens", len(content) // 4)

        return content, input_tokens, output_tokens
    except Exception as e:
        return f"[Error: {str(e)[:100]}]", 0, 0


def calculate_cost(model_key: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in dollars."""
    model = MODELS[model_key]
    return (input_tokens * model["input_cost"] / 1_000_000) + (output_tokens * model["output_cost"] / 1_000_000)


def run_comparison():
    """Compare both models on 30 random days."""
    print("=" * 70)
    print("GEMINI MODEL COMPARISON: 2.5 Flash Lite vs 3 Flash Preview")
    print("=" * 70)

    con = duckdb.connect()

    # Get 30 random days with good message counts
    days = con.execute(f"""
        SELECT
            CAST(idx.timestamp AS DATE) as date,
            LIST(LEFT(c.full_content, 500)) as previews,
            COUNT(*) as message_count
        FROM '{FACTS_DIR}/brain/content.parquet' c
        JOIN '{FACTS_DIR}/brain/index.parquet' idx ON c.event_id = idx.event_id
        WHERE c.event_type = 'message'
          AND c.full_content IS NOT NULL
          AND idx.timestamp >= '2024-01-01'
        GROUP BY CAST(idx.timestamp AS DATE)
        HAVING COUNT(*) >= 5
        ORDER BY RANDOM()
        LIMIT 30
    """).fetchall()

    print(f"\nTesting on {len(days)} days from 2024+\n")

    results = []
    lite_total_cost = 0.0
    preview_total_cost = 0.0
    lite_total_input = 0
    lite_total_output = 0
    preview_total_input = 0
    preview_total_output = 0

    for i, (date, previews, msg_count) in enumerate(days):
        sample_text = "\n---\n".join(previews[:10])
        prompt = f"""Date: {date}
Number of messages: {msg_count}

Sample messages from this day:
{sample_text}

What was the main focus or theme of thinking on this day?"""

        # Test both models
        print(f"[{i+1}/30] {date}...")

        lite_response, lite_in, lite_out = call_model("lite", prompt)
        time.sleep(2)  # Rate limit

        preview_response, preview_in, preview_out = call_model("preview", prompt)
        time.sleep(2)

        lite_cost = calculate_cost("lite", lite_in, lite_out)
        preview_cost = calculate_cost("preview", preview_in, preview_out)

        lite_total_cost += lite_cost
        preview_total_cost += preview_cost
        lite_total_input += lite_in
        lite_total_output += lite_out
        preview_total_input += preview_in
        preview_total_output += preview_out

        results.append({
            "date": str(date),
            "lite": lite_response,
            "preview": preview_response,
            "lite_cost": lite_cost,
            "preview_cost": preview_cost
        })

        # Show progress every 5
        if (i + 1) % 5 == 0:
            print(f"  Lite: ${lite_total_cost:.6f} | Preview: ${preview_total_cost:.6f}")

    # Summary
    print("\n" + "=" * 70)
    print("COST ANALYSIS")
    print("=" * 70)

    print(f"\n30 Conversations:")
    print(f"  Gemini 2.5 Flash Lite:    ${lite_total_cost:.6f}")
    print(f"  Gemini 3 Flash Preview:   ${preview_total_cost:.6f}")
    print(f"  Preview is {preview_total_cost/lite_total_cost:.1f}x more expensive")

    # Extrapolate to full dataset (957 days)
    scale = 957 / 30
    print(f"\nExtrapolated to 957 days:")
    print(f"  Gemini 2.5 Flash Lite:    ${lite_total_cost * scale:.4f}")
    print(f"  Gemini 3 Flash Preview:   ${preview_total_cost * scale:.4f}")
    print(f"  Difference:               ${(preview_total_cost - lite_total_cost) * scale:.4f}")

    # Token usage
    print(f"\nToken Usage (30 conversations):")
    print(f"  Lite:    {lite_total_input:,} input, {lite_total_output:,} output")
    print(f"  Preview: {preview_total_input:,} input, {preview_total_output:,} output")

    # Quality comparison - show samples
    print("\n" + "=" * 70)
    print("QUALITY COMPARISON (5 samples)")
    print("=" * 70)

    for r in results[:5]:
        print(f"\n📅 {r['date']}")
        print(f"  💰 Lite (${r['lite_cost']:.6f}):")
        print(f"     {r['lite'][:200]}...")
        print(f"  💎 Preview (${r['preview_cost']:.6f}):")
        print(f"     {r['preview'][:200]}...")

    # Save full results
    output_path = BASE_DIR / "data" / "gemini_comparison.json"
    with open(output_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "lite_total_cost": lite_total_cost,
                "preview_total_cost": preview_total_cost,
                "cost_ratio": preview_total_cost / lite_total_cost if lite_total_cost > 0 else 0,
                "extrapolated_957_lite": lite_total_cost * scale,
                "extrapolated_957_preview": preview_total_cost * scale
            },
            "results": results
        }, f, indent=2)

    print(f"\nFull results saved to: {output_path}")

    return results


if __name__ == '__main__':
    run_comparison()
