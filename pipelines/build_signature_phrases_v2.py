#!/usr/bin/env python3
"""
Signature Phrases v2 Pipeline
Extracts authentic communication patterns from conversation corpus
"""

import duckdb
import re
import json
from collections import Counter
from pathlib import Path
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq

# Paths
BASE_DIR = Path("/Users/mordechai/intellectual_dna")
CONVERSATIONS = BASE_DIR / "data/facts/sources/all_conversations.parquet"
OUTPUT_DIR = BASE_DIR / "data/interpretations/signature_phrases/v2"

# IDE/System message patterns to exclude (Claude Code artifacts)
IDE_PATTERNS = [
    # Context/metadata
    r'context window', r'tokens used', r'tokens remaining',
    r'vscode', r'VSCode', r'visible files', r'open tabs',
    r'timezone', r'utc', r'jerusalem', r'asia/jerusalem',
    r'localhost', r'node_modules', r'package\.json',
    # File paths
    r'/Users/', r'/home/', r'\\Users\\', r'\.parquet', r'\.tsx', r'\.jsx',
    # IDE notifications
    r'user opened the file', r'may or may not', r'may not be related',
    r'the current task', r'the ide this', r'Tool Result',
    r'request interrupted', r'user uploaded image',
    r'bash tool output', r'tool output in',
    # System messages
    r'null null', r'frontend frontend', r'apps apps',
    r'mode act mode', r'command message command',
    r'mordechais macbook', r'mordechaipotash',
    r'caveat the messages', r'messages below were',
    r'consider them in your', r'unless the user explicitly',
    r'running local commands', r'do not respond to these',
]

# Compile patterns for speed
IDE_REGEX = re.compile('|'.join(IDE_PATTERNS), re.IGNORECASE)

# Category definitions
CATEGORIES = {
    'command': {
        'patterns': ['give me', 'show me', 'help me', 'tell me', 'make me', 'create me'],
        'description': 'Direct instructions to AI'
    },
    'intention': {
        'patterns': ['i want to', 'i need to', 'i\'m trying to', 'i\'m going to', 'want to be', 'need to be'],
        'description': 'Expressing desires and goals'
    },
    'question': {
        'patterns': ['how do', 'how can', 'what is', 'what are', 'whats the', 'why is', 'where is'],
        'description': 'Seeking information'
    },
    'correction': {
        'patterns': ['no no', 'not that', 'dont give', 'dont want', 'thats not', 'thats wrong'],
        'description': 'Correcting or rejecting'
    },
    'process': {
        'patterns': ['step by step', 'one by one', 'list of', 'each of', 'all of', 'first then'],
        'description': 'Requesting structured output'
    },
    'evaluation': {
        'patterns': ['the best', 'is better', 'would be', 'should be', 'is good', 'is great'],
        'description': 'Evaluating options'
    },
    'thinking': {
        'patterns': ['i think', 'i believe', 'maybe we', 'perhaps', 'it seems', 'looks like'],
        'description': 'Expressing thoughts'
    },
    'emphasis': {
        'patterns': ['very important', 'make sure', 'dont forget', 'remember to', 'always', 'never'],
        'description': 'Emphasizing importance'
    }
}


def is_conversational(text: str) -> bool:
    """Check if message is genuine conversation vs system noise"""
    if not text or len(text) < 10:
        return False
    if len(text) > 1000:  # Very long messages are often code/data
        return False
    if IDE_REGEX.search(text):
        return False
    if '```' in text or '{}' in text:
        return False
    return True


def extract_ngrams(text: str, n: int) -> list:
    """Extract n-grams from text"""
    words = re.findall(r'\b[a-z][a-z]+\b', text.lower())
    ngrams = []
    for i in range(len(words) - n + 1):
        phrase = ' '.join(words[i:i+n])
        # Skip if contains technical words
        if any(w in phrase for w in ['http', 'www', 'src', 'lib', 'var', 'const', 'def', 'class']):
            continue
        ngrams.append(phrase)
    return ngrams


def categorize_phrase(phrase: str) -> str:
    """Assign phrase to a category"""
    for cat_name, cat_info in CATEGORIES.items():
        if any(p in phrase for p in cat_info['patterns']):
            return cat_name
    return 'general'


def run_extraction():
    """Main extraction pipeline"""
    print("=" * 60)
    print("SIGNATURE PHRASES V2 EXTRACTION")
    print("=" * 60)

    con = duckdb.connect()

    # Load user messages
    print("\n📥 Loading user messages...")
    messages = con.execute(f"""
        SELECT content
        FROM '{CONVERSATIONS}'
        WHERE role = 'user'
        AND LENGTH(content) BETWEEN 10 AND 1000
    """).fetchall()
    print(f"   Total messages: {len(messages):,}")

    # Filter to conversational only
    print("\n🔍 Filtering to conversational messages...")
    conversational = [(m[0],) for m in messages if is_conversational(m[0])]
    print(f"   Conversational: {len(conversational):,} ({100*len(conversational)/len(messages):.1f}%)")

    # Extract n-grams
    print("\n📊 Extracting n-grams...")
    ngram_counts = Counter()
    for (content,) in conversational:
        for n in [3, 4, 5]:
            ngram_counts.update(extract_ngrams(content, n))
    print(f"   Unique n-grams: {len(ngram_counts):,}")
    print(f"   Appearing 5+ times: {len([p for p,c in ngram_counts.items() if c >= 5]):,}")

    # Categorize and rank
    print("\n🏷️ Categorizing phrases...")
    categorized = {}
    for phrase, count in ngram_counts.items():
        if count < 5:  # Minimum frequency
            continue
        cat = categorize_phrase(phrase)
        if cat not in categorized:
            categorized[cat] = []
        categorized[cat].append((phrase, count))

    # Sort each category by count
    for cat in categorized:
        categorized[cat].sort(key=lambda x: -x[1])

    # Print summary
    print("\n📈 RESULTS BY CATEGORY:")
    print("-" * 60)
    results = []
    for cat_name, phrases in sorted(categorized.items()):
        print(f"\n{cat_name.upper()} ({len(phrases)} phrases):")
        for phrase, count in phrases[:10]:
            print(f"  {count:4d}x  \"{phrase}\"")

        # Take top 50 per category for output
        for phrase, count in phrases[:50]:
            results.append({
                'phrase': phrase,
                'count': count,
                'category': cat_name,
                'ngram_size': len(phrase.split()),
                'first_word': phrase.split()[0],
            })

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    print(f"\n💾 Saving {len(results)} phrases to parquet...")

    # Create DataFrame-like structure
    table = pa.table({
        'phrase': [r['phrase'] for r in results],
        'count': [r['count'] for r in results],
        'category': [r['category'] for r in results],
        'ngram_size': [r['ngram_size'] for r in results],
        'first_word': [r['first_word'] for r in results],
        'meaning': [None for _ in results],  # To be filled by LLM
        'style_insight': [None for _ in results],  # To be filled by LLM
        'processed_at': [datetime.now().isoformat() for _ in results],
    })

    pq.write_table(table, OUTPUT_DIR / "phrases.parquet")

    # Save config
    config = {
        'version': 'v2',
        'created_at': datetime.now().isoformat(),
        'total_messages': len(messages),
        'conversational_messages': len(conversational),
        'total_phrases': len(results),
        'categories': list(CATEGORIES.keys()),
        'min_frequency': 5,
        'ngram_sizes': [3, 4, 5],
    }
    with open(OUTPUT_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Complete! Output: {OUTPUT_DIR}")
    print(f"   Phrases extracted: {len(results)}")

    return results


if __name__ == "__main__":
    run_extraction()
