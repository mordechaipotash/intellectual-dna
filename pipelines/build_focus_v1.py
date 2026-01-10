#!/usr/bin/env python3
"""
Build focus/v1 interpretation: Daily focus detection using keyword extraction.

This is an INTERPRETATION (derived, versioned) not a FACT.
Can be rebuilt anytime from facts/brain/content.parquet.
"""

import re
import json
import duckdb
from pathlib import Path
from collections import Counter
from datetime import datetime

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
DATA_DIR = BASE_DIR / "data"
FACTS_DIR = DATA_DIR / "facts"
INTERP_DIR = DATA_DIR / "interpretations" / "focus" / "v1"

# Simple English stopwords
STOPWORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
    'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
    'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
    'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see',
    'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
    'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work',
    'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
    'give', 'day', 'most', 'us', 'is', 'are', 'was', 'were', 'been', 'being',
    'has', 'had', 'did', 'does', 'doing', 'should', 'could', 'would', 'might',
    'must', 'shall', 'need', 'really', 'very', 'much', 'more', 'many', 'still',
    'here', 'where', 'when', 'why', 'how', 'each', 'every', 'both', 'few',
    'same', 'such', 'once', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'yeah', 'okay', 'sure', 'right', 'going', 'thing', 'things', 'something',
    'anything', 'everything', 'nothing', 'someone', 'anyone', 'everyone',
    'actually', 'basically', 'probably', 'maybe', 'perhaps', 'however', 'though',
    'because', 'since', 'while', 'although', 'whether', 'instead', 'rather',
    'claude', 'please', 'thanks', 'thank', 'help', 'lets', "let's", 'dont', "don't",
    'cant', "can't", 'wont', "won't", 'thats', "that's", 'whats', "what's",
    'heres', "here's", 'theres', "there's", 'youre', "you're", 'were', "we're",
    'theyre', "they're", 'ive', "i've", 'youve', "you've", 'weve', "we've",
    # Conversation/tool artifacts
    'tool', 'tools', 'result', 'results', 'user', 'assistant', 'message', 'messages',
    'function', 'parameter', 'parameters', 'invoke', 'response', 'request', 'content',
    'true', 'false', 'null', 'none', 'error', 'success', 'failed', 'return', 'value',
    'file', 'files', 'path', 'directory', 'folder', 'code', 'data', 'name', 'type',
    'string', 'number', 'boolean', 'array', 'object', 'list', 'dict', 'output', 'input',
    'command', 'commands', 'system', 'process', 'call', 'calls', 'query', 'using',
    'warmup', 'cache', 'token', 'tokens', 'model', 'claude', 'anthropic', 'openai',
}


def extract_keywords(text: str, min_length: int = 4, max_keywords: int = 10) -> list:
    """Extract top keywords from text using simple word frequency."""
    if not text:
        return []

    # Tokenize and clean
    words = re.findall(r'\b[a-zA-Z]{%d,}\b' % min_length, text.lower())

    # Filter stopwords
    words = [w for w in words if w not in STOPWORDS]

    # Count and return top keywords
    counts = Counter(words)
    return [word for word, count in counts.most_common(max_keywords)]


def build_focus_daily():
    """Build daily focus interpretation."""
    print("Building focus/v1 interpretation...")

    con = duckdb.connect()
    output_path = INTERP_DIR / "daily.parquet"

    # Get messages by day - join content with index for timestamps
    print("  Loading messages by day...")
    daily_messages = con.execute(f"""
        SELECT
            CAST(idx.timestamp AS DATE) as date,
            LIST(c.full_content) as messages,
            COUNT(*) as message_count,
            SUM(c.word_count) as total_words
        FROM '{FACTS_DIR}/brain/content.parquet' c
        JOIN '{FACTS_DIR}/brain/index.parquet' idx ON c.event_id = idx.event_id
        WHERE c.event_type = 'message'
          AND c.full_content IS NOT NULL
          AND idx.timestamp >= '2022-11-01'
        GROUP BY CAST(idx.timestamp AS DATE)
        HAVING COUNT(*) >= 3
        ORDER BY date
    """).fetchall()

    print(f"  Processing {len(daily_messages)} days with 3+ messages...")

    # Process each day
    results = []
    for date, messages, msg_count, total_words in daily_messages:
        # Combine all messages for the day
        combined_text = " ".join([m for m in messages if m])

        # Extract keywords
        keywords = extract_keywords(combined_text, min_length=4, max_keywords=10)

        # Calculate focus score (based on keyword concentration)
        if keywords and total_words:
            # Count how often top keyword appears
            top_keyword = keywords[0] if keywords else ""
            top_count = combined_text.lower().count(top_keyword)
            focus_score = min(1.0, top_count / (total_words / 100))  # Normalize
        else:
            focus_score = 0.0

        results.append({
            'date': str(date),
            'keywords': json.dumps(keywords),
            'top_keyword': keywords[0] if keywords else "",
            'message_count': msg_count,
            'total_words': total_words,
            'focus_score': round(focus_score, 3),
        })

    # Write to parquet
    print(f"  Writing {len(results)} days to parquet...")

    # Create table from results
    con.execute("""
        CREATE TABLE focus_daily (
            date DATE,
            keywords VARCHAR,
            top_keyword VARCHAR,
            message_count INTEGER,
            total_words INTEGER,
            focus_score DOUBLE
        )
    """)

    for r in results:
        con.execute("""
            INSERT INTO focus_daily VALUES (?, ?, ?, ?, ?, ?)
        """, [r['date'], r['keywords'], r['top_keyword'],
              r['message_count'], r['total_words'], r['focus_score']])

    con.execute(f"COPY focus_daily TO '{output_path}' (FORMAT PARQUET)")

    # Show sample results
    print("\n  Sample daily focus (recent 10 days):")
    sample = con.execute(f"""
        SELECT date, top_keyword, keywords, focus_score
        FROM '{output_path}'
        ORDER BY date DESC
        LIMIT 10
    """).fetchall()

    for date, top, keywords, score in sample:
        kw_list = json.loads(keywords)[:3]
        print(f"    {date}: [{top}] {', '.join(kw_list)} (score: {score:.2f})")

    # Stats
    total_days = len(results)
    high_focus_days = len([r for r in results if r['focus_score'] > 0.5])
    print(f"\n  Total days processed: {total_days}")
    print(f"  High focus days (>0.5): {high_focus_days} ({100*high_focus_days/total_days:.1f}%)")

    return total_days


def create_readme():
    """Create README for this interpretation version."""
    readme_path = INTERP_DIR / "README.md"
    readme_content = """# focus/v1 - Daily Focus Detection

## Algorithm
Uses TF-IDF keyword extraction to identify daily focus areas.

## Parameters
- `min_word_length`: 4 (ignore short words)
- `max_keywords_per_day`: 10
- `min_messages_per_day`: 3

## Output Schema
| Column | Type | Description |
|--------|------|-------------|
| date | DATE | Day of focus |
| keywords | JSON | Top 10 keywords as JSON array |
| top_keyword | VARCHAR | Most frequent keyword |
| message_count | INTEGER | Messages sent that day |
| total_words | INTEGER | Total words written |
| focus_score | DOUBLE | Focus concentration (0-1) |

## Limitations
- v1 uses simple word frequency, not semantic understanding
- Doesn't distinguish between multiple focus areas in one day
- No context awareness (keywords may be ambiguous)

## Future Improvements (v2)
- Use LLM to generate focus summaries
- Cluster messages by topic before keyword extraction
- Add focus area classification (coding, writing, research, etc.)

## Rebuild
```bash
./mordelab/02-monotropic-prosthetic/mcp-env/bin/python pipelines/build_focus_v1.py
```

## Created
2025-12-25
"""
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"  Created README at {readme_path}")


if __name__ == '__main__':
    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    build_focus_daily()
    create_readme()
