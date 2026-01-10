#!/usr/bin/env python3
"""
Import YouTube search history from Google Takeout export.

Usage:
    python -m pipelines import-youtube-search           # Dry run
    python -m pipelines import-youtube-search --import  # Actually import
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from urllib.parse import unquote, parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
from config import DATA_DIR

YOUTUBE_SEARCH_ACTIVITY = Path.home() / "Documents/google takeout/YouTube and YouTube Music/history/search-history.html"
YOUTUBE_SEARCH_PARQUET = DATA_DIR / "youtube_searches.parquet"


def parse_timestamp(ts_str: str) -> datetime:
    """Parse Google timestamp format."""
    ts_clean = re.sub(r'\s+(IST|EST|PST|UTC|GMT|[A-Z]{2,4})$', '', ts_str.strip())
    formats = [
        '%b %d, %Y, %I:%M:%S %p',
        '%b %d, %Y, %I:%M %p',
        '%B %d, %Y, %I:%M:%S %p',
    ]
    for fmt in formats:
        try:
            return datetime.strptime(ts_clean, fmt)
        except ValueError:
            continue
    return datetime.now()


def parse_youtube_search_html(filepath: Path) -> list[dict]:
    """Parse YouTube search history HTML."""
    print(f"Reading {filepath}...")

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()

    searches = []

    # Pattern: Searched for <a href="...">QUERY</a><br>TIMESTAMP<br>
    pattern = r'Searched for\s*<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>\s*<br>\s*([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}(?::\d{2})?\s*[AP]M\s*\w*)'

    matches = re.findall(pattern, html_content)
    print(f"Found {len(matches)} YouTube searches")

    for url, query, ts_str in matches:
        query = query.strip()
        if not query or len(query) < 2:
            continue

        timestamp = parse_timestamp(ts_str)
        content_hash = hashlib.md5(f"{query}{timestamp}".encode()).hexdigest()[:16]

        searches.append({
            'search_id': f"yt_search_{content_hash}",
            'query': query,
            'url': url,
            'timestamp': timestamp,
            'year': timestamp.year,
            'month': timestamp.month,
            'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][timestamp.weekday()],
            'hour': timestamp.hour,
            'word_count': len(query.split()),
            'char_count': len(query),
            'has_question': 1 if '?' in query else 0,
        })

    return searches


def import_youtube_search(do_import: bool = False):
    """Import YouTube search history."""
    print("=" * 70)
    print("YOUTUBE SEARCH TAKEOUT IMPORT")
    print("=" * 70)

    if not YOUTUBE_SEARCH_ACTIVITY.exists():
        print(f"\nERROR: YouTube search export not found at:")
        print(f"  {YOUTUBE_SEARCH_ACTIVITY}")
        return

    searches = parse_youtube_search_html(YOUTUBE_SEARCH_ACTIVITY)

    if not searches:
        print("No searches to import")
        return

    dates = [s['timestamp'] for s in searches]
    min_date, max_date = min(dates), max(dates)

    # Top queries
    from collections import Counter
    all_words = []
    for s in searches:
        all_words.extend(s['query'].lower().split())
    word_freq = Counter(all_words)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'to', 'of', 'in', 'for', 'on', 'with', 'how', 'what'}
    top_words = [(w, c) for w, c in word_freq.most_common(20) if w not in stopwords and len(w) > 2][:10]

    print(f"\n## Statistics")
    print(f"  Searches: {len(searches):,}")
    print(f"  Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

    print(f"\n## Top Terms")
    for word, count in top_words:
        print(f"  {word}: {count}")

    print(f"\n## Sample Searches")
    for s in searches[:10]:
        q = s['query'][:50]
        print(f"  [{s['timestamp'].strftime('%Y-%m-%d')}] {q}")

    if not do_import:
        print("\n" + "=" * 70)
        print("DRY RUN - Run with --import to save")
        print("=" * 70)
        return

    print(f"\n## Saving to {YOUTUBE_SEARCH_PARQUET}...")
    import pandas as pd
    df = pd.DataFrame(searches)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.to_parquet(YOUTUBE_SEARCH_PARQUET, index=False)
    print(f"  Saved {len(df)} searches")
    print(f"\n✓ Import complete!")


def main():
    parser = argparse.ArgumentParser(description='Import YouTube search history')
    parser.add_argument('--import', dest='do_import', action='store_true', help='Actually import')
    args = parser.parse_args()
    import_youtube_search(do_import=args.do_import)


if __name__ == '__main__':
    main()
