#!/usr/bin/env python3
"""
Import Google Search history from Google Takeout export.

Parses MyActivity.html from Search export and saves to parquet.
Search queries are stored separately (not in all_conversations) and
added to brain L0 as event_type='google_search'.

Usage:
    python -m pipelines import-search                # Dry run
    python -m pipelines import-search --import       # Actually import
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from urllib.parse import unquote, parse_qs, urlparse

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb

from config import (
    DATA_DIR,
)

# Search export location
SEARCH_ACTIVITY = Path.home() / "Documents/google takeout/My Activity/Search/MyActivity.html"
SEARCH_PARQUET = DATA_DIR / "google_searches.parquet"


def parse_timestamp(ts_str: str) -> datetime:
    """Parse Google timestamp format: 'Dec 21, 2025, 9:53:03 PM IST'"""
    # Remove timezone suffix (IST, EST, etc.)
    ts_clean = re.sub(r'\s+(IST|EST|PST|UTC|GMT|[A-Z]{2,4})$', '', ts_str.strip())

    # Try various formats
    formats = [
        '%b %d, %Y, %I:%M:%S %p',  # Dec 21, 2025, 9:53:03 PM
        '%b %d, %Y, %I:%M %p',      # Dec 21, 2025, 9:53 PM
        '%B %d, %Y, %I:%M:%S %p',   # December 21, 2025, 9:53:03 PM
    ]

    for fmt in formats:
        try:
            return datetime.strptime(ts_clean, fmt)
        except ValueError:
            continue

    # Fallback: try to extract date parts
    match = re.search(r'(\w+)\s+(\d+),\s+(\d{4})', ts_str)
    if match:
        month, day, year = match.groups()
        try:
            return datetime.strptime(f"{month} {day}, {year}", '%b %d, %Y')
        except:
            pass

    return datetime.now()


def extract_query_from_url(url: str) -> str:
    """Extract search query from Google search URL."""
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        if 'q' in params:
            return unquote(params['q'][0])
    except:
        pass
    return None


def parse_search_html(filepath: Path) -> tuple[list[dict], list[dict]]:
    """
    Parse Google Search HTML export.

    Returns:
        (searches, visits) - Two lists of records
    """
    print(f"Reading {filepath}...")

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()

    searches = []
    visits = []

    # Pattern for content cells containing search data
    cell_pattern = r'<div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">(.*?)</div>'
    cells = re.findall(cell_pattern, html_content, re.DOTALL)

    print(f"Found {len(cells)} content cells")

    for cell in cells:
        # Skip empty or metadata cells
        if not cell.strip() or 'Products:' in cell:
            continue

        # Determine type: Searched vs Visited
        is_search = 'Searched for' in cell
        is_visit = 'Visited' in cell and 'Searched for' not in cell

        if not is_search and not is_visit:
            continue

        # Extract timestamp (pattern: "Dec 21, 2025, 9:53:03 PM IST")
        ts_pattern = r'([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}(?::\d{2})?\s*[AP]M\s*\w*)'
        ts_match = re.search(ts_pattern, cell)
        timestamp = parse_timestamp(ts_match.group(1)) if ts_match else datetime.now()

        if is_search:
            # Extract search query from link
            link_match = re.search(r'Searched for\s*<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>', cell)
            if link_match:
                url, query_text = link_match.groups()
                # Prefer URL-extracted query (more accurate)
                query = extract_query_from_url(url) or query_text

                # Clean up query
                query = query.strip()
                if not query or len(query) < 2:
                    continue

                # Generate unique ID
                content_hash = hashlib.md5(
                    f"{query}{timestamp}".encode()
                ).hexdigest()[:16]

                searches.append({
                    'search_id': f"search_{content_hash}",
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

        elif is_visit:
            # Extract visited URL
            link_match = re.search(r'Visited\s*<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>', cell)
            if link_match:
                url, title = link_match.groups()
                title = title.strip()

                content_hash = hashlib.md5(
                    f"{url}{timestamp}".encode()
                ).hexdigest()[:16]

                visits.append({
                    'visit_id': f"visit_{content_hash}",
                    'url': url,
                    'title': title,
                    'timestamp': timestamp,
                    'year': timestamp.year,
                    'month': timestamp.month,
                    'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][timestamp.weekday()],
                    'hour': timestamp.hour,
                })

    return searches, visits


def import_search(do_import: bool = False):
    """
    Import Google Search history from Takeout.

    Args:
        do_import: Actually perform import (False = dry run)
    """
    print("=" * 70)
    print("GOOGLE SEARCH TAKEOUT IMPORT")
    print("=" * 70)

    if not SEARCH_ACTIVITY.exists():
        print(f"\nERROR: Search export not found at:")
        print(f"  {SEARCH_ACTIVITY}")
        print("\nExpected location: ~/Documents/google takeout/My Activity/Search/MyActivity.html")
        return

    # Parse HTML
    searches, visits = parse_search_html(SEARCH_ACTIVITY)

    print(f"\nParsed {len(searches)} searches and {len(visits)} visits")

    if not searches:
        print("No searches to import")
        return

    # Statistics
    dates = [s['timestamp'] for s in searches]
    min_date = min(dates)
    max_date = max(dates)

    # Word analysis
    all_words = []
    for s in searches:
        all_words.extend(s['query'].lower().split())

    from collections import Counter
    word_freq = Counter(all_words)
    # Filter out common words
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'for', 'on', 'with', 'how', 'what', 'why', 'when', 'where', 'who'}
    top_words = [(w, c) for w, c in word_freq.most_common(30) if w not in stopwords and len(w) > 2][:15]

    # By year
    by_year = defaultdict(int)
    for s in searches:
        by_year[s['year']] += 1

    # Questions vs statements
    questions = len([s for s in searches if s['has_question']])

    print(f"\n## Statistics")
    print(f"  Searches:       {len(searches):,}")
    print(f"  Visits:         {len(visits):,}")
    print(f"  Date range:     {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    print(f"  Questions (?):  {questions:,} ({100*questions/len(searches):.1f}%)")

    print(f"\n## By Year")
    for year in sorted(by_year.keys()):
        print(f"  {year}: {by_year[year]:,}")

    print(f"\n## Top Search Terms")
    for word, count in top_words:
        print(f"  {word}: {count}")

    print(f"\n## Sample Searches")
    for s in searches[:10]:
        q = s['query'][:60].replace('\n', ' ')
        print(f"  [{s['timestamp'].strftime('%Y-%m-%d')}] {q}...")

    if not do_import:
        print("\n" + "=" * 70)
        print("DRY RUN - Run with --import to save to parquet")
        print("=" * 70)
        return

    # Save to parquet
    print(f"\n## Saving to {SEARCH_PARQUET}...")

    import pandas as pd
    df = pd.DataFrame(searches)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.to_parquet(SEARCH_PARQUET, index=False)
    print(f"  Saved {len(df)} searches")

    # Also save visits if any
    if visits:
        visits_path = DATA_DIR / "google_visits.parquet"
        df_visits = pd.DataFrame(visits)
        df_visits['timestamp'] = pd.to_datetime(df_visits['timestamp'])
        df_visits.to_parquet(visits_path, index=False)
        print(f"  Saved {len(df_visits)} visits to {visits_path.name}")

    print(f"\n✓ Import complete!")
    print(f"\nTo add searches to brain L0, rebuild the brain layers:")
    print(f"  python pipelines/rebuild.py brain")


def main():
    parser = argparse.ArgumentParser(description='Import Google Search history from Takeout')
    parser.add_argument('--import', dest='do_import', action='store_true', help='Actually import')
    args = parser.parse_args()

    import_search(
        do_import=args.do_import
    )


if __name__ == '__main__':
    main()
