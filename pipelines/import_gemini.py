#!/usr/bin/env python3
"""
Import Gemini conversations from Google Takeout export.

Parses MyActivity.html and adds to the brain parquet archive.

Usage:
    python -m pipelines import-gemini                    # Dry run
    python -m pipelines import-gemini --import           # Actually import
    python -m pipelines import-gemini --import --merge   # Import + merge with archive
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path
from datetime import datetime
from html.parser import HTMLParser
from collections import defaultdict

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb

from config import (
    PARQUET_PATH,
    BACKUP_DIR,
    DATA_DIR,
)

# Gemini export location
GEMINI_ACTIVITY = Path.home() / "Documents/google takeout/My Activity/Gemini Apps/MyActivity.html"
GEMINI_IMPORT_PARQUET = DATA_DIR / "gemini_import.parquet"


class GeminiActivityParser(HTMLParser):
    """Parse Gemini MyActivity.html export."""

    def __init__(self):
        super().__init__()
        self.records = []
        self.current_record = {}
        self.in_content_cell = False
        self.in_caption = False
        self.current_text = []
        self.current_class = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        class_val = attrs_dict.get('class', '')

        # Track when we enter a content cell (main content)
        if tag == 'div' and 'content-cell' in class_val and 'mdl-cell--6-col' in class_val:
            if 'text-right' not in class_val:  # Skip the empty right column
                self.in_content_cell = True
                self.current_text = []
                self.current_class = class_val

        # Track caption cells (metadata)
        if tag == 'div' and 'content-cell' in class_val and 'caption' in class_val:
            self.in_caption = True

        # Start of new outer cell = new record
        if tag == 'div' and 'outer-cell' in class_val:
            if self.current_record:
                self.records.append(self.current_record)
            self.current_record = {}

    def handle_endtag(self, tag):
        if tag == 'div' and self.in_content_cell:
            content = ' '.join(self.current_text).strip()
            if content and not self.current_record.get('content'):
                self.current_record['raw_content'] = content
            self.in_content_cell = False
            self.current_text = []

        if tag == 'div' and self.in_caption:
            self.in_caption = False

    def handle_data(self, data):
        if self.in_content_cell:
            self.current_text.append(data.strip())

    def finalize(self):
        """Add last record if exists."""
        if self.current_record:
            self.records.append(self.current_record)
        return self.records


def parse_timestamp(ts_str: str) -> datetime:
    """Parse Gemini timestamp format: 'Dec 21, 2025, 9:53:03 PM IST'"""
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


def parse_raw_content(raw: str) -> dict:
    """
    Parse raw content cell into structured data.

    Examples:
    - "Prompted tell me about me Attached 1 file. - brain_backup_latest.json Dec 6, 2025, 7:42:20 PM IST Based on..."
    - "Prompted I always knew that I was different Dec 15, 2025, 7:57:38 AM IST That's a very profound..."
    - "Created Gemini Canvas titled Low-Impact Fitness Guide # The Ultimate..."
    - "Gave feedback: Good response Dec 15, 2025..."
    """
    result = {
        'action': None,
        'prompt': None,
        'response': None,
        'timestamp': None,
        'attachments': [],
        'has_audio': False,
    }

    # Detect action type
    if raw.startswith('Prompted'):
        result['action'] = 'prompt'
        raw = raw[8:].strip()  # Remove "Prompted "
    elif raw.startswith('Created Gemini Canvas'):
        result['action'] = 'canvas'
        raw = raw[21:].strip()
    elif raw.startswith('Gave feedback'):
        result['action'] = 'feedback'
        return result  # Skip feedback entries
    elif raw.startswith('Opened'):
        result['action'] = 'opened'
        return result  # Skip opened entries
    else:
        result['action'] = 'other'

    # Check for audio
    if 'Audio included' in raw:
        result['has_audio'] = True
        raw = re.sub(r'Audio included\.?\s*', '', raw)

    # Extract timestamp (pattern: "Dec 21, 2025, 9:53:03 PM IST")
    ts_pattern = r'([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}(?::\d{2})?\s*[AP]M\s*\w*)'
    ts_match = re.search(ts_pattern, raw)
    if ts_match:
        result['timestamp'] = parse_timestamp(ts_match.group(1))
        ts_pos = ts_match.start()

        # Split into prompt (before timestamp) and response (after)
        before = raw[:ts_pos].strip()
        after = raw[ts_match.end():].strip()

        # Clean up prompt
        before = re.sub(r'Attached \d+ files?\..*?(?=\s*$|\s*[A-Z])', '', before, flags=re.DOTALL).strip()
        before = re.sub(r'-\s*\S+\.(json|pdf|wav|mp3)\s*', '', before).strip()

        result['prompt'] = before if before else None
        result['response'] = after if after else None
    else:
        # No timestamp found, treat entire content as prompt
        result['prompt'] = raw

    return result


def parse_gemini_html(filepath: Path) -> list[dict]:
    """Parse Gemini HTML export into conversation records."""
    print(f"Reading {filepath}...")

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()

    # Simple regex-based extraction (more reliable than HTML parser for this format)
    records = []

    # Pattern for content cells
    cell_pattern = r'<div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">(.*?)</div>'

    # Find all content cells
    cells = re.findall(cell_pattern, html_content, re.DOTALL)
    print(f"Found {len(cells)} content cells")

    for i, cell in enumerate(cells):
        # Skip empty or metadata cells
        if not cell.strip() or 'Products:' in cell:
            continue

        # Clean HTML entities and tags
        text = re.sub(r'<[^>]+>', ' ', cell)  # Remove tags
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        text = text.replace('&amp;', '&')
        text = text.replace('&emsp;', ' ')
        text = text.replace('&nbsp;', ' ')
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

        if not text or len(text) < 10:
            continue

        # Parse structured content
        parsed = parse_raw_content(text)

        # Skip non-prompt actions
        if parsed['action'] not in ('prompt', 'canvas'):
            continue

        if not parsed['prompt']:
            continue

        # Generate unique ID
        content_hash = hashlib.md5(
            f"{parsed['prompt']}{parsed.get('timestamp', '')}".encode()
        ).hexdigest()[:16]

        ts = parsed['timestamp'] or datetime.now()

        record = {
            'source': 'gemini',
            'model': 'gemini-1.5',
            'project': None,
            'conversation_id': f"gemini_{content_hash}",
            'conversation_title': parsed['prompt'][:50] if parsed['prompt'] else 'Gemini',
            'created': ts,
            'updated': ts,
            'year': ts.year,
            'month': ts.month,
            'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][ts.weekday()],
            'hour': ts.hour,
            'message_id': f"gemini_{content_hash}_user",
            'parent_id': None,
            'msg_index': 0,
            'msg_timestamp': ts,
            'timestamp_is_fallback': 0,
            'role': 'user',
            'content_type': 'audio' if parsed['has_audio'] else 'text',
            'is_first': 1,
            'is_last': 0 if parsed['response'] else 1,
            'word_count': len(parsed['prompt'].split()) if parsed['prompt'] else 0,
            'char_count': len(parsed['prompt']) if parsed['prompt'] else 0,
            'conversation_msg_count': 2 if parsed['response'] else 1,
            'has_code': 1 if re.search(r'```|def |function |class |import ', parsed['prompt'] or '') else 0,
            'has_url': 1 if re.search(r'https?://', parsed['prompt'] or '') else 0,
            'has_question': 1 if '?' in (parsed['prompt'] or '') else 0,
            'has_attachment': 1 if parsed['attachments'] else 0,
            'has_citation': 0,
            'content': parsed['prompt'][:50000] if parsed['prompt'] else '',
            'temporal_precision': 'minute',
        }
        records.append(record)

        # Add response as assistant message
        if parsed['response'] and len(parsed['response']) > 20:
            response_record = record.copy()
            response_record['message_id'] = f"gemini_{content_hash}_assistant"
            response_record['role'] = 'assistant'
            response_record['msg_index'] = 1
            response_record['is_first'] = 0
            response_record['is_last'] = 1
            response_record['word_count'] = len(parsed['response'].split())
            response_record['char_count'] = len(parsed['response'])
            response_record['has_code'] = 1 if re.search(r'```|def |function |class |import ', parsed['response']) else 0
            response_record['has_url'] = 1 if re.search(r'https?://', parsed['response']) else 0
            response_record['has_question'] = 1 if '?' in parsed['response'] else 0
            response_record['content'] = parsed['response'][:50000]
            records.append(response_record)

    return records


def merge_with_archive(con):
    """Merge import into main parquet archive."""
    import shutil

    BACKUP_DIR.mkdir(exist_ok=True)
    backup_path = BACKUP_DIR / f"all_conversations_{datetime.now():%Y%m%d_%H%M%S}.parquet"
    shutil.copy(PARQUET_PATH, backup_path)
    print(f"  Backed up to: {backup_path.name}")

    orig_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{PARQUET_PATH}')").fetchone()[0]
    import_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{GEMINI_IMPORT_PARQUET}')").fetchone()[0]

    merged_path = PARQUET_PATH.with_suffix('.merged.parquet')

    # Get column list from main archive for consistent ordering
    cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{PARQUET_PATH}')").fetchall()
    col_names = [c[0] for c in cols]

    # Cast gemini timestamps to match archive (TIMESTAMPTZ)
    gemini_select = []
    for col in col_names:
        if col in ('created', 'updated', 'msg_timestamp'):
            gemini_select.append(f"CAST({col} AS TIMESTAMPTZ) as {col}")
        else:
            gemini_select.append(col)

    gemini_cols = ', '.join(gemini_select)

    con.execute(f"""
        COPY (
            WITH combined AS (
                SELECT * FROM read_parquet('{PARQUET_PATH}')
                UNION ALL
                SELECT {gemini_cols} FROM read_parquet('{GEMINI_IMPORT_PARQUET}')
            ),
            deduplicated AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY COALESCE(message_id, md5(content || conversation_id || role))
                        ORDER BY created
                    ) as rn
                FROM combined
            )
            SELECT * EXCLUDE (rn) FROM deduplicated WHERE rn = 1
        ) TO '{merged_path}' (FORMAT PARQUET)
    """)

    merged_path.replace(PARQUET_PATH)

    final_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{PARQUET_PATH}')").fetchone()[0]
    new_msgs = final_count - orig_count

    print(f"  Original: {orig_count:,}")
    print(f"  Imported: {import_count:,}")
    print(f"  New (after dedup): {new_msgs:,}")
    print(f"  Final total: {final_count:,}")
    print(f"\n  Archive updated!")


def import_gemini(do_import: bool = False, merge: bool = False):
    """
    Import Gemini conversations from Google Takeout.

    Args:
        do_import: Actually perform import (False = dry run)
        merge: Merge into main archive after import
    """
    print("=" * 70)
    print("GEMINI TAKEOUT IMPORT")
    print("=" * 70)

    if not GEMINI_ACTIVITY.exists():
        print(f"\nERROR: Gemini export not found at:")
        print(f"  {GEMINI_ACTIVITY}")
        print("\nExpected location: ~/Documents/google takeout/My Activity/Gemini Apps/MyActivity.html")
        return

    # Parse HTML
    records = parse_gemini_html(GEMINI_ACTIVITY)

    print(f"\nParsed {len(records)} message records")

    if not records:
        print("No records to import")
        return

    # Statistics
    user_msgs = [r for r in records if r['role'] == 'user']
    asst_msgs = [r for r in records if r['role'] == 'assistant']
    audio_msgs = [r for r in records if r['content_type'] == 'audio']

    dates = [r['created'] for r in records if r['created']]
    min_date = min(dates) if dates else None
    max_date = max(dates) if dates else None

    print(f"\n## Statistics")
    print(f"  User messages:      {len(user_msgs):,}")
    print(f"  Assistant messages: {len(asst_msgs):,}")
    print(f"  Audio transcripts:  {len(audio_msgs):,}")
    print(f"  Date range:         {min_date} to {max_date}")

    # Sample prompts
    print(f"\n## Sample prompts:")
    for r in user_msgs[:5]:
        prompt = r['content'][:80].replace('\n', ' ')
        print(f"  [{r['created'].strftime('%Y-%m-%d')}] {prompt}...")

    if not do_import:
        print("\n" + "=" * 70)
        print("DRY RUN - Run with --import to save to parquet")
        print("=" * 70)
        return

    # Save to parquet
    print(f"\n## Saving to {GEMINI_IMPORT_PARQUET}...")

    import pandas as pd
    df = pd.DataFrame(records)

    for col in ['created', 'updated', 'msg_timestamp']:
        df[col] = pd.to_datetime(df[col])

    df.to_parquet(GEMINI_IMPORT_PARQUET, index=False)
    print(f"  Saved {len(df)} records")

    con = duckdb.connect()

    if merge:
        print("\n## Merging with main archive...")
        merge_with_archive(con)
    else:
        print(f"\nTo merge with main archive, run with --merge flag")


def main():
    parser = argparse.ArgumentParser(description='Import Gemini conversations from Takeout')
    parser.add_argument('--import', dest='do_import', action='store_true', help='Actually import')
    parser.add_argument('--merge', action='store_true', help='Merge into main archive')
    args = parser.parse_args()

    import_gemini(
        do_import=args.do_import,
        merge=args.merge
    )


if __name__ == '__main__':
    main()
