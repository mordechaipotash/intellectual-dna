#!/usr/bin/env python3
"""
Import Claude Code conversations from local ~/.claude/projects directory.

Parses JSONL conversation files and adds them to the brain parquet archive.

Usage:
    python -m pipelines import-claude --days 8          # Last 8 days (dry run)
    python -m pipelines import-claude --days 8 --import # Actually import
    python -m pipelines import-claude --all --import    # Import all
"""

import json
import argparse
import re
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
from config import (
    CLAUDE_PROJECTS,
    PARQUET_PATH,
    CLAUDE_CODE_IMPORT_PARQUET,
    BACKUP_DIR,
)


def find_recent_jsonl(days: int = None) -> list[Path]:
    """Find JSONL files, optionally limited to recent days."""
    if not CLAUDE_PROJECTS.exists():
        print(f"ERROR: {CLAUDE_PROJECTS} does not exist")
        return []

    all_files = list(CLAUDE_PROJECTS.glob("**/*.jsonl"))

    if days:
        cutoff = datetime.now() - timedelta(days=days)
        recent = [f for f in all_files if datetime.fromtimestamp(f.stat().st_mtime) > cutoff]
        return sorted(recent, key=lambda f: f.stat().st_mtime, reverse=True)

    return sorted(all_files, key=lambda f: f.stat().st_mtime, reverse=True)


def extract_project_name(project_path: Path) -> str:
    """Extract readable project name from path."""
    name = project_path.name
    name = name.lstrip('-')
    parts = name.split('-')
    meaningful = [p for p in parts if p not in ('Users', 'mordechai', 'Library', 'Mobile', 'Documents', 'com', 'apple', 'CloudDocs')]
    if meaningful:
        return '-'.join(meaningful[-3:])
    return name[-50:]


def extract_content(message: dict) -> str:
    """Extract text content from message structure."""
    if not message:
        return ""

    content = message.get('content', '')

    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                # Only extract actual text content, skip tool_use and tool_result blocks
                if block.get('type') == 'text':
                    texts.append(block.get('text', ''))
                # Skip tool_use and tool_result entirely - they're noise
            elif isinstance(block, str):
                texts.append(block)
        return '\n'.join(texts)

    if isinstance(content, str):
        return content

    return str(content) if content else ""


def parse_jsonl_file(filepath: Path) -> list[dict]:
    """Parse a single JSONL file into message records."""
    records = []
    project_name = extract_project_name(filepath.parent)

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  Error reading {filepath.name}: {e}")
        return []

    session_id = None
    messages = []

    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = data.get('type')
        if msg_type not in ('user', 'assistant'):
            continue

        if not session_id:
            session_id = data.get('sessionId', filepath.stem)

        message = data.get('message', {})
        role = message.get('role') or data.get('type')
        content = extract_content(message)

        # Quality filter: Skip noise before it enters the archive
        if not content or len(content.strip()) < 5:
            continue

        # Skip known noise patterns
        content_lower = content.lower().strip()
        noise_patterns = ['warmup', 'prompt is too long', 'request interrupted']
        if any(content_lower == pattern for pattern in noise_patterns):
            continue

        # Skip very short messages (likely not substantive)
        if role == 'user' and len(content) < 15:
            continue
        if role == 'assistant' and len(content) < 10:
            continue

        ts_str = data.get('timestamp', '')
        try:
            if ts_str:
                ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            else:
                ts = datetime.fromtimestamp(filepath.stat().st_mtime)
        except:
            ts = datetime.fromtimestamp(filepath.stat().st_mtime)

        msg_id = data.get('uuid', f"{session_id}_{line_num}")

        messages.append({
            'source': 'claude-code',
            'model': message.get('model'),
            'project': project_name,
            'conversation_id': f"cc_local_{session_id}",
            'conversation_title': project_name,
            'created': ts,
            'updated': ts,
            'year': ts.year,
            'month': ts.month,
            'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][ts.weekday()],
            'hour': ts.hour,
            'message_id': msg_id,
            'parent_id': None,
            'msg_index': len(messages),
            'msg_timestamp': ts,
            'timestamp_is_fallback': 0,
            'role': role,
            'content_type': 'text',
            'is_first': 0,
            'is_last': 0,
            'word_count': len(content.split()),
            'char_count': len(content),
            'conversation_msg_count': 0,
            'has_code': 1 if re.search(r'```|def |function |class |import |const |let |var ', content) else 0,
            'has_url': 1 if re.search(r'https?://', content) else 0,
            'has_question': 1 if '?' in content else 0,
            'has_attachment': 0,
            'has_citation': 0,
            'content': content[:50000],
            'temporal_precision': 'day',
        })

    if messages:
        msg_count = len(messages)
        for i, msg in enumerate(messages):
            msg['conversation_msg_count'] = msg_count
            msg['is_first'] = 1 if i == 0 else 0
            msg['is_last'] = 1 if i == msg_count - 1 else 0
            msg['msg_index'] = i

    return messages


def merge_with_archive(con):
    """Merge import into main parquet archive."""
    import shutil

    BACKUP_DIR.mkdir(exist_ok=True)
    backup_path = BACKUP_DIR / f"all_conversations_{datetime.now():%Y%m%d_%H%M%S}.parquet"
    shutil.copy(PARQUET_PATH, backup_path)
    print(f"  Backed up to: {backup_path.name}")

    orig_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{PARQUET_PATH}')").fetchone()[0]
    import_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{CLAUDE_CODE_IMPORT_PARQUET}')").fetchone()[0]

    merged_path = PARQUET_PATH.with_suffix('.merged.parquet')
    con.execute(f"""
        COPY (
            WITH combined AS (
                SELECT * FROM read_parquet('{PARQUET_PATH}')
                UNION ALL
                SELECT * FROM read_parquet('{CLAUDE_CODE_IMPORT_PARQUET}')
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
    print(f"\n✓ Archive updated!")


def import_claude_code(days: int = 8, import_all: bool = False, do_import: bool = False, merge: bool = False):
    """
    Import Claude Code conversations.

    Args:
        days: Number of days to look back (default: 8)
        import_all: Import all conversations, not just recent
        do_import: Actually perform import (False = dry run)
        merge: Merge into main archive after import
    """
    print("=" * 70)
    print("CLAUDE CODE LOCAL IMPORT")
    print("=" * 70)

    days_param = None if import_all else days
    files = find_recent_jsonl(days_param)

    days_desc = "all time" if import_all else f"last {days} days"
    print(f"\nFound {len(files)} JSONL files from {days_desc}")

    if not files:
        print("No files to import")
        return

    if not do_import:
        print("\n## DRY RUN - Analyzing files...")

        by_project = defaultdict(int)
        by_date = defaultdict(int)
        sample_files = []

        for f in files[:50]:
            project = extract_project_name(f.parent)
            by_project[project] += 1
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            by_date[mtime.strftime('%Y-%m-%d')] += 1
            if len(sample_files) < 10:
                sample_files.append((f.name, project, mtime.strftime('%Y-%m-%d %H:%M')))

        for f in files[50:]:
            project = extract_project_name(f.parent)
            by_project[project] += 1
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            by_date[mtime.strftime('%Y-%m-%d')] += 1

        print("\nFiles by project:")
        for proj, cnt in sorted(by_project.items(), key=lambda x: -x[1])[:10]:
            print(f"  {proj}: {cnt}")

        print("\nFiles by date:")
        for date, cnt in sorted(by_date.items())[-10:]:
            print(f"  {date}: {cnt}")

        print("\nSample files:")
        for name, proj, mtime in sample_files:
            print(f"  [{mtime}] {proj}/{name[:30]}...")

        print("\n" + "=" * 70)
        print("Run with --import to process files")
        print("=" * 70)
        return

    print("\nProcessing files...")
    all_records = []
    errors = []

    for i, filepath in enumerate(files):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(files)} files ({len(all_records)} messages)...")

        try:
            records = parse_jsonl_file(filepath)
            all_records.extend(records)
        except Exception as e:
            errors.append((filepath.name, str(e)))

    print(f"\nProcessed {len(files)} files")
    print(f"Generated {len(all_records)} message records")

    if errors:
        print(f"Errors: {len(errors)}")
        for name, err in errors[:5]:
            print(f"  {name}: {err}")

    if not all_records:
        print("No messages to import")
        return

    print(f"\nSaving to {CLAUDE_CODE_IMPORT_PARQUET}...")

    con = duckdb.connect()

    import pandas as pd
    df = pd.DataFrame(all_records)

    for col in ['created', 'updated', 'msg_timestamp']:
        df[col] = pd.to_datetime(df[col])

    df.to_parquet(CLAUDE_CODE_IMPORT_PARQUET, index=False)

    print(f"Saved {len(df)} records")

    print("\n## Import Summary")
    print(f"  Messages: {len(df):,}")
    print(f"  Conversations: {df['conversation_id'].nunique():,}")
    print(f"  Date range: {df['created'].min()} to {df['created'].max()}")
    print(f"  User messages: {len(df[df['role'] == 'user']):,}")
    print(f"  Assistant messages: {len(df[df['role'] == 'assistant']):,}")

    by_project = df.groupby('project').size().sort_values(ascending=False)
    print("\n  By project:")
    for proj, cnt in by_project.head(10).items():
        print(f"    {proj}: {cnt}")

    if merge:
        print("\n## Merging with main archive...")
        merge_with_archive(con)
    else:
        print("\nTo merge with main archive, run with --merge flag")


def main():
    parser = argparse.ArgumentParser(description='Import Claude Code local conversations')
    parser.add_argument('--days', type=int, default=8, help='Days of history to import')
    parser.add_argument('--all', action='store_true', help='Import all conversations')
    parser.add_argument('--import', dest='do_import', action='store_true', help='Actually import')
    parser.add_argument('--merge', action='store_true', help='Merge into main archive')
    args = parser.parse_args()

    import_claude_code(
        days=args.days,
        import_all=args.all,
        do_import=args.do_import,
        merge=args.merge
    )


if __name__ == '__main__':
    main()
