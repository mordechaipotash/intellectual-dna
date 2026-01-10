#!/usr/bin/env python3
"""
Harvest Markdown Pipeline
=========================
Scans directories for .md files, filters noise, extracts metadata,
and creates a parquet for intellectual IP harvesting.

Schema:
- path: full path to file
- filename: just the filename
- domain: top-level project (Sparkii, TCS, intellectual_dna, mordharvest)
- subdomain: second-level folder
- date_created: file creation time
- date_modified: file modification time
- word_count: words in content
- char_count: characters in content
- line_count: lines in content
- content: full markdown content
- title: extracted from first # heading or filename
- is_original_ip: True if not boilerplate
- category: detected category (DECISION, ARCHITECTURE, BRAINSTORM, CONFIG, etc.)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import re

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

import pyarrow as pa
import pyarrow.parquet as pq

# Directories to scan
SCAN_DIRS = [
    Path.home() / "intellectual_dna",
    Path.home() / "mordharvest",
    Path.home() / "Mordechai Dev 2025",
    # Added 2026-01-03 - older content discovery
    Path.home() / "notion",                 # 35K files from Notion export
    Path.home() / "wotc-tax-automation",    # WOTC project docs
    Path.home() / "Transcripts",            # Torah/video transcripts
    Path.home() / "wotc_archived_variants", # Archived project versions
    Path.home() / "wotcfy-sh-reference",    # Reference implementations
]

# Paths to exclude (patterns)
EXCLUDE_PATTERNS = [
    "node_modules",
    ".next",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "SuperClaude",  # External framework, not original IP
    ".claude-backup",
    # Conversation logs and scraped data
    "claude-code-chats",
    "conversation",
    "chat-log",
    "chat_log",
    "Archives",
    "archive",
    # Scraped/generated data
    "github structured",
    "Github 2021 vs 2025",
    "Sparkii-Knowledge1",
    "Sparkii-Knowledge",
    # Job application bulk data
    "unmasked jobs",
    # Generated artifacts
    "sparkii-command-center",
    # More conversation logs
    "20aug_chats",
    "conversations_md",
    "conversations_json",
    "conversations_html",
    "_chats",
]

# Boilerplate filenames to filter
BOILERPLATE_NAMES = {
    "LICENSE.md",
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
}

# Filename patterns to exclude (regex)
EXCLUDE_FILENAME_PATTERNS = [
    r'^\d{4}-\d{2}-\d{2}_\d{4}_[a-f0-9-]+\.md$',  # UUID-dated conversation logs
    r'^chat_?\d*\.md$',  # Chat files (chat.md, chat_45.md, chat45.md)
    r'^chat_\w+\.md$',  # Named chat files (chat_thursdaynight.md)
    r'^Unmasking.?Autism',  # OCR'd book
    r'^The 7 Habits',  # OCR'd book
    r'^Divergent.?Mind',  # OCR'd book
    r'^Rich.?Dad',  # OCR'd book
    r'^DDD Autism',  # OCR'd book variant
    r'^Nonvisible Part of the Autism',  # OCR'd book/paper
    r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\.md$',  # Pure UUID filenames
]

# Category detection patterns
CATEGORY_PATTERNS = {
    "DECISION": ["decision", "chose", "decided", "why we", "rationale"],
    "ARCHITECTURE": ["architecture", "system design", "data model", "schema"],
    "PLAN": ["plan", "roadmap", "milestone", "phase", "implementation"],
    "BRAINSTORM": ["brainstorm", "ideas", "thoughts", "exploration"],
    "CONFIG": ["claude.md", "config", "settings", "setup"],
    "EVIDENCE": ["evidence", "proof", "claims", "receipts", "verified"],
    "JOURNEY": ["journey", "evolution", "timeline", "history"],
    "WHITEPAPER": ["whitepaper", "thesis", "proposal", "framework"],
}


def should_exclude(path: Path) -> bool:
    """Check if path should be excluded."""
    path_str = str(path).lower()
    for pattern in EXCLUDE_PATTERNS:
        if pattern.lower() in path_str:
            return True
    return False


def is_boilerplate(path: Path, content: str) -> bool:
    """Check if file is boilerplate."""
    filename = path.name

    # Known boilerplate names
    if filename in BOILERPLATE_NAMES:
        return True

    # Empty or tiny README.md
    if filename == "README.md" and len(content.strip()) < 200:
        return True

    # Generic "Untitled" files
    if filename.startswith("Untitled"):
        return True

    # Check filename patterns (conversation logs, OCR'd books, etc.)
    for pattern in EXCLUDE_FILENAME_PATTERNS:
        if re.match(pattern, filename, re.IGNORECASE):
            return True

    return False


def extract_title(content: str, filename: str) -> str:
    """Extract title from first # heading or use filename."""
    # Look for first # heading
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # Fall back to filename without extension
    return Path(filename).stem


def detect_category(path: Path, content: str) -> str:
    """Detect category from path and content."""
    path_lower = str(path).lower()
    content_lower = content[:2000].lower()  # Check first 2000 chars

    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if pattern in path_lower or pattern in content_lower:
                return category

    return "OTHER"


def extract_domain(path: Path) -> tuple[str, str]:
    """Extract domain and subdomain from path."""
    parts = path.parts

    # Find the root directory index
    home = str(Path.home())
    try:
        home_idx = parts.index(Path.home().name)
    except ValueError:
        home_idx = 0

    # Domain is the first meaningful directory after home
    if len(parts) > home_idx + 1:
        domain = parts[home_idx + 1]
    else:
        domain = "unknown"

    # Subdomain is the next level
    if len(parts) > home_idx + 2:
        subdomain = parts[home_idx + 2]
    else:
        subdomain = ""

    return domain, subdomain


def get_file_dates(path: Path) -> tuple[datetime, datetime]:
    """Get file creation and modification times."""
    stat = path.stat()

    # macOS has st_birthtime for creation time
    if hasattr(stat, 'st_birthtime'):
        created = datetime.fromtimestamp(stat.st_birthtime)
    else:
        created = datetime.fromtimestamp(stat.st_ctime)

    modified = datetime.fromtimestamp(stat.st_mtime)

    return created, modified


def scan_markdown_files(directories: list[Path], verbose: bool = True) -> list[dict]:
    """Scan directories for markdown files."""
    records = []
    scanned = 0
    excluded = 0
    boilerplate = 0
    errors = 0

    for base_dir in directories:
        if not base_dir.exists():
            if verbose:
                print(f"Skipping {base_dir} - does not exist")
            continue

        if verbose:
            print(f"Scanning {base_dir}...")

        for path in base_dir.rglob("*.md"):
            scanned += 1

            # Check exclusions
            if should_exclude(path):
                excluded += 1
                continue

            # Read content
            try:
                content = path.read_text(encoding='utf-8', errors='replace')
            except Exception as e:
                errors += 1
                continue

            # Check boilerplate
            if is_boilerplate(path, content):
                boilerplate += 1
                continue

            # Extract metadata
            domain, subdomain = extract_domain(path)
            created, modified = get_file_dates(path)
            title = extract_title(content, path.name)
            category = detect_category(path, content)

            record = {
                'path': str(path),
                'filename': path.name,
                'domain': domain,
                'subdomain': subdomain,
                'date_created': created,
                'date_modified': modified,
                'word_count': len(content.split()),
                'char_count': len(content),
                'line_count': content.count('\n') + 1,
                'content': content,
                'title': title,
                'is_original_ip': True,  # Already filtered boilerplate
                'category': category,
            }
            records.append(record)

    if verbose:
        print(f"\nScan complete:")
        print(f"  Total scanned: {scanned:,}")
        print(f"  Excluded (node_modules, etc): {excluded:,}")
        print(f"  Boilerplate filtered: {boilerplate:,}")
        print(f"  Errors: {errors:,}")
        print(f"  Original IP files: {len(records):,}")

    return records


def create_parquet(records: list[dict], output_path: Path) -> None:
    """Create parquet file from records."""
    if not records:
        print("No records to write!")
        return

    # Define schema
    schema = pa.schema([
        ('path', pa.string()),
        ('filename', pa.string()),
        ('domain', pa.string()),
        ('subdomain', pa.string()),
        ('date_created', pa.timestamp('us')),
        ('date_modified', pa.timestamp('us')),
        ('word_count', pa.int32()),
        ('char_count', pa.int32()),
        ('line_count', pa.int32()),
        ('content', pa.string()),
        ('title', pa.string()),
        ('is_original_ip', pa.bool_()),
        ('category', pa.string()),
    ])

    # Convert to columnar format
    columns = {key: [r[key] for r in records] for key in records[0].keys()}

    # Create table
    table = pa.table(columns, schema=schema)

    # Write parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression='snappy')

    print(f"\nWrote {len(records):,} records to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def show_stats(records: list[dict]) -> None:
    """Show statistics about harvested files."""
    print("\n" + "="*50)
    print("HARVEST STATISTICS")
    print("="*50)

    # By domain
    print("\nBy Domain:")
    domains = {}
    for r in records:
        domains[r['domain']] = domains.get(r['domain'], 0) + 1
    for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
        print(f"  {domain}: {count:,}")

    # By category
    print("\nBy Category:")
    categories = {}
    for r in records:
        categories[r['category']] = categories.get(r['category'], 0) + 1
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count:,}")

    # Size stats
    total_words = sum(r['word_count'] for r in records)
    total_chars = sum(r['char_count'] for r in records)
    print(f"\nTotal Words: {total_words:,}")
    print(f"Total Characters: {total_chars:,}")
    print(f"Average Words/File: {total_words // len(records):,}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Harvest markdown files to parquet")
    parser.add_argument('--output', '-o', type=str,
                       default=str(DATA_DIR / "facts" / "markdown_files.parquet"),
                       help="Output parquet path")
    parser.add_argument('--stats-only', action='store_true',
                       help="Only show stats, don't write parquet")
    parser.add_argument('--quiet', '-q', action='store_true',
                       help="Minimal output")

    args = parser.parse_args()

    # Scan files
    records = scan_markdown_files(SCAN_DIRS, verbose=not args.quiet)

    if not records:
        print("No files found!")
        return 1

    # Show stats
    if not args.quiet:
        show_stats(records)

    # Write parquet
    if not args.stats_only:
        create_parquet(records, Path(args.output))

    return 0


if __name__ == "__main__":
    sys.exit(main())
