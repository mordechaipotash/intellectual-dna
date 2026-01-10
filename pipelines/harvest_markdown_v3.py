#!/usr/bin/env python3
"""
Harvest Markdown Pipeline v3 - Deep Mining
===========================================
Plums through markdown files to extract maximum intellectual value.

NEW v3 COLUMNS (beyond v2):
- content_date: Date extracted from content (not filesystem)
- voice: FIRST_PERSON (my thoughts) vs THIRD_PERSON (documentation)
- project: Associated project name if detected
- content_hash: SHA256 for duplicate detection
- is_duplicate: Boolean if content exists elsewhere
- question_count: Number of ? in content
- decision_count: Explicit decisions made
- draft_status: DRAFT, WIP, FINAL, TODO
- energy: FRUSTRATED, EXCITED, BREAKTHROUGH, NEUTRAL
- key_people: People mentioned
- key_tools: Specific tools mentioned
- depth_score: 0-100 substantiveness score
- first_line: First meaningful line (for quick preview)
- todos_open: Count of [ ] unchecked
- todos_done: Count of [x] checked
"""

import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime
import re
from typing import Optional, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

import pyarrow as pa
import pyarrow.parquet as pq

# Import from v1 and v2
from harvest_markdown import (
    SCAN_DIRS, EXCLUDE_PATTERNS, BOILERPLATE_NAMES,
    EXCLUDE_FILENAME_PATTERNS, should_exclude, is_boilerplate,
    extract_title, extract_domain, get_file_dates
)
from harvest_markdown_v2 import (
    SEED_CONCEPTS, FRAMEWORKS, TECH_STACK, CODE_LANGUAGES,
    detect_doc_type, extract_code_languages, extract_tech_stack,
    extract_seed_concepts, extract_frameworks, detect_action_type,
    count_headings, count_links, count_file_refs, calculate_harvest_score
)

# Known projects to detect
KNOWN_PROJECTS = [
    'sparkii', 'wotc', 'tcs', 'intellectual_dna', 'mordharvest',
    'brain-mcp', 'seedgarden', 'hpi', 'jewtube', 'compligenie',
    'sefaria', 'gesture-mouse', 'realtime-audio', 'nivelty'
]

# Key people to detect
KEY_PEOPLE = [
    'claude', 'gemini', 'gpt', 'anthropic', 'openai', 'google',
    'mordechai', 'aviva', 'avrham', 'shifra'
]

# Energy/emotion markers
ENERGY_MARKERS = {
    'FRUSTRATED': ['frustrat', 'stuck', 'broken', 'waste', 'annoying', 'why the', 'doesnt work'],
    'EXCITED': ['amazing', 'breakthrough', 'finally', 'perfect', 'love it', 'holy', 'incredible'],
    'BREAKTHROUGH': ['realized', 'insight', 'aha', 'figured out', 'the key is', 'now i understand'],
}


def extract_content_date(content: str, filename: str) -> Optional[datetime]:
    """Extract date from content or filename."""
    # Try filename patterns first
    # Pattern: 2025-12-13, 20251213, 13-12-2025
    patterns = [
        r'(\d{4})-(\d{2})-(\d{2})',  # ISO format
        r'(\d{4})(\d{2})(\d{2})',     # Compact format
        r'(\d{2})-(\d{2})-(\d{4})',   # Day-Month-Year
    ]

    # Check filename first
    for pattern in patterns[:2]:  # ISO and compact only for filename
        match = re.search(pattern, filename)
        if match:
            try:
                groups = match.groups()
                return datetime(int(groups[0]), int(groups[1]), int(groups[2]))
            except ValueError:
                continue

    # Check content (first 500 chars)
    content_head = content[:500]
    for pattern in patterns:
        match = re.search(pattern, content_head)
        if match:
            try:
                groups = match.groups()
                if len(groups[0]) == 4:  # Year first
                    return datetime(int(groups[0]), int(groups[1]), int(groups[2]))
                else:  # Day first
                    return datetime(int(groups[2]), int(groups[1]), int(groups[0]))
            except ValueError:
                continue

    return None


def detect_voice(content: str) -> str:
    """Detect if content is first person (my thoughts) or third person (documentation)."""
    content_lower = content[:2000].lower()

    first_person = len(re.findall(r'\bi\s|\bmy\s|\bme\s|\bwe\s|\bour\s', content_lower))
    third_person = len(re.findall(r'\bthe\s+user|\bthe\s+system|\bit\s+is|\bthis\s+is', content_lower))

    if first_person > third_person * 2:
        return 'FIRST_PERSON'
    elif third_person > first_person * 2:
        return 'THIRD_PERSON'
    return 'MIXED'


def detect_project(content: str, path: str) -> Optional[str]:
    """Detect which project this file belongs to."""
    combined = (content[:1000] + path).lower()

    for project in KNOWN_PROJECTS:
        if project in combined:
            return project
    return None


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content for duplicate detection."""
    # Normalize whitespace for comparison
    normalized = ' '.join(content.split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def detect_draft_status(content: str, filename: str) -> str:
    """Detect if content is draft, WIP, or final."""
    combined = (content[:500] + filename).lower()

    if 'wip' in combined or 'work in progress' in combined:
        return 'WIP'
    if 'draft' in combined:
        return 'DRAFT'
    if 'todo' in combined or '[ ]' in content[:1000]:
        return 'TODO'
    return 'FINAL'


def detect_energy(content: str) -> str:
    """Detect emotional energy in content."""
    content_lower = content[:3000].lower()

    for energy, markers in ENERGY_MARKERS.items():
        for marker in markers:
            if marker in content_lower:
                return energy
    return 'NEUTRAL'


def extract_key_people(content: str) -> list[str]:
    """Extract key people/entities mentioned."""
    content_lower = content.lower()
    found = []
    for person in KEY_PEOPLE:
        if person in content_lower:
            found.append(person)
    return found


def count_questions(content: str) -> int:
    """Count questions in content."""
    return content.count('?')


def count_decisions(content: str) -> int:
    """Count explicit decision markers."""
    patterns = [
        r'decided to', r'chose to', r'going with', r'selected',
        r'decision:', r'we will', r'i will', r'the plan is'
    ]
    content_lower = content.lower()
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, content_lower))
    return count


def count_todos(content: str) -> Tuple[int, int]:
    """Count open and completed todos."""
    open_todos = len(re.findall(r'- \[ \]', content))
    done_todos = len(re.findall(r'- \[x\]', content, re.IGNORECASE))
    return open_todos, done_todos


def extract_first_line(content: str) -> str:
    """Extract first meaningful line for preview."""
    lines = content.strip().split('\n')
    for line in lines:
        line = line.strip()
        # Skip empty, headings, and metadata
        if line and not line.startswith('#') and not line.startswith('---'):
            return line[:200]
    return lines[0][:200] if lines else ''


def calculate_depth_score(row: dict) -> int:
    """Calculate substantiveness/depth score 0-100."""
    score = 0

    # Length contribution (max 25)
    wc = row['word_count']
    if wc > 100:
        score += min(25, wc // 100)

    # Structure (max 20)
    if row['heading_count'] > 3:
        score += 10
    if row['has_code']:
        score += 5
    if row['has_tables']:
        score += 5

    # Intellectual markers (max 25)
    score += len(row['seed_concepts']) * 3
    score += row['decision_count'] * 2

    # First person = more personal depth (max 15)
    if row['voice'] == 'FIRST_PERSON':
        score += 15
    elif row['voice'] == 'MIXED':
        score += 7

    # Energy = engagement (max 15)
    if row['energy'] in ['BREAKTHROUGH', 'EXCITED']:
        score += 15
    elif row['energy'] == 'FRUSTRATED':
        score += 10  # Frustration shows engagement

    return min(100, score)


def scan_markdown_files_v3(directories: list[Path], verbose: bool = True) -> list[dict]:
    """Scan directories with deep mining extraction."""
    records = []
    content_hashes = defaultdict(list)  # For duplicate detection
    scanned = 0
    excluded = 0
    boilerplate = 0
    errors = 0

    for base_dir in directories:
        if not base_dir.exists():
            continue

        if verbose:
            print(f"Scanning {base_dir}...")

        for path in base_dir.rglob("*.md"):
            scanned += 1

            if should_exclude(path):
                excluded += 1
                continue

            try:
                content = path.read_text(encoding='utf-8', errors='replace')
            except Exception:
                errors += 1
                continue

            if is_boilerplate(path, content):
                boilerplate += 1
                continue

            # Basic metadata (from v1)
            domain, subdomain = extract_domain(path)
            created, modified = get_file_dates(path)
            title = extract_title(content, path.name)

            # Enhanced metadata (from v2)
            doc_type = detect_doc_type(content, path.name, subdomain)
            code_languages = extract_code_languages(content)
            tech_stack = extract_tech_stack(content)
            seed_concepts = extract_seed_concepts(content)
            frameworks = extract_frameworks(content)
            action_type = detect_action_type(content)

            # Deep mining (v3)
            content_date = extract_content_date(content, path.name)
            voice = detect_voice(content)
            project = detect_project(content, str(path))
            content_hash = compute_content_hash(content)
            draft_status = detect_draft_status(content, path.name)
            energy = detect_energy(content)
            key_people = extract_key_people(content)
            question_count = count_questions(content)
            decision_count = count_decisions(content)
            todos_open, todos_done = count_todos(content)
            first_line = extract_first_line(content)

            # Track for duplicate detection
            content_hashes[content_hash].append(str(path))

            record = {
                # Basic (v1)
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

                # Document type (v2)
                'doc_type': doc_type,
                'action_type': action_type,

                # Structure flags (v2)
                'has_code': '```' in content,
                'has_tables': bool(re.search(r'\|.*\|.*\|', content)),
                'has_checklists': bool(re.search(r'- \[ \]|- \[x\]', content, re.IGNORECASE)),
                'has_mermaid': '```mermaid' in content,

                # Lists (v2)
                'code_languages': code_languages,
                'tech_stack': tech_stack,
                'seed_concepts': seed_concepts,
                'frameworks': frameworks,

                # Counts (v2)
                'heading_count': count_headings(content),
                'link_count': count_links(content),
                'file_ref_count': count_file_refs(content),
                'estimated_reading_mins': max(1, len(content.split()) // 200),

                # Deep mining (v3)
                'content_date': content_date,
                'voice': voice,
                'project': project or '',
                'content_hash': content_hash,
                'draft_status': draft_status,
                'energy': energy,
                'key_people': key_people,
                'question_count': question_count,
                'decision_count': decision_count,
                'todos_open': todos_open,
                'todos_done': todos_done,
                'first_line': first_line,
            }

            records.append(record)

    # Mark duplicates
    for r in records:
        paths = content_hashes[r['content_hash']]
        r['is_duplicate'] = len(paths) > 1
        r['duplicate_count'] = len(paths)

    # Calculate scores
    for r in records:
        r['harvest_score'] = calculate_harvest_score(r)
        r['depth_score'] = calculate_depth_score(r)

    if verbose:
        print(f"\nScan complete:")
        print(f"  Total scanned: {scanned:,}")
        print(f"  Excluded: {excluded:,}")
        print(f"  Boilerplate: {boilerplate:,}")
        print(f"  Original IP files: {len(records):,}")

        # Duplicate stats
        dupes = sum(1 for r in records if r['is_duplicate'])
        print(f"  Duplicates detected: {dupes:,}")

    return records


def create_parquet_v3(records: list[dict], output_path: Path) -> None:
    """Create enhanced parquet file with v3 columns."""
    if not records:
        print("No records!")
        return

    # Convert lists to strings for parquet
    for r in records:
        r['code_languages'] = ','.join(r['code_languages']) if r['code_languages'] else ''
        r['tech_stack'] = ','.join(r['tech_stack']) if r['tech_stack'] else ''
        r['seed_concepts'] = ','.join(r['seed_concepts']) if r['seed_concepts'] else ''
        r['frameworks'] = ','.join(r['frameworks']) if r['frameworks'] else ''
        r['key_people'] = ','.join(r['key_people']) if r['key_people'] else ''
        r['action_type'] = r['action_type'] or ''

    schema = pa.schema([
        # Basic (v1)
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

        # Document type (v2)
        ('doc_type', pa.string()),
        ('action_type', pa.string()),

        # Structure flags (v2)
        ('has_code', pa.bool_()),
        ('has_tables', pa.bool_()),
        ('has_checklists', pa.bool_()),
        ('has_mermaid', pa.bool_()),

        # Lists (v2)
        ('code_languages', pa.string()),
        ('tech_stack', pa.string()),
        ('seed_concepts', pa.string()),
        ('frameworks', pa.string()),

        # Counts (v2)
        ('heading_count', pa.int32()),
        ('link_count', pa.int32()),
        ('file_ref_count', pa.int32()),
        ('estimated_reading_mins', pa.int32()),
        ('harvest_score', pa.int32()),

        # Deep mining (v3)
        ('content_date', pa.timestamp('us')),
        ('voice', pa.string()),
        ('project', pa.string()),
        ('content_hash', pa.string()),
        ('is_duplicate', pa.bool_()),
        ('duplicate_count', pa.int32()),
        ('draft_status', pa.string()),
        ('energy', pa.string()),
        ('key_people', pa.string()),
        ('question_count', pa.int32()),
        ('decision_count', pa.int32()),
        ('todos_open', pa.int32()),
        ('todos_done', pa.int32()),
        ('first_line', pa.string()),
        ('depth_score', pa.int32()),
    ])

    columns = {key: [r[key] for r in records] for key in records[0].keys()}
    table = pa.table(columns, schema=schema)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression='snappy')

    print(f"\nWrote {len(records):,} records to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def show_stats_v3(records: list[dict]) -> None:
    """Show v3 enhanced statistics."""
    print("\n" + "="*60)
    print("DEEP MINING STATISTICS (v3)")
    print("="*60)

    # By voice
    print("\nBy Voice:")
    voices = {}
    for r in records:
        voices[r['voice']] = voices.get(r['voice'], 0) + 1
    for v, c in sorted(voices.items(), key=lambda x: -x[1]):
        print(f"  {v:<15} {c:>6} ({c/len(records)*100:.1f}%)")

    # By energy
    print("\nBy Energy:")
    energies = {}
    for r in records:
        energies[r['energy']] = energies.get(r['energy'], 0) + 1
    for e, c in sorted(energies.items(), key=lambda x: -x[1]):
        print(f"  {e:<15} {c:>6} ({c/len(records)*100:.1f}%)")

    # By draft status
    print("\nBy Draft Status:")
    statuses = {}
    for r in records:
        statuses[r['draft_status']] = statuses.get(r['draft_status'], 0) + 1
    for s, c in sorted(statuses.items(), key=lambda x: -x[1]):
        print(f"  {s:<15} {c:>6} ({c/len(records)*100:.1f}%)")

    # By project
    print("\nBy Project (top 15):")
    projects = {}
    for r in records:
        p = r['project'] or 'UNASSIGNED'
        projects[p] = projects.get(p, 0) + 1
    for p, c in sorted(projects.items(), key=lambda x: -x[1])[:15]:
        print(f"  {p:<20} {c:>6}")

    # By depth score
    print("\nBy Depth Score:")
    deep = sum(1 for r in records if r['depth_score'] >= 70)
    medium = sum(1 for r in records if 40 <= r['depth_score'] < 70)
    shallow = sum(1 for r in records if r['depth_score'] < 40)
    print(f"  Deep (70+):    {deep:>6} ({deep/len(records)*100:.1f}%)")
    print(f"  Medium (40-69): {medium:>6} ({medium/len(records)*100:.1f}%)")
    print(f"  Shallow (<40): {shallow:>6} ({shallow/len(records)*100:.1f}%)")

    # Content dates found
    with_dates = sum(1 for r in records if r['content_date'])
    print(f"\nContent Dates Found: {with_dates:,} ({with_dates/len(records)*100:.1f}%)")

    # Duplicates
    dupes = sum(1 for r in records if r['is_duplicate'])
    print(f"Duplicates: {dupes:,} ({dupes/len(records)*100:.1f}%)")

    # Decision density
    total_decisions = sum(r['decision_count'] for r in records)
    total_questions = sum(r['question_count'] for r in records)
    print(f"\nTotal Decisions: {total_decisions:,}")
    print(f"Total Questions: {total_questions:,}")

    # Todos
    total_open = sum(r['todos_open'] for r in records)
    total_done = sum(r['todos_done'] for r in records)
    print(f"TODOs Open: {total_open:,}")
    print(f"TODOs Done: {total_done:,}")

    # Top depth files
    print("\nTop 10 by Depth Score:")
    sorted_by_depth = sorted(records, key=lambda x: -x['depth_score'])[:10]
    for r in sorted_by_depth:
        print(f"  {r['depth_score']:3} | {r['filename'][:50]:<50} | {r['energy']}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Deep mining markdown harvest v3")
    parser.add_argument('--output', '-o', type=str,
                       default=str(DATA_DIR / "facts" / "markdown_files_v3.parquet"))
    parser.add_argument('--stats-only', action='store_true')
    parser.add_argument('--quiet', '-q', action='store_true')

    args = parser.parse_args()

    records = scan_markdown_files_v3(SCAN_DIRS, verbose=not args.quiet)

    if not records:
        print("No files found!")
        return 1

    if not args.quiet:
        show_stats_v3(records)

    if not args.stats_only:
        create_parquet_v3(records, Path(args.output))

    return 0


if __name__ == "__main__":
    sys.exit(main())
