#!/usr/bin/env python3
"""
Harvest Markdown Pipeline v2 - Enhanced Schema
===============================================
Adds rich metadata columns for intellectual IP extraction.

NEW COLUMNS:
- doc_type: PLAN, ANALYSIS, BRAINSTORM, ARCHITECTURE, CAREER, TORAH, etc.
- has_code: boolean - contains code blocks
- has_tables: boolean - contains markdown tables
- has_checklists: boolean - contains [ ] items
- has_mermaid: boolean - contains diagrams
- code_languages: list of languages in code blocks
- tech_stack: list of technologies mentioned
- seed_concepts: list of SEED principles mentioned
- frameworks: list of frameworks (SHELET, SMAT, ZMAT, HPI)
- action_type: DECISION, PROBLEM_SOLVED, LEARNING, PLAN, QUESTION
- heading_count: number of headings (structure indicator)
- link_count: number of URLs
- file_refs: number of file references
- estimated_reading_time: minutes based on word count
- harvest_score: 0-100 score of IP value
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import re
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

import pyarrow as pa
import pyarrow.parquet as pq

# Import from v1
from harvest_markdown import (
    SCAN_DIRS, EXCLUDE_PATTERNS, BOILERPLATE_NAMES,
    EXCLUDE_FILENAME_PATTERNS, should_exclude, is_boilerplate,
    extract_title, extract_domain, get_file_dates
)

# SEED concepts to detect
SEED_CONCEPTS = [
    'inversion', 'compression', 'bottleneck', 'agency',
    'seeds', 'translation', 'temporal', 'cognitive architecture',
    'monotropic', 'hyperfocus'
]

# Frameworks to detect
FRAMEWORKS = ['SHELET', 'SMAT', 'ZMAT', 'HPI', 'SeedGarden', 'MORDETROPIC']

# Tech stack keywords
TECH_STACK = [
    'supabase', 'next.js', 'nextjs', 'react', 'typescript', 'python',
    'duckdb', 'openrouter', 'claude', 'shadcn', 'mcp', 'parquet',
    'postgresql', 'railway', 'vercel', 'github', 'obsidian'
]

# Code languages to detect
CODE_LANGUAGES = [
    'python', 'typescript', 'javascript', 'ts', 'js', 'sql', 'bash',
    'json', 'yaml', 'markdown', 'mermaid', 'html', 'css', 'tsx', 'jsx'
]


def detect_doc_type(content: str, filename: str, subdomain: str) -> str:
    """Detect document type from content and metadata."""
    content_lower = content[:3000].lower()
    filename_lower = filename.lower()
    subdomain_lower = subdomain.lower()

    # Command/Tool docs
    if content_lower.startswith('# /') or '/shelet' in content_lower[:100]:
        return 'COMMAND_DOC'

    if 'warmup' in content_lower[:100]:
        return 'WARMUP'

    if 'extracted from chatgpt' in content_lower[:200]:
        return 'CHATGPT_EXTRACT'

    if 'claude.md' in filename_lower or filename_lower == 'config.md':
        return 'CONFIG'

    if 'plan' in filename_lower or 'roadmap' in filename_lower or '## phase' in content_lower:
        return 'PLAN'

    if 'status' in filename_lower or 'report' in filename_lower:
        return 'STATUS_REPORT'

    if 'analysis' in filename_lower or '## findings' in content_lower:
        return 'ANALYSIS'

    if 'guide' in filename_lower or '## installation' in content_lower:
        return 'GUIDE'

    if 'brainstorm' in content_lower[:200] or 'ideas' in filename_lower:
        return 'BRAINSTORM'

    if 'torah' in subdomain_lower or 'parsha' in content_lower or 'midrash' in content_lower:
        return 'TORAH'

    if 'job' in filename_lower or 'resume' in filename_lower or 'linkedin' in content_lower[:500]:
        return 'CAREER'

    if 'architecture' in filename_lower or '## schema' in content_lower:
        return 'ARCHITECTURE'

    if 'whitepaper' in filename_lower or 'thesis' in content_lower[:500]:
        return 'WHITEPAPER'

    if 'evidence' in filename_lower or 'claims' in filename_lower:
        return 'EVIDENCE'

    return 'OTHER'


def extract_code_languages(content: str) -> list[str]:
    """Extract programming languages from code blocks."""
    languages = set()
    for match in re.finditer(r'```(\w+)', content):
        lang = match.group(1).lower()
        if lang in CODE_LANGUAGES:
            languages.add(lang)
    return sorted(languages)


def extract_tech_stack(content: str) -> list[str]:
    """Extract technologies mentioned."""
    content_lower = content.lower()
    found = []
    for tech in TECH_STACK:
        if tech.lower() in content_lower:
            found.append(tech)
    return found


def extract_seed_concepts(content: str) -> list[str]:
    """Extract SEED principles mentioned."""
    content_lower = content.lower()
    found = []
    for concept in SEED_CONCEPTS:
        if concept.lower() in content_lower:
            found.append(concept)
    return found


def extract_frameworks(content: str) -> list[str]:
    """Extract frameworks mentioned."""
    found = []
    for fw in FRAMEWORKS:
        if fw in content or fw.lower() in content.lower():
            found.append(fw)
    return found


def detect_action_type(content: str) -> Optional[str]:
    """Detect primary action type in document."""
    content_lower = content.lower()

    # Check patterns in order of specificity
    if re.search(r'decided|chose|selected|picked|decision', content_lower):
        return 'DECISION'
    if re.search(r'fixed|solved|resolved|figured out', content_lower):
        return 'PROBLEM_SOLVED'
    if re.search(r'learned|realized|discovered|insight', content_lower):
        return 'LEARNING'
    if re.search(r'will|going to|plan to|roadmap|next steps', content_lower):
        return 'PLAN'
    if content.count('?') > 3:
        return 'QUESTION'

    return None


def count_headings(content: str) -> int:
    """Count total headings."""
    return len(re.findall(r'^#{1,6}\s', content, re.MULTILINE))


def count_links(content: str) -> int:
    """Count URLs."""
    return len(re.findall(r'https?://', content))


def count_file_refs(content: str) -> int:
    """Count file references."""
    return len(re.findall(r'\.(md|ts|tsx|py|json|js|jsx)\b', content))


def calculate_harvest_score(row: dict) -> int:
    """Calculate IP harvest value score 0-100."""
    score = 0

    # Word count contribution (max 20)
    wc = row['word_count']
    if wc > 500:
        score += min(20, wc // 250)

    # SEED concepts (max 20)
    score += len(row['seed_concepts']) * 4

    # Frameworks (max 15)
    score += len(row['frameworks']) * 5

    # Action type bonus (max 15)
    if row['action_type'] == 'DECISION':
        score += 15
    elif row['action_type'] == 'PROBLEM_SOLVED':
        score += 12
    elif row['action_type'] == 'LEARNING':
        score += 10
    elif row['action_type'] == 'PLAN':
        score += 8

    # Doc type bonus (max 15)
    high_value_types = ['WHITEPAPER', 'ARCHITECTURE', 'ANALYSIS', 'EVIDENCE']
    medium_value_types = ['PLAN', 'BRAINSTORM', 'STATUS_REPORT']
    if row['doc_type'] in high_value_types:
        score += 15
    elif row['doc_type'] in medium_value_types:
        score += 10

    # Structure bonus (max 10)
    if row['heading_count'] > 5:
        score += 5
    if row['has_tables']:
        score += 3
    if row['has_mermaid']:
        score += 2

    # Tech stack (max 5)
    score += min(5, len(row['tech_stack']))

    return min(100, score)


def scan_markdown_files_v2(directories: list[Path], verbose: bool = True) -> list[dict]:
    """Scan directories with enhanced metadata extraction."""
    records = []
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

            # Basic metadata
            domain, subdomain = extract_domain(path)
            created, modified = get_file_dates(path)
            title = extract_title(content, path.name)

            # Enhanced metadata
            doc_type = detect_doc_type(content, path.name, subdomain)
            code_languages = extract_code_languages(content)
            tech_stack = extract_tech_stack(content)
            seed_concepts = extract_seed_concepts(content)
            frameworks = extract_frameworks(content)
            action_type = detect_action_type(content)

            record = {
                # Basic
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

                # Document type
                'doc_type': doc_type,
                'action_type': action_type,

                # Structure flags
                'has_code': '```' in content,
                'has_tables': bool(re.search(r'\|.*\|.*\|', content)),
                'has_checklists': bool(re.search(r'- \[ \]|- \[x\]', content, re.IGNORECASE)),
                'has_mermaid': '```mermaid' in content,

                # Lists
                'code_languages': code_languages,
                'tech_stack': tech_stack,
                'seed_concepts': seed_concepts,
                'frameworks': frameworks,

                # Counts
                'heading_count': count_headings(content),
                'link_count': count_links(content),
                'file_ref_count': count_file_refs(content),
                'estimated_reading_mins': max(1, len(content.split()) // 200),
            }

            # Calculate harvest score
            record['harvest_score'] = calculate_harvest_score(record)

            records.append(record)

    if verbose:
        print(f"\nScan complete:")
        print(f"  Total scanned: {scanned:,}")
        print(f"  Excluded: {excluded:,}")
        print(f"  Boilerplate: {boilerplate:,}")
        print(f"  Original IP files: {len(records):,}")

    return records


def create_parquet_v2(records: list[dict], output_path: Path) -> None:
    """Create enhanced parquet file."""
    if not records:
        print("No records!")
        return

    # Convert lists to strings for parquet
    for r in records:
        r['code_languages'] = ','.join(r['code_languages']) if r['code_languages'] else ''
        r['tech_stack'] = ','.join(r['tech_stack']) if r['tech_stack'] else ''
        r['seed_concepts'] = ','.join(r['seed_concepts']) if r['seed_concepts'] else ''
        r['frameworks'] = ','.join(r['frameworks']) if r['frameworks'] else ''
        r['action_type'] = r['action_type'] or ''

    schema = pa.schema([
        # Basic
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

        # Document type
        ('doc_type', pa.string()),
        ('action_type', pa.string()),

        # Structure flags
        ('has_code', pa.bool_()),
        ('has_tables', pa.bool_()),
        ('has_checklists', pa.bool_()),
        ('has_mermaid', pa.bool_()),

        # Lists (as comma-separated strings)
        ('code_languages', pa.string()),
        ('tech_stack', pa.string()),
        ('seed_concepts', pa.string()),
        ('frameworks', pa.string()),

        # Counts
        ('heading_count', pa.int32()),
        ('link_count', pa.int32()),
        ('file_ref_count', pa.int32()),
        ('estimated_reading_mins', pa.int32()),

        # Score
        ('harvest_score', pa.int32()),
    ])

    columns = {key: [r[key] for r in records] for key in records[0].keys()}
    table = pa.table(columns, schema=schema)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression='snappy')

    print(f"\nWrote {len(records):,} records to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def show_stats_v2(records: list[dict]) -> None:
    """Show enhanced statistics."""
    print("\n" + "="*60)
    print("ENHANCED HARVEST STATISTICS")
    print("="*60)

    # By doc type
    print("\nBy Document Type:")
    types = {}
    for r in records:
        types[r['doc_type']] = types.get(r['doc_type'], 0) + 1
    for t, c in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {t:<20} {c:>5}")

    # By harvest score
    print("\nBy Harvest Score:")
    high = sum(1 for r in records if r['harvest_score'] >= 70)
    medium = sum(1 for r in records if 40 <= r['harvest_score'] < 70)
    low = sum(1 for r in records if r['harvest_score'] < 40)
    print(f"  High (70+):   {high:>5} ({high/len(records)*100:.1f}%)")
    print(f"  Medium (40-69): {medium:>5} ({medium/len(records)*100:.1f}%)")
    print(f"  Low (<40):    {low:>5} ({low/len(records)*100:.1f}%)")

    # SEED concepts coverage
    print("\nSEED Concept Coverage:")
    for concept in SEED_CONCEPTS:
        count = sum(1 for r in records if concept in r.get('seed_concepts', []))
        if count > 0:
            print(f"  {concept:<25} {count:>5}")

    # Top frameworks
    print("\nFramework Mentions:")
    for fw in FRAMEWORKS:
        count = sum(1 for r in records if fw in r.get('frameworks', []))
        if count > 0:
            print(f"  {fw:<20} {count:>5}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced markdown harvest")
    parser.add_argument('--output', '-o', type=str,
                       default=str(DATA_DIR / "facts" / "markdown_files_v2.parquet"))
    parser.add_argument('--stats-only', action='store_true')
    parser.add_argument('--quiet', '-q', action='store_true')

    args = parser.parse_args()

    records = scan_markdown_files_v2(SCAN_DIRS, verbose=not args.quiet)

    if not records:
        print("No files found!")
        return 1

    if not args.quiet:
        show_stats_v2(records)

    if not args.stats_only:
        create_parquet_v2(records, Path(args.output))

    return 0


if __name__ == "__main__":
    sys.exit(main())
