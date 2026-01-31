#!/usr/bin/env python3
"""
Build markdown knowledge analysis from 5.5K documents.
Phase 6 of 55x Mining Plan.

Creates:
- interpretations/markdown_stats/v1/ - Document statistics
- interpretations/markdown_categories/v1/ - Category analysis
- interpretations/markdown_projects/v1/ - Project clustering
- interpretations/markdown_curated/v1/ - Curated document indexes
"""

import duckdb
from pathlib import Path
from datetime import datetime
import csv

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
MARKDOWN_PATH = BASE_DIR / "data" / "facts" / "markdown_files_v20.parquet"
TOP_IP_PATH = BASE_DIR / "data" / "facts" / "TOP_TIER_IP.csv"
DEEP_DOCS_PATH = BASE_DIR / "data" / "facts" / "TOP_261_DEEP_DOCS.csv"
INTERP_DIR = BASE_DIR / "data" / "interpretations"


def ensure_dir(path: Path):
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def build_markdown_stats():
    """
    Build document-level statistics.
    """
    print("\n=== Building Markdown Stats ===")

    out_dir = INTERP_DIR / "markdown_stats" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Document summary
    con.execute(f"""
        COPY (
            SELECT
                path,
                filename,
                domain,
                final_category,
                word_count,
                char_count,
                line_count,
                heading_count,
                link_count,
                has_code,
                has_tables,
                has_checklists,
                harvest_score,
                depth_score,
                estimated_reading_mins,
                date_created,
                date_modified,
                project,
                voice,
                energy
            FROM '{MARKDOWN_PATH}'
            ORDER BY word_count DESC
        ) TO '{out_dir}/documents.parquet' (FORMAT PARQUET)
    """)

    count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/documents.parquet'").fetchone()[0]
    print(f"  Documents indexed: {count:,}")

    # High-value documents (harvest_score > 0 or depth_score > 0)
    con.execute(f"""
        COPY (
            SELECT *
            FROM '{out_dir}/documents.parquet'
            WHERE harvest_score > 0 OR depth_score > 0
            ORDER BY harvest_score DESC, depth_score DESC
        ) TO '{out_dir}/high_value.parquet' (FORMAT PARQUET)
    """)

    high_value = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/high_value.parquet'").fetchone()[0]
    print(f"  High-value documents: {high_value}")

    # Monthly document creation
    con.execute(f"""
        COPY (
            SELECT
                strftime(date_created, '%Y-%m') as month,
                COUNT(*) as docs,
                SUM(word_count) as words
            FROM '{MARKDOWN_PATH}'
            WHERE date_created IS NOT NULL
            GROUP BY month
            ORDER BY month
        ) TO '{out_dir}/monthly.parquet' (FORMAT PARQUET)
    """)

    return count


def build_category_analysis():
    """
    Build category-level analysis.
    """
    print("\n=== Building Category Analysis ===")

    out_dir = INTERP_DIR / "markdown_categories" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Category summary
    con.execute(f"""
        COPY (
            SELECT
                final_category,
                COUNT(*) as doc_count,
                SUM(word_count) as total_words,
                AVG(word_count) as avg_words,
                SUM(CASE WHEN has_code THEN 1 ELSE 0 END) as with_code,
                AVG(harvest_score) as avg_harvest_score,
                AVG(depth_score) as avg_depth_score,
                COUNT(DISTINCT project) as projects
            FROM '{MARKDOWN_PATH}'
            GROUP BY final_category
            ORDER BY doc_count DESC
        ) TO '{out_dir}/summary.parquet' (FORMAT PARQUET)
    """)

    count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/summary.parquet'").fetchone()[0]
    print(f"  Categories analyzed: {count}")

    # Domain summary
    con.execute(f"""
        COPY (
            SELECT
                domain,
                COUNT(*) as doc_count,
                SUM(word_count) as total_words,
                COUNT(DISTINCT final_category) as categories,
                COUNT(DISTINCT project) as projects
            FROM '{MARKDOWN_PATH}'
            GROUP BY domain
            ORDER BY doc_count DESC
        ) TO '{out_dir}/domains.parquet' (FORMAT PARQUET)
    """)

    domains = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/domains.parquet'").fetchone()[0]
    print(f"  Domains analyzed: {domains}")

    return count


def build_project_analysis():
    """
    Build project-level clustering.
    """
    print("\n=== Building Project Analysis ===")

    out_dir = INTERP_DIR / "markdown_projects" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Project summary
    con.execute(f"""
        COPY (
            SELECT
                project,
                COUNT(*) as doc_count,
                SUM(word_count) as total_words,
                COUNT(DISTINCT final_category) as categories,
                SUM(CASE WHEN has_code THEN 1 ELSE 0 END) as with_code,
                AVG(harvest_score) as avg_harvest_score,
                MIN(date_created) as first_doc,
                MAX(date_modified) as last_modified
            FROM '{MARKDOWN_PATH}'
            WHERE project IS NOT NULL AND project != ''
            GROUP BY project
            HAVING doc_count >= 3
            ORDER BY doc_count DESC
        ) TO '{out_dir}/projects.parquet' (FORMAT PARQUET)
    """)

    count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/projects.parquet'").fetchone()[0]
    print(f"  Projects analyzed: {count}")

    # Top projects
    top = con.execute(f"""
        SELECT project, doc_count, total_words
        FROM '{out_dir}/projects.parquet'
        ORDER BY doc_count DESC
        LIMIT 10
    """).fetchall()
    print("  Top projects:")
    for proj, docs, words in top:
        print(f"    {proj}: {docs} docs, {words:,} words")

    return count


def build_curated_index():
    """
    Build curated document indexes from manual review files.
    """
    print("\n=== Building Curated Index ===")

    out_dir = INTERP_DIR / "markdown_curated" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Import TOP_TIER_IP if exists
    if TOP_IP_PATH.exists():
        con.execute(f"""
            COPY (
                SELECT * FROM read_csv('{TOP_IP_PATH}', auto_detect=true)
            ) TO '{out_dir}/top_tier_ip.parquet' (FORMAT PARQUET)
        """)
        ip_count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/top_tier_ip.parquet'").fetchone()[0]
        print(f"  TOP_TIER_IP: {ip_count} documents")
    else:
        ip_count = 0
        print("  TOP_TIER_IP: not found")

    # Import DEEP_DOCS if exists
    if DEEP_DOCS_PATH.exists():
        con.execute(f"""
            COPY (
                SELECT * FROM read_csv('{DEEP_DOCS_PATH}', auto_detect=true)
            ) TO '{out_dir}/deep_docs.parquet' (FORMAT PARQUET)
        """)
        deep_count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/deep_docs.parquet'").fetchone()[0]
        print(f"  DEEP_DOCS: {deep_count} documents")
    else:
        deep_count = 0
        print("  DEEP_DOCS: not found")

    return ip_count + deep_count


def build_content_patterns():
    """
    Build content pattern analysis.
    """
    print("\n=== Building Content Patterns ===")

    out_dir = INTERP_DIR / "markdown_patterns" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Voice distribution
    con.execute(f"""
        COPY (
            SELECT
                voice,
                COUNT(*) as doc_count,
                SUM(word_count) as total_words
            FROM '{MARKDOWN_PATH}'
            WHERE voice IS NOT NULL
            GROUP BY voice
            ORDER BY doc_count DESC
        ) TO '{out_dir}/voice_distribution.parquet' (FORMAT PARQUET)
    """)

    # Energy distribution
    con.execute(f"""
        COPY (
            SELECT
                energy,
                COUNT(*) as doc_count,
                AVG(word_count) as avg_words
            FROM '{MARKDOWN_PATH}'
            WHERE energy IS NOT NULL
            GROUP BY energy
            ORDER BY doc_count DESC
        ) TO '{out_dir}/energy_distribution.parquet' (FORMAT PARQUET)
    """)

    # Code language usage
    con.execute(f"""
        COPY (
            SELECT
                code_languages,
                COUNT(*) as doc_count
            FROM '{MARKDOWN_PATH}'
            WHERE code_languages IS NOT NULL AND code_languages != '[]'
            GROUP BY code_languages
            ORDER BY doc_count DESC
            LIMIT 50
        ) TO '{out_dir}/code_languages.parquet' (FORMAT PARQUET)
    """)

    # Tech stack usage
    con.execute(f"""
        COPY (
            SELECT
                tech_stack,
                COUNT(*) as doc_count
            FROM '{MARKDOWN_PATH}'
            WHERE tech_stack IS NOT NULL AND tech_stack != '[]'
            GROUP BY tech_stack
            ORDER BY doc_count DESC
            LIMIT 50
        ) TO '{out_dir}/tech_stacks.parquet' (FORMAT PARQUET)
    """)

    patterns = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/voice_distribution.parquet'").fetchone()[0]
    print(f"  Voice patterns: {patterns}")

    return patterns


def build_all():
    """Build all markdown analysis layers."""
    print(f"Building markdown analysis from {MARKDOWN_PATH}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    if not MARKDOWN_PATH.exists():
        print(f"ERROR: Markdown data not found at {MARKDOWN_PATH}")
        return

    # Build all layers
    build_markdown_stats()
    build_category_analysis()
    build_project_analysis()
    build_curated_index()
    build_content_patterns()

    print("\n✅ All markdown analysis layers built!")


if __name__ == "__main__":
    build_all()
