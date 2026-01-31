#!/usr/bin/env python3
"""
Build code productivity analysis from GitHub commits and repos.
Phase 5 of 55x Mining Plan.

Creates:
- interpretations/code_velocity/v1/ - Commit velocity over time
- interpretations/code_repos/v1/ - Repository analysis
- interpretations/code_languages/v1/ - Language evolution
- interpretations/code_patterns/v1/ - Commit message patterns
"""

import duckdb
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
COMMITS_PATH = BASE_DIR / "data" / "github_commits.parquet"
REPOS_PATH = BASE_DIR / "data" / "github_repos.parquet"
INTERP_DIR = BASE_DIR / "data" / "interpretations"


def ensure_dir(path: Path):
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def build_code_velocity():
    """
    Build commit velocity analysis - how fast you code.
    """
    print("\n=== Building Code Velocity ===")

    out_dir = INTERP_DIR / "code_velocity" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Daily commit velocity
    con.execute(f"""
        COPY (
            SELECT
                DATE_TRUNC('day', timestamp) as date,
                COUNT(*) as commits,
                COUNT(DISTINCT repo_name) as repos_touched,
                COUNT(DISTINCT author) as authors
            FROM '{COMMITS_PATH}'
            GROUP BY date
            ORDER BY date
        ) TO '{out_dir}/daily.parquet' (FORMAT PARQUET)
    """)

    daily = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/daily.parquet'").fetchone()[0]
    print(f"  Daily records: {daily}")

    # Weekly commit velocity
    con.execute(f"""
        COPY (
            SELECT
                DATE_TRUNC('week', timestamp) as week_start,
                COUNT(*) as commits,
                COUNT(DISTINCT repo_name) as repos_touched,
                COUNT(DISTINCT DATE_TRUNC('day', timestamp)) as active_days
            FROM '{COMMITS_PATH}'
            GROUP BY week_start
            ORDER BY week_start
        ) TO '{out_dir}/weekly.parquet' (FORMAT PARQUET)
    """)

    # Monthly commit velocity
    con.execute(f"""
        COPY (
            SELECT
                DATE_TRUNC('month', timestamp) as month_start,
                COUNT(*) as commits,
                COUNT(DISTINCT repo_name) as repos_touched,
                COUNT(DISTINCT DATE_TRUNC('day', timestamp)) as active_days,
                COUNT(DISTINCT DATE_TRUNC('week', timestamp)) as active_weeks
            FROM '{COMMITS_PATH}'
            GROUP BY month_start
            ORDER BY month_start
        ) TO '{out_dir}/monthly.parquet' (FORMAT PARQUET)
    """)

    monthly = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/monthly.parquet'").fetchone()[0]
    print(f"  Monthly records: {monthly}")

    # High productivity days (5+ commits)
    con.execute(f"""
        COPY (
            SELECT
                DATE_TRUNC('day', timestamp) as date,
                COUNT(*) as commits,
                COUNT(DISTINCT repo_name) as repos,
                ARRAY_AGG(DISTINCT repo_name) as repo_list
            FROM '{COMMITS_PATH}'
            GROUP BY date
            HAVING commits >= 5
            ORDER BY commits DESC
        ) TO '{out_dir}/high_productivity_days.parquet' (FORMAT PARQUET)
    """)

    high_days = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/high_productivity_days.parquet'").fetchone()[0]
    print(f"  High productivity days (5+ commits): {high_days}")

    return daily


def build_repo_analysis():
    """
    Build repository analysis - project health and activity.
    """
    print("\n=== Building Repository Analysis ===")

    out_dir = INTERP_DIR / "code_repos" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Repo stats with commit counts
    con.execute(f"""
        COPY (
            SELECT
                r.repo_name,
                r.language,
                r.created_at,
                r.pushed_at,
                r.stars,
                r.is_private,
                COALESCE(c.commit_count, 0) as commits,
                COALESCE(c.first_commit, r.created_at) as first_commit,
                COALESCE(c.last_commit, r.pushed_at) as last_commit,
                COALESCE(c.active_days, 0) as active_days
            FROM '{REPOS_PATH}' r
            LEFT JOIN (
                SELECT
                    repo_name,
                    COUNT(*) as commit_count,
                    MIN(timestamp) as first_commit,
                    MAX(timestamp) as last_commit,
                    COUNT(DISTINCT DATE_TRUNC('day', timestamp)) as active_days
                FROM '{COMMITS_PATH}'
                GROUP BY repo_name
            ) c ON r.repo_name = c.repo_name
            ORDER BY commits DESC
        ) TO '{out_dir}/repos_enriched.parquet' (FORMAT PARQUET)
    """)

    count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/repos_enriched.parquet'").fetchone()[0]
    print(f"  Enriched repos: {count}")

    # Active repos (with commits)
    con.execute(f"""
        COPY (
            SELECT *
            FROM '{out_dir}/repos_enriched.parquet'
            WHERE commits > 0
            ORDER BY commits DESC
        ) TO '{out_dir}/active_repos.parquet' (FORMAT PARQUET)
    """)

    active = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/active_repos.parquet'").fetchone()[0]
    print(f"  Active repos (with commits): {active}")

    return count


def build_language_analysis():
    """
    Build language evolution analysis.
    """
    print("\n=== Building Language Analysis ===")

    out_dir = INTERP_DIR / "code_languages" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Language distribution
    con.execute(f"""
        COPY (
            SELECT
                language,
                COUNT(*) as repo_count,
                SUM(CASE WHEN is_private THEN 1 ELSE 0 END) as private_count
            FROM '{REPOS_PATH}'
            WHERE language IS NOT NULL
            GROUP BY language
            ORDER BY repo_count DESC
        ) TO '{out_dir}/distribution.parquet' (FORMAT PARQUET)
    """)

    langs = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/distribution.parquet'").fetchone()[0]
    print(f"  Languages tracked: {langs}")

    # Monthly language activity (based on commits to repos of each language)
    con.execute(f"""
        COPY (
            SELECT
                DATE_TRUNC('month', c.timestamp) as month_start,
                r.language,
                COUNT(*) as commits
            FROM '{COMMITS_PATH}' c
            JOIN '{REPOS_PATH}' r ON c.repo_name = r.repo_name
            WHERE r.language IS NOT NULL
            GROUP BY month_start, r.language
            ORDER BY month_start, commits DESC
        ) TO '{out_dir}/monthly_activity.parquet' (FORMAT PARQUET)
    """)

    monthly = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/monthly_activity.parquet'").fetchone()[0]
    print(f"  Monthly language records: {monthly}")

    # Top languages
    top = con.execute(f"""
        SELECT language, repo_count
        FROM '{out_dir}/distribution.parquet'
        ORDER BY repo_count DESC
        LIMIT 5
    """).fetchall()
    print("  Top languages:")
    for lang, cnt in top:
        print(f"    {lang}: {cnt} repos")

    return langs


def build_commit_patterns():
    """
    Build commit message pattern analysis.
    """
    print("\n=== Building Commit Patterns ===")

    out_dir = INTERP_DIR / "code_patterns" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Categorize commit types
    con.execute(f"""
        COPY (
            SELECT
                sha,
                repo_name,
                timestamp,
                message,
                CASE
                    WHEN LOWER(message) LIKE 'fix%' OR LOWER(message) LIKE '%fix %' OR LOWER(message) LIKE '%bug%' THEN 'fix'
                    WHEN LOWER(message) LIKE 'add%' OR LOWER(message) LIKE '%add %' THEN 'add'
                    WHEN LOWER(message) LIKE 'update%' OR LOWER(message) LIKE '%update %' THEN 'update'
                    WHEN LOWER(message) LIKE 'refactor%' OR LOWER(message) LIKE '%refactor%' THEN 'refactor'
                    WHEN LOWER(message) LIKE 'remove%' OR LOWER(message) LIKE '%delete%' THEN 'remove'
                    WHEN LOWER(message) LIKE 'initial%' OR LOWER(message) LIKE 'init%' THEN 'initial'
                    WHEN LOWER(message) LIKE 'merge%' THEN 'merge'
                    WHEN LOWER(message) LIKE 'wip%' OR LOWER(message) LIKE '%work in progress%' THEN 'wip'
                    WHEN LOWER(message) LIKE 'test%' OR LOWER(message) LIKE '%test %' THEN 'test'
                    WHEN LOWER(message) LIKE 'docs%' OR LOWER(message) LIKE '%documentation%' THEN 'docs'
                    ELSE 'other'
                END as commit_type,
                LENGTH(message) as message_length
            FROM '{COMMITS_PATH}'
            ORDER BY timestamp
        ) TO '{out_dir}/commits_categorized.parquet' (FORMAT PARQUET)
    """)

    count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/commits_categorized.parquet'").fetchone()[0]
    print(f"  Categorized commits: {count}")

    # Commit type distribution
    con.execute(f"""
        COPY (
            SELECT
                commit_type,
                COUNT(*) as count,
                AVG(message_length) as avg_msg_length
            FROM '{out_dir}/commits_categorized.parquet'
            GROUP BY commit_type
            ORDER BY count DESC
        ) TO '{out_dir}/type_distribution.parquet' (FORMAT PARQUET)
    """)

    # Monthly commit types
    con.execute(f"""
        COPY (
            SELECT
                DATE_TRUNC('month', timestamp) as month_start,
                commit_type,
                COUNT(*) as count
            FROM '{out_dir}/commits_categorized.parquet'
            GROUP BY month_start, commit_type
            ORDER BY month_start, count DESC
        ) TO '{out_dir}/monthly_types.parquet' (FORMAT PARQUET)
    """)

    # Show distribution
    dist = con.execute(f"""
        SELECT commit_type, count
        FROM '{out_dir}/type_distribution.parquet'
        ORDER BY count DESC
    """).fetchall()
    print("  Commit type distribution:")
    for ctype, cnt in dist:
        print(f"    {ctype}: {cnt}")

    return count


def build_all():
    """Build all code productivity layers."""
    print(f"Building code productivity analysis")
    print(f"Timestamp: {datetime.now().isoformat()}")

    if not COMMITS_PATH.exists():
        print(f"ERROR: Commits not found at {COMMITS_PATH}")
        return

    # Build all layers
    build_code_velocity()
    build_repo_analysis()
    build_language_analysis()
    build_commit_patterns()

    print("\n✅ All code productivity layers built!")


if __name__ == "__main__":
    build_all()
