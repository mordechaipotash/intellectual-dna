#!/usr/bin/env python3
"""
Build behavioral archaeology from Google searches/visits and YouTube.
Phase 4 of 55x Mining Plan.

Creates:
- interpretations/search_patterns/v1/ - Search intent and clustering
- interpretations/browsing_patterns/v1/ - Website visit analysis
- interpretations/research_velocity/v1/ - Deep dive detection
- interpretations/curiosity_domains/v1/ - Interest mapping
"""

import duckdb
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
GOOGLE_SEARCHES = BASE_DIR / "data" / "google_searches.parquet"
GOOGLE_VISITS = BASE_DIR / "data" / "google_visits.parquet"
YOUTUBE_SEARCHES = BASE_DIR / "data" / "youtube_searches.parquet"
INTERP_DIR = BASE_DIR / "data" / "interpretations"


def ensure_dir(path: Path):
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def build_search_patterns():
    """
    Build search pattern analysis - categorize searches by intent.
    """
    print("\n=== Building Search Patterns ===")

    out_dir = INTERP_DIR / "search_patterns" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Categorize searches by intent
    con.execute(f"""
        COPY (
            SELECT
                search_id,
                query,
                timestamp,
                year,
                month,
                hour,
                word_count,
                has_question,
                CASE
                    WHEN LOWER(query) LIKE '%error%' OR LOWER(query) LIKE '%bug%'
                         OR LOWER(query) LIKE '%fix%' OR LOWER(query) LIKE '%not working%'
                         OR LOWER(query) LIKE '%failed%' THEN 'debugging'
                    WHEN LOWER(query) LIKE '%how to%' OR LOWER(query) LIKE '%how do%'
                         OR LOWER(query) LIKE '%tutorial%' OR LOWER(query) LIKE '%guide%' THEN 'how-to'
                    WHEN LOWER(query) LIKE '%what is%' OR LOWER(query) LIKE '%define%'
                         OR LOWER(query) LIKE '%meaning%' THEN 'definition'
                    WHEN LOWER(query) LIKE '%vs%' OR LOWER(query) LIKE '%versus%'
                         OR LOWER(query) LIKE '%comparison%' OR LOWER(query) LIKE '%better%' THEN 'comparison'
                    WHEN LOWER(query) LIKE '%python%' OR LOWER(query) LIKE '%javascript%'
                         OR LOWER(query) LIKE '%react%' OR LOWER(query) LIKE '%api%'
                         OR LOWER(query) LIKE '%code%' OR LOWER(query) LIKE '%function%' THEN 'programming'
                    WHEN LOWER(query) LIKE '%buy%' OR LOWER(query) LIKE '%price%'
                         OR LOWER(query) LIKE '%amazon%' OR LOWER(query) LIKE '%shop%' THEN 'shopping'
                    WHEN LOWER(query) LIKE '%news%' OR LOWER(query) LIKE '%latest%'
                         OR LOWER(query) LIKE '%2024%' OR LOWER(query) LIKE '%2025%' THEN 'news'
                    WHEN has_question = true THEN 'question'
                    ELSE 'general'
                END as search_intent
            FROM '{GOOGLE_SEARCHES}'
            ORDER BY timestamp
        ) TO '{out_dir}/searches_categorized.parquet' (FORMAT PARQUET)
    """)

    count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/searches_categorized.parquet'").fetchone()[0]
    print(f"  Categorized searches: {count:,}")

    # Monthly search intent distribution
    con.execute(f"""
        COPY (
            SELECT
                year || '-' || LPAD(CAST(month AS VARCHAR), 2, '0') as month_str,
                search_intent,
                COUNT(*) as count
            FROM '{out_dir}/searches_categorized.parquet'
            GROUP BY year, month, search_intent
            ORDER BY year, month, count DESC
        ) TO '{out_dir}/monthly_intents.parquet' (FORMAT PARQUET)
    """)

    # Hour of day patterns
    con.execute(f"""
        COPY (
            SELECT
                hour,
                search_intent,
                COUNT(*) as count
            FROM '{out_dir}/searches_categorized.parquet'
            GROUP BY hour, search_intent
            ORDER BY hour, count DESC
        ) TO '{out_dir}/hourly_intents.parquet' (FORMAT PARQUET)
    """)

    # Overall intent distribution
    intents = con.execute(f"""
        SELECT search_intent, COUNT(*) as cnt
        FROM '{out_dir}/searches_categorized.parquet'
        GROUP BY search_intent
        ORDER BY cnt DESC
    """).fetchall()

    print("  Intent distribution:")
    for intent, cnt in intents:
        print(f"    {intent}: {cnt:,}")

    return count


def build_browsing_patterns():
    """
    Build browsing pattern analysis - domain clustering and visit analysis.
    """
    print("\n=== Building Browsing Patterns ===")

    out_dir = INTERP_DIR / "browsing_patterns" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Extract domains and categorize
    con.execute(f"""
        COPY (
            SELECT
                visit_id,
                url,
                title,
                timestamp,
                year,
                month,
                hour,
                -- Extract domain from URL
                REGEXP_EXTRACT(url, 'https?://([^/]+)', 1) as domain,
                CASE
                    WHEN url LIKE '%github.com%' THEN 'github'
                    WHEN url LIKE '%stackoverflow.com%' OR url LIKE '%stackexchange.com%' THEN 'stackoverflow'
                    WHEN url LIKE '%google.com%' THEN 'google'
                    WHEN url LIKE '%youtube.com%' THEN 'youtube'
                    WHEN url LIKE '%reddit.com%' THEN 'reddit'
                    WHEN url LIKE '%twitter.com%' OR url LIKE '%x.com%' THEN 'twitter'
                    WHEN url LIKE '%linkedin.com%' THEN 'linkedin'
                    WHEN url LIKE '%medium.com%' THEN 'medium'
                    WHEN url LIKE '%docs.%' OR url LIKE '%documentation%' THEN 'documentation'
                    WHEN url LIKE '%amazon.com%' THEN 'amazon'
                    WHEN url LIKE '%.edu%' THEN 'academic'
                    WHEN url LIKE '%news%' OR url LIKE '%bbc%' OR url LIKE '%cnn%' THEN 'news'
                    ELSE 'other'
                END as site_category
            FROM '{GOOGLE_VISITS}'
            ORDER BY timestamp
        ) TO '{out_dir}/visits_categorized.parquet' (FORMAT PARQUET)
    """)

    count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/visits_categorized.parquet'").fetchone()[0]
    print(f"  Categorized visits: {count:,}")

    # Top domains
    con.execute(f"""
        COPY (
            SELECT
                domain,
                COUNT(*) as visit_count,
                COUNT(DISTINCT year || '-' || month) as months_active,
                MIN(timestamp) as first_visit,
                MAX(timestamp) as last_visit
            FROM '{out_dir}/visits_categorized.parquet'
            WHERE domain IS NOT NULL
            GROUP BY domain
            HAVING visit_count >= 5
            ORDER BY visit_count DESC
        ) TO '{out_dir}/top_domains.parquet' (FORMAT PARQUET)
    """)

    domains = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/top_domains.parquet'").fetchone()[0]
    print(f"  Unique domains (5+ visits): {domains:,}")

    # Site category distribution
    con.execute(f"""
        COPY (
            SELECT
                site_category,
                COUNT(*) as visit_count,
                COUNT(DISTINCT domain) as unique_domains
            FROM '{out_dir}/visits_categorized.parquet'
            GROUP BY site_category
            ORDER BY visit_count DESC
        ) TO '{out_dir}/category_summary.parquet' (FORMAT PARQUET)
    """)

    # Hour patterns
    con.execute(f"""
        COPY (
            SELECT
                hour,
                site_category,
                COUNT(*) as count
            FROM '{out_dir}/visits_categorized.parquet'
            GROUP BY hour, site_category
            ORDER BY hour, count DESC
        ) TO '{out_dir}/hourly_categories.parquet' (FORMAT PARQUET)
    """)

    # Monthly browsing volume
    con.execute(f"""
        COPY (
            SELECT
                year || '-' || LPAD(CAST(month AS VARCHAR), 2, '0') as month_str,
                site_category,
                COUNT(*) as visits
            FROM '{out_dir}/visits_categorized.parquet'
            GROUP BY year, month, site_category
            ORDER BY year, month, visits DESC
        ) TO '{out_dir}/monthly_categories.parquet' (FORMAT PARQUET)
    """)

    return count


def build_research_velocity():
    """
    Build research velocity - detect deep dive sessions.
    """
    print("\n=== Building Research Velocity ===")

    out_dir = INTERP_DIR / "research_velocity" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Daily search volume (research intensity)
    con.execute(f"""
        COPY (
            SELECT
                DATE_TRUNC('day', timestamp) as date,
                COUNT(*) as search_count,
                COUNT(DISTINCT query) as unique_queries,
                SUM(word_count) as total_words,
                SUM(CASE WHEN has_question THEN 1 ELSE 0 END) as questions
            FROM '{GOOGLE_SEARCHES}'
            GROUP BY date
            ORDER BY date
        ) TO '{out_dir}/daily_search_intensity.parquet' (FORMAT PARQUET)
    """)

    # Find high-intensity research days (top 10%)
    con.execute(f"""
        COPY (
            SELECT
                date,
                search_count,
                unique_queries,
                questions,
                'high_intensity' as label
            FROM '{out_dir}/daily_search_intensity.parquet'
            WHERE search_count >= (
                SELECT PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY search_count)
                FROM '{out_dir}/daily_search_intensity.parquet'
            )
            ORDER BY search_count DESC
        ) TO '{out_dir}/deep_dive_days.parquet' (FORMAT PARQUET)
    """)

    deep_days = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/deep_dive_days.parquet'").fetchone()[0]
    print(f"  Deep dive days (top 10%): {deep_days}")

    # Weekly research patterns
    con.execute(f"""
        COPY (
            SELECT
                DATE_TRUNC('week', timestamp) as week_start,
                COUNT(*) as searches,
                COUNT(DISTINCT DATE_TRUNC('day', timestamp)) as active_days,
                AVG(word_count) as avg_query_length
            FROM '{GOOGLE_SEARCHES}'
            GROUP BY week_start
            ORDER BY week_start
        ) TO '{out_dir}/weekly_velocity.parquet' (FORMAT PARQUET)
    """)

    weeks = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/weekly_velocity.parquet'").fetchone()[0]
    print(f"  Weekly velocity records: {weeks}")

    return deep_days


def build_curiosity_domains():
    """
    Build curiosity domain analysis - what topics you research.
    """
    print("\n=== Building Curiosity Domains ===")

    out_dir = INTERP_DIR / "curiosity_domains" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Extract key terms from searches (simple word frequency)
    con.execute(f"""
        COPY (
            SELECT
                LOWER(word) as term,
                COUNT(*) as frequency,
                COUNT(DISTINCT year || '-' || month) as months_appeared,
                MIN(timestamp) as first_search,
                MAX(timestamp) as last_search
            FROM (
                SELECT
                    UNNEST(STRING_SPLIT(LOWER(REGEXP_REPLACE(query, '[^a-zA-Z0-9 ]', ' ', 'g')), ' ')) as word,
                    timestamp,
                    year,
                    month
                FROM '{GOOGLE_SEARCHES}'
            )
            WHERE LENGTH(word) > 3
              AND word NOT IN ('http', 'https', 'www', 'com', 'the', 'and', 'for', 'with', 'that', 'this', 'from', 'have', 'what', 'how')
            GROUP BY word
            HAVING frequency >= 10
            ORDER BY frequency DESC
        ) TO '{out_dir}/search_terms.parquet' (FORMAT PARQUET)
    """)

    terms = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/search_terms.parquet'").fetchone()[0]
    print(f"  Significant search terms: {terms}")

    # Monthly term evolution
    con.execute(f"""
        COPY (
            SELECT
                year || '-' || LPAD(CAST(month AS VARCHAR), 2, '0') as month_str,
                LOWER(word) as term,
                COUNT(*) as frequency
            FROM (
                SELECT
                    UNNEST(STRING_SPLIT(LOWER(REGEXP_REPLACE(query, '[^a-zA-Z0-9 ]', ' ', 'g')), ' ')) as word,
                    year,
                    month
                FROM '{GOOGLE_SEARCHES}'
            )
            WHERE LENGTH(word) > 3
              AND word NOT IN ('http', 'https', 'www', 'com', 'the', 'and', 'for', 'with', 'that', 'this', 'from', 'have', 'what', 'how')
            GROUP BY year, month, word
            HAVING frequency >= 3
            ORDER BY year, month, frequency DESC
        ) TO '{out_dir}/monthly_terms.parquet' (FORMAT PARQUET)
    """)

    monthly = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/monthly_terms.parquet'").fetchone()[0]
    print(f"  Monthly term records: {monthly:,}")

    # Top terms
    top = con.execute(f"""
        SELECT term, frequency
        FROM '{out_dir}/search_terms.parquet'
        ORDER BY frequency DESC
        LIMIT 20
    """).fetchall()

    print("  Top search terms:")
    for term, freq in top[:10]:
        print(f"    {term}: {freq}")

    return terms


def build_all():
    """Build all behavioral analysis layers."""
    print(f"Building behavioral analysis")
    print(f"Timestamp: {datetime.now().isoformat()}")

    if not GOOGLE_SEARCHES.exists():
        print(f"ERROR: Google searches not found at {GOOGLE_SEARCHES}")
        return

    # Build all layers
    build_search_patterns()
    build_browsing_patterns()
    build_research_velocity()
    build_curiosity_domains()

    print("\n✅ All behavioral analysis layers built!")


if __name__ == "__main__":
    build_all()
