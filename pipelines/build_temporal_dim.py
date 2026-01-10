#!/usr/bin/env python3
"""
Build temporal dimension table for star schema analytics.
Contains date attributes and pre-computed daily activity metrics.
"""

import duckdb
from pathlib import Path

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
DATA_DIR = BASE_DIR / "data"
FACTS_DIR = DATA_DIR / "facts"


def build_temporal_dim():
    """Build temporal dimension table with activity metrics."""
    print("Building temporal_dim.parquet...")

    con = duckdb.connect()
    output_path = FACTS_DIR / "temporal_dim.parquet"

    # Get date range from all sources
    min_date = con.execute(f"""
        SELECT MIN(d) FROM (
            SELECT MIN(CAST(msg_timestamp AS DATE)) as d FROM '{DATA_DIR}/all_conversations.parquet'
            UNION ALL
            SELECT MIN(CAST(watched_date AS DATE)) as d FROM '{DATA_DIR}/youtube_rows.parquet'
            UNION ALL
            SELECT MIN(CAST(timestamp AS DATE)) as d FROM '{DATA_DIR}/github_commits.parquet'
        )
    """).fetchone()[0]

    max_date = con.execute(f"""
        SELECT MAX(d) FROM (
            SELECT MAX(CAST(msg_timestamp AS DATE)) as d FROM '{DATA_DIR}/all_conversations.parquet'
            UNION ALL
            SELECT MAX(CAST(watched_date AS DATE)) as d FROM '{DATA_DIR}/youtube_rows.parquet'
            UNION ALL
            SELECT MAX(CAST(timestamp AS DATE)) as d FROM '{DATA_DIR}/github_commits.parquet'
        )
    """).fetchone()[0]

    print(f"  Date range: {min_date} → {max_date}")

    # Build dimension table
    con.execute(f"""
        COPY (
            WITH dates AS (
                SELECT CAST(unnest(generate_series(
                    CAST('{min_date}' AS DATE),
                    CAST('{max_date}' AS DATE),
                    INTERVAL '1 day'
                )) AS DATE) as date
            ),
            -- Pre-aggregate messages by day
            msg_daily AS (
                SELECT
                    CAST(msg_timestamp AS DATE) as date,
                    COUNT(*) as messages_sent,
                    COUNT(DISTINCT conversation_id) as convos_started,
                    SUM(word_count) as words_written
                FROM '{DATA_DIR}/all_conversations.parquet'
                WHERE role = 'user' AND msg_timestamp IS NOT NULL
                GROUP BY CAST(msg_timestamp AS DATE)
            ),
            -- Pre-aggregate YouTube by day
            yt_daily AS (
                SELECT
                    CAST(watched_date AS DATE) as date,
                    COUNT(*) as videos_watched
                FROM '{DATA_DIR}/youtube_rows.parquet'
                WHERE watched_date IS NOT NULL
                GROUP BY CAST(watched_date AS DATE)
            ),
            -- Pre-aggregate commits by day
            git_daily AS (
                SELECT
                    CAST(timestamp AS DATE) as date,
                    COUNT(*) as commits_made
                FROM '{DATA_DIR}/github_commits.parquet'
                GROUP BY CAST(timestamp AS DATE)
            ),
            -- Pre-aggregate spend by day
            spend_daily AS (
                SELECT
                    CAST(date AS DATE) as date,
                    SUM(cost_usd) as cost_total,
                    SUM(tokens_total) as tokens_used
                FROM '{FACTS_DIR}/spend/raw.parquet'
                WHERE date IS NOT NULL AND date != ''
                GROUP BY CAST(date AS DATE)
            )
            SELECT
                d.date,
                -- Date attributes
                EXTRACT(year FROM d.date) as year,
                'Q' || EXTRACT(quarter FROM d.date) as quarter,
                EXTRACT(month FROM d.date) as month,
                strftime(d.date, '%B') as month_name,
                EXTRACT(week FROM d.date) as week_of_year,
                EXTRACT(dow FROM d.date) as day_of_week,
                strftime(d.date, '%A') as day_name,
                CASE WHEN EXTRACT(dow FROM d.date) IN (0, 6) THEN true ELSE false END as is_weekend,

                -- Activity metrics
                COALESCE(m.messages_sent, 0) as messages_sent,
                COALESCE(m.convos_started, 0) as convos_started,
                COALESCE(m.words_written, 0) as words_written,
                COALESCE(y.videos_watched, 0) as videos_watched,
                COALESCE(g.commits_made, 0) as commits_made,
                COALESCE(s.cost_total, 0) as cost_total,
                COALESCE(s.tokens_used, 0) as tokens_used,

                -- Derived metrics
                COALESCE(m.messages_sent, 0) + COALESCE(g.commits_made, 0) as productivity_score,
                CASE WHEN m.messages_sent > 0 OR g.commits_made > 0 THEN true ELSE false END as was_active

            FROM dates d
            LEFT JOIN msg_daily m ON d.date = m.date
            LEFT JOIN yt_daily y ON d.date = y.date
            LEFT JOIN git_daily g ON d.date = g.date
            LEFT JOIN spend_daily s ON d.date = s.date
            ORDER BY d.date
        ) TO '{output_path}' (FORMAT PARQUET)
    """)

    # Stats
    count = con.execute(f"SELECT COUNT(*) FROM '{output_path}'").fetchone()[0]
    active_days = con.execute(f"SELECT COUNT(*) FROM '{output_path}' WHERE was_active = true").fetchone()[0]
    total_cost = con.execute(f"SELECT SUM(cost_total) FROM '{output_path}'").fetchone()[0]
    total_tokens = con.execute(f"SELECT SUM(tokens_used) FROM '{output_path}'").fetchone()[0]

    print(f"\n  Total days:  {count:,}")
    print(f"  Active days: {active_days:,} ({100*active_days/count:.1f}%)")
    print(f"  Total cost:  ${total_cost:,.2f}")
    print(f"  Total tokens: {total_tokens:,}")

    # Show monthly summary
    print("\n  Monthly activity (recent 6 months):")
    monthly = con.execute(f"""
        SELECT
            year || '-' || LPAD(CAST(month AS VARCHAR), 2, '0') as month,
            SUM(messages_sent) as msgs,
            SUM(videos_watched) as vids,
            SUM(commits_made) as commits,
            SUM(cost_total) as cost
        FROM '{output_path}'
        GROUP BY year, month
        ORDER BY year DESC, month DESC
        LIMIT 6
    """).fetchall()

    for month, msgs, vids, commits, cost in monthly:
        print(f"    {month}: {msgs:>6,} msgs | {vids:>4,} vids | {commits:>3,} commits | ${cost:>7,.2f}")

    return count


if __name__ == '__main__':
    build_temporal_dim()
