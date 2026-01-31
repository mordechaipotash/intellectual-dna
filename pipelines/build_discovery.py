#!/usr/bin/env python3
"""
Build discovery pipelines - automated insight generation.
Phase 9 of 55x Mining Plan.

Creates:
- interpretations/anomalies/v1/ - Spending spikes, productivity outliers
- interpretations/trends/v1/ - Rising/declining concepts
- interpretations/recommendations/v1/ - Dormant valuable topics, connections
- interpretations/weekly_synthesis/v1/ - Auto-generated weekly reports
"""

import duckdb
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
DATA_DIR = BASE_DIR / "data"
INTERP_DIR = DATA_DIR / "interpretations"

# Source paths
PRODUCTIVITY = INTERP_DIR / "productivity_matrix" / "v1" / "daily.parquet"
SPEND_DAILY = INTERP_DIR / "spend_temporal" / "v1" / "daily.parquet"
SPEND_MONTHLY = INTERP_DIR / "spend_temporal" / "v1" / "monthly.parquet"
CURIOSITY_TERMS = INTERP_DIR / "curiosity_domains" / "v1" / "monthly_terms.parquet"
SEARCH_TERMS = INTERP_DIR / "curiosity_domains" / "v1" / "search_terms.parquet"
CONVERSATIONS = DATA_DIR / "all_conversations.parquet"
UNIFIED_TIMELINE = INTERP_DIR / "unified_timeline" / "v1" / "monthly.parquet"
LEARNING_ARCS = INTERP_DIR / "learning_arcs" / "v1" / "monthly_learning.parquet"
PROJECT_SUCCESS = INTERP_DIR / "project_success" / "v1" / "projects_enriched.parquet"


def ensure_dir(path: Path):
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def build_anomaly_detection():
    """
    Detect anomalies: spending spikes, productivity outliers, behavior breaks.
    """
    print("\n=== Building Anomaly Detection ===")

    out_dir = INTERP_DIR / "anomalies" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Spending anomalies (days with spend > 2 std devs above mean)
    if SPEND_DAILY.exists():
        con.execute(f"""
            COPY (
                WITH stats AS (
                    SELECT
                        AVG(cost_usd) as mean_spend,
                        STDDEV(cost_usd) as std_spend
                    FROM '{SPEND_DAILY}'
                    WHERE cost_usd > 0
                )
                SELECT
                    d.date,
                    d.cost_usd,
                    d.tokens,
                    d.call_count,
                    d.models_used,
                    (d.cost_usd - s.mean_spend) / NULLIF(s.std_spend, 0) as z_score,
                    'spending_spike' as anomaly_type
                FROM '{SPEND_DAILY}' d, stats s
                WHERE d.cost_usd > s.mean_spend + (2 * s.std_spend)
                ORDER BY d.cost_usd DESC
            ) TO '{out_dir}/spending_spikes.parquet' (FORMAT PARQUET)
        """)
        spikes = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/spending_spikes.parquet'").fetchone()[0]
        print(f"  Spending spikes detected: {spikes}")

    # Productivity anomalies (unusually high or low days)
    if PRODUCTIVITY.exists():
        con.execute(f"""
            COPY (
                WITH stats AS (
                    SELECT
                        AVG(productivity_score) as mean_prod,
                        STDDEV(productivity_score) as std_prod
                    FROM '{PRODUCTIVITY}'
                    WHERE productivity_score > 0
                )
                SELECT
                    p.date,
                    p.productivity_score,
                    p.accomplishments,
                    p.commits,
                    p.api_calls,
                    (p.productivity_score - s.mean_prod) / NULLIF(s.std_prod, 0) as z_score,
                    CASE
                        WHEN p.productivity_score > s.mean_prod + (2 * s.std_prod) THEN 'high_productivity'
                        WHEN p.productivity_score < s.mean_prod - (2 * s.std_prod) THEN 'low_productivity'
                    END as anomaly_type
                FROM '{PRODUCTIVITY}' p, stats s
                WHERE p.productivity_score > s.mean_prod + (2 * s.std_prod)
                   OR p.productivity_score < s.mean_prod - (2 * s.std_prod)
                ORDER BY ABS((p.productivity_score - s.mean_prod) / NULLIF(s.std_prod, 0)) DESC
            ) TO '{out_dir}/productivity_outliers.parquet' (FORMAT PARQUET)
        """)
        outliers = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/productivity_outliers.parquet'").fetchone()[0]
        print(f"  Productivity outliers detected: {outliers}")

    # Monthly activity breaks (months with unusual patterns)
    if UNIFIED_TIMELINE.exists():
        con.execute(f"""
            COPY (
                WITH monthly_stats AS (
                    SELECT
                        AVG(activity_score) as mean_activity,
                        STDDEV(activity_score) as std_activity
                    FROM '{UNIFIED_TIMELINE}'
                    WHERE activity_score > 0
                ),
                with_lag AS (
                    SELECT
                        month,
                        activity_score,
                        conversations,
                        commits,
                        LAG(activity_score) OVER (ORDER BY month) as prev_activity,
                        LEAD(activity_score) OVER (ORDER BY month) as next_activity
                    FROM '{UNIFIED_TIMELINE}'
                )
                SELECT
                    w.month,
                    w.activity_score,
                    w.conversations,
                    w.commits,
                    w.prev_activity,
                    w.next_activity,
                    CASE
                        WHEN w.activity_score > 2 * COALESCE(w.prev_activity, w.activity_score) THEN 'activity_surge'
                        WHEN w.activity_score < 0.5 * COALESCE(w.prev_activity, w.activity_score) THEN 'activity_drop'
                    END as pattern_break
                FROM with_lag w, monthly_stats s
                WHERE w.activity_score > 2 * COALESCE(w.prev_activity, w.activity_score)
                   OR w.activity_score < 0.5 * COALESCE(w.prev_activity, w.activity_score)
                ORDER BY w.month DESC
            ) TO '{out_dir}/activity_breaks.parquet' (FORMAT PARQUET)
        """)
        breaks = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/activity_breaks.parquet'").fetchone()[0]
        print(f"  Activity breaks detected: {breaks}")

    return spikes if SPEND_DAILY.exists() else 0


def build_trend_prediction():
    """
    Detect rising and declining concepts/interests.
    """
    print("\n=== Building Trend Prediction ===")

    out_dir = INTERP_DIR / "trends" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Rising concepts (terms appearing more frequently in recent months)
    if CURIOSITY_TERMS.exists():
        con.execute(f"""
            COPY (
                WITH recent AS (
                    SELECT term, SUM(frequency) as recent_freq
                    FROM '{CURIOSITY_TERMS}'
                    WHERE month_str >= '2025-01'
                    GROUP BY term
                ),
                older AS (
                    SELECT term, SUM(frequency) as older_freq
                    FROM '{CURIOSITY_TERMS}'
                    WHERE month_str < '2025-01' AND month_str >= '2024-01'
                    GROUP BY term
                )
                SELECT
                    COALESCE(r.term, o.term) as term,
                    COALESCE(r.recent_freq, 0) as recent_frequency,
                    COALESCE(o.older_freq, 0) as older_frequency,
                    COALESCE(r.recent_freq, 0) - COALESCE(o.older_freq, 0) as frequency_delta,
                    CASE
                        WHEN COALESCE(o.older_freq, 0) > 0 THEN
                            (COALESCE(r.recent_freq, 0) - COALESCE(o.older_freq, 0)) * 100.0 / o.older_freq
                        ELSE NULL
                    END as growth_pct,
                    CASE
                        WHEN COALESCE(r.recent_freq, 0) > COALESCE(o.older_freq, 0) * 1.5 THEN 'rising'
                        WHEN COALESCE(r.recent_freq, 0) < COALESCE(o.older_freq, 0) * 0.5 THEN 'declining'
                        ELSE 'stable'
                    END as trend
                FROM recent r
                FULL OUTER JOIN older o ON r.term = o.term
                WHERE COALESCE(r.recent_freq, 0) + COALESCE(o.older_freq, 0) >= 10
                ORDER BY frequency_delta DESC
            ) TO '{out_dir}/concept_trends.parquet' (FORMAT PARQUET)
        """)

        rising = con.execute(f"""
            SELECT COUNT(*) FROM '{out_dir}/concept_trends.parquet' WHERE trend = 'rising'
        """).fetchone()[0]
        declining = con.execute(f"""
            SELECT COUNT(*) FROM '{out_dir}/concept_trends.parquet' WHERE trend = 'declining'
        """).fetchone()[0]
        print(f"  Rising concepts: {rising}")
        print(f"  Declining concepts: {declining}")

        # Top rising terms
        top_rising = con.execute(f"""
            SELECT term, recent_frequency, older_frequency, frequency_delta
            FROM '{out_dir}/concept_trends.parquet'
            WHERE trend = 'rising'
            ORDER BY frequency_delta DESC
            LIMIT 10
        """).fetchall()
        print("  Top rising:")
        for term, recent, older, delta in top_rising[:5]:
            print(f"    {term}: {older} → {recent} (+{delta})")

    # Learning velocity trends
    if LEARNING_ARCS.exists():
        con.execute(f"""
            COPY (
                SELECT
                    month_str,
                    learning_velocity,
                    LAG(learning_velocity, 3) OVER (ORDER BY month_str) as velocity_3m_ago,
                    LAG(learning_velocity, 6) OVER (ORDER BY month_str) as velocity_6m_ago,
                    learning_velocity - LAG(learning_velocity, 3) OVER (ORDER BY month_str) as velocity_delta_3m,
                    CASE
                        WHEN learning_velocity > LAG(learning_velocity, 3) OVER (ORDER BY month_str) * 1.5 THEN 'accelerating'
                        WHEN learning_velocity < LAG(learning_velocity, 3) OVER (ORDER BY month_str) * 0.7 THEN 'decelerating'
                        ELSE 'steady'
                    END as velocity_trend
                FROM '{LEARNING_ARCS}'
                ORDER BY month_str DESC
            ) TO '{out_dir}/learning_velocity_trend.parquet' (FORMAT PARQUET)
        """)
        print("  Learning velocity trends computed")

    return rising if CURIOSITY_TERMS.exists() else 0


def build_recommendations():
    """
    Generate recommendations: dormant topics, cross-domain connections.
    """
    print("\n=== Building Recommendations ===")

    out_dir = INTERP_DIR / "recommendations" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Dormant but valuable topics (high past activity, low recent activity)
    if CURIOSITY_TERMS.exists():
        con.execute(f"""
            COPY (
                WITH recent AS (
                    SELECT term, SUM(frequency) as recent_freq
                    FROM '{CURIOSITY_TERMS}'
                    WHERE month_str >= '2025-07'
                    GROUP BY term
                ),
                peak AS (
                    SELECT term, MAX(frequency) as peak_freq
                    FROM '{CURIOSITY_TERMS}'
                    GROUP BY term
                ),
                total AS (
                    SELECT term, SUM(frequency) as total_freq
                    FROM '{CURIOSITY_TERMS}'
                    GROUP BY term
                )
                SELECT
                    t.term,
                    t.total_freq as total_mentions,
                    p.peak_freq as peak_mentions,
                    COALESCE(r.recent_freq, 0) as recent_mentions,
                    'Consider revisiting: historically important topic with declining attention' as recommendation
                FROM total t
                JOIN peak p ON t.term = p.term
                LEFT JOIN recent r ON t.term = r.term
                WHERE t.total_freq >= 20  -- Was significant
                  AND p.peak_freq >= 5    -- Had peak activity
                  AND COALESCE(r.recent_freq, 0) <= 2  -- Now dormant
                ORDER BY t.total_freq DESC
            ) TO '{out_dir}/dormant_topics.parquet' (FORMAT PARQUET)
        """)
        dormant = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/dormant_topics.parquet'").fetchone()[0]
        print(f"  Dormant valuable topics: {dormant}")

    # Stalled projects worth revisiting
    if PROJECT_SUCCESS.exists():
        con.execute(f"""
            COPY (
                SELECT
                    project,
                    weeks_active,
                    total_commits,
                    conversations,
                    investment_score,
                    outcome,
                    'High investment, incomplete outcome - consider finishing' as recommendation
                FROM '{PROJECT_SUCCESS}'
                WHERE investment_score >= 50
                  AND success_score < 70
                ORDER BY investment_score DESC
            ) TO '{out_dir}/stalled_projects.parquet' (FORMAT PARQUET)
        """)
        stalled = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/stalled_projects.parquet'").fetchone()[0]
        print(f"  Stalled projects worth revisiting: {stalled}")

    # High productivity patterns (what leads to productive days)
    if PRODUCTIVITY.exists():
        con.execute(f"""
            COPY (
                WITH high_prod AS (
                    SELECT *
                    FROM '{PRODUCTIVITY}'
                    WHERE productivity_score >= (
                        SELECT PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY productivity_score)
                        FROM '{PRODUCTIVITY}'
                    )
                )
                SELECT
                    CASE
                        WHEN accomplishments > 15 THEN 'high_accomplishments'
                        WHEN commits > 5 THEN 'high_commits'
                        WHEN api_calls > 100 THEN 'high_api_usage'
                        WHEN searches > 20 THEN 'high_research'
                        ELSE 'balanced'
                    END as productivity_driver,
                    COUNT(*) as days,
                    AVG(productivity_score) as avg_score,
                    AVG(accomplishments) as avg_accomplishments,
                    AVG(commits) as avg_commits
                FROM high_prod
                GROUP BY productivity_driver
                ORDER BY days DESC
            ) TO '{out_dir}/productivity_patterns.parquet' (FORMAT PARQUET)
        """)
        patterns = con.execute(f"SELECT * FROM '{out_dir}/productivity_patterns.parquet'").fetchall()
        print("  Productivity patterns:")
        for driver, days, score, accom, commits in patterns:
            print(f"    {driver}: {days} days, avg score {score:.0f}")

    return dormant if CURIOSITY_TERMS.exists() else 0


def build_weekly_synthesis():
    """
    Build weekly synthesis reports.
    """
    print("\n=== Building Weekly Synthesis ===")

    out_dir = INTERP_DIR / "weekly_synthesis" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Aggregate weekly metrics across all sources
    if PRODUCTIVITY.exists() and UNIFIED_TIMELINE.exists():
        con.execute(f"""
            COPY (
                WITH weekly_prod AS (
                    SELECT
                        DATE_TRUNC('week', date) as week_start,
                        SUM(accomplishments) as accomplishments,
                        SUM(commits) as commits,
                        SUM(api_calls) as api_calls,
                        SUM(searches) as searches,
                        AVG(productivity_score) as avg_productivity,
                        MAX(productivity_score) as peak_productivity,
                        COUNT(*) as days_tracked
                    FROM '{PRODUCTIVITY}'
                    GROUP BY week_start
                )
                SELECT
                    week_start,
                    accomplishments,
                    commits,
                    api_calls,
                    searches,
                    ROUND(avg_productivity, 1) as avg_productivity,
                    ROUND(peak_productivity, 1) as peak_productivity,
                    days_tracked,
                    -- Week characterization
                    CASE
                        WHEN commits > 20 THEN 'coding_sprint'
                        WHEN accomplishments > 100 THEN 'high_output'
                        WHEN searches > 50 THEN 'research_heavy'
                        WHEN api_calls > 500 THEN 'ai_intensive'
                        ELSE 'balanced'
                    END as week_type
                FROM weekly_prod
                WHERE week_start IS NOT NULL
                ORDER BY week_start DESC
            ) TO '{out_dir}/weekly_summary.parquet' (FORMAT PARQUET)
        """)

        weeks = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/weekly_summary.parquet'").fetchone()[0]
        print(f"  Weekly summaries generated: {weeks}")

        # Recent weeks
        recent = con.execute(f"""
            SELECT week_start, week_type, avg_productivity, accomplishments, commits
            FROM '{out_dir}/weekly_summary.parquet'
            ORDER BY week_start DESC
            LIMIT 5
        """).fetchall()
        print("  Recent weeks:")
        for week, wtype, prod, accom, commits in recent:
            week_str = str(week)[:10] if week else "N/A"
            print(f"    {week_str}: {wtype}, prod={prod:.0f}, {accom} accomplishments, {commits} commits")

    return weeks if PRODUCTIVITY.exists() else 0


def build_all():
    """Build all discovery pipelines."""
    print(f"Building discovery pipelines")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Build all layers
    build_anomaly_detection()
    build_trend_prediction()
    build_recommendations()
    build_weekly_synthesis()

    print("\n✅ All discovery pipelines built!")


if __name__ == "__main__":
    build_all()
