#!/usr/bin/env python3
"""
Build cross-dimensional synthesis - connecting ALL data sources.
Phase 7 of 55x Mining Plan.

Creates:
- interpretations/productivity_matrix/v1/ - Daily productivity scores
- interpretations/learning_arcs/v1/ - Topic learning velocity tracking
- interpretations/project_success/v1/ - Project outcome prediction
- interpretations/unified_timeline/v1/ - Cross-source unified timeline
"""

import duckdb
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
DATA_DIR = BASE_DIR / "data"
INTERP_DIR = DATA_DIR / "interpretations"

# Source paths
ACCOMPLISHMENTS = INTERP_DIR / "daily_accomplishments" / "v1" / "daily.parquet"
MOOD = INTERP_DIR / "mood" / "v1" / "daily.parquet"
SPEND_DAILY = INTERP_DIR / "spend_temporal" / "v1" / "daily.parquet"
CODE_VELOCITY = INTERP_DIR / "code_velocity" / "v1" / "daily.parquet"
RESEARCH_VELOCITY = INTERP_DIR / "research_velocity" / "v1" / "daily_search_intensity.parquet"
CONVERSATION_STATS = INTERP_DIR / "conversation_stats" / "v1" / "conversations.parquet"
PROJECT_ARCS = INTERP_DIR / "project_arcs" / "v1" / "projects.parquet"
COMMITS = DATA_DIR / "github_commits.parquet"
YOUTUBE = DATA_DIR / "youtube_rows.parquet"
CONVERSATIONS = DATA_DIR / "all_conversations.parquet"
GOOGLE_SEARCHES = DATA_DIR / "google_searches.parquet"


def ensure_dir(path: Path):
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def build_productivity_matrix():
    """
    Build daily productivity matrix combining multiple dimensions.

    Formula: Accomplishments × Mood × Commits × Spend × Research = Productivity Score
    """
    print("\n=== Building Productivity Matrix ===")

    out_dir = INTERP_DIR / "productivity_matrix" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Create unified daily view joining all sources
    con.execute(f"""
        COPY (
            WITH dates AS (
                -- Generate all unique dates from all sources
                SELECT DISTINCT DATE_TRUNC('day', date) as date FROM '{ACCOMPLISHMENTS}'
                UNION
                SELECT DISTINCT DATE_TRUNC('day', date) as date FROM '{MOOD}'
                UNION
                SELECT DISTINCT date FROM '{CODE_VELOCITY}'
                UNION
                SELECT DISTINCT date FROM '{SPEND_DAILY}'
                UNION
                SELECT DISTINCT date FROM '{RESEARCH_VELOCITY}'
            ),
            accomplishment_data AS (
                SELECT
                    DATE_TRUNC('day', date) as date,
                    accomplishment_count,
                    message_count as accomplishment_msgs
                FROM '{ACCOMPLISHMENTS}'
            ),
            mood_data AS (
                SELECT
                    DATE_TRUNC('day', date) as date,
                    mood,
                    energy,
                    cognitive_state,
                    stress
                FROM '{MOOD}'
            ),
            code_data AS (
                SELECT
                    date,
                    commits,
                    repos_touched
                FROM '{CODE_VELOCITY}'
            ),
            spend_data AS (
                SELECT
                    date,
                    call_count as api_calls,
                    tokens,
                    cost_usd
                FROM '{SPEND_DAILY}'
            ),
            research_data AS (
                SELECT
                    date,
                    search_count,
                    unique_queries,
                    questions as research_questions
                FROM '{RESEARCH_VELOCITY}'
            )
            SELECT
                d.date,
                -- Accomplishments dimension
                COALESCE(a.accomplishment_count, 0) as accomplishments,
                COALESCE(a.accomplishment_msgs, 0) as accomplishment_msgs,
                -- Mood dimension
                m.mood,
                m.energy,
                m.cognitive_state,
                m.stress,
                -- Code dimension
                COALESCE(c.commits, 0) as commits,
                COALESCE(c.repos_touched, 0) as repos_touched,
                -- Spend dimension
                COALESCE(s.api_calls, 0) as api_calls,
                COALESCE(s.tokens, 0) as tokens,
                COALESCE(s.cost_usd, 0) as cost_usd,
                -- Research dimension
                COALESCE(r.search_count, 0) as searches,
                COALESCE(r.unique_queries, 0) as unique_queries,
                COALESCE(r.research_questions, 0) as research_questions,
                -- Productivity score (weighted composite)
                (
                    COALESCE(a.accomplishment_count, 0) * 10 +  -- Accomplishments weight high
                    COALESCE(c.commits, 0) * 5 +                 -- Commits weight medium
                    COALESCE(s.api_calls, 0) * 0.01 +            -- API usage minor
                    COALESCE(r.search_count, 0) * 0.5            -- Research minor
                ) as productivity_score,
                -- Dimension counts (for coverage tracking)
                CASE WHEN a.date IS NOT NULL THEN 1 ELSE 0 END +
                CASE WHEN m.date IS NOT NULL THEN 1 ELSE 0 END +
                CASE WHEN c.date IS NOT NULL THEN 1 ELSE 0 END +
                CASE WHEN s.date IS NOT NULL THEN 1 ELSE 0 END +
                CASE WHEN r.date IS NOT NULL THEN 1 ELSE 0 END as dimensions_present
            FROM dates d
            LEFT JOIN accomplishment_data a ON d.date = a.date
            LEFT JOIN mood_data m ON d.date = m.date
            LEFT JOIN code_data c ON d.date = c.date
            LEFT JOIN spend_data s ON d.date = s.date
            LEFT JOIN research_data r ON d.date = r.date
            WHERE d.date IS NOT NULL
            ORDER BY d.date DESC
        ) TO '{out_dir}/daily.parquet' (FORMAT PARQUET)
    """)

    count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/daily.parquet'").fetchone()[0]
    print(f"  Daily productivity records: {count}")

    # Build weekly aggregates
    con.execute(f"""
        COPY (
            SELECT
                DATE_TRUNC('week', date) as week_start,
                SUM(accomplishments) as accomplishments,
                SUM(commits) as commits,
                SUM(api_calls) as api_calls,
                SUM(tokens) as tokens,
                SUM(cost_usd) as cost_usd,
                SUM(searches) as searches,
                AVG(productivity_score) as avg_productivity,
                MAX(productivity_score) as peak_productivity,
                COUNT(CASE WHEN dimensions_present >= 3 THEN 1 END) as rich_data_days
            FROM '{out_dir}/daily.parquet'
            GROUP BY week_start
            ORDER BY week_start DESC
        ) TO '{out_dir}/weekly.parquet' (FORMAT PARQUET)
    """)

    weekly = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/weekly.parquet'").fetchone()[0]
    print(f"  Weekly aggregates: {weekly}")

    # Build monthly summary
    con.execute(f"""
        COPY (
            SELECT
                DATE_TRUNC('month', date) as month_start,
                SUM(accomplishments) as total_accomplishments,
                SUM(commits) as total_commits,
                SUM(api_calls) as total_api_calls,
                ROUND(SUM(cost_usd), 2) as total_spend,
                SUM(searches) as total_searches,
                AVG(productivity_score) as avg_productivity,
                MAX(productivity_score) as peak_productivity,
                COUNT(*) as days_tracked
            FROM '{out_dir}/daily.parquet'
            GROUP BY month_start
            ORDER BY month_start DESC
        ) TO '{out_dir}/monthly.parquet' (FORMAT PARQUET)
    """)

    monthly = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/monthly.parquet'").fetchone()[0]
    print(f"  Monthly summaries: {monthly}")

    # High productivity days (top 10%)
    con.execute(f"""
        COPY (
            SELECT *
            FROM '{out_dir}/daily.parquet'
            WHERE productivity_score >= (
                SELECT PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY productivity_score)
                FROM '{out_dir}/daily.parquet'
            )
            ORDER BY productivity_score DESC
        ) TO '{out_dir}/peak_days.parquet' (FORMAT PARQUET)
    """)

    peak_days = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/peak_days.parquet'").fetchone()[0]
    print(f"  Peak productivity days (top 10%): {peak_days}")

    return count


def build_learning_arcs():
    """
    Track learning velocity: Search → Watch → Discuss → Build
    """
    print("\n=== Building Learning Arcs ===")

    out_dir = INTERP_DIR / "learning_arcs" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Monthly cross-source activity
    con.execute(f"""
        COPY (
            WITH search_activity AS (
                SELECT
                    year || '-' || LPAD(CAST(month AS VARCHAR), 2, '0') as month_str,
                    COUNT(*) as google_searches,
                    COUNT(DISTINCT query) as unique_searches
                FROM '{GOOGLE_SEARCHES}'
                GROUP BY year, month
            ),
            youtube_activity AS (
                SELECT
                    strftime(watched_date, '%Y-%m') as month_str,
                    COUNT(*) as videos_watched
                FROM '{YOUTUBE}'
                WHERE watched_date IS NOT NULL
                GROUP BY month_str
            ),
            conversation_activity AS (
                SELECT
                    year || '-' || LPAD(CAST(month AS VARCHAR), 2, '0') as month_str,
                    COUNT(DISTINCT conversation_id) as conversations,
                    COUNT(*) as messages,
                    SUM(word_count) as words
                FROM '{CONVERSATIONS}'
                GROUP BY year, month
            ),
            code_activity AS (
                SELECT
                    strftime(timestamp, '%Y-%m') as month_str,
                    COUNT(*) as commits,
                    COUNT(DISTINCT repo_name) as repos
                FROM '{COMMITS}'
                GROUP BY month_str
            )
            SELECT
                s.month_str,
                COALESCE(s.google_searches, 0) as google_searches,
                COALESCE(s.unique_searches, 0) as unique_searches,
                COALESCE(y.videos_watched, 0) as videos_watched,
                COALESCE(c.conversations, 0) as conversations,
                COALESCE(c.messages, 0) as messages,
                COALESCE(c.words, 0) as words,
                COALESCE(code.commits, 0) as commits,
                COALESCE(code.repos, 0) as repos,
                -- Learning velocity score (all sources combined)
                (
                    COALESCE(s.google_searches, 0) * 0.5 +   -- Curiosity
                    COALESCE(y.videos_watched, 0) * 1.0 +     -- Consumption
                    COALESCE(c.conversations, 0) * 2.0 +      -- Processing
                    COALESCE(code.commits, 0) * 3.0           -- Application
                ) as learning_velocity
            FROM search_activity s
            FULL OUTER JOIN youtube_activity y ON s.month_str = y.month_str
            FULL OUTER JOIN conversation_activity c ON s.month_str = c.month_str
            FULL OUTER JOIN code_activity code ON s.month_str = code.month_str
            WHERE s.month_str IS NOT NULL
               OR y.month_str IS NOT NULL
               OR c.month_str IS NOT NULL
               OR code.month_str IS NOT NULL
            ORDER BY COALESCE(s.month_str, y.month_str, c.month_str, code.month_str) DESC
        ) TO '{out_dir}/monthly_learning.parquet' (FORMAT PARQUET)
    """)

    count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/monthly_learning.parquet'").fetchone()[0]
    print(f"  Monthly learning records: {count}")

    # Topic-specific learning arcs (common programming topics)
    topics = ['python', 'react', 'typescript', 'api', 'database', 'ai', 'machine learning']

    topic_data = []
    for topic in topics:
        try:
            # Count occurrences across sources
            searches = con.execute(f"""
                SELECT COUNT(*) FROM '{GOOGLE_SEARCHES}'
                WHERE LOWER(query) LIKE '%{topic}%'
            """).fetchone()[0]

            videos = con.execute(f"""
                SELECT COUNT(*) FROM '{YOUTUBE}'
                WHERE LOWER(title) LIKE '%{topic}%'
                AND watched_date IS NOT NULL
            """).fetchone()[0]

            convos = con.execute(f"""
                SELECT COUNT(*) FROM '{CONVERSATIONS}'
                WHERE LOWER(content) LIKE '%{topic}%'
            """).fetchone()[0]

            commits = con.execute(f"""
                SELECT COUNT(*) FROM '{COMMITS}'
                WHERE LOWER(message) LIKE '%{topic}%'
                OR LOWER(repo_name) LIKE '%{topic}%'
            """).fetchone()[0]

            topic_data.append((topic, searches, videos, convos, commits))
        except:
            pass

    if topic_data:
        con.execute(f"""
            COPY (
                SELECT * FROM (VALUES {', '.join([f"('{t}', {s}, {v}, {c}, {cm})" for t, s, v, c, cm in topic_data])})
                AS t(topic, google_searches, videos_watched, conversations, commits)
                ORDER BY (google_searches + videos_watched + conversations + commits) DESC
            ) TO '{out_dir}/topic_arcs.parquet' (FORMAT PARQUET)
        """)
        print(f"  Topic arcs tracked: {len(topic_data)}")

    return count


def build_project_success():
    """
    Project success prediction combining project arcs with activity metrics.
    """
    print("\n=== Building Project Success Analysis ===")

    out_dir = INTERP_DIR / "project_success" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Enrich project arcs with conversation and commit data
    con.execute(f"""
        COPY (
            WITH project_commits AS (
                SELECT
                    repo_name as project,
                    COUNT(*) as total_commits,
                    MIN(timestamp) as first_commit,
                    MAX(timestamp) as last_commit,
                    COUNT(DISTINCT DATE_TRUNC('day', timestamp)) as active_days
                FROM '{COMMITS}'
                GROUP BY repo_name
            ),
            project_conversations AS (
                SELECT
                    project,
                    COUNT(DISTINCT conversation_id) as conversations,
                    COUNT(*) as messages,
                    SUM(word_count) as words
                FROM (
                    SELECT
                        LOWER(REGEXP_EXTRACT(content, '([a-zA-Z][a-zA-Z0-9_-]{{2,30}})', 1)) as project,
                        conversation_id,
                        word_count
                    FROM '{CONVERSATIONS}'
                    WHERE role = 'user'
                ) sub
                WHERE project IS NOT NULL
                GROUP BY project
                HAVING conversations >= 3
            )
            SELECT
                p.name as project,
                p.first_seen,
                p.weeks_active,
                p.stage_progression,
                p.final_stage,
                p.outcome,
                COALESCE(c.total_commits, 0) as total_commits,
                COALESCE(c.active_days, 0) as commit_days,
                COALESCE(conv.conversations, 0) as conversations,
                COALESCE(conv.messages, 0) as discussion_messages,
                -- Success score based on shipping indicators
                CASE
                    WHEN p.outcome LIKE '%shipped%' OR p.outcome LIKE '%complete%' OR p.outcome LIKE '%live%' THEN 100
                    WHEN p.outcome LIKE '%active%' OR p.outcome LIKE '%ongoing%' THEN 70
                    WHEN p.outcome LIKE '%stalled%' OR p.outcome LIKE '%paused%' THEN 30
                    WHEN p.outcome LIKE '%abandoned%' THEN 0
                    ELSE 50  -- Unknown
                END as success_score,
                -- Investment score
                (
                    p.weeks_active * 10 +
                    COALESCE(c.total_commits, 0) * 2 +
                    COALESCE(conv.conversations, 0) * 5
                ) as investment_score
            FROM '{PROJECT_ARCS}' p
            LEFT JOIN project_commits c ON LOWER(p.name) = LOWER(c.project)
            LEFT JOIN project_conversations conv ON LOWER(p.name) = LOWER(conv.project)
            ORDER BY investment_score DESC
        ) TO '{out_dir}/projects_enriched.parquet' (FORMAT PARQUET)
    """)

    count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/projects_enriched.parquet'").fetchone()[0]
    print(f"  Enriched projects: {count}")

    # Identify shipped vs abandoned patterns
    con.execute(f"""
        COPY (
            SELECT
                CASE
                    WHEN success_score >= 70 THEN 'shipped'
                    WHEN success_score >= 30 THEN 'in_progress'
                    ELSE 'stalled'
                END as category,
                COUNT(*) as project_count,
                AVG(weeks_active) as avg_weeks,
                AVG(total_commits) as avg_commits,
                AVG(conversations) as avg_conversations,
                AVG(investment_score) as avg_investment
            FROM '{out_dir}/projects_enriched.parquet'
            GROUP BY category
            ORDER BY project_count DESC
        ) TO '{out_dir}/outcome_patterns.parquet' (FORMAT PARQUET)
    """)

    patterns = con.execute(f"SELECT * FROM '{out_dir}/outcome_patterns.parquet'").fetchall()
    print("  Outcome patterns:")
    for cat, cnt, weeks, commits, convos, invest in patterns:
        print(f"    {cat}: {cnt} projects, {weeks:.1f} weeks avg, {commits:.0f} commits avg")

    return count


def build_unified_timeline():
    """
    Create unified timeline across all sources.
    """
    print("\n=== Building Unified Timeline ===")

    out_dir = INTERP_DIR / "unified_timeline" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Build monthly cross-source summary
    con.execute(f"""
        COPY (
            WITH monthly_convos AS (
                SELECT
                    year || '-' || LPAD(CAST(month AS VARCHAR), 2, '0') as month_str,
                    COUNT(DISTINCT conversation_id) as conversations,
                    COUNT(*) as messages,
                    SUM(word_count) as conv_words
                FROM '{CONVERSATIONS}'
                GROUP BY year, month
            ),
            monthly_commits AS (
                SELECT
                    strftime(timestamp, '%Y-%m') as month_str,
                    COUNT(*) as commits,
                    COUNT(DISTINCT repo_name) as repos
                FROM '{COMMITS}'
                GROUP BY month_str
            ),
            monthly_searches AS (
                SELECT
                    year || '-' || LPAD(CAST(month AS VARCHAR), 2, '0') as month_str,
                    COUNT(*) as google_searches
                FROM '{GOOGLE_SEARCHES}'
                GROUP BY year, month
            ),
            monthly_youtube AS (
                SELECT
                    strftime(watched_date, '%Y-%m') as month_str,
                    COUNT(*) as videos_watched
                FROM '{YOUTUBE}'
                WHERE watched_date IS NOT NULL
                GROUP BY month_str
            ),
            monthly_spend AS (
                SELECT
                    strftime(date, '%Y-%m') as month_str,
                    SUM(cost_usd) as spend,
                    SUM(tokens) as tokens
                FROM '{SPEND_DAILY}'
                GROUP BY month_str
            )
            SELECT
                COALESCE(c.month_str, cm.month_str, s.month_str, y.month_str, sp.month_str) as month,
                COALESCE(c.conversations, 0) as conversations,
                COALESCE(c.messages, 0) as messages,
                COALESCE(c.conv_words, 0) as words,
                COALESCE(cm.commits, 0) as commits,
                COALESCE(cm.repos, 0) as active_repos,
                COALESCE(s.google_searches, 0) as google_searches,
                COALESCE(y.videos_watched, 0) as videos_watched,
                ROUND(COALESCE(sp.spend, 0), 2) as api_spend,
                COALESCE(sp.tokens, 0) as api_tokens,
                -- Activity score
                (
                    COALESCE(c.messages, 0) * 0.1 +
                    COALESCE(cm.commits, 0) * 5 +
                    COALESCE(s.google_searches, 0) * 0.5 +
                    COALESCE(y.videos_watched, 0) * 1 +
                    COALESCE(sp.spend, 0) * 10
                ) as activity_score
            FROM monthly_convos c
            FULL OUTER JOIN monthly_commits cm ON c.month_str = cm.month_str
            FULL OUTER JOIN monthly_searches s ON COALESCE(c.month_str, cm.month_str) = s.month_str
            FULL OUTER JOIN monthly_youtube y ON COALESCE(c.month_str, cm.month_str, s.month_str) = y.month_str
            FULL OUTER JOIN monthly_spend sp ON COALESCE(c.month_str, cm.month_str, s.month_str, y.month_str) = sp.month_str
            WHERE COALESCE(c.month_str, cm.month_str, s.month_str, y.month_str, sp.month_str) IS NOT NULL
            ORDER BY month DESC
        ) TO '{out_dir}/monthly.parquet' (FORMAT PARQUET)
    """)

    count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/monthly.parquet'").fetchone()[0]
    print(f"  Monthly timeline records: {count}")

    # Year-over-year summary
    con.execute(f"""
        COPY (
            SELECT
                LEFT(month, 4) as year,
                SUM(conversations) as total_conversations,
                SUM(messages) as total_messages,
                SUM(commits) as total_commits,
                SUM(google_searches) as total_searches,
                SUM(videos_watched) as total_videos,
                ROUND(SUM(api_spend), 2) as total_spend,
                AVG(activity_score) as avg_activity
            FROM '{out_dir}/monthly.parquet'
            GROUP BY year
            ORDER BY year DESC
        ) TO '{out_dir}/yearly.parquet' (FORMAT PARQUET)
    """)

    yearly = con.execute(f"SELECT * FROM '{out_dir}/yearly.parquet'").fetchall()
    print("  Yearly summary:")
    for row in yearly:
        year, convos, msgs, commits, searches, videos, spend, activity = row
        print(f"    {year}: {convos:,} convos, {commits} commits, ${spend:.0f} spend")

    return count


def build_all():
    """Build all cross-dimensional synthesis layers."""
    print(f"Building cross-dimensional synthesis")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Build all layers
    build_productivity_matrix()
    build_learning_arcs()
    build_project_success()
    build_unified_timeline()

    print("\n✅ All cross-dimensional synthesis layers built!")


if __name__ == "__main__":
    build_all()
