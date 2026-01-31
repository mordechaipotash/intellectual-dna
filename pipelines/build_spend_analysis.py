#!/usr/bin/env python3
"""
Build spend analysis interpretations from 732K OpenRouter API calls.
Phase 2 of 55x Mining Plan.

Creates:
- interpretations/spend_model_efficiency/v1/ - ROI per model
- interpretations/spend_provider_analysis/v1/ - Provider performance
- interpretations/spend_cache_analysis/v1/ - Cache patterns
- interpretations/spend_temporal/v1/ - Usage patterns over time
"""

import duckdb
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
RAW_CSV = BASE_DIR / "data" / "spend" / "openrouter_raw.csv"
INTERP_DIR = BASE_DIR / "data" / "interpretations"


def ensure_dir(path: Path):
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def build_model_efficiency():
    """
    Build model efficiency analysis - cost per million tokens by model.
    Shows which models give best value.
    """
    print("\n=== Building Model Efficiency Analysis ===")

    out_dir = INTERP_DIR / "spend_model_efficiency" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Aggregate by model - calculate efficiency metrics
    con.execute(f"""
        COPY (
            SELECT
                model,
                COUNT(*) as call_count,
                SUM(tokens_input) as total_input,
                SUM(tokens_output) as total_output,
                SUM(tokens_cached) as total_cached,
                SUM(tokens_input + tokens_output) as total_tokens,
                SUM(cost_usd) as total_cost_usd,
                -- Cost per million tokens (key efficiency metric)
                CASE
                    WHEN SUM(tokens_input + tokens_output) > 0
                    THEN (SUM(cost_usd) / SUM(tokens_input + tokens_output)) * 1000000
                    ELSE 0
                END as cost_per_million_tokens,
                -- Cache efficiency (what % of input tokens were cached)
                CASE
                    WHEN SUM(tokens_input) > 0
                    THEN (SUM(tokens_cached) * 100.0 / SUM(tokens_input))
                    ELSE 0
                END as cache_hit_pct,
                -- Output ratio (how much output per input)
                CASE
                    WHEN SUM(tokens_input) > 0
                    THEN (SUM(tokens_output) * 1.0 / SUM(tokens_input))
                    ELSE 0
                END as output_ratio,
                MIN(date) as first_used,
                MAX(date) as last_used,
                COUNT(DISTINCT date) as days_used
            FROM read_csv('{RAW_CSV}', auto_detect=true)
            GROUP BY model
            HAVING SUM(tokens_input + tokens_output) > 1000  -- Filter noise
            ORDER BY total_cost_usd DESC
        ) TO '{out_dir}/models.parquet' (FORMAT PARQUET)
    """)

    count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/models.parquet'").fetchone()[0]
    print(f"  Written: {count} models analyzed")

    # Summary stats
    stats = con.execute(f"""
        SELECT
            SUM(total_cost_usd) as total_spent,
            SUM(total_tokens) as total_tokens,
            COUNT(*) as model_count,
            MIN(cost_per_million_tokens) as min_cost_per_m,
            MAX(cost_per_million_tokens) as max_cost_per_m
        FROM '{out_dir}/models.parquet'
    """).fetchone()

    print(f"  Total spent: ${stats[0]:.2f}")
    print(f"  Total tokens: {stats[1]:,}")
    print(f"  Cost range: ${stats[3]:.4f} - ${stats[4]:.4f} per million tokens")

    return count


def build_provider_analysis():
    """
    Build provider analysis - performance by provider (Together, DeepInfra, Novita, etc).
    """
    print("\n=== Building Provider Analysis ===")

    out_dir = INTERP_DIR / "spend_provider_analysis" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Aggregate by provider
    con.execute(f"""
        COPY (
            SELECT
                provider,
                COUNT(*) as call_count,
                SUM(tokens_input + tokens_output) as total_tokens,
                SUM(cost_usd) as total_cost_usd,
                CASE
                    WHEN SUM(tokens_input + tokens_output) > 0
                    THEN (SUM(cost_usd) / SUM(tokens_input + tokens_output)) * 1000000
                    ELSE 0
                END as cost_per_million_tokens,
                COUNT(DISTINCT model) as models_used,
                COUNT(DISTINCT date) as days_active,
                MIN(date) as first_used,
                MAX(date) as last_used
            FROM read_csv('{RAW_CSV}', auto_detect=true)
            WHERE provider IS NOT NULL AND provider != ''
            GROUP BY provider
            ORDER BY total_cost_usd DESC
        ) TO '{out_dir}/providers.parquet' (FORMAT PARQUET)
    """)

    count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/providers.parquet'").fetchone()[0]
    print(f"  Written: {count} providers analyzed")

    # Monthly provider usage
    con.execute(f"""
        COPY (
            SELECT
                strftime(date, '%Y-%m') as month,
                provider,
                COUNT(*) as call_count,
                SUM(cost_usd) as cost_usd,
                SUM(tokens_input + tokens_output) as tokens
            FROM read_csv('{RAW_CSV}', auto_detect=true)
            WHERE provider IS NOT NULL AND provider != ''
            GROUP BY month, provider
            ORDER BY month, cost_usd DESC
        ) TO '{out_dir}/monthly.parquet' (FORMAT PARQUET)
    """)

    monthly_count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/monthly.parquet'").fetchone()[0]
    print(f"  Written: {monthly_count} monthly provider records")

    return count


def build_cache_analysis():
    """
    Build cache hit analysis - understand caching patterns.
    """
    print("\n=== Building Cache Analysis ===")

    out_dir = INTERP_DIR / "spend_cache_analysis" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Daily cache metrics
    con.execute(f"""
        COPY (
            SELECT
                date,
                COUNT(*) as call_count,
                SUM(tokens_input) as input_tokens,
                SUM(tokens_cached) as cached_tokens,
                SUM(cost_usd) as cost_usd,
                CASE
                    WHEN SUM(tokens_input) > 0
                    THEN (SUM(tokens_cached) * 100.0 / SUM(tokens_input))
                    ELSE 0
                END as cache_hit_pct,
                -- Estimated savings (assume cached tokens cost 0.1x normal)
                SUM(tokens_cached) * 0.9 as estimated_tokens_saved
            FROM read_csv('{RAW_CSV}', auto_detect=true)
            GROUP BY date
            ORDER BY date
        ) TO '{out_dir}/daily.parquet' (FORMAT PARQUET)
    """)

    # Model-level cache patterns
    con.execute(f"""
        COPY (
            SELECT
                model,
                SUM(tokens_input) as input_tokens,
                SUM(tokens_cached) as cached_tokens,
                CASE
                    WHEN SUM(tokens_input) > 0
                    THEN (SUM(tokens_cached) * 100.0 / SUM(tokens_input))
                    ELSE 0
                END as cache_hit_pct,
                COUNT(*) as call_count
            FROM read_csv('{RAW_CSV}', auto_detect=true)
            GROUP BY model
            HAVING SUM(tokens_input) > 10000
            ORDER BY cache_hit_pct DESC
        ) TO '{out_dir}/by_model.parquet' (FORMAT PARQUET)
    """)

    daily_count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/daily.parquet'").fetchone()[0]
    model_count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/by_model.parquet'").fetchone()[0]

    # Summary
    stats = con.execute(f"""
        SELECT
            SUM(input_tokens) as total_input,
            SUM(cached_tokens) as total_cached,
            (SUM(cached_tokens) * 100.0 / SUM(input_tokens)) as overall_cache_pct
        FROM '{out_dir}/daily.parquet'
    """).fetchone()

    print(f"  Daily records: {daily_count}")
    print(f"  Models analyzed: {model_count}")
    print(f"  Overall cache hit rate: {stats[2]:.1f}%")
    print(f"  Total cached tokens: {stats[1]:,.0f}")

    return daily_count


def build_temporal_patterns():
    """
    Build temporal usage patterns - when you use AI, hourly/daily/weekly patterns.
    """
    print("\n=== Building Temporal Patterns ===")

    out_dir = INTERP_DIR / "spend_temporal" / "v1"
    ensure_dir(out_dir)

    con = duckdb.connect()

    # Daily patterns
    con.execute(f"""
        COPY (
            SELECT
                date,
                COUNT(*) as call_count,
                SUM(tokens_input + tokens_output) as tokens,
                SUM(cost_usd) as cost_usd,
                COUNT(DISTINCT model) as models_used
            FROM read_csv('{RAW_CSV}', auto_detect=true)
            GROUP BY date
            ORDER BY date
        ) TO '{out_dir}/daily.parquet' (FORMAT PARQUET)
    """)

    # Monthly patterns
    con.execute(f"""
        COPY (
            SELECT
                strftime(date, '%Y-%m') as month,
                COUNT(*) as call_count,
                SUM(tokens_input + tokens_output) as tokens,
                SUM(cost_usd) as cost_usd,
                COUNT(DISTINCT model) as models_used,
                COUNT(DISTINCT provider) as providers_used,
                COUNT(DISTINCT date) as active_days
            FROM read_csv('{RAW_CSV}', auto_detect=true)
            GROUP BY month
            ORDER BY month
        ) TO '{out_dir}/monthly.parquet' (FORMAT PARQUET)
    """)

    # Day of week patterns
    con.execute(f"""
        COPY (
            SELECT
                dayofweek(date) as dow,
                CASE dayofweek(date)
                    WHEN 0 THEN 'Sunday'
                    WHEN 1 THEN 'Monday'
                    WHEN 2 THEN 'Tuesday'
                    WHEN 3 THEN 'Wednesday'
                    WHEN 4 THEN 'Thursday'
                    WHEN 5 THEN 'Friday'
                    WHEN 6 THEN 'Saturday'
                END as day_name,
                COUNT(*) as call_count,
                SUM(cost_usd) as cost_usd,
                AVG(tokens_input + tokens_output) as avg_tokens_per_call
            FROM read_csv('{RAW_CSV}', auto_detect=true)
            GROUP BY dow
            ORDER BY dow
        ) TO '{out_dir}/day_of_week.parquet' (FORMAT PARQUET)
    """)

    daily_count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/daily.parquet'").fetchone()[0]
    monthly_count = con.execute(f"SELECT COUNT(*) FROM '{out_dir}/monthly.parquet'").fetchone()[0]

    print(f"  Daily records: {daily_count}")
    print(f"  Monthly records: {monthly_count}")
    print(f"  Day of week patterns: 7")

    return daily_count


def build_all():
    """Build all spend analysis layers."""
    print(f"Building spend analysis from {RAW_CSV}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    if not RAW_CSV.exists():
        print(f"ERROR: Raw CSV not found at {RAW_CSV}")
        return

    # Build all layers
    build_model_efficiency()
    build_provider_analysis()
    build_cache_analysis()
    build_temporal_patterns()

    print("\n✅ All spend analysis layers built!")


if __name__ == "__main__":
    build_all()
