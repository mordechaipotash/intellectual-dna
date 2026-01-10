#!/usr/bin/env python3
"""
Build facts/spend/ three-tier parquet structure:
- raw.parquet: Individual records (immutable source of truth)
- daily.parquet: Aggregated by date + source
- monthly.parquet: Aggregated by month + source
"""

import duckdb
from pathlib import Path

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
SPEND_DIR = BASE_DIR / "data" / "spend"
FACTS_SPEND = BASE_DIR / "data" / "facts" / "spend"


def build_spend_layers():
    """Build all three spend layers from unified_spend.parquet."""
    print("Building facts/spend/ layers...")

    source_file = SPEND_DIR / "unified_spend.parquet"
    if not source_file.exists():
        print(f"ERROR: Source file not found: {source_file}")
        return False

    con = duckdb.connect()

    # Layer 1: raw.parquet - Copy as-is (immutable truth)
    print("\n=== Layer 1: raw.parquet ===")
    raw_path = FACTS_SPEND / "raw.parquet"
    con.execute(f"""
        COPY (
            SELECT
                date,
                source,
                type,
                model,
                tokens_input,
                tokens_output,
                tokens_cached,
                tokens_total,
                cost_usd,
                app_name,
                receipt_id
            FROM '{source_file}'
            ORDER BY date, source
        ) TO '{raw_path}' (FORMAT PARQUET)
    """)
    raw_count = con.execute(f"SELECT COUNT(*) FROM '{raw_path}'").fetchone()[0]
    print(f"  Written: {raw_count:,} records")

    # Layer 2: daily.parquet - Aggregate by date + source
    print("\n=== Layer 2: daily.parquet ===")
    daily_path = FACTS_SPEND / "daily.parquet"
    con.execute(f"""
        COPY (
            SELECT
                date,
                source,
                type,
                SUM(tokens_input) as tokens_input,
                SUM(tokens_output) as tokens_output,
                SUM(tokens_cached) as tokens_cached,
                SUM(tokens_total) as tokens_total,
                SUM(cost_usd) as cost_usd,
                COUNT(*) as record_count
            FROM '{source_file}'
            WHERE date IS NOT NULL AND date != ''
            GROUP BY date, source, type
            ORDER BY date, source
        ) TO '{daily_path}' (FORMAT PARQUET)
    """)
    daily_count = con.execute(f"SELECT COUNT(*) FROM '{daily_path}'").fetchone()[0]
    print(f"  Written: {daily_count:,} records")

    # Layer 3: monthly.parquet - Aggregate by month + source
    print("\n=== Layer 3: monthly.parquet ===")
    monthly_path = FACTS_SPEND / "monthly.parquet"
    con.execute(f"""
        COPY (
            SELECT
                strftime(TRY_CAST(date AS DATE), '%Y-%m') as month,
                source,
                type,
                SUM(tokens_input) as tokens_input,
                SUM(tokens_output) as tokens_output,
                SUM(tokens_cached) as tokens_cached,
                SUM(tokens_total) as tokens_total,
                SUM(cost_usd) as cost_usd,
                COUNT(*) as record_count
            FROM '{source_file}'
            WHERE date IS NOT NULL AND date != ''
            GROUP BY strftime(TRY_CAST(date AS DATE), '%Y-%m'), source, type
            ORDER BY month, source
        ) TO '{monthly_path}' (FORMAT PARQUET)
    """)
    monthly_count = con.execute(f"SELECT COUNT(*) FROM '{monthly_path}'").fetchone()[0]
    print(f"  Written: {monthly_count:,} records")

    # Summary
    print("\n=== Summary ===")
    print(f"  raw.parquet:     {raw_count:>6,} records")
    print(f"  daily.parquet:   {daily_count:>6,} records")
    print(f"  monthly.parquet: {monthly_count:>6,} records")

    # Verify totals match
    raw_total = con.execute(f"SELECT SUM(cost_usd) FROM '{raw_path}'").fetchone()[0]
    daily_total = con.execute(f"SELECT SUM(cost_usd) FROM '{daily_path}'").fetchone()[0]
    monthly_total = con.execute(f"SELECT SUM(cost_usd) FROM '{monthly_path}'").fetchone()[0]

    print(f"\n=== Cost Verification ===")
    print(f"  raw total:     ${raw_total:,.2f}")
    print(f"  daily total:   ${daily_total:,.2f}")
    print(f"  monthly total: ${monthly_total:,.2f}")

    if abs(raw_total - daily_total) < 0.01 and abs(raw_total - monthly_total) < 0.01:
        print("  ✅ Totals match!")
    else:
        print("  ⚠️ Totals mismatch - check for NULL dates")

    return True


def show_monthly_summary():
    """Display monthly spend summary."""
    monthly_path = FACTS_SPEND / "monthly.parquet"
    if not monthly_path.exists():
        print("Monthly parquet not found")
        return

    con = duckdb.connect()
    print("\n=== Monthly Spend by Source ===")
    result = con.execute(f"""
        SELECT
            month,
            source,
            cost_usd,
            tokens_total
        FROM '{monthly_path}'
        WHERE cost_usd > 0
        ORDER BY month DESC, cost_usd DESC
        LIMIT 30
    """).fetchall()

    for row in result:
        month, source, cost, tokens = row
        tokens_str = f"{tokens:,}" if tokens else "N/A"
        print(f"  {month} | {source:<25} | ${cost:>8,.2f} | {tokens_str}")


if __name__ == '__main__':
    FACTS_SPEND.mkdir(parents=True, exist_ok=True)
    build_spend_layers()
    show_monthly_summary()
