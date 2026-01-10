#!/usr/bin/env python3
"""
Create unified spend parquet from all sources:
- Receipt CSVs (anthropic, cline, windsurf, augment)
- API usage CSVs (anthropic, openai, openrouter)
"""

import csv
from datetime import datetime
from pathlib import Path
from collections import defaultdict

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
SPEND_DIR = BASE_DIR / "data" / "spend"

def load_claude_subscription():
    """Load Claude Pro/Max subscription data."""
    print("\n=== Loading Claude Subscription ===")
    records = []

    sub_file = SPEND_DIR / "claude_subscription.csv"
    if sub_file.exists():
        with open(sub_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    'date': row.get('date', ''),
                    'source': row.get('source', ''),  # claude_pro or claude_max
                    'type': 'subscription',
                    'model': None,
                    'tokens_input': 0,
                    'tokens_output': 0,
                    'tokens_cached': 0,
                    'tokens_total': 0,
                    'cost_usd': float(row.get('amount_usd', 0) or 0),
                    'app_name': row.get('description', ''),
                    'receipt_id': row.get('receipt_id', ''),
                })

    print(f"  Loaded {len(records)} Claude subscription records")
    return records


def load_chatgpt_subscription():
    """Load ChatGPT Plus/Pro subscription data."""
    print("\n=== Loading ChatGPT Subscription ===")
    records = []

    sub_file = SPEND_DIR / "chatgpt_subscription.csv"
    if sub_file.exists():
        with open(sub_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    'date': row.get('date', ''),
                    'source': row.get('source', ''),  # chatgpt_plus or chatgpt_pro
                    'type': 'subscription',
                    'model': None,
                    'tokens_input': 0,
                    'tokens_output': 0,
                    'tokens_cached': 0,
                    'tokens_total': 0,
                    'cost_usd': float(row.get('amount_usd', 0) or 0),
                    'app_name': row.get('description', ''),
                    'receipt_id': row.get('receipt_id', ''),
                })

    print(f"  Loaded {len(records)} ChatGPT subscription records")
    return records


def load_claude_code_tokens():
    """Load Claude Code session token usage from metadata parquet."""
    print("\n=== Loading Claude Code Tokens ===")
    records = []

    if not HAS_DUCKDB:
        print("  DuckDB not available, skipping Claude Code tokens")
        return records

    metadata_file = BASE_DIR / "data" / "claude_code_metadata.parquet"
    if not metadata_file.exists():
        print(f"  File not found: {metadata_file}")
        return records

    con = duckdb.connect()
    # Aggregate by date and model
    result = con.execute(f"""
        SELECT
            CAST(first_timestamp AS DATE) as date,
            models_used,
            SUM(total_input_tokens) as tokens_input,
            SUM(total_output_tokens) as tokens_output,
            SUM(total_cache_creation_tokens + total_cache_read_tokens) as tokens_cached,
            COUNT(*) as session_count
        FROM '{metadata_file}'
        WHERE first_timestamp IS NOT NULL
        GROUP BY CAST(first_timestamp AS DATE), models_used
        ORDER BY date
    """).fetchall()

    for row in result:
        date, models_json, tokens_in, tokens_out, tokens_cached, sessions = row
        # Parse model from JSON array
        model = None
        if models_json:
            try:
                import json
                models = json.loads(models_json)
                model = models[0] if models else None
            except:
                model = str(models_json)

        records.append({
            'date': str(date) if date else '',
            'source': 'claude_code',
            'type': 'code_session',
            'model': model,
            'tokens_input': int(tokens_in or 0),
            'tokens_output': int(tokens_out or 0),
            'tokens_cached': int(tokens_cached or 0),
            'tokens_total': int((tokens_in or 0) + (tokens_out or 0)),
            'cost_usd': 0.0,  # Included in Claude Max subscription
            'app_name': f'{sessions} sessions',
            'receipt_id': None,
        })

    print(f"  Loaded {len(records)} Claude Code daily records")

    # Summary stats
    total_input = sum(r['tokens_input'] for r in records)
    total_output = sum(r['tokens_output'] for r in records)
    total_cached = sum(r['tokens_cached'] for r in records)
    print(f"    Input: {total_input:,} | Output: {total_output:,} | Cached: {total_cached:,}")

    return records


def load_receipts():
    """Load receipt data."""
    print("\n=== Loading Receipts ===")
    records = []

    receipt_file = SPEND_DIR / "all_receipts.csv"
    if receipt_file.exists():
        with open(receipt_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize source names
                source = row.get('source', '')
                if source == 'anthropic':
                    source = 'anthropic_subscription'
                elif source == 'cline':
                    source = 'cline_subscription'
                elif source == 'windsurf':
                    source = 'windsurf_subscription'
                elif source == 'augment':
                    source = 'augment_subscription'

                records.append({
                    'date': row.get('date', ''),
                    'source': source,
                    'type': 'subscription',
                    'model': None,
                    'tokens_input': 0,
                    'tokens_output': 0,
                    'tokens_cached': 0,
                    'tokens_total': 0,
                    'cost_usd': float(row.get('amount_usd', 0) or 0),
                    'app_name': row.get('description', ''),
                    'receipt_id': row.get('receipt_id', ''),
                })

    print(f"  Loaded {len(records)} receipt records")
    return records


def load_api_daily():
    """Load daily API aggregates."""
    print("\n=== Loading API Usage (Daily) ===")
    records = []

    daily_file = SPEND_DIR / "api_usage_daily.csv"
    if daily_file.exists():
        with open(daily_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    'date': row.get('date', ''),
                    'source': row.get('source', ''),
                    'type': 'api_usage',
                    'model': row.get('model', ''),
                    'tokens_input': int(row.get('tokens_input', 0) or 0),
                    'tokens_output': int(row.get('tokens_output', 0) or 0),
                    'tokens_cached': int(row.get('tokens_cached', 0) or 0),
                    'tokens_total': int(row.get('tokens_total', 0) or 0),
                    'cost_usd': float(row.get('cost_usd', 0) or 0),
                    'app_name': row.get('app_name', ''),
                    'receipt_id': None,
                })

    print(f"  Loaded {len(records)} daily API records")
    return records


def aggregate_monthly(records: list) -> list:
    """Aggregate to monthly summaries."""
    print("\n=== Aggregating to Monthly ===")

    monthly = defaultdict(lambda: {
        'tokens_input': 0,
        'tokens_output': 0,
        'tokens_cached': 0,
        'tokens_total': 0,
        'cost_usd': 0,
        'records': 0,
    })

    for r in records:
        date = r.get('date', '')
        if len(date) >= 7:
            month = date[:7]  # YYYY-MM
        else:
            continue

        key = (month, r['source'], r['type'])
        monthly[key]['tokens_input'] += r.get('tokens_input', 0)
        monthly[key]['tokens_output'] += r.get('tokens_output', 0)
        monthly[key]['tokens_cached'] += r.get('tokens_cached', 0)
        monthly[key]['tokens_total'] += r.get('tokens_total', 0)
        monthly[key]['cost_usd'] += r.get('cost_usd', 0)
        monthly[key]['records'] += 1

    result = []
    for (month, source, type_), agg in monthly.items():
        result.append({
            'month': month,
            'source': source,
            'type': type_,
            'tokens_input': agg['tokens_input'],
            'tokens_output': agg['tokens_output'],
            'tokens_cached': agg['tokens_cached'],
            'tokens_total': agg['tokens_total'],
            'cost_usd': agg['cost_usd'],
            'records': agg['records'],
        })

    result.sort(key=lambda x: (x['month'], x['source']))
    print(f"  Aggregated to {len(result)} monthly records")
    return result


def write_csv(records: list, filename: str, fieldnames: list = None):
    """Write records to CSV."""
    if not records:
        return

    filepath = SPEND_DIR / filename
    if not fieldnames:
        fieldnames = list(records[0].keys())

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)

    print(f"  Wrote {len(records)} to {filepath}")


def write_parquet(records: list, filename: str):
    """Write records to parquet using DuckDB."""
    if not HAS_DUCKDB:
        print("  DuckDB not available, skipping parquet")
        return

    filepath = SPEND_DIR / filename

    con = duckdb.connect()

    # Create table from records
    con.execute("""
        CREATE TABLE spend (
            date VARCHAR,
            source VARCHAR,
            type VARCHAR,
            model VARCHAR,
            tokens_input BIGINT,
            tokens_output BIGINT,
            tokens_cached BIGINT,
            tokens_total BIGINT,
            cost_usd DOUBLE,
            app_name VARCHAR,
            receipt_id VARCHAR
        )
    """)

    for r in records:
        con.execute("""
            INSERT INTO spend VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            r.get('date'),
            r.get('source'),
            r.get('type'),
            r.get('model'),
            r.get('tokens_input', 0),
            r.get('tokens_output', 0),
            r.get('tokens_cached', 0),
            r.get('tokens_total', 0),
            r.get('cost_usd', 0),
            r.get('app_name'),
            r.get('receipt_id'),
        ])

    con.execute(f"COPY spend TO '{filepath}' (FORMAT PARQUET)")
    print(f"  Wrote {len(records)} to {filepath}")


def main():
    print("Creating Unified Spend Data")
    print("=" * 50)

    # Load all data
    claude_sub = load_claude_subscription()
    chatgpt_sub = load_chatgpt_subscription()
    claude_code = load_claude_code_tokens()
    receipts = load_receipts()
    api_daily = load_api_daily()

    # Combine
    all_records = claude_sub + chatgpt_sub + claude_code + receipts + api_daily
    print(f"\n=== Combined: {len(all_records)} records ===")

    # Write unified CSV
    fieldnames = ['date', 'source', 'type', 'model', 'tokens_input', 'tokens_output',
                  'tokens_cached', 'tokens_total', 'cost_usd', 'app_name', 'receipt_id']
    write_csv(all_records, 'unified_spend.csv', fieldnames)

    # Write parquet
    write_parquet(all_records, 'unified_spend.parquet')

    # Monthly aggregates
    monthly = aggregate_monthly(all_records)
    monthly_fields = ['month', 'source', 'type', 'tokens_input', 'tokens_output',
                      'tokens_cached', 'tokens_total', 'cost_usd', 'records']
    write_csv(monthly, 'spend_monthly.csv', monthly_fields)

    # Summary by source
    print("\n=== Cost Summary by Source ===")
    by_source = defaultdict(lambda: {'tokens': 0, 'cost': 0, 'records': 0})
    for r in all_records:
        src = r['source']
        by_source[src]['tokens'] += r.get('tokens_total', 0)
        by_source[src]['cost'] += r.get('cost_usd', 0)
        by_source[src]['records'] += 1

    total_cost = 0
    total_tokens = 0
    for src, data in sorted(by_source.items(), key=lambda x: -x[1]['cost']):
        print(f"  {src}:")
        if data['tokens'] > 0:
            print(f"    Tokens: {data['tokens']:>15,}")
        print(f"    Cost:   ${data['cost']:>14,.2f}")
        print(f"    Records: {data['records']:>14,}")
        total_cost += data['cost']
        total_tokens += data['tokens']

    print(f"\n  GRAND TOTAL:")
    print(f"    Tokens: {total_tokens:>15,}")
    print(f"    Cost:   ${total_cost:>14,.2f}")


if __name__ == '__main__':
    main()
