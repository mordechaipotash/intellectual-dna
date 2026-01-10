#!/usr/bin/env python3
"""
Parse API token usage CSVs from Anthropic, OpenAI, and OpenRouter.
Creates unified spend CSVs for analysis.
"""

import csv
from datetime import datetime
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
OUTPUT_DIR = BASE_DIR / "data" / "spend"

# Source directories
ANTHROPIC_DIR = BASE_DIR / "anthropic_api_tokens_mordechai"
OPENAI_DIR = BASE_DIR / "openai_api_tokens_mordechai"
OPENROUTER_DIR = BASE_DIR / "data" / "openrouter_history_mordechai"


def parse_anthropic():
    """Parse Anthropic API token CSVs."""
    print("\n=== Processing Anthropic API Tokens ===")
    records = []

    for csv_file in sorted(ANTHROPIC_DIR.glob("*.csv")):
        print(f"  Parsing: {csv_file.name}")
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Calculate total tokens
                input_tokens = (
                    int(row.get('usage_input_tokens_no_cache', 0) or 0) +
                    int(row.get('usage_input_tokens_cache_write_5m', 0) or 0) +
                    int(row.get('usage_input_tokens_cache_write_1h', 0) or 0) +
                    int(row.get('usage_input_tokens_cache_read', 0) or 0)
                )
                output_tokens = int(row.get('usage_output_tokens', 0) or 0)
                cache_read = int(row.get('usage_input_tokens_cache_read', 0) or 0)

                records.append({
                    'source': 'anthropic_direct',
                    'date': row.get('usage_date_utc', ''),
                    'model': row.get('model_version', ''),
                    'tokens_input': input_tokens,
                    'tokens_output': output_tokens,
                    'tokens_cached': cache_read,
                    'cost_usd': None,  # Anthropic CSVs don't include cost
                    'app_name': row.get('api_key', ''),
                    'workspace': row.get('workspace', ''),
                })

    print(f"  Total records: {len(records)}")
    return records


def parse_openai():
    """Parse OpenAI API token CSVs."""
    print("\n=== Processing OpenAI API Tokens ===")
    records = []

    for csv_file in sorted(OPENAI_DIR.glob("*.csv")):
        print(f"  Parsing: {csv_file.name}")
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip empty rows
                input_tokens = row.get('input_tokens', '')
                if not input_tokens or input_tokens == '':
                    continue

                try:
                    input_tokens = int(float(input_tokens))
                    output_tokens = int(float(row.get('output_tokens', 0) or 0))
                    cached_tokens = int(float(row.get('input_cached_tokens', 0) or 0))
                except (ValueError, TypeError):
                    continue

                # Parse date from ISO string
                date_str = row.get('start_time_iso', '')
                if date_str:
                    try:
                        date = datetime.fromisoformat(date_str.replace('+00:00', '')).strftime('%Y-%m-%d')
                    except:
                        date = date_str[:10]
                else:
                    date = ''

                records.append({
                    'source': 'openai_direct',
                    'date': date,
                    'model': row.get('model', ''),
                    'tokens_input': input_tokens,
                    'tokens_output': output_tokens,
                    'tokens_cached': cached_tokens,
                    'cost_usd': None,  # OpenAI CSVs don't include cost
                    'app_name': row.get('api_key_id', ''),
                    'workspace': row.get('project_id', ''),
                })

    print(f"  Total records: {len(records)}")
    return records


def parse_openrouter():
    """Parse OpenRouter history CSVs."""
    print("\n=== Processing OpenRouter History ===")
    records = []

    for csv_file in sorted(OPENROUTER_DIR.glob("*.csv")):
        print(f"  Parsing: {csv_file.name}")
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse date from created_at
                created_at = row.get('created_at', '')
                if created_at:
                    try:
                        date = created_at[:10]  # YYYY-MM-DD
                    except:
                        date = ''
                else:
                    date = ''

                # Parse tokens
                try:
                    prompt_tokens = int(row.get('tokens_prompt', 0) or 0)
                    completion_tokens = int(row.get('tokens_completion', 0) or 0)
                    reasoning_tokens = int(row.get('tokens_reasoning', 0) or 0)
                    cost = float(row.get('cost_total', 0) or 0)
                except (ValueError, TypeError):
                    prompt_tokens = 0
                    completion_tokens = 0
                    reasoning_tokens = 0
                    cost = 0

                records.append({
                    'source': 'openrouter',
                    'date': date,
                    'model': row.get('model_permaslug', ''),
                    'tokens_input': prompt_tokens,
                    'tokens_output': completion_tokens + reasoning_tokens,
                    'tokens_cached': 0,  # OpenRouter tracks differently
                    'cost_usd': cost,
                    'app_name': row.get('app_name', ''),
                    'provider': row.get('provider_name', ''),
                })

    print(f"  Total records: {len(records)}")
    return records


def aggregate_daily(records: list) -> list:
    """Aggregate records by date, source, model."""
    print("\n=== Aggregating to Daily ===")

    # Group by (date, source, model, app_name)
    daily = defaultdict(lambda: {
        'tokens_input': 0,
        'tokens_output': 0,
        'tokens_cached': 0,
        'cost_usd': 0,
        'count': 0,
    })

    for r in records:
        key = (r['date'], r['source'], r['model'], r.get('app_name', ''))
        daily[key]['tokens_input'] += r['tokens_input']
        daily[key]['tokens_output'] += r['tokens_output']
        daily[key]['tokens_cached'] += r.get('tokens_cached', 0) or 0
        daily[key]['cost_usd'] += r.get('cost_usd', 0) or 0
        daily[key]['count'] += 1

    result = []
    for (date, source, model, app_name), agg in daily.items():
        result.append({
            'date': date,
            'source': source,
            'model': model,
            'app_name': app_name,
            'tokens_input': agg['tokens_input'],
            'tokens_output': agg['tokens_output'],
            'tokens_cached': agg['tokens_cached'],
            'tokens_total': agg['tokens_input'] + agg['tokens_output'],
            'cost_usd': agg['cost_usd'],
            'api_calls': agg['count'],
        })

    result.sort(key=lambda x: (x['date'], x['source'], x['model']))
    print(f"  Aggregated to {len(result)} daily records")
    return result


def write_csv(records: list, filename: str, fieldnames: list = None):
    """Write records to CSV file."""
    if not records:
        print(f"  No records to write for {filename}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / filename

    if not fieldnames:
        fieldnames = list(records[0].keys())

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)

    print(f"  Wrote {len(records)} records to {filepath}")


def main():
    print("API Token Parser - Extracting usage data")
    print("=" * 50)

    # Parse each source
    anthropic_records = parse_anthropic()
    openai_records = parse_openai()
    openrouter_records = parse_openrouter()

    # Write raw CSVs (preserving all data)
    print("\n=== Writing Raw CSVs ===")
    write_csv(anthropic_records, 'anthropic_api_raw.csv')
    write_csv(openai_records, 'openai_api_raw.csv')
    write_csv(openrouter_records, 'openrouter_raw.csv')

    # Combine all records
    all_records = anthropic_records + openai_records + openrouter_records

    # Aggregate to daily
    daily_records = aggregate_daily(all_records)

    # Write daily aggregates
    print("\n=== Writing Aggregated CSVs ===")
    daily_fieldnames = ['date', 'source', 'model', 'app_name', 'tokens_input',
                        'tokens_output', 'tokens_cached', 'tokens_total', 'cost_usd', 'api_calls']
    write_csv(daily_records, 'api_usage_daily.csv', daily_fieldnames)

    # Summary
    print("\n=== Summary ===")
    print(f"  Anthropic Direct: {len(anthropic_records):,} records")
    print(f"  OpenAI Direct:    {len(openai_records):,} records")
    print(f"  OpenRouter:       {len(openrouter_records):,} records")
    print(f"  TOTAL:            {len(all_records):,} records")
    print(f"  Daily aggregates: {len(daily_records):,} records")

    # Token totals by source
    print("\n=== Token Totals by Source ===")
    totals = defaultdict(lambda: {'input': 0, 'output': 0, 'cost': 0})
    for r in all_records:
        totals[r['source']]['input'] += r['tokens_input']
        totals[r['source']]['output'] += r['tokens_output']
        totals[r['source']]['cost'] += r.get('cost_usd', 0) or 0

    grand_input = 0
    grand_output = 0
    grand_cost = 0
    for source, t in sorted(totals.items()):
        total_tokens = t['input'] + t['output']
        print(f"  {source}:")
        print(f"    Input:  {t['input']:>15,} tokens")
        print(f"    Output: {t['output']:>15,} tokens")
        print(f"    Total:  {total_tokens:>15,} tokens")
        if t['cost'] > 0:
            print(f"    Cost:   ${t['cost']:>14,.2f}")
        grand_input += t['input']
        grand_output += t['output']
        grand_cost += t['cost']

    print(f"\n  GRAND TOTAL:")
    print(f"    Input:  {grand_input:>15,} tokens")
    print(f"    Output: {grand_output:>15,} tokens")
    print(f"    Total:  {grand_input + grand_output:>15,} tokens")
    print(f"    Cost:   ${grand_cost:>14,.2f}")


if __name__ == '__main__':
    main()
