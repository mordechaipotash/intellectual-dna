#!/usr/bin/env python3
"""
Parse receipt files (PDFs and EMLs) from extracted zips into CSVs.
Handles: Anthropic, Cline, Windsurf, Augment receipts
"""

import csv
import email
import re
from datetime import datetime
from pathlib import Path
from email import policy
from email.parser import BytesParser

import pdfplumber

BASE_DIR = Path("/Users/mordechai/intellectual_dna/data/receipts_extracted")
OUTPUT_DIR = Path("/Users/mordechai/intellectual_dna/data/spend")


def parse_amount(text: str) -> float | None:
    """Extract dollar amount from text."""
    match = re.search(r'\$?([\d,]+\.?\d*)', text)
    if match:
        return float(match.group(1).replace(',', ''))
    return None


def parse_date(text: str) -> str | None:
    """Parse various date formats to ISO format."""
    patterns = [
        (r'(\w+ \d{1,2}, \d{4})', '%B %d, %Y'),  # May 20, 2025
        (r'(\w{3} \d{1,2}, \d{4})', '%b %d, %Y'),  # Jun 6, 2025 (abbreviated month)
        (r'(\d{1,2}/\d{1,2}/\d{4})', '%m/%d/%Y'),  # 05/20/2025
        (r'(\d{4}-\d{2}-\d{2})', '%Y-%m-%d'),  # 2025-05-20
        (r'(\d{1,2} \w+ \d{4})', '%d %B %Y'),  # 20 May 2025
    ]
    for pattern, fmt in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                dt = datetime.strptime(match.group(1), fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
    return None


def parse_cline_eml(filepath: Path) -> dict | None:
    """Parse Cline Stripe receipt email."""
    try:
        with open(filepath, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)

        # Get receipt ID from subject
        subject = msg.get('Subject', '')
        receipt_match = re.search(r'\[#([\d-]+)\]', subject)
        receipt_id = receipt_match.group(1) if receipt_match else None

        # Get date from header
        date_str = msg.get('Date', '')
        date = None
        if date_str:
            try:
                # Parse RFC 2822 date
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(date_str)
                date = dt.strftime('%Y-%m-%d')
            except:
                pass

        # Get body content
        body = ''
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    body = part.get_content()
                    break
        else:
            body = msg.get_content()

        # Extract amount
        amount_match = re.search(r'Amount paid\s*\n?\s*\$?([\d.,]+)', body)
        amount = float(amount_match.group(1).replace(',', '')) if amount_match else None

        return {
            'source': 'cline',
            'receipt_id': receipt_id,
            'date': date,
            'amount_usd': amount,
            'description': 'Cline Bot Inc. subscription',
            'payment_method': 'Link/Stripe',
            'file': filepath.name,
        }
    except Exception as e:
        print(f"  Error parsing {filepath.name}: {e}")
        return None


def parse_augment_eml(filepath: Path) -> dict | None:
    """Parse Augment payment confirmation email."""
    try:
        with open(filepath, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)

        subject = msg.get('Subject', '')

        # Extract invoice number from subject
        invoice_match = re.search(r'#([A-Z0-9-]+)', subject)
        receipt_id = invoice_match.group(1) if invoice_match else None

        # Get date from header
        date_str = msg.get('Date', '')
        date = None
        if date_str:
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(date_str)
                date = dt.strftime('%Y-%m-%d')
            except:
                pass

        # Get body content
        body = ''
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    body = part.get_content()
                    break
        else:
            body = msg.get_content()

        # Extract amount - look for Total or Amount patterns
        amount = None
        amount_patterns = [
            r'Total\s*:?\s*\$?([\d.,]+)',
            r'Amount\s*:?\s*\$?([\d.,]+)',
            r'\$\s*([\d.,]+)',
        ]
        for pattern in amount_patterns:
            match = re.search(pattern, body, re.IGNORECASE)
            if match:
                amount = float(match.group(1).replace(',', ''))
                break

        return {
            'source': 'augment',
            'receipt_id': receipt_id,
            'date': date,
            'amount_usd': amount,
            'description': 'Augment Code subscription',
            'payment_method': 'Stripe',
            'file': filepath.name,
        }
    except Exception as e:
        print(f"  Error parsing {filepath.name}: {e}")
        return None


def parse_invoice_pdf(filepath: Path, source: str) -> dict | None:
    """Parse invoice PDF using pdfplumber."""
    try:
        with pdfplumber.open(filepath) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text() or ''

        # Extract invoice number from filename
        invoice_match = re.search(r'Invoice-([A-Z0-9-]+)', filepath.name)
        receipt_id = invoice_match.group(1) if invoice_match else filepath.stem

        # If it's a Receipt PDF, extract from filename
        if filepath.name.startswith('Receipt-'):
            receipt_match = re.search(r'Receipt-(\d+-\d+)', filepath.name)
            receipt_id = receipt_match.group(1) if receipt_match else filepath.stem

        # Extract date
        date = parse_date(text)

        # Extract amount - look for explicit dollar amounts with $ sign
        amount = None

        # First, try to find "Amount due" or "Total" with explicit $ amounts
        amount_patterns = [
            r'Amount due\s*\$?([\d,]+\.?\d*)\s*USD',  # Amount due $20.00 USD
            r'Total\s+\$(\d+\.?\d*)',  # Total $20.00
            r'Amount due\s+\$(\d+\.?\d*)',  # Amount due $20.00
            r'due.*?\$(\d+\.?\d*)',  # ...due $20.00
        ]
        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount = float(match.group(1).replace(',', ''))
                break

        # If still no amount, find all dollar amounts and take the most common reasonable one
        if amount is None:
            # Find all amounts that are preceded by $
            dollar_amounts = re.findall(r'\$([\d,]+\.\d{2})', text)
            if dollar_amounts:
                # Convert to floats and filter out likely invoice numbers (> $10000 is suspicious for subscriptions)
                amounts = [float(a.replace(',', '')) for a in dollar_amounts]
                # For subscriptions, expect reasonable amounts
                reasonable_amounts = [a for a in amounts if a < 10000]
                if reasonable_amounts:
                    # Take the most common amount (usually the total appears multiple times)
                    from collections import Counter
                    most_common = Counter(reasonable_amounts).most_common(1)
                    if most_common:
                        amount = most_common[0][0]
                elif amounts:
                    amount = min(amounts)  # Take smallest if all are large

        # Determine description based on source
        descriptions = {
            'anthropic': 'Anthropic API usage',
            'windsurf': 'Windsurf/Codeium subscription',
            'augment': 'Augment Code subscription',
        }

        return {
            'source': source,
            'receipt_id': receipt_id,
            'date': date,
            'amount_usd': amount,
            'description': descriptions.get(source, f'{source} invoice'),
            'payment_method': 'Invoice',
            'file': filepath.name,
        }
    except Exception as e:
        print(f"  Error parsing {filepath.name}: {e}")
        return None


def process_anthropic():
    """Process Anthropic receipts."""
    print("\n=== Processing Anthropic ===")
    folder = BASE_DIR / "anthropic"
    records = []

    # Only process Invoice-*.pdf files
    for f in sorted(folder.glob("Invoice-*.pdf")):
        print(f"  Parsing: {f.name}")
        record = parse_invoice_pdf(f, 'anthropic')
        if record:
            records.append(record)

    print(f"  Found {len(records)} records")
    return records


def process_cline():
    """Process Cline receipts."""
    print("\n=== Processing Cline ===")
    folder = BASE_DIR / "cline"
    records = []

    for f in sorted(folder.glob("*.eml")):
        print(f"  Parsing: {f.name}")
        record = parse_cline_eml(f)
        if record:
            records.append(record)

    print(f"  Found {len(records)} records")
    return records


def process_windsurf():
    """Process Windsurf receipts."""
    print("\n=== Processing Windsurf ===")
    folder = BASE_DIR / "windsurf"
    records = []

    # Process Invoice PDFs
    for f in sorted(folder.glob("Invoice-*.pdf")):
        print(f"  Parsing: {f.name}")
        record = parse_invoice_pdf(f, 'windsurf')
        if record:
            records.append(record)

    # Also check for Receipt PDFs
    for f in sorted(folder.glob("Receipt-*.pdf")):
        print(f"  Parsing: {f.name}")
        record = parse_invoice_pdf(f, 'windsurf')
        if record:
            records.append(record)

    print(f"  Found {len(records)} records")
    return records


def process_augment():
    """Process Augment receipts."""
    print("\n=== Processing Augment ===")
    folder = BASE_DIR / "augment"
    records = []

    # Process Invoice PDFs
    for f in sorted(folder.glob("Invoice-*.pdf")):
        print(f"  Parsing: {f.name}")
        record = parse_invoice_pdf(f, 'augment')
        if record:
            records.append(record)

    # Process receipt emails (skip "Payment received" emails - they're just confirmations)
    for f in sorted(folder.glob("*.eml")):
        # Skip payment confirmation emails - they don't have amounts
        if f.name.startswith("Payment received"):
            print(f"  Skipping confirmation: {f.name}")
            continue
        print(f"  Parsing: {f.name}")
        record = parse_augment_eml(f)
        if record:
            records.append(record)

    print(f"  Found {len(records)} records")
    return records


def write_csv(records: list, filename: str):
    """Write records to CSV file."""
    if not records:
        print(f"  No records to write for {filename}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / filename

    fieldnames = ['source', 'receipt_id', 'date', 'amount_usd', 'description', 'payment_method', 'file']

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"  Wrote {len(records)} records to {filepath}")


def main():
    print("Receipt Parser - Extracting data from receipts")
    print("=" * 50)

    # Process each source
    anthropic_records = process_anthropic()
    cline_records = process_cline()
    windsurf_records = process_windsurf()
    augment_records = process_augment()

    # Write individual CSVs
    print("\n=== Writing CSVs ===")
    write_csv(anthropic_records, 'anthropic_receipts.csv')
    write_csv(cline_records, 'cline_receipts.csv')
    write_csv(windsurf_records, 'windsurf_receipts.csv')
    write_csv(augment_records, 'augment_receipts.csv')

    # Write combined CSV
    all_records = anthropic_records + cline_records + windsurf_records + augment_records
    write_csv(all_records, 'all_receipts.csv')

    # Summary
    print("\n=== Summary ===")
    print(f"  Anthropic: {len(anthropic_records)} receipts")
    print(f"  Cline:     {len(cline_records)} receipts")
    print(f"  Windsurf:  {len(windsurf_records)} receipts")
    print(f"  Augment:   {len(augment_records)} receipts")
    print(f"  TOTAL:     {len(all_records)} receipts")

    # Calculate totals
    total_by_source = {}
    for r in all_records:
        src = r['source']
        amt = r['amount_usd'] or 0
        total_by_source[src] = total_by_source.get(src, 0) + amt

    print("\n=== Spend Totals ===")
    for src, total in sorted(total_by_source.items()):
        print(f"  {src}: ${total:,.2f}")
    print(f"  TOTAL: ${sum(total_by_source.values()):,.2f}")


if __name__ == '__main__':
    main()
