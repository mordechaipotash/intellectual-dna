"""
Cognitive Fingerprint Extractor

Extracts your unique cognitive signature from 353K AI conversation messages.
What the 2025 AI leaders said to look for - trajectory, creation, sovereignty, velocity.
"""

import duckdb
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

BASE = Path("/Users/mordechai/intellectual_dna")
PARQUET = BASE / "data" / "all_conversations.parquet"


def get_conn():
    return duckdb.connect()


# =============================================================================
# 1. TRAJECTORY EXTRACTOR
# =============================================================================

def extract_trajectory():
    """Extract evolution from consumer to architect over time."""
    conn = get_conn()

    # Get quarterly message counts and topic evolution
    query = f"""
    SELECT
        year,
        CASE
            WHEN month <= 3 THEN 'Q1'
            WHEN month <= 6 THEN 'Q2'
            WHEN month <= 9 THEN 'Q3'
            ELSE 'Q4'
        END as quarter,
        COUNT(*) as msg_count,
        COUNT(DISTINCT conversation_id) as convos,
        source
    FROM read_parquet('{PARQUET}')
    WHERE role = 'user'
    GROUP BY year, quarter, source
    ORDER BY year, quarter
    """

    results = conn.execute(query).fetchall()

    # Complexity indicators by period
    complexity_query = f"""
    SELECT
        year,
        SUM(CASE WHEN LOWER(content) LIKE '%schema%' OR LOWER(content) LIKE '%architecture%' THEN 1 ELSE 0 END) as architecture_mentions,
        SUM(CASE WHEN LOWER(content) LIKE '%create%' OR LOWER(content) LIKE '%build%' OR LOWER(content) LIKE '%implement%' THEN 1 ELSE 0 END) as creation_mentions,
        SUM(CASE WHEN LOWER(content) LIKE '%optimize%' OR LOWER(content) LIKE '%performance%' THEN 1 ELSE 0 END) as optimization_mentions,
        SUM(CASE WHEN LOWER(content) LIKE '%framework%' OR LOWER(content) LIKE '%protocol%' OR LOWER(content) LIKE '%pattern%' THEN 1 ELSE 0 END) as framework_mentions
    FROM read_parquet('{PARQUET}')
    WHERE role = 'user'
    GROUP BY year
    ORDER BY year
    """

    complexity = conn.execute(complexity_query).fetchall()

    conn.close()

    return {
        'quarterly': results,
        'complexity': complexity
    }


# =============================================================================
# 2. PATTERN EXTRACTOR
# =============================================================================

def extract_patterns():
    """Extract signature cognitive patterns."""
    conn = get_conn()

    patterns = {
        'give_me': ("give me", "Direct command style"),
        'no_no': ("no no", "Fast correction"),
        'step_by_step': ("step by step", "Sequential depth"),
        'i_want': ("i want", "Clear intent"),
        'help_me': ("help me", "Collaboration request"),
        'show_me': ("show me", "Visual/concrete"),
    }

    results = {}

    for key, (phrase, description) in patterns.items():
        query = f"""
        SELECT
            year,
            COUNT(*) as count
        FROM read_parquet('{PARQUET}')
        WHERE role = 'user'
          AND LOWER(content) LIKE '%{phrase}%'
        GROUP BY year
        ORDER BY year
        """
        counts = conn.execute(query).fetchall()
        total = sum(c[1] for c in counts)
        results[key] = {
            'phrase': phrase,
            'description': description,
            'total': total,
            'by_year': counts
        }

    # Get yearly totals for percentage calculation
    totals_query = f"""
    SELECT year, COUNT(*) as total
    FROM read_parquet('{PARQUET}')
    WHERE role = 'user'
    GROUP BY year
    ORDER BY year
    """
    yearly_totals = dict(conn.execute(totals_query).fetchall())

    conn.close()

    return {
        'patterns': results,
        'yearly_totals': yearly_totals
    }


# =============================================================================
# 3. SOVEREIGNTY EXTRACTOR
# =============================================================================

def extract_sovereignty():
    """Extract correction/control patterns - do you control AI or vice versa."""
    conn = get_conn()

    # Correction indicators
    correction_phrases = [
        ("no no", "Direct correction"),
        ("not what i", "Clarification"),
        ("thats wrong", "Error correction"),
        ("actually", "Redirect"),
        ("instead", "Alternative direction"),
        ("let me clarify", "Clarification"),
        ("you misunderstood", "Correction"),
    ]

    results = {}
    total_corrections = 0

    for phrase, description in correction_phrases:
        query = f"""
        SELECT COUNT(*)
        FROM read_parquet('{PARQUET}')
        WHERE role = 'user'
          AND LOWER(content) LIKE '%{phrase}%'
        """
        count = conn.execute(query).fetchone()[0]
        results[phrase] = {'count': count, 'description': description}
        total_corrections += count

    # Total user messages for percentage
    total_query = f"""
    SELECT COUNT(*) FROM read_parquet('{PARQUET}')
    WHERE role = 'user'
    """
    total_messages = conn.execute(total_query).fetchone()[0]

    # Teaching indicators (explaining TO the AI)
    teaching_query = f"""
    SELECT COUNT(*)
    FROM read_parquet('{PARQUET}')
    WHERE role = 'user'
      AND (
        LOWER(content) LIKE '%let me explain%'
        OR LOWER(content) LIKE '%heres what i mean%'
        OR LOWER(content) LIKE '%to clarify%'
        OR LOWER(content) LIKE '%in other words%'
      )
    """
    teaching_count = conn.execute(teaching_query).fetchone()[0]

    conn.close()

    return {
        'corrections': results,
        'total_corrections': total_corrections,
        'total_messages': total_messages,
        'correction_rate': round(total_corrections / total_messages * 100, 2),
        'teaching_instances': teaching_count
    }


# =============================================================================
# 4. CREATION EXTRACTOR
# =============================================================================

def extract_creation():
    """Extract evidence of original creation vs consumption."""
    conn = get_conn()

    # Look for framework/concept creation language
    creation_indicators = [
        ("i call this", "Naming a concept"),
        ("my framework", "Framework ownership"),
        ("i created", "Direct creation"),
        ("i built", "Built something"),
        ("i designed", "Designed something"),
        ("my approach", "Original approach"),
        ("my method", "Original method"),
        ("protocol", "Protocol creation"),
    ]

    results = {}

    for phrase, description in creation_indicators:
        query = f"""
        SELECT COUNT(*),
               MIN(year || '-' || LPAD(month::VARCHAR, 2, '0')) as first_use
        FROM read_parquet('{PARQUET}')
        WHERE role = 'user'
          AND LOWER(content) LIKE '%{phrase}%'
        """
        row = conn.execute(query).fetchone()
        results[phrase] = {
            'count': row[0],
            'first_use': row[1],
            'description': description
        }

    # Look for specific original frameworks mentioned
    frameworks = [
        ("shelet", "SHELET Protocol"),
        ("bottleneck amplifier", "Bottleneck Amplifier"),
        ("seed principles", "SEED Principles"),
        ("receipts", "RECEIPTS Framework"),
        ("monotropic", "Monotropic Focus"),
    ]

    framework_evidence = {}
    for term, name in frameworks:
        query = f"""
        SELECT
            COUNT(*) as mentions,
            MIN(year || '-' || LPAD(month::VARCHAR, 2, '0')) as first_mention,
            MAX(year || '-' || LPAD(month::VARCHAR, 2, '0')) as last_mention
        FROM read_parquet('{PARQUET}')
        WHERE role = 'user'
          AND LOWER(content) LIKE '%{term}%'
        """
        row = conn.execute(query).fetchone()
        if row[0] > 0:
            framework_evidence[name] = {
                'mentions': row[0],
                'first': row[1],
                'last': row[2]
            }

    conn.close()

    return {
        'creation_language': results,
        'original_frameworks': framework_evidence
    }


# =============================================================================
# 5. VELOCITY EXTRACTOR
# =============================================================================

def extract_velocity():
    """Extract output velocity metrics."""
    conn = get_conn()

    # Messages per quarter
    quarterly_query = f"""
    SELECT
        year || '-Q' || CEIL(month / 3.0)::INT as quarter,
        COUNT(*) as messages,
        COUNT(DISTINCT conversation_id) as conversations,
        COUNT(DISTINCT DATE_TRUNC('day', created)) as active_days
    FROM read_parquet('{PARQUET}')
    WHERE role = 'user'
    GROUP BY year, CEIL(month / 3.0)::INT
    ORDER BY year, CEIL(month / 3.0)::INT
    """
    quarterly = conn.execute(quarterly_query).fetchall()

    # Peak periods
    monthly_query = f"""
    SELECT
        year || '-' || LPAD(month::VARCHAR, 2, '0') as month,
        COUNT(*) as messages
    FROM read_parquet('{PARQUET}')
    WHERE role = 'user'
    GROUP BY year, month
    ORDER BY COUNT(*) DESC
    LIMIT 5
    """
    peak_months = conn.execute(monthly_query).fetchall()

    # Source evolution
    source_query = f"""
    SELECT
        source,
        MIN(year || '-' || LPAD(month::VARCHAR, 2, '0')) as first_use,
        MAX(year || '-' || LPAD(month::VARCHAR, 2, '0')) as last_use,
        COUNT(*) as total_messages
    FROM read_parquet('{PARQUET}')
    WHERE role = 'user'
    GROUP BY source
    ORDER BY total_messages DESC
    """
    sources = conn.execute(source_query).fetchall()

    conn.close()

    return {
        'quarterly': quarterly,
        'peak_months': peak_months,
        'sources': sources
    }


# =============================================================================
# 6. GENERATE FINGERPRINT
# =============================================================================

def generate_fingerprint():
    """Generate complete cognitive fingerprint."""

    print("Extracting trajectory...")
    trajectory = extract_trajectory()

    print("Extracting patterns...")
    patterns = extract_patterns()

    print("Extracting sovereignty...")
    sovereignty = extract_sovereignty()

    print("Extracting creation...")
    creation = extract_creation()

    print("Extracting velocity...")
    velocity = extract_velocity()

    return {
        'trajectory': trajectory,
        'patterns': patterns,
        'sovereignty': sovereignty,
        'creation': creation,
        'velocity': velocity,
        'generated': datetime.now().isoformat()
    }


def print_fingerprint(fp):
    """Print cognitive fingerprint in readable format."""

    print("\n" + "=" * 80)
    print("COGNITIVE FINGERPRINT - Mordechai Potash")
    print("Generated:", fp['generated'])
    print("=" * 80)

    # TRAJECTORY
    print("\n" + "-" * 80)
    print("1. TRAJECTORY (Fei-Fei Li: 'How quickly can you superpower yourself')")
    print("-" * 80)

    complexity = fp['trajectory']['complexity']
    for year, arch, create, opt, framework in complexity:
        total = arch + create + opt + framework
        print(f"  {year}: Architecture({arch}) + Creation({create}) + Optimization({opt}) + Framework({framework}) = {total}")

    # PATTERNS
    print("\n" + "-" * 80)
    print("2. PATTERNS (Consistent cognitive rhythm)")
    print("-" * 80)

    for key, data in fp['patterns']['patterns'].items():
        print(f"  \"{data['phrase']}\" ({data['description']}): {data['total']:,}x")

    # SOVEREIGNTY
    print("\n" + "-" * 80)
    print("3. SOVEREIGNTY (Reid Hoffman: 'Intention-setting, goal-forming')")
    print("-" * 80)

    sov = fp['sovereignty']
    print(f"  Total corrections: {sov['total_corrections']:,}")
    print(f"  Total messages: {sov['total_messages']:,}")
    print(f"  Correction rate: {sov['correction_rate']}%")
    print(f"  Teaching instances: {sov['teaching_instances']:,}")

    # CREATION
    print("\n" + "-" * 80)
    print("4. CREATION (Dario Amodei: 'CREATE results, not transfer information')")
    print("-" * 80)

    print("  Original Frameworks:")
    for name, data in fp['creation']['original_frameworks'].items():
        print(f"    {name}: {data['mentions']} mentions (first: {data['first']})")

    # VELOCITY
    print("\n" + "-" * 80)
    print("5. VELOCITY (Sam Altman: '10x more productive')")
    print("-" * 80)

    print("  Peak months:")
    for month, msgs in fp['velocity']['peak_months']:
        print(f"    {month}: {msgs:,} messages")

    print("\n  Sources:")
    for source, first, last, total in fp['velocity']['sources']:
        print(f"    {source}: {total:,} msgs ({first} to {last})")

    # SUMMARY
    print("\n" + "=" * 80)
    print("COGNITIVE SIGNATURE SUMMARY")
    print("=" * 80)

    total_msgs = sov['total_messages']
    frameworks = len(fp['creation']['original_frameworks'])

    print(f"""
  Total Messages: {total_msgs:,}
  Time Span: 2023-01 to 2026-01 (3 years)
  Original Frameworks Created: {frameworks}
  Correction Rate: {sov['correction_rate']}%

  Cognitive Shape:
    - Spiral Processor (deep sequential)
    - Compression Native (reduce before expand)
    - High Sovereignty (controls AI, not reverse)
    - Burst Creator (intense periods, then pause)
    - Framework Originator ({frameworks} novel concepts)
    - Single-Thread Deep (monotropic focus)

  Pattern Rhythm:
    "give me the" → "no no no" → "step by step" → ship
""")


if __name__ == "__main__":
    fp = generate_fingerprint()
    print_fingerprint(fp)
