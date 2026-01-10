#!/usr/bin/env python3
"""
Intellectual DNA - Pipeline CLI

Single entry point for all data pipeline operations.

Usage:
    python -m pipelines status                              # Show brain status
    python -m pipelines import-claude --days 8              # Dry run import
    python -m pipelines import-claude --days 8 --import     # Actually import
    python -m pipelines import-claude --all --import --merge # Import all + merge
    python -m pipelines embed --batches 100                 # Embed 100 batches
    python -m pipelines embed --all                         # Embed all remaining
    python -m pipelines search "query"                      # Search embeddings
    python -m pipelines extract-metadata --all              # Extract JSONL metadata
    python -m pipelines extract-metadata --days 30          # Recent metadata only
    python -m pipelines metadata-stats                      # Show metadata stats
"""

import sys
import argparse
from pathlib import Path

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description='Intellectual DNA - Data Pipelines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pipelines status                          Show brain status
  python -m pipelines import-claude --days 8          Preview import (dry run)
  python -m pipelines import-claude --days 8 --import Actually import
  python -m pipelines embed --batches 50              Embed 50 batches
  python -m pipelines embed --all                     Embed all remaining
  python -m pipelines search "bottleneck"             Search embeddings
  python -m pipelines extract-metadata --all          Extract all JSONL metadata
  python -m pipelines metadata-stats                  Show metadata statistics
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show brain status')

    # Import Claude Code command
    import_parser = subparsers.add_parser('import-claude', help='Import Claude Code conversations')
    import_parser.add_argument('--days', type=int, default=8, help='Days of history (default: 8)')
    import_parser.add_argument('--all', action='store_true', help='Import all conversations')
    import_parser.add_argument('--import', dest='do_import', action='store_true', help='Actually import (not dry run)')
    import_parser.add_argument('--merge', action='store_true', help='Merge into main archive')

    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Run embedding pipeline')
    embed_parser.add_argument('--batches', type=int, default=5, help='Number of batches (default: 5)')
    embed_parser.add_argument('--all', action='store_true', help='Embed all remaining messages')
    embed_parser.add_argument('--checkpoint', type=int, default=500, help='Log progress every N messages')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search embeddings')
    search_parser.add_argument('query', nargs='+', help='Search query')
    search_parser.add_argument('--limit', type=int, default=10, help='Number of results (default: 10)')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show embedding stats')

    # Extract metadata command
    metadata_parser = subparsers.add_parser('extract-metadata', help='Extract Claude Code JSONL metadata')
    metadata_parser.add_argument('--days', type=int, help='Only process files from last N days')
    metadata_parser.add_argument('--all', dest='include_old', action='store_true', help='Include old archive')
    metadata_parser.add_argument('--dry-run', action='store_true', help='Show what would be processed')

    # Metadata stats command
    metadata_stats_parser = subparsers.add_parser('metadata-stats', help='Show Claude Code metadata stats')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == 'status':
        from pipelines.status import show_status
        show_status()

    elif args.command == 'import-claude':
        from pipelines.import_claude_code import import_claude_code
        import_claude_code(
            days=args.days,
            import_all=args.all,
            do_import=args.do_import,
            merge=args.merge
        )

    elif args.command == 'embed':
        if args.all:
            from pipelines.embed_all import embed_all
            embed_all(checkpoint_every=args.checkpoint)
        else:
            from pipelines.embed_messages import run_embedding_pipeline
            run_embedding_pipeline(batch_count=args.batches)

    elif args.command == 'search':
        from pipelines.embed_messages import search_similar
        query = ' '.join(args.query)
        print(f"Searching for: {query}\n")
        results = search_similar(query, limit=args.limit)
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['year']}-{r['month']:02d}] {r['title']}")
            print(f"   Similarity: {r['similarity']}")
            print(f"   {r['content'][:200]}...")
            print()

    elif args.command == 'stats':
        from pipelines.embed_messages import get_stats
        import duckdb
        from config import EMBEDDINGS_DB
        con = duckdb.connect(str(EMBEDDINGS_DB), read_only=True)
        stats = get_stats(con)
        print(f"Total embedded: {stats['total_embedded']:,}")
        print(f"By year: {stats['by_year']}")
        con.close()

    elif args.command == 'extract-metadata':
        from pipelines.extract_claude_code_metadata import extract_metadata
        extract_metadata(
            days=args.days,
            include_old=args.include_old,
            dry_run=args.dry_run
        )

    elif args.command == 'metadata-stats':
        from pipelines.extract_claude_code_metadata import show_stats
        show_stats()


if __name__ == '__main__':
    main()
