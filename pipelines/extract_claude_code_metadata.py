#!/usr/bin/env python3
"""
Extract rich metadata from Claude Code JSONL files.

Captures tool usage, web searches, files touched, token usage, and project context
that isn't stored in the main conversations parquet.

Usage:
    python -m pipelines extract-metadata [--all] [--days N]
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CLAUDE_PROJECTS, CLAUDE_CODE_METADATA_PARQUET,
)

# Output path
METADATA_PARQUET = CLAUDE_CODE_METADATA_PARQUET

# Old archive location
OLD_JSONL_DIR = Path("/Users/mordechai/Mordechai Dev 2025/Sparkii/sparkii-command-center/oct-5-2-2025/sparkii-rag/RAG/claude_code_chats/old jsonl files")


def extract_tool_calls(content_blocks: list) -> list[dict]:
    """Extract tool calls from message content blocks."""
    tools = []
    if not isinstance(content_blocks, list):
        return tools

    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if block.get('type') == 'tool_use':
            tool_name = block.get('name', 'unknown')
            tool_input = block.get('input', {})

            # Summarize input based on tool type
            input_summary = summarize_tool_input(tool_name, tool_input)

            tools.append({
                'name': tool_name,
                'input_summary': input_summary,
                'tool_use_id': block.get('id', ''),
            })

    return tools


def summarize_tool_input(tool_name: str, tool_input: dict) -> str:
    """Extract key info from tool input - no truncation, 1.2MB is tiny."""
    if not isinstance(tool_input, dict):
        return str(tool_input)

    # Web search - extract query
    if tool_name == 'WebSearch':
        return tool_input.get('query', '')

    # File operations - extract path
    if tool_name in ('Read', 'Write', 'Edit', 'Glob', 'Grep'):
        path = tool_input.get('file_path') or tool_input.get('path') or tool_input.get('pattern', '')
        return str(path)

    # Bash - extract command
    if tool_name == 'Bash':
        return tool_input.get('command', '')

    # MCP tools - extract key params
    if tool_name.startswith('mcp__'):
        # Brain tools
        if 'brain' in tool_name:
            return tool_input.get('term') or tool_input.get('query') or tool_input.get('topic') or str(tool_input)
        # Desktop commander
        if 'desktop-commander' in tool_name:
            return tool_input.get('command') or tool_input.get('path') or str(tool_input)
        # GitHub
        if 'github' in tool_name:
            return f"{tool_input.get('owner', '')}/{tool_input.get('repo', '')}"

    # Task/Agent - extract prompt
    if tool_name == 'Task':
        return tool_input.get('prompt', '') or tool_input.get('description', '')

    # Default - full string representation
    return str(tool_input)


def extract_web_searches(tools: list[dict]) -> list[str]:
    """Extract web search queries from tool calls."""
    searches = []
    for tool in tools:
        if tool['name'] == 'WebSearch' and tool['input_summary']:
            searches.append(tool['input_summary'])
    return searches


def extract_files_touched(tools: list[dict]) -> dict[str, list[str]]:
    """Extract files read/written/edited from tool calls."""
    files = {
        'read': [],
        'written': [],
        'edited': [],
        'searched': [],
    }

    for tool in tools:
        name = tool['name']
        path = tool['input_summary']

        if not path or path.startswith('{') or path.startswith('['):
            continue

        if name == 'Read':
            files['read'].append(path)
        elif name == 'Write':
            files['written'].append(path)
        elif name == 'Edit':
            files['edited'].append(path)
        elif name in ('Glob', 'Grep'):
            files['searched'].append(path)
        elif name.startswith('mcp__desktop-commander__read'):
            files['read'].append(path)
        elif name.startswith('mcp__desktop-commander__write'):
            files['written'].append(path)

    # Deduplicate
    for key in files:
        files[key] = list(set(files[key]))

    return files


def extract_bash_commands(tools: list[dict]) -> list[str]:
    """Extract bash commands from tool calls."""
    commands = []
    for tool in tools:
        if tool['name'] == 'Bash' and tool['input_summary']:
            commands.append(tool['input_summary'])
    return commands


def extract_usage_stats(message: dict) -> dict:
    """Extract token usage from message."""
    usage = message.get('message', {}).get('usage', {})
    if not usage:
        return {}

    return {
        'input_tokens': usage.get('input_tokens', 0),
        'output_tokens': usage.get('output_tokens', 0),
        'cache_creation_tokens': usage.get('cache_creation_input_tokens', 0),
        'cache_read_tokens': usage.get('cache_read_input_tokens', 0),
    }


def extract_thinking_blocks(content_blocks: list) -> list[str]:
    """Extract extended thinking content from message blocks."""
    thinking = []
    if not isinstance(content_blocks, list):
        return thinking
    for block in content_blocks:
        if isinstance(block, dict) and block.get('type') == 'thinking':
            text = block.get('thinking', '')
            if text:
                thinking.append(text)
    return thinking


def process_jsonl_file(jsonl_path: Path) -> Optional[dict]:
    """Process a single JSONL file and extract metadata."""
    try:
        lines = jsonl_path.read_text().strip().split('\n')
    except Exception as e:
        return None

    if not lines or not lines[0].strip():
        return None

    # Extract conversation-level metadata
    conversation_id = jsonl_path.stem
    project = jsonl_path.parent.name.replace('-Users-mordechai-', '').replace('-', '/')[:100]

    all_tools = []
    total_usage = defaultdict(int)
    working_directory = None
    git_branch = None
    first_timestamp = None
    last_timestamp = None
    msg_count = 0

    # NEW: Additional metadata
    slug = None
    claude_code_version = None
    models_used = set()
    thinking_blocks = []
    todos_snapshots = []
    stop_reasons = []

    # Even more metadata
    title = None
    summary = None
    agent_ids = set()
    sidechain_count = 0
    tool_errors = []
    ephemeral_5m_tokens = 0
    ephemeral_1h_tokens = 0
    request_ids = set()

    for line in lines:
        if not line.strip():
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Capture todos snapshots (any entry type)
        if obj.get('todos'):
            todos_snapshots.append(obj['todos'])

        # Capture conversation-level metadata (often in first entries)
        if not title and obj.get('title'):
            title = obj['title']
        if not summary and obj.get('summary'):
            summary = obj['summary']

        # Track agents and sidechains
        if obj.get('agentId'):
            agent_ids.add(obj['agentId'])
        if obj.get('isSidechain'):
            sidechain_count += 1
        if obj.get('requestId'):
            request_ids.add(obj['requestId'])

        # Skip non-message entries (accept user, assistant, and legacy 'message' types)
        msg_type = obj.get('type')
        if msg_type not in (None, 'message', 'user', 'assistant'):
            continue

        msg_count += 1

        # Extract context
        if not working_directory and obj.get('cwd'):
            working_directory = obj['cwd']
        if not git_branch and obj.get('gitBranch'):
            git_branch = obj['gitBranch']
        if not slug and obj.get('slug'):
            slug = obj['slug']
        if not claude_code_version and obj.get('version'):
            claude_code_version = obj['version']

        # Track timestamps
        ts = obj.get('timestamp')
        if ts:
            if not first_timestamp:
                first_timestamp = ts
            last_timestamp = ts

        # Extract from message object
        msg = obj.get('message', {})
        content = msg.get('content', [])

        # Track model
        model = msg.get('model')
        if model and model != '<synthetic>':
            models_used.add(model)

        # Track stop reason
        stop_reason = msg.get('stop_reason')
        if stop_reason:
            stop_reasons.append(stop_reason)

        if isinstance(content, list):
            # Extract tool calls
            tools = extract_tool_calls(content)
            for tool in tools:
                tool['timestamp'] = ts
            all_tools.extend(tools)

            # Extract thinking blocks
            thinking = extract_thinking_blocks(content)
            thinking_blocks.extend(thinking)

            # Extract tool errors
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'tool_result' and block.get('is_error'):
                    err_content = block.get('content', '')
                    if isinstance(err_content, str) and err_content:
                        tool_errors.append(err_content)

        # Accumulate usage
        usage = extract_usage_stats(obj)
        for key, val in usage.items():
            total_usage[key] += val

        # Extract cache breakdown
        msg_usage = msg.get('usage', {})
        cache = msg_usage.get('cache_creation', {})
        if isinstance(cache, dict):
            ephemeral_5m_tokens += cache.get('ephemeral_5m_input_tokens', 0)
            ephemeral_1h_tokens += cache.get('ephemeral_1h_input_tokens', 0)

    if msg_count == 0:
        return None

    # Aggregate extractions
    web_searches = extract_web_searches(all_tools)
    files = extract_files_touched(all_tools)
    bash_commands = extract_bash_commands(all_tools)

    # Count tools by name
    tool_counts = defaultdict(int)
    for tool in all_tools:
        tool_counts[tool['name']] += 1

    # Count stop reasons
    stop_reason_counts = defaultdict(int)
    for sr in stop_reasons:
        stop_reason_counts[sr] += 1

    return {
        'conversation_id': conversation_id,
        'project': project,
        'slug': slug,
        'title': title,
        'summary': summary,
        'claude_code_version': claude_code_version,
        'models_used': list(models_used),
        'working_directory': working_directory,
        'git_branch': git_branch,
        'first_timestamp': first_timestamp,
        'last_timestamp': last_timestamp,
        'msg_count': msg_count,

        # Agent/sidechain tracking
        'agent_ids': list(agent_ids),
        'agent_count': len(agent_ids),
        'sidechain_count': sidechain_count,
        'request_ids': list(request_ids),

        # Tool stats
        'total_tool_calls': len(all_tools),
        'tool_counts': dict(tool_counts),
        'unique_tools': list(tool_counts.keys()),

        # Tool errors
        'tool_errors': tool_errors,
        'tool_error_count': len(tool_errors),

        # Extracted data
        'web_searches': web_searches,
        'files_read': files['read'],
        'files_written': files['written'],
        'files_edited': files['edited'],
        'files_searched': files['searched'],
        'bash_commands': bash_commands,

        # Extended thinking
        'thinking_blocks': thinking_blocks,
        'thinking_count': len(thinking_blocks),

        # Todos
        'todos_snapshots': todos_snapshots,

        # Stop reasons
        'stop_reason_counts': dict(stop_reason_counts),

        # Token usage
        'total_input_tokens': total_usage['input_tokens'],
        'total_output_tokens': total_usage['output_tokens'],
        'total_cache_creation_tokens': total_usage['cache_creation_tokens'],
        'total_cache_read_tokens': total_usage['cache_read_tokens'],
        'ephemeral_5m_tokens': ephemeral_5m_tokens,
        'ephemeral_1h_tokens': ephemeral_1h_tokens,

        # Source tracking
        'jsonl_path': str(jsonl_path),
        'extracted_at': datetime.now().isoformat(),
    }


def find_jsonl_files(days: int = None, include_old: bool = True) -> list[Path]:
    """Find JSONL files to process."""
    files = []

    # Current Claude projects
    if CLAUDE_PROJECTS.exists():
        for jsonl in CLAUDE_PROJECTS.rglob('*.jsonl'):
            # Skip agent files for now (sub-conversations)
            if jsonl.name.startswith('agent-'):
                continue
            files.append(jsonl)

    # Old archive
    if include_old and OLD_JSONL_DIR.exists():
        for jsonl in OLD_JSONL_DIR.glob('*.jsonl'):
            files.append(jsonl)

    # Filter by days if specified
    if days:
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_ts = cutoff.timestamp()
        files = [f for f in files if f.stat().st_mtime >= cutoff_ts]

    return files


def extract_metadata(days: int = None, include_old: bool = True, dry_run: bool = False):
    """Main extraction function."""
    print("=" * 60)
    print("CLAUDE CODE METADATA EXTRACTION")
    print("=" * 60)
    print()

    files = find_jsonl_files(days=days, include_old=include_old)
    print(f"Found {len(files)} JSONL files to process")

    if days:
        print(f"  (filtered to last {days} days)")

    if dry_run:
        print("\n## DRY RUN - showing sample files")
        for f in files[:10]:
            print(f"  {f.name}")
        print(f"  ... and {len(files) - 10} more")
        return

    # Process files
    results = []
    errors = 0
    empty = 0

    print(f"\nProcessing {len(files)} files...")

    for i, jsonl in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(files)}...")

        try:
            metadata = process_jsonl_file(jsonl)
            if metadata:
                results.append(metadata)
            else:
                empty += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error processing {jsonl.name}: {e}")

    print(f"\nExtraction complete:")
    print(f"  Conversations with metadata: {len(results)}")
    print(f"  Empty/skipped: {empty}")
    print(f"  Errors: {errors}")

    if not results:
        print("No metadata extracted.")
        return

    # Convert to parquet
    try:
        import duckdb
        import pandas as pd

        df = pd.DataFrame(results)

        # Convert lists to JSON strings for parquet compatibility
        list_cols = ['tool_counts', 'unique_tools', 'web_searches',
                     'files_read', 'files_written', 'files_edited',
                     'files_searched', 'bash_commands', 'models_used',
                     'thinking_blocks', 'todos_snapshots', 'stop_reason_counts',
                     'agent_ids', 'tool_errors', 'request_ids']

        for col in list_cols:
            if col in df.columns:
                df[col] = df[col].apply(json.dumps)

        # Save to parquet
        df.to_parquet(METADATA_PARQUET, index=False)
        print(f"\nSaved to: {METADATA_PARQUET}")
        print(f"  Rows: {len(df)}")
        print(f"  Size: {METADATA_PARQUET.stat().st_size / 1024 / 1024:.1f} MB")

        # Summary stats
        print(f"\n=== Summary Stats ===")
        print(f"Total tool calls: {df['total_tool_calls'].sum():,}")
        print(f"Total web searches: {sum(len(json.loads(x)) for x in df['web_searches']):,}")
        print(f"Total files read: {sum(len(json.loads(x)) for x in df['files_read']):,}")
        print(f"Total files written: {sum(len(json.loads(x)) for x in df['files_written']):,}")
        print(f"Total input tokens: {df['total_input_tokens'].sum():,}")
        print(f"Total output tokens: {df['total_output_tokens'].sum():,}")

    except ImportError as e:
        print(f"Error: {e}")
        print("Install pandas: pip install pandas")


def show_stats():
    """Show stats from existing metadata parquet."""
    if not METADATA_PARQUET.exists():
        print(f"No metadata file found at {METADATA_PARQUET}")
        print("Run: python -m pipelines extract-metadata")
        return

    import duckdb
    con = duckdb.connect()

    print("=" * 60)
    print("CLAUDE CODE METADATA STATS")
    print("=" * 60)

    df = con.execute(f"SELECT * FROM read_parquet('{METADATA_PARQUET}')").fetchdf()

    print(f"\nTotal conversations: {len(df)}")
    # Filter out null timestamps for date range
    valid_timestamps = df['first_timestamp'].dropna()
    if len(valid_timestamps) > 0:
        print(f"Date range: {valid_timestamps.min()} to {df['last_timestamp'].dropna().max()}")

    print(f"\n=== Token Usage ===")
    print(f"Total input tokens: {df['total_input_tokens'].sum():,}")
    print(f"Total output tokens: {df['total_output_tokens'].sum():,}")
    print(f"Total cache creation: {df['total_cache_creation_tokens'].sum():,}")

    print(f"\n=== Tool Usage ===")
    print(f"Total tool calls: {df['total_tool_calls'].sum():,}")

    # Aggregate tool counts
    import json
    all_tools = defaultdict(int)
    for tc in df['tool_counts']:
        counts = json.loads(tc)
        for tool, count in counts.items():
            all_tools[tool] += count

    print("\nTop 20 tools:")
    for tool, count in sorted(all_tools.items(), key=lambda x: -x[1])[:20]:
        print(f"  {count:6,} {tool}")

    print(f"\n=== File Operations ===")
    total_read = sum(len(json.loads(x)) for x in df['files_read'])
    total_written = sum(len(json.loads(x)) for x in df['files_written'])
    total_edited = sum(len(json.loads(x)) for x in df['files_edited'])
    print(f"Files read: {total_read:,}")
    print(f"Files written: {total_written:,}")
    print(f"Files edited: {total_edited:,}")

    print(f"\n=== Web Searches ===")
    all_searches = []
    for ws in df['web_searches']:
        all_searches.extend(json.loads(ws))
    print(f"Total searches: {len(all_searches):,}")
    print(f"Unique searches: {len(set(all_searches)):,}")

    print(f"\n=== Agent Delegation ===")
    if 'agent_count' in df.columns:
        total_agents = df['agent_count'].sum()
        convos_with_agents = (df['agent_count'] > 0).sum()
        print(f"Conversations with agents: {convos_with_agents:,}")
        print(f"Total agent delegations: {total_agents:,}")
    if 'sidechain_count' in df.columns:
        print(f"Total sidechain messages: {df['sidechain_count'].sum():,}")

    print(f"\n=== Tool Errors ===")
    if 'tool_error_count' in df.columns:
        total_errors = df['tool_error_count'].sum()
        convos_with_errors = (df['tool_error_count'] > 0).sum()
        print(f"Conversations with errors: {convos_with_errors:,}")
        print(f"Total tool errors: {total_errors:,}")

    print(f"\n=== Extended Thinking ===")
    if 'thinking_count' in df.columns:
        total_thinking = df['thinking_count'].sum()
        convos_with_thinking = (df['thinking_count'] > 0).sum()
        print(f"Conversations with thinking: {convos_with_thinking:,}")
        print(f"Total thinking blocks: {total_thinking:,}")

    print(f"\n=== Ephemeral Tokens ===")
    if 'ephemeral_5m_tokens' in df.columns:
        print(f"Ephemeral 5m tokens: {df['ephemeral_5m_tokens'].sum():,}")
    if 'ephemeral_1h_tokens' in df.columns:
        print(f"Ephemeral 1h tokens: {df['ephemeral_1h_tokens'].sum():,}")

    print(f"\n=== Conversation Metadata ===")
    if 'title' in df.columns:
        with_title = df['title'].notna().sum()
        print(f"Conversations with title: {with_title:,}")
    if 'summary' in df.columns:
        with_summary = df['summary'].notna().sum()
        print(f"Conversations with summary: {with_summary:,}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract Claude Code metadata")
    parser.add_argument('--days', type=int, help="Only process files from last N days")
    parser.add_argument('--all', action='store_true', help="Include old archive")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be processed")
    parser.add_argument('--stats', action='store_true', help="Show stats from existing file")

    args = parser.parse_args()

    if args.stats:
        show_stats()
    else:
        extract_metadata(
            days=args.days,
            include_old=args.all,
            dry_run=args.dry_run
        )
