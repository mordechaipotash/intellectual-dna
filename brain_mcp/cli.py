#!/usr/bin/env python3
"""brain-mcp — CLI for brain-mcp."""

import argparse
import json
import sys
from pathlib import Path

from brain_mcp.platform import app_data_dir, config_dir as _platform_config_dir, claude_desktop_config, claude_desktop_conversations, cursor_data_dir, cursor_vscdb_paths

DEFAULT_CONFIG_DIR = _platform_config_dir()
DEFAULT_CONFIG_PATH_TOML = DEFAULT_CONFIG_DIR / "config.toml"
DEFAULT_CONFIG_PATH_YAML = DEFAULT_CONFIG_DIR / "brain.yaml"
# Prefer TOML if it exists, fall back to YAML
DEFAULT_CONFIG_PATH = (
    DEFAULT_CONFIG_PATH_TOML if DEFAULT_CONFIG_PATH_TOML.exists()
    else DEFAULT_CONFIG_PATH_YAML
)


def _has_config() -> bool:
    """Check if brain-mcp has been configured."""
    return DEFAULT_CONFIG_PATH_TOML.exists() or DEFAULT_CONFIG_PATH_YAML.exists()


def discover_sources():
    """Auto-discover conversation sources on this machine."""
    from rich.console import Console
    console = Console()
    sources = []

    checks = [
        ("Claude Code", "claude-code", Path.home() / ".claude" / "projects", "jsonl"),
        ("Claude Desktop", "claude-desktop", claude_desktop_conversations(), "jsonl"),
        ("Clawdbot", "clawdbot", Path.home() / ".clawdbot" / "agents", "jsonl"),
    ]

    console.print("\n[bold]Discovering AI conversations...[/bold]\n")

    for name, source_type, path, fmt in checks:
        if path.exists():
            # Count files
            if fmt == "jsonl":
                files = list(path.rglob("*.jsonl"))
                count = len(files)
            else:
                count = 1

            if count > 0:
                console.print(f"   [green]found[/green] {name:<16} {count:>6} sessions    {path}")
                sources.append({
                    "type": source_type,
                    "path": str(path),
                    "name": name,
                })
            else:
                console.print(f"   [dim]--  {name:<16} empty           {path}[/dim]")
        else:
            console.print(f"   [dim]--  {name:<16} not found[/dim]")

    # Check for ChatGPT exports in Downloads
    downloads = Path.home() / "Downloads"
    if downloads.exists():
        chatgpt_files = list(downloads.glob("chatgpt*.json")) + list(downloads.glob("conversations.json"))
        # Also check subdirectories one level deep
        for d in downloads.iterdir():
            if d.is_dir() and "chatgpt" in d.name.lower():
                chatgpt_files.extend(d.glob("conversations.json"))

        if chatgpt_files:
            console.print(f"   [green]found[/green] {'ChatGPT':<16} {len(chatgpt_files):>6} exports     {chatgpt_files[0].parent}")
            sources.append({
                "type": "chatgpt",
                "path": str(chatgpt_files[0].parent),
                "name": "ChatGPT",
            })
        else:
            console.print(f"   [dim]--  {'ChatGPT':<16} no exports in ~/Downloads[/dim]")

    # Check Cursor
    cursor_path = Path.home() / ".cursor"
    if cursor_path.exists():
        # Count Cursor data sources
        cursor_count = 0
        for vscdb_path in cursor_vscdb_paths():
            if vscdb_path.exists():
                cursor_count += 1
        # Also check agent transcripts
        for proj_dir in (Path.home() / ".cursor" / "projects").glob("*"):
            transcripts = proj_dir / "agent-transcripts"
            if transcripts.exists():
                cursor_count += len(list(transcripts.glob("*.jsonl")))

        if cursor_count > 0:
            console.print(f"   [green]found[/green] {'Cursor':<16} {cursor_count:>6} sources     {cursor_path}")
            sources.append({
                "type": "cursor",
                "path": str(cursor_path),
                "name": "Cursor",
            })
        else:
            console.print(f"   [dim]--  {'Cursor':<16} installed but no data found[/dim]")

    # Check Windsurf
    windsurf_path = Path.home() / ".windsurf"
    if windsurf_path.exists():
        console.print(f"   [yellow]!![/yellow]  {'Windsurf':<16} found but ingester not yet supported")

    console.print()
    return sources


def create_config(sources, config_dir=None):
    """Create config.toml (preferred) or brain.yaml (fallback) from discovered sources."""
    config_dir = config_dir or DEFAULT_CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)

    data_dir = config_dir / "data"
    vectors_dir = config_dir / "vectors"
    data_dir.mkdir(exist_ok=True)
    vectors_dir.mkdir(exist_ok=True)

    config = {
        "data_dir": str(data_dir),
        "vectors_dir": str(vectors_dir),
        "sources": sources,
        "embedding": {
            "model": "nomic-ai/nomic-embed-text-v1.5",
            "dim": 768,
            "batch_size": 50,
            "max_chars": 8000,
        },
        "summarizer": {
            "enabled": False,
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "api_key_env": "ANTHROPIC_API_KEY",
        },
        "domains": [
            "ai-dev", "backend-dev", "frontend-dev", "data-engineering",
            "devops", "database", "python", "web-scraping", "automation",
            "prompt-engineering", "documentation", "business-strategy",
            "career", "finance", "personal", "health", "education",
        ],
    }

    # Try TOML first (Python 3.11+), fall back to YAML
    try:
        import tomli_w
        config_path = config_dir / "config.toml"
        with open(config_path, "wb") as f:
            tomli_w.dump(config, f)
        return config_path
    except ImportError:
        pass

    # Fallback: YAML
    import yaml
    config_path = config_dir / "brain.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config_path


def _auto_detect_mcp_clients():
    """Auto-detect installed MCP clients. Returns list of (label, config_path) tuples."""
    clients = []

    # Claude Desktop
    desktop_path = claude_desktop_config()
    if desktop_path.parent.exists():
        clients.append(("Claude Desktop", desktop_path))

    # Claude Code
    code_path = Path.home() / ".claude.json"
    if code_path.exists() or (Path.home() / ".claude").exists():
        clients.append(("Claude Code", code_path))

    # Cursor
    cursor_path = Path.home() / ".cursor" / "mcp.json"
    if cursor_path.parent.exists():
        clients.append(("Cursor", cursor_path))

    # Windsurf
    windsurf_path = Path.home() / ".windsurf" / "mcp.json"
    if windsurf_path.parent.exists():
        clients.append(("Windsurf", windsurf_path))

    return clients


def _run_test_query(console):
    """Run a test query to verify the brain works. Returns (success, msg_count)."""
    try:
        from brain_mcp.config import get_config
        cfg = get_config()
        if not cfg.parquet_path.exists():
            return False, 0

        import duckdb
        con = duckdb.connect()
        con.execute(f"""
            CREATE VIEW conversations
            AS SELECT * FROM read_parquet('{cfg.parquet_path}')
        """)

        # Get 3 recent user messages as proof
        results = con.execute("""
            SELECT substr(content, 1, 120) as preview,
                   conversation_title, source
            FROM conversations
            WHERE role = 'user' AND length(content) > 20
            ORDER BY created DESC
            LIMIT 3
        """).fetchall()

        if results:
            console.print("\n[bold]📋 Sample from your brain:[/bold]\n")
            for preview, title, source in results:
                preview_clean = preview.replace('\n', ' ').strip()
                console.print(f'   [dim]{source}[/dim] "{preview_clean}..."')

        con.close()
        return True, len(results)
    except Exception as e:
        console.print(f"   [yellow]Test query failed: {e}[/yellow]")
        return False, 0


def cmd_init(args):
    """Discover sources and create config."""
    from rich.console import Console
    console = Console()

    sources = discover_sources()

    if not sources:
        console.print("[yellow]No conversation sources found.[/yellow]\n")
        console.print("💡 [bold]ChatGPT[/bold]: Go to Settings → Data Controls → Export in ChatGPT,")
        console.print("   download the zip, extract it to ~/Downloads, then run [cyan]brain-mcp init[/cyan] again.\n")
        console.print("💡 [bold]Claude Code[/bold]: Sessions are auto-detected from ~/.claude/projects/\n")
        console.print("You can also manually add sources to config.toml.")
        console.print("Supported: Claude Code, ChatGPT export, Claude Desktop, Clawdbot, Cursor, Gemini CLI")

    config_path = create_config(sources)
    total = sum(1 for s in sources)
    console.print(f"[bold green]Config saved to {config_path}[/bold green]")
    console.print(f"Found {total} source(s).\n")

    if args.full and sources:
        cmd_ingest(args)
        try:
            cmd_embed(args)
        except Exception as e:
            if "fastembed" in str(e) or "No module named" in str(e) or "No embedding backend" in str(e):
                console.print("\n[yellow]Embedding skipped — no embedding backend installed.[/yellow]")
                console.print("   Keyword search works now. For semantic search:")
                import os as _os
                if "pipx" in (_os.environ.get("VIRTUAL_ENV", "") + sys.prefix):
                    console.print("   pipx inject brain-mcp fastembed")
                else:
                    console.print("   pip install 'brain-mcp[embed]'")
                console.print("   brain-mcp embed")
            else:
                raise
        console.print("\n[bold green]Your brain is ready![/bold green]\n")
        console.print("Next steps:")
        console.print("   brain-mcp setup claude          Connect to Claude Desktop + Code")
        console.print("   brain-mcp setup claude-desktop   Claude Desktop only")
        console.print("   brain-mcp setup claude-code      Claude Code only")
        console.print("   brain-mcp setup cursor           Connect to Cursor")
        console.print("   brain-mcp setup windsurf         Connect to Windsurf")
        console.print("   brain-mcp serve                  Start MCP server manually")
    elif sources:
        console.print("Next steps:")
        console.print("   brain-mcp init --full     Import everything now")
        console.print("   brain-mcp ingest          Import without embedding")


def cmd_ingest(args):
    """Import conversations from configured sources."""
    from rich.console import Console
    console = Console()

    from brain_mcp.config import load_config, set_config, get_config
    config_path = getattr(args, 'config', None) or DEFAULT_CONFIG_PATH
    if config_path and Path(config_path).exists():
        set_config(load_config(str(config_path)))

    cfg = get_config()
    console.print("\n[bold]Importing conversations...[/bold]\n")

    # Run ingest pipeline
    from brain_mcp.ingest import run_all_ingesters
    total = run_all_ingesters(cfg)

    console.print(f"\n[green]Total: {total:,} messages imported[/green]\n")
    return total


def cmd_embed(args):
    """Create/update vector embeddings."""
    from rich.console import Console
    console = Console()

    from brain_mcp.config import load_config, set_config, get_config
    config_path = getattr(args, 'config', None) or DEFAULT_CONFIG_PATH
    if config_path and Path(config_path).exists():
        set_config(load_config(str(config_path)))

    rebuild = getattr(args, 'rebuild', False)
    force_provider = getattr(args, 'provider', None)

    # Show which provider will be used
    try:
        from brain_mcp.embed.provider import _detect_available_provider
        if force_provider:
            console.print(f"[dim]Provider: {force_provider} (forced)[/dim]")
        else:
            detected = _detect_available_provider()
            console.print(f"[dim]Provider: {detected} (auto-detected)[/dim]")
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        return

    console.print("\n[bold]Creating embeddings...[/bold]\n")

    from brain_mcp.embed.embed import embed_messages
    embed_messages(rebuild=rebuild, force_provider=force_provider)

    console.print("\n[green]Embedding complete.[/green]\n")


def _incremental_sync(cfg, stderr_print):
    """Check for new files and do incremental ingest if needed. Returns count of new sessions or 0."""
    import os

    parquet_path = cfg.parquet_path
    if not parquet_path.exists():
        return 0

    parquet_mtime = parquet_path.stat().st_mtime

    # Check each source for files newer than parquet
    new_count = 0
    for source in (cfg.sources or []):
        source_path = Path(source.path if hasattr(source, 'path') else source.get("path", ""))
        if not source_path.exists():
            continue
        for f in source_path.rglob("*.jsonl"):
            try:
                if f.stat().st_mtime > parquet_mtime:
                    new_count += 1
            except OSError:
                continue

    if new_count == 0:
        return 0

    stderr_print(f"🔄 Found {new_count} new session{'s' if new_count != 1 else ''}, syncing...")

    try:
        import concurrent.futures
        from brain_mcp.ingest import run_all_ingesters
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_all_ingesters, cfg)
            future.result(timeout=30)  # Cap sync at 30 seconds on startup
        stderr_print(f"✅ Sync complete.")
    except concurrent.futures.TimeoutError:
        stderr_print(f"⚠️  Sync taking too long, continuing with existing data. Run 'brain-mcp sync' later.")
    except Exception as e:
        stderr_print(f"⚠️  Incremental sync failed: {e}")

    return new_count


def cmd_serve(args):
    """Start the MCP server."""
    import sys as _sys
    from brain_mcp.config import load_config, set_config, get_config
    from brain_mcp.telemetry import track
    config_path = getattr(args, 'config', None) or DEFAULT_CONFIG_PATH
    if config_path and Path(config_path).exists():
        set_config(load_config(str(config_path)))

    cfg = get_config()

    def stderr_print(msg):
        print(msg, file=_sys.stderr, flush=True)

    # Auto-sync: check for new files before starting
    _incremental_sync(cfg, stderr_print)

    # Show startup message on stderr (stdout is for MCP stdio transport)
    msg_count = 0
    if cfg.parquet_path.exists():
        try:
            import duckdb
            con = duckdb.connect()
            msg_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{cfg.parquet_path}')").fetchone()[0]
            con.close()
        except Exception:
            pass
    stderr_print(f"brain-mcp MCP server running on stdio. Messages: {msg_count:,}.")
    track("serve_started", {"messages": msg_count})

    from brain_mcp.server.server import create_server
    mcp = create_server()
    mcp.run()


def _write_mcp_config(config_file, console, label=None):
    """Write brain-mcp entry to an MCP config file."""
    import shutil

    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    else:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config = {}

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Use "my-brain" as key (not "brain") to avoid colliding with
    # users who may have a personal brain server already configured
    server_key = "my-brain"

    if server_key in config["mcpServers"]:
        console.print(f"   [green]✓[/green] {label or config_file} [dim](already configured)[/dim]")
        return

    # Use full path to brain-mcp so the client can find it
    brain_cmd = shutil.which("brain-mcp") or "brain-mcp"

    config["mcpServers"][server_key] = {
        "command": brain_cmd,
        "args": ["serve"],
    }

    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    display = label or config_file
    console.print(f"   [green]✓[/green] {display}  [dim]({config_file})[/dim]")


def cmd_setup(args):
    """Setup wizard or configure a specific MCP client."""
    client = getattr(args, 'client', None)

    if client:
        # Specific client setup — original behavior
        _setup_single_client(args)
    else:
        # Unified setup wizard
        _setup_wizard(args)


def _setup_wizard(args):
    """Unified one-command setup wizard."""
    import os
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from brain_mcp.telemetry import track, maybe_show_notice
    console = Console()

    maybe_show_notice()
    track("setup_started")

    console.print("\n[bold]🧠 Setting up brain-mcp...[/bold]\n")

    # ── Step 1: Discover sources ──
    sources = discover_sources()

    if not sources:
        console.print("[yellow]No conversation sources found.[/yellow]\n")
        console.print("💡 Start using Claude Code or Claude Desktop, then re-run [cyan]brain-mcp setup[/cyan]")
        console.print("💡 Or add sources manually to [cyan]~/.config/brain-mcp/config.toml[/cyan] and run [cyan]brain-mcp sync[/cyan]\n")
        return

    # ── Step 2: Create config ──
    config_path = create_config(sources)
    console.print(f"[dim]Config saved to {config_path}[/dim]\n")

    # Load config for subsequent steps
    from brain_mcp.config import load_config, set_config, get_config
    set_config(load_config(str(config_path)))
    cfg = get_config()

    # ── Step 3: Ingest ──
    console.print("[bold]📥 Importing conversations...[/bold]\n")
    from brain_mcp.ingest import run_all_ingesters
    total_messages = run_all_ingesters(cfg)
    console.print(f"\n[green]   {total_messages:,} messages imported[/green]\n")

    # ── Step 4: Embed ──
    total_embeddings = 0
    if total_messages > 0:
        # Check if fastembed is available; if not, auto-install it
        fastembed_available = False
        try:
            import fastembed  # noqa: F401
            fastembed_available = True
        except ImportError:
            console.print("[bold]🔮 Installing embedding model...[/bold]\n")
            console.print("   [dim]This is a one-time download (~107MB). Future runs skip this.[/dim]\n")
            try:
                import subprocess
                # Detect if running in pipx or regular pip
                in_pipx = "pipx" in (os.environ.get("VIRTUAL_ENV", "") + sys.prefix)
                if in_pipx:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "fastembed>=0.5"],
                        capture_output=True, text=True, timeout=300
                    )
                else:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "fastembed>=0.5"],
                        capture_output=True, text=True, timeout=300
                    )
                if result.returncode == 0:
                    console.print("   [green]✓ Embedding model installed[/green]\n")
                    fastembed_available = True
                else:
                    console.print(f"   [yellow]⚠️  Could not install embedding model[/yellow]")
                    console.print(f"   [dim]Install manually: pipx inject brain-mcp fastembed[/dim]\n")
            except Exception as e:
                console.print(f"   [yellow]⚠️  Could not install embedding model: {e}[/yellow]")
                console.print(f"   [dim]Install manually: pipx inject brain-mcp fastembed[/dim]\n")

        if fastembed_available:
            console.print("[bold]🔮 Creating embeddings...[/bold]\n")
            try:
                from brain_mcp.embed.embed import embed_messages
                embed_messages()
                # Count embeddings
                try:
                    import lancedb
                    db = lancedb.connect(str(cfg.lance_path))
                    tables = db.table_names() if hasattr(db, "table_names") else list(db.list_tables())
                    for t_name in ("message", "brain"):
                        if t_name in tables:
                            total_embeddings = db.open_table(t_name).count_rows()
                            break
                except Exception:
                    pass
                console.print(f"\n[green]   {total_embeddings:,} embeddings created[/green]\n")
            except Exception as e:
                console.print(f"\n[yellow]   Embedding skipped: {e}[/yellow]")
                console.print("   Keyword search works now. Run [cyan]brain-mcp embed[/cyan] later for semantic search.\n")
        else:
            console.print("[dim]   Skipping embeddings — keyword search works without them.[/dim]")
            console.print("   [dim]For semantic search later: pipx inject brain-mcp fastembed && brain-mcp embed[/dim]\n")

    # ── Step 5: Auto-detect and configure MCP clients ──
    console.print("[bold]🔌 Configuring MCP clients...[/bold]\n")
    clients = _auto_detect_mcp_clients()
    configured_names = []

    if clients:
        for label, client_path in clients:
            try:
                _write_mcp_config(client_path, console, label)
                configured_names.append(label)
            except Exception as e:
                console.print(f"   [yellow]⚠️  {label}: {e}[/yellow]")
    else:
        console.print("   [dim]No MCP clients detected[/dim]")

    console.print()

    # ── Step 6: Test query ──
    if total_messages > 0:
        _run_test_query(console)

    # ── Step 7: Summary ──
    console.print()
    configured_str = ", ".join(configured_names) if configured_names else "none"
    console.print(f"[bold green]✅ Your brain is ready![/bold green] "
                  f"{total_messages:,} messages indexed, {total_embeddings:,} embeddings created. "
                  f"Configured: {configured_str}.")
    console.print()
    console.print("   Open Claude and ask: [italic]'what was I working on last week?'[/italic]")
    console.print()

    track("setup_completed", {
        "messages": total_messages,
        "embeddings": total_embeddings,
        "sources": len(sources),
        "clients": configured_names,
    })


def _setup_single_client(args):
    """Auto-configure a specific MCP client."""
    from rich.console import Console
    console = Console()

    client = args.client

    # Determine config file path
    if client in ("claude-desktop", "desktop"):
        config_file = claude_desktop_config()
    elif client in ("claude-code", "code"):
        config_file = Path.home() / ".claude.json"
    elif client == "claude":
        # Auto-detect: set up BOTH if they exist, else whichever is found
        desktop = claude_desktop_config()
        code = Path.home() / ".claude.json"

        targets = []
        if desktop.parent.exists():
            targets.append(("Claude Desktop", desktop))
        if code.parent.exists():
            targets.append(("Claude Code", code))

        if not targets:
            # Neither installed — create Claude Code config (most common)
            targets.append(("Claude Code", code))

        for label, target_file in targets:
            _write_mcp_config(target_file, console, label)

        return
    elif client == "cursor":
        config_file = Path.home() / ".cursor" / "mcp.json"
    elif client == "windsurf":
        config_file = Path.home() / ".windsurf" / "mcp.json"
    else:
        console.print(f"[red]Unknown client: {client}[/red]")
        console.print("Supported: claude, claude-desktop, claude-code, cursor, windsurf")
        return

    _write_mcp_config(config_file, console)
    console.print(f"\nRestart {client.replace('-', ' ').title()} to activate.")


def cmd_doctor(args):
    """Health check."""
    from rich.console import Console
    console = Console()

    console.print("\n[bold]Brain MCP Health Check[/bold]\n")

    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    console.print(f"   [green]ok[/green] Python {py_ver}")

    # Config
    if DEFAULT_CONFIG_PATH.exists():
        console.print(f"   [green]ok[/green] Config: {DEFAULT_CONFIG_PATH}")
    else:
        console.print(f"   [red]missing[/red] Config: not found")
        console.print(f"      -> Run: brain-mcp init")
        return

    # Load config
    from brain_mcp.config import load_config, set_config, get_config
    set_config(load_config(str(DEFAULT_CONFIG_PATH)))
    cfg = get_config()

    # Parquet
    parquet_path = Path(cfg.data_dir) / "all_conversations.parquet"
    if parquet_path.exists():
        import duckdb
        con = duckdb.connect()
        count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet_path}')").fetchone()[0]
        con.close()
        console.print(f"   [green]ok[/green] Parquet: {count:,} messages")
    else:
        console.print(f"   [red]missing[/red] Parquet: not found")
        console.print(f"      -> Run: brain-mcp ingest")

    # Vectors
    vectors_path = Path(cfg.vectors_dir) / "brain.lance"
    if vectors_path.exists():
        import lancedb
        db = lancedb.connect(str(cfg.lance_path))
        try:
            tables = db.table_names() if hasattr(db, "table_names") else list(db.list_tables())
            vec_count = 0
            for t_name in ("message", "brain"):
                if t_name in tables:
                    vec_count = len(db.open_table(t_name))
                    break
            console.print(f"   [green]ok[/green] Vectors: {vec_count:,} embeddings")
        except Exception:
            console.print(f"   [yellow]warn[/yellow]  Vectors: directory exists but no table")
    else:
        console.print(f"   [yellow]warn[/yellow]  Vectors: not found")
        console.print(f"      -> Run: brain-mcp embed")

    # Embedding provider check — with timeout
    import concurrent.futures
    def _check_embedding():
        try:
            from brain_mcp.embed.provider import _detect_available_provider
            return _detect_available_provider()
        except ImportError:
            return None

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_check_embedding)
            provider = future.result(timeout=5)
            if provider:
                console.print(f"   [green]ok[/green] Embedding: {provider}")
            else:
                console.print(f"   [yellow]warn[/yellow]  Embedding: no backend found")
    except concurrent.futures.TimeoutError:
        console.print(f"   [yellow]⚠️[/yellow]  Embedding check timed out")
    except Exception as e:
        console.print(f"   [yellow]warn[/yellow]  Embedding: {e}")

    # Summaries
    summaries_path = Path(cfg.data_dir) / "brain_summaries_v6.parquet"
    if summaries_path.exists():
        console.print(f"   [green]ok[/green] Summaries: available (prosthetic tools in full mode)")
    else:
        console.print(f"   [yellow]warn[/yellow]  Summaries: not generated (prosthetic tools in basic mode)")
        console.print(f"      -> Run: brain-mcp summarize (requires ANTHROPIC_API_KEY)")

    # MCP client configs
    for name, path in [
        ("Claude Desktop", claude_desktop_config()),
        ("Claude Code", Path.home() / ".claude.json"),
        ("Cursor", Path.home() / ".cursor" / "mcp.json"),
        ("Windsurf", Path.home() / ".windsurf" / "mcp.json"),
    ]:
        if path.exists():
            with open(path) as f:
                try:
                    cfg_json = json.load(f)
                    if "my-brain" in cfg_json.get("mcpServers", {}):
                        console.print(f"   [green]ok[/green] {name}: configured")
                    else:
                        console.print(f"   [dim]--  {name}: not configured[/dim]")
                        setup_name = name.lower().replace(" ", "-")
                        console.print(f"      -> Run: brain-mcp setup {setup_name}")
                except json.JSONDecodeError:
                    console.print(f"   [yellow]warn[/yellow]  {name}: config exists but invalid JSON")
        else:
            console.print(f"   [dim]--  {name}: not installed[/dim]")

    console.print()


def cmd_status(args):
    """Quick status one-liner."""
    from brain_mcp.config import load_config, set_config, get_config

    if not DEFAULT_CONFIG_PATH.exists():
        print("Brain not initialized. Run: brain-mcp init")
        return

    set_config(load_config(str(DEFAULT_CONFIG_PATH)))
    cfg = get_config()

    parquet_path = Path(cfg.data_dir) / "all_conversations.parquet"
    msg_count = 0
    if parquet_path.exists():
        import duckdb
        con = duckdb.connect()
        msg_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet_path}')").fetchone()[0]
        con.close()

    vec_count = 0
    lance_path = cfg.lance_path
    if lance_path.exists():
        try:
            import lancedb
            db = lancedb.connect(str(lance_path))
            tables = db.table_names() if hasattr(db, "table_names") else list(db.list_tables())
            for t_name in ("message", "brain"):
                if t_name in tables:
                    vec_count = len(db.open_table(t_name))
                    break
        except Exception:
            pass

    sources = len(cfg.sources) if cfg.sources else 0
    print(f"Brain: {msg_count:,} messages | {vec_count:,} vectors | {sources} sources")


def _smart_status():
    """Show smart status dashboard when brain is already configured."""
    from rich.console import Console
    console = Console()

    try:
        from importlib.metadata import version as _get_version
        ver = _get_version("brain-mcp")
    except Exception:
        try:
            from brain_mcp import __version__ as ver
        except Exception:
            ver = "unknown"

    from brain_mcp.config import load_config, set_config, get_config
    set_config(load_config(str(DEFAULT_CONFIG_PATH)))
    cfg = get_config()

    msg_count = 0
    parquet_path = cfg.parquet_path
    last_sync = None
    if parquet_path.exists():
        try:
            import duckdb
            con = duckdb.connect()
            msg_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet_path}')").fetchone()[0]
            con.close()
            # Last sync = parquet modification time
            mtime = parquet_path.stat().st_mtime
            import datetime
            delta = datetime.datetime.now() - datetime.datetime.fromtimestamp(mtime)
            if delta.total_seconds() < 3600:
                last_sync = f"{int(delta.total_seconds() / 60)} minutes ago"
            elif delta.total_seconds() < 86400:
                last_sync = f"{int(delta.total_seconds() / 3600)} hours ago"
            else:
                last_sync = f"{int(delta.total_seconds() / 86400)} days ago"
        except Exception:
            pass

    vec_count = 0
    if cfg.lance_path.exists():
        try:
            import lancedb
            db = lancedb.connect(str(cfg.lance_path))
            tables = db.table_names() if hasattr(db, "table_names") else list(db.list_tables())
            for t_name in ("message", "brain"):
                if t_name in tables:
                    tbl = db.open_table(t_name)
                    vec_count = tbl.count_rows()
                    break
        except Exception:
            pass

    sources = len(cfg.sources) if cfg.sources else 0

    console.print(f"\n[bold]🧠 brain-mcp v{ver}[/bold]")
    stats_line = f"   Messages: {msg_count:,} | Embeddings: {vec_count:,} | Sources: {sources}"
    console.print(stats_line)
    if last_sync:
        console.print(f"   Last sync: {last_sync}")
    console.print()
    console.print("   [dim]Commands: setup, sync, doctor, serve[/dim]")
    console.print()


def cmd_version(args):
    """Print version."""
    try:
        from importlib.metadata import version as _get_version
        ver = _get_version("brain-mcp")
    except Exception:
        try:
            from brain_mcp import __version__ as ver
        except Exception:
            ver = "unknown"
    print(f"brain-mcp {ver}")


def cmd_summarize(args):
    """Generate structured conversation summaries."""
    from rich.console import Console
    console = Console()

    from brain_mcp.config import load_config, set_config, get_config
    config_path = getattr(args, 'config', None) or DEFAULT_CONFIG_PATH
    if config_path and Path(config_path).exists():
        set_config(load_config(str(config_path)))

    cfg = get_config()

    if not cfg.summarizer.enabled:
        console.print("\n[yellow]Summarization is not configured.[/yellow]\n")
        console.print("Summaries extract decisions, open questions, and key quotes")
        console.print("from each conversation. Powers the 8 prosthetic tools.\n")
        console.print("To enable, add this to your config.toml:")
        console.print('  [summarizer]')
        console.print('  enabled = true')
        console.print('  provider = "anthropic"  # or "openai"')
        console.print(f'  api_key_env = "ANTHROPIC_API_KEY"\n')
        console.print("Or configure via the dashboard:")
        console.print("  brain-mcp dashboard\n")
        console.print("[dim]Without summaries, all search + basic prosthetic tools still work.[/dim]")
        return

    try:
        from brain_mcp.summarize.summarize import run_summarizer
        console.print("\n[bold]Generating conversation summaries...[/bold]\n")
        run_summarizer(cfg)
        console.print("\n[green]Summarization complete.[/green]\n")
    except ImportError:
        console.print("[red]Summarizer module not found.[/red]")
        console.print("Install with: pip install brain-mcp[summarize]")
    except Exception as e:
        console.print(f"[red]Summarization error: {e}[/red]")


def cmd_dashboard(args):
    """Start the Brain MCP dashboard."""
    try:
        from brain_mcp.dashboard.app import create_app
        import uvicorn

        port = getattr(args, 'port', 8742) or 8742
        no_open = getattr(args, 'no_open', False)

        from brain_mcp.config import load_config, set_config
        config_path = getattr(args, 'config', None) or DEFAULT_CONFIG_PATH
        if config_path and Path(config_path).exists():
            set_config(load_config(str(config_path)))

        app = create_app()

        if not no_open:
            import threading
            import webbrowser
            threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()

        print(f"🧠 Brain MCP Dashboard → http://localhost:{port}")
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    except ImportError:
        from rich.console import Console
        console = Console()
        console.print("\n[yellow]Dashboard not yet available.[/yellow]")
        console.print("The dashboard is coming in v0.2.0.\n")
        console.print("For now, use the CLI commands:")
        console.print("  brain-mcp setup             One-command setup")
        console.print("  brain-mcp setup claude      Connect to Claude")
        console.print("  brain-mcp doctor            Health check")
        console.print("  brain-mcp status            Quick status\n")


def cmd_sync(args):
    """Incremental sync -- ingest new + embed new."""
    from rich.console import Console
    console = Console()

    console.print("[bold]Syncing...[/bold]")
    cmd_ingest(args)
    cmd_embed(args)
    console.print("[green]Sync complete[/green]")


def main():
    # Import version - try importlib.metadata first (works in pipx), then __init__
    try:
        from importlib.metadata import version as _get_version
        _version = _get_version("brain-mcp")
    except Exception:
        try:
            from brain_mcp import __version__ as _version
        except Exception:
            _version = "unknown"

    parser = argparse.ArgumentParser(
        prog="brain-mcp",
        description="Turn your AI conversations into a searchable second brain",
    )
    parser.add_argument("--version", action="version", version=f"brain-mcp {_version}")
    parser.add_argument("--config", help="Path to config.toml config file")

    sub = parser.add_subparsers(dest="command")

    # setup — unified wizard or per-client
    p_setup = sub.add_parser("setup", help="Setup wizard (or configure a specific MCP client)")
    p_setup.add_argument(
        "client",
        nargs="?",
        default=None,
        choices=["claude", "claude-desktop", "desktop", "claude-code", "code", "cursor", "windsurf"],
        help="Specific client to configure (omit for guided wizard)",
    )

    # init
    p_init = sub.add_parser("init", help="Discover sources and create config")
    p_init.add_argument("--full", action="store_true", help="Also ingest and embed")

    # ingest
    sub.add_parser("ingest", help="Import conversations from configured sources")

    # embed
    embed_parser = sub.add_parser("embed", help="Create/update vector embeddings")
    embed_parser.add_argument("--rebuild", action="store_true", help="Drop and recreate all vectors from scratch")
    embed_parser.add_argument(
        "--provider",
        choices=["fastembed", "sentence-transformers"],
        default=None,
        help="Force a specific embedding provider (default: auto-detect)",
    )

    # serve
    sub.add_parser("serve", help="Start the MCP server")

    # doctor
    sub.add_parser("doctor", help="Health check")

    # status
    sub.add_parser("status", help="Quick status")

    # sync
    sub.add_parser("sync", help="Incremental update")

    # version
    sub.add_parser("version", help="Print version")

    # summarize
    sub.add_parser("summarize", help="Generate conversation summaries")

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Open the web dashboard")
    p_dash.add_argument("--port", type=int, default=8742, help="Port (default: 8742)")
    p_dash.add_argument("--no-open", action="store_true", help="Don't open browser")

    # telemetry
    p_telem = sub.add_parser("telemetry", help="Manage anonymous usage telemetry")
    p_telem.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=["on", "off", "status"],
        help="Enable, disable, or check telemetry status",
    )

    args = parser.parse_args()

    def cmd_telemetry(args):
        from brain_mcp.telemetry import is_enabled, set_enabled
        action = getattr(args, 'action', 'status')
        if action == "on":
            set_enabled(True)
            print("✅ Telemetry enabled. Anonymous usage data will be collected.")
        elif action == "off":
            set_enabled(False)
            print("🔇 Telemetry disabled. No data will be collected.")
        else:
            status = "enabled" if is_enabled() else "disabled"
            print(f"Telemetry: {status}")
            print(f"  Disable: brain-mcp telemetry off")
            print(f"  Enable:  brain-mcp telemetry on")
            print(f"  Details: https://brainmcp.dev/telemetry")

    commands = {
        "init": cmd_init,
        "ingest": cmd_ingest,
        "embed": cmd_embed,
        "serve": cmd_serve,
        "setup": cmd_setup,
        "doctor": cmd_doctor,
        "status": cmd_status,
        "sync": cmd_sync,
        "version": cmd_version,
        "summarize": cmd_summarize,
        "dashboard": cmd_dashboard,
        "telemetry": cmd_telemetry,
    }

    if args.command in commands:
        commands[args.command](args)
    elif args.command is None:
        # No subcommand: smart status if configured, setup wizard if not
        if _has_config():
            _smart_status()
        else:
            # Run setup wizard
            args.client = None
            cmd_setup(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
