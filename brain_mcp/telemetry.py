"""
brain-mcp — Anonymous telemetry.

Tracks usage patterns to improve the product. No conversation content is
ever sent — only tool names, latencies, error types, and anonymous machine IDs.

Opt-out: brain-mcp telemetry off
Check:   brain-mcp telemetry status

Respects DO_NOT_TRACK=1 environment variable (https://consoledonottrack.com).
"""

import atexit
import hashlib
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

_PROJECT_ID = "xsjyfneizfkbitmzbrta"
_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhzanlmbmVpemZrYml0bXpicnRhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE4MDcxMzUsImV4cCI6MjA2NzM4MzEzNX0."
    "wgrPV43hdznmYGYQIRgH2gxl_mByAS52S3d7vQLgYB8"
)
_ENDPOINT = os.environ.get(
    "BRAIN_MCP_TELEMETRY_URL",
    f"https://{_PROJECT_ID}.supabase.co/rest/v1/telemetry_events",
)
_FLUSH_INTERVAL = 60.0  # seconds between flushes
_FLUSH_THRESHOLD = 10   # flush when buffer hits this size
_MAX_BUFFER = 100       # hard cap to prevent memory leak
_TIMEOUT = 5            # HTTP timeout seconds

# ═══════════════════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════════════════

_buffer: list[dict] = []
_lock = threading.Lock()
_flush_timer: Optional[threading.Timer] = None
_enabled: Optional[bool] = None  # lazy-loaded
_machine_id: Optional[str] = None
_version: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# OPT-IN / OPT-OUT
# ═══════════════════════════════════════════════════════════════════════════════

def _config_path() -> Path:
    """Path to telemetry config file."""
    xdg = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    return Path(xdg) / "brain-mcp" / "telemetry.json"


def is_enabled() -> bool:
    """Check if telemetry is enabled.

    Disabled if:
    - User ran `brain-mcp telemetry off`
    - DO_NOT_TRACK=1 environment variable is set
    - BRAIN_MCP_TELEMETRY=0 environment variable is set
    - CI=true (don't track CI runs)
    """
    global _enabled
    if _enabled is not None:
        return _enabled

    # Environment overrides
    if os.environ.get("DO_NOT_TRACK", "").strip() == "1":
        _enabled = False
        return False
    if os.environ.get("BRAIN_MCP_TELEMETRY", "").strip() == "0":
        _enabled = False
        return False
    if os.environ.get("CI", "").strip().lower() == "true":
        _enabled = False
        return False

    # Check config file
    cfg = _config_path()
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text())
            _enabled = data.get("enabled", True)
        except Exception:
            _enabled = True
    else:
        _enabled = True

    return _enabled


def set_enabled(enabled: bool) -> None:
    """Set telemetry enabled/disabled. Persists to config file."""
    global _enabled
    _enabled = enabled
    cfg = _config_path()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps({"enabled": enabled}, indent=2))


# ═══════════════════════════════════════════════════════════════════════════════
# MACHINE ID
# ═══════════════════════════════════════════════════════════════════════════════

def _get_machine_id() -> str:
    """Generate anonymous machine ID. Cannot be reversed to identify the user.

    Uses sha256(hostname + login + salt) truncated to 16 hex chars.
    """
    global _machine_id
    if _machine_id:
        return _machine_id

    try:
        login = os.getlogin()
    except OSError:
        login = os.environ.get("USER", "unknown")

    raw = f"{platform.node()}-{login}-brain-mcp-v1"
    _machine_id = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return _machine_id


def _get_version() -> str:
    """Get brain-mcp version."""
    global _version
    if _version:
        return _version
    try:
        from brain_mcp import __version__
        _version = __version__
    except Exception:
        _version = "unknown"
    return _version


def _get_os() -> str:
    """Get OS identifier."""
    return f"{platform.system().lower()}-{platform.machine()}"


def _get_python() -> str:
    """Get Python version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


# ═══════════════════════════════════════════════════════════════════════════════
# TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

def track(event: str, props: Optional[dict[str, Any]] = None) -> None:
    """Track an anonymous event. Non-blocking, fire-and-forget.

    Args:
        event: Event name (must be in server allowlist)
        props: Optional properties dict (tool name, latency, etc.)
    """
    if not is_enabled():
        return

    entry = {
        "event": event,
        "props": props or {},
        "version": _get_version(),
        "os": _get_os(),
        "python": _get_python(),
        "ts": time.time(),
    }

    with _lock:
        if len(_buffer) >= _MAX_BUFFER:
            _buffer.pop(0)  # drop oldest
        _buffer.append(entry)

        if len(_buffer) >= _FLUSH_THRESHOLD:
            _schedule_flush(immediate=True)
        elif not _flush_timer:
            _schedule_flush()


def track_tool(tool_name: str, latency_ms: float, result_count: int = 0,
               empty: bool = False) -> None:
    """Convenience: track a tool call."""
    track("tool_called", {
        "tool": tool_name,
        "latency_ms": round(latency_ms, 1),
        "result_count": result_count,
    })
    if empty:
        track("tool_empty_result", {"tool": tool_name})


def track_error(tool_name: str, error_type: str) -> None:
    """Convenience: track a tool error."""
    track("error", {
        "tool": tool_name,
        "error_type": error_type,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# FLUSHING
# ═══════════════════════════════════════════════════════════════════════════════

def _schedule_flush(immediate: bool = False) -> None:
    """Schedule a flush (cancels any pending timer)."""
    global _flush_timer
    if _flush_timer:
        _flush_timer.cancel()
    delay = 0.1 if immediate else _FLUSH_INTERVAL
    _flush_timer = threading.Timer(delay, _do_flush)
    _flush_timer.daemon = True
    _flush_timer.start()


def _do_flush() -> None:
    """Flush buffered events to Supabase. Runs in background thread.

    Uses Supabase REST API (PostgREST) with anon key.
    RLS policy allows insert-only — no reads with anon key.
    """
    global _flush_timer
    _flush_timer = None

    with _lock:
        if not _buffer:
            return
        events = _buffer.copy()
        _buffer.clear()

    machine_id = _get_machine_id()

    # Transform events into Supabase row format
    rows = []
    for e in events:
        rows.append({
            "machine_id": machine_id,
            "event": e["event"],
            "props": e.get("props", {}),
            "version": e.get("version", ""),
            "os": e.get("os", ""),
            "python": e.get("python", ""),
        })

    payload = json.dumps(rows).encode("utf-8")

    try:
        from urllib.request import urlopen, Request
        req = Request(
            _ENDPOINT,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {_ANON_KEY}",
                "apikey": _ANON_KEY,
                "Prefer": "return=minimal",
            },
            method="POST",
        )
        urlopen(req, timeout=_TIMEOUT)
    except Exception:
        pass  # silently drop — never crash the tool for telemetry


def flush() -> None:
    """Force flush (call on shutdown)."""
    _do_flush()


# Register flush on exit so we don't lose buffered events
atexit.register(flush)


# ═══════════════════════════════════════════════════════════════════════════════
# FIRST RUN NOTICE
# ═══════════════════════════════════════════════════════════════════════════════

def maybe_show_notice() -> None:
    """Show telemetry notice on first run. Idempotent."""
    cfg = _config_path()
    if cfg.exists():
        return  # already shown

    # Create config (enabled by default) and show notice
    set_enabled(True)
    print(
        "\n📊 brain-mcp collects anonymous usage statistics (tool popularity, "
        "error rates) to improve the product.\n"
        "   No conversation content is ever sent.\n"
        "   Disable: brain-mcp telemetry off\n"
        "   Details: https://brainmcp.dev/telemetry\n",
        file=sys.stderr,
    )
