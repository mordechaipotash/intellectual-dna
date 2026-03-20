"""Cross-platform path resolution for brain-mcp.

Handles macOS, Linux, and Windows path conventions for:
- brain-mcp config directory
- AI client config files (Claude Desktop, Cursor, Windsurf)
- AI conversation data directories
"""

import os
import sys
from pathlib import Path

__all__ = ["app_data_dir", "config_dir", "claude_desktop_config", "claude_desktop_conversations"]


def app_data_dir(app_name: str) -> Path:
    """Return the platform-appropriate application data directory.

    - macOS:   ~/Library/Application Support/{app_name}
    - Windows: %APPDATA%/{app_name}
    - Linux:   ~/.config/{app_name}
    """
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / app_name
    elif sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / app_name
        return Path.home() / "AppData" / "Roaming" / app_name
    else:
        return Path.home() / ".config" / app_name


def config_dir() -> Path:
    """Return the brain-mcp config directory.
    
    Always uses ~/.config/brain-mcp for backward compatibility across all
    platforms. This is brain-mcp's own convention, not tied to OS conventions.
    """
    return Path.home() / ".config" / "brain-mcp"


def claude_desktop_config() -> Path:
    """Return the Claude Desktop MCP config file path."""
    return app_data_dir("Claude") / "claude_desktop_config.json"


def claude_desktop_conversations() -> Path:
    """Return the Claude Desktop chat conversations directory."""
    return app_data_dir("Claude") / "chat_conversations"


def cursor_data_dir() -> Path:
    """Return the Cursor IDE global storage directory."""
    return app_data_dir("Cursor") / "User" / "globalStorage"


def cursor_vscdb_paths() -> list[Path]:
    """Return all possible Cursor vscdb file paths for the current platform."""
    gdir = cursor_data_dir()
    return [
        gdir / "state.vscdb",
        gdir / "cursor.vscdb",
    ]
