#!/usr/bin/env python3
"""
brain-mcp — MCP Server entry point.

A self-hosted MCP server that turns your AI conversations into a
searchable "brain" with 25 tools for searching, synthesizing, and
navigating your intellectual history.

Usage:
    python -m server.server
    # or
    python server/server.py
"""

from mcp.server.fastmcp import FastMCP

from brain_mcp.config import get_config, load_config
from .db import prewarm_async, get_principles


def create_server(config_path: str = None) -> FastMCP:
    """Create and configure the MCP server with all tools registered."""
    if config_path:
        from brain_mcp.config import set_config
        set_config(load_config(config_path))

    cfg = get_config()

    mcp = FastMCP(
        cfg.server_name,
        instructions=cfg.server_instructions,
    )

    # Register all tool modules
    from . import tools_conversations
    from . import tools_search
    from . import tools_synthesis
    from . import tools_stats
    from . import tools_prosthetic
    from . import tools_github
    from . import tools_analytics

    tools_conversations.register(mcp)
    tools_search.register(mcp)
    tools_synthesis.register(mcp)
    tools_stats.register(mcp)
    tools_prosthetic.register(mcp)
    tools_github.register(mcp)
    tools_analytics.register(mcp)

    # Register principles tools (if principles are configured)
    _register_principles(mcp)

    # ── Telemetry: wrap tool calls for latency + usage tracking ──
    _register_tool_telemetry(mcp)

    # Register resources
    @mcp.resource("brain://stats")
    def resource_stats() -> str:
        """Current brain statistics."""
        # Import locally to use the registered tool
        from .db import get_conversations, lance_count
        try:
            con = get_conversations()
            total = con.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            embedded = lance_count("message")
            return f"Messages: {total:,} | Embeddings: {embedded:,}"
        except Exception:
            return "Brain stats unavailable"

    return mcp


def _register_tool_telemetry(mcp):
    """Wrap tool handlers to track latency and usage via telemetry.

    Patches FastMCP's internal tool handler to add timing around each call.
    Falls back silently if the internal API changes.
    """
    try:
        import time as _time
        from brain_mcp.telemetry import track_tool, track_error

        # Get reference to the internal MCP server's call_tool handler
        original_call_tool = mcp._mcp_server.request_handlers.get("tools/call")
        if not original_call_tool:
            return

        async def tracked_call_tool(request):
            tool_name = request.params.name if hasattr(request, 'params') else "unknown"
            start = _time.monotonic()
            try:
                result = await original_call_tool(request)
                elapsed_ms = (_time.monotonic() - start) * 1000
                # Check if result is empty
                text = ""
                if hasattr(result, 'content') and result.content:
                    text = getattr(result.content[0], 'text', '')
                empty = (
                    "No conversations found" in text[:60]
                    or "not found" in text[:60].lower()
                    or len(text) < 30
                )
                track_tool(tool_name, elapsed_ms, empty=empty)
                return result
            except Exception as e:
                elapsed_ms = (_time.monotonic() - start) * 1000
                track_error(tool_name, type(e).__name__)
                raise

        mcp._mcp_server.request_handlers["tools/call"] = tracked_call_tool
    except Exception:
        pass  # never break the server for telemetry


def _register_principles(mcp):
    """Register principles tools if a principles file is configured."""

    @mcp.tool()
    def list_principles() -> str:
        """
        List your configured principles.
        Principles are loaded from the YAML/JSON file specified in config.toml.
        """
        principles = get_principles()

        if not principles:
            return "No principles configured. Add a principles file to config.toml."

        # Support both flat list and nested dict formats
        section = principles.get("principles", principles)

        if isinstance(section, list):
            output = ["## Your Principles\n"]
            for i, p in enumerate(section, 1):
                if isinstance(p, dict):
                    name = p.get("name", f"Principle {i}")
                    desc = p.get("description", p.get("definition", ""))[:200]
                    output.append(f"**{i}. {name}**")
                    output.append(f"> {desc}\n")
                else:
                    output.append(f"**{i}.** {p}\n")
            return "\n".join(output)

        elif isinstance(section, dict):
            output = ["## Your Principles\n"]
            for key in sorted(section.keys()):
                principle = section[key]
                if isinstance(principle, dict):
                    name = principle.get("name", key)
                    definition = principle.get("definition", "")[:200]
                    output.append(f"**{key.upper()}. {name}**")
                    output.append(f"> {definition}\n")
                else:
                    output.append(f"**{key}**: {principle}\n")
            return "\n".join(output)

        return "Principles format not recognized."

    @mcp.tool()
    def get_principle(name: str) -> str:
        """
        Get detailed info about a specific principle.
        """
        principles = get_principles()
        if not principles:
            return "No principles configured."

        section = principles.get("principles", principles)
        name_lower = name.lower()

        if isinstance(section, list):
            for p in section:
                if isinstance(p, dict):
                    p_name = p.get("name", "").lower()
                    if name_lower in p_name:
                        output = [f"## {p.get('name', 'Unknown')}\n"]
                        if "definition" in p:
                            output.append(f"**Definition**: {p['definition']}\n")
                        if "description" in p:
                            output.append(f"**Description**: {p['description']}\n")
                        if "formula" in p:
                            output.append(f"**Formula**: `{p['formula']}`\n")
                        if "applications" in p:
                            output.append("### Applications")
                            for app in p["applications"]:
                                output.append(f"- {app}")
                        return "\n".join(output)

        elif isinstance(section, dict):
            for key, principle in section.items():
                if isinstance(principle, dict):
                    p_name = principle.get("name", "").lower()
                    if name_lower in p_name or name_lower in key.lower():
                        output = [f"## {principle.get('name', key)}\n"]
                        if "definition" in principle:
                            output.append(f"**Definition**: {principle['definition']}\n")
                        if "core_formula" in principle:
                            output.append(f"**Formula**: `{principle['core_formula']}`\n")
                        if "applications" in principle:
                            output.append("### Applications")
                            apps = principle["applications"]
                            if isinstance(apps, dict):
                                for domain, app in apps.items():
                                    output.append(f"\n**{domain}**:")
                                    if isinstance(app, dict):
                                        for k, v in app.items():
                                            output.append(f"- {k}: {v}")
                                    else:
                                        output.append(f"- {app}")
                            elif isinstance(apps, list):
                                for app in apps:
                                    output.append(f"- {app}")
                        return "\n".join(output)

        return f"Principle '{name}' not found."


def main():
    """Run the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="brain-mcp server")
    parser.add_argument("--config", help="Path to config.toml config file")
    parser.add_argument("--no-prewarm", action="store_true", help="Skip model pre-warming")
    args = parser.parse_args()

    mcp = create_server(config_path=args.config)

    if not args.no_prewarm:
        prewarm_async()

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
