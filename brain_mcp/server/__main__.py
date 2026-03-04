"""
brain-mcp — Server entry point for `python -m brain_mcp.server`.

Allows running the MCP server with:
    python -m brain_mcp.server
    python -m brain_mcp.server --config /path/to/brain.yaml
    python -m brain_mcp.server --no-prewarm
"""

from brain_mcp.server.server import main

if __name__ == "__main__":
    main()
