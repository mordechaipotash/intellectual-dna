#!/bin/bash
# brain-mcp — Start the MCP server.
#
# Usage:
#   ./start.sh                    # Start with default config
#   ./start.sh --config my.yaml   # Start with custom config

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
elif [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "❌ No venv found. Run ./setup.sh first or: pip install brain-mcp"
    exit 1
fi

exec python -m brain_mcp.server "$@"
