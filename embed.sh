#!/bin/bash
# brain-mcp — Run the embedding pipeline.
#
# Usage:
#   ./embed.sh          # Incremental (only new messages)
#   ./embed.sh full     # Re-embed everything
#   ./embed.sh stats    # Show embedding stats

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
elif [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "❌ No venv found. Run ./setup.sh first or: pip install brain-mcp"
    exit 1
fi

echo "🔮 brain-mcp — Embedding pipeline"
python -m brain_mcp.embed.embed "${1:-incremental}"
