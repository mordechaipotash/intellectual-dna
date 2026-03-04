#!/bin/bash
# brain-mcp — Run all configured ingesters and merge into main parquet.
#
# Usage:
#   ./ingest.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    true  # already activated
else
    echo "❌ No venv found. Run ./setup.sh first or: pip install brain-mcp"
    exit 1
fi

echo "🧠 brain-mcp — Ingesting conversations"
echo "======================================="
echo ""

# Run the ingest pipeline via the installed package
python -c "
from brain_mcp.config import load_config, set_config, get_config
from brain_mcp.ingest import run_all_ingesters

cfg = load_config()
set_config(cfg)
cfg = get_config()
total = run_all_ingesters(cfg)
"

echo ""
echo "Done! Run ./embed.sh to create vector embeddings."
