#!/bin/bash
# brain-mcp — Setup script
# Creates Python virtual environment, installs dependencies, and prepares data directories.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════╗"
echo "║    brain-mcp setup                     ║"
echo "╚════════════════════════════════════════╝"

# Check Python version
PYTHON="${PYTHON:-python3}"
PY_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)

if [[ -z "$PY_VERSION" ]]; then
    echo "❌ Python 3 not found. Install Python 3.11+ and try again."
    exit 1
fi

PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MINOR" -lt 11 ]]; then
    echo "❌ Python $PY_VERSION detected. Python 3.11+ required."
    exit 1
fi

echo "✅ Python $PY_VERSION"

# Create virtual environment
if [[ ! -d "venv" ]]; then
    echo "📦 Creating virtual environment..."
    "$PYTHON" -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment exists"
fi

# Activate and install
echo "📦 Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -e ".[dev]" -q
echo "✅ Dependencies installed"

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data vectors logs

# Copy example config if needed
if [[ ! -f "brain.yaml" ]]; then
    echo "📝 Creating brain.yaml from example..."
    cp brain.yaml.example brain.yaml
    echo "   → Edit brain.yaml to configure your sources"
else
    echo "✅ brain.yaml exists"
fi

echo ""
echo "╔════════════════════════════════════════╗"
echo "║    Setup complete!                     ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Edit brain.yaml — add your conversation sources"
echo "  2. ./ingest.sh    — import conversations"
echo "  3. ./embed.sh     — create vector embeddings"
echo "  4. ./start.sh     — start the MCP server"
echo ""
echo "For detailed docs: docs/getting-started.md"
