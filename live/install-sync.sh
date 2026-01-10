#!/bin/bash
# Install brain-sync launchd agent
#
# This agent runs hourly to catch up on any Claude Code conversations
# that weren't synced by the Stop hook.

set -e

PLIST_NAME="com.mordechai.brain-sync.plist"
PLIST_SRC="/Users/mordechai/intellectual_dna/live/$PLIST_NAME"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME"

echo "Installing brain-sync launchd agent..."

# Unload if already loaded
if launchctl list | grep -q "com.mordechai.brain-sync"; then
    echo "  Unloading existing agent..."
    launchctl unload "$PLIST_DST" 2>/dev/null || true
fi

# Link plist
echo "  Linking plist to ~/Library/LaunchAgents/"
ln -sf "$PLIST_SRC" "$PLIST_DST"

# Load agent
echo "  Loading agent..."
launchctl load "$PLIST_DST"

# Verify
if launchctl list | grep -q "com.mordechai.brain-sync"; then
    echo "  ✓ Agent installed and running"
    echo ""
    echo "The agent will run every hour to sync Claude Code conversations."
    echo "Logs: ~/intellectual_dna/logs/sync-launchd.log"
else
    echo "  ✗ Agent failed to load"
    exit 1
fi
