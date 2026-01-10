#!/bin/bash
# Install Daily Briefing Agent (runs at 6am)

set -e

PLIST_SRC="/Users/mordechai/intellectual_dna/live/com.mordechai.daily-briefing.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.mordechai.daily-briefing.plist"

echo "Installing Daily Briefing Agent..."

# Ensure logs directory exists
mkdir -p /Users/mordechai/intellectual_dna/logs

# Stop if running
launchctl unload "$PLIST_DST" 2>/dev/null || true

# Copy plist
cp "$PLIST_SRC" "$PLIST_DST"

# Load
launchctl load "$PLIST_DST"

echo "Done! Daily briefing will run at 6am."
echo ""
echo "Commands:"
echo "  View status:   launchctl list | grep daily-briefing"
echo "  Run now:       launchctl start com.mordechai.daily-briefing"
echo "  View logs:     tail -f ~/intellectual_dna/logs/daily-briefing.log"
echo "  Uninstall:     launchctl unload ~/Library/LaunchAgents/com.mordechai.daily-briefing.plist"
