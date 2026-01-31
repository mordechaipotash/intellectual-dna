#!/usr/bin/env python3
"""
Sync Clawdbot chat history into Brain's conversation archive.

Reads: ~/.clawdbot/agents/*/sessions/*.jsonl
Writes: data/clawdbot_import.parquet (then merge into all_conversations.parquet)

Run: python pipelines/sync_clawdbot.py
Cron: Every hour or on-demand

Schema mapping:
  Clawdbot JSONL → Brain Parquet
"""

import json
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Paths
CLAWDBOT_SESSIONS = Path.home() / ".clawdbot" / "agents"
OUTPUT_DIR = Path("/Users/mordechai/intellectual_dna/data")
OUTPUT_FILE = OUTPUT_DIR / "clawdbot_import.parquet"
MAIN_PARQUET = OUTPUT_DIR / "all_conversations.parquet"
SYNC_STATE_FILE = OUTPUT_DIR / "clawdbot_sync_state.json"


def load_sync_state():
    """Load last sync state to avoid re-processing."""
    if SYNC_STATE_FILE.exists():
        with open(SYNC_STATE_FILE) as f:
            return json.load(f)
    return {"last_sync": None, "processed_files": {}}


def save_sync_state(state):
    """Save sync state."""
    with open(SYNC_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def parse_clawdbot_session(jsonl_path: Path) -> list[dict]:
    """Parse a Clawdbot session JSONL into Brain-compatible records."""
    records = []
    session_id = None
    session_start = None
    model = "unknown"
    
    with open(jsonl_path) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            
            # Extract session metadata
            if entry.get("type") == "session":
                session_id = entry.get("id")
                session_start = entry.get("timestamp")
            
            # Extract model info
            if entry.get("type") == "model_change":
                model = entry.get("modelId", "unknown")
            
            # Extract messages
            if entry.get("type") == "message":
                msg = entry.get("message", {})
                role = msg.get("role")
                
                # Skip tool results and other non-user/assistant messages
                if role not in ("user", "assistant"):
                    continue
                
                # Extract text content
                content_parts = msg.get("content", [])
                text_parts = []
                for part in content_parts:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                
                content = "\n".join(text_parts).strip()
                if not content:
                    continue
                
                # Parse timestamp
                ts_str = entry.get("timestamp") or msg.get("timestamp")
                if isinstance(ts_str, str):
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    except:
                        ts = datetime.now()
                elif isinstance(ts_str, (int, float)):
                    ts = datetime.fromtimestamp(ts_str / 1000)
                else:
                    ts = datetime.now()
                
                # Build record matching Brain schema
                record = {
                    "source": "clawdbot",
                    "model": msg.get("model") or model,
                    "project": None,
                    "conversation_id": session_id or jsonl_path.stem,
                    "conversation_title": f"Clawdbot {jsonl_path.stem[:8]}",
                    "created": ts,
                    "updated": ts,
                    "year": ts.year,
                    "month": ts.month,
                    "day_of_week": ts.strftime("%A"),
                    "hour": ts.hour,
                    "message_id": entry.get("id"),
                    "parent_id": entry.get("parentId"),
                    "msg_index": len(records),
                    "msg_timestamp": ts,
                    "timestamp_is_fallback": 0,
                    "role": role,
                    "content_type": "text",
                    "is_first": 1 if len(records) == 0 else 0,
                    "is_last": 0,  # Will fix later
                    "word_count": len(content.split()),
                    "char_count": len(content),
                    "conversation_msg_count": 0,  # Will fix later
                    "has_code": 1 if "```" in content else 0,
                    "has_url": 1 if re.search(r"https?://", content) else 0,
                    "has_question": 1 if "?" in content else 0,
                    "has_attachment": 0,
                    "has_citation": 0,
                    "content": content,
                    "temporal_precision": "exact",
                }
                records.append(record)
    
    # Fix is_last and conversation_msg_count
    if records:
        records[-1]["is_last"] = 1
        msg_count = len(records)
        for r in records:
            r["conversation_msg_count"] = msg_count
    
    return records


def sync_all_sessions(incremental: bool = True):
    """Sync all Clawdbot sessions to Brain format."""
    state = load_sync_state() if incremental else {"processed_files": {}}
    all_records = []
    
    # Find all session files
    session_files = list(CLAWDBOT_SESSIONS.glob("*/sessions/*.jsonl"))
    print(f"Found {len(session_files)} session files")
    
    new_files = 0
    for jsonl_path in session_files:
        # Skip deleted files
        if ".deleted." in jsonl_path.name:
            continue
        
        # Skip already processed (by mtime)
        file_key = str(jsonl_path)
        file_mtime = jsonl_path.stat().st_mtime
        
        if incremental:
            last_mtime = state["processed_files"].get(file_key, 0)
            if file_mtime <= last_mtime:
                continue
        
        # Parse session
        try:
            records = parse_clawdbot_session(jsonl_path)
            if records:
                all_records.extend(records)
                new_files += 1
                print(f"  Parsed {jsonl_path.name}: {len(records)} messages")
        except Exception as e:
            print(f"  Error parsing {jsonl_path.name}: {e}")
        
        # Update state
        state["processed_files"][file_key] = file_mtime
    
    print(f"\nProcessed {new_files} new/updated files")
    print(f"Total new records: {len(all_records)}")
    
    if not all_records:
        print("No new records to sync.")
        save_sync_state(state)
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    
    # Save to import parquet
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")
    
    # Merge with main parquet
    if MAIN_PARQUET.exists():
        print(f"Merging with {MAIN_PARQUET}...")
        main_df = pd.read_parquet(MAIN_PARQUET)
        
        # Remove old clawdbot records (we'll replace them)
        main_df = main_df[main_df["source"] != "clawdbot"]
        
        # Load all clawdbot records (not just new ones)
        all_clawdbot = pd.read_parquet(OUTPUT_FILE)
        
        # Combine
        merged_df = pd.concat([main_df, all_clawdbot], ignore_index=True)
        merged_df = merged_df.sort_values(["created", "msg_index"])
        
        # Save back
        merged_df.to_parquet(MAIN_PARQUET, index=False)
        print(f"Merged! Total records: {len(merged_df)}")
    
    # Update sync state
    state["last_sync"] = datetime.now().isoformat()
    save_sync_state(state)
    
    print(f"\n✅ Sync complete! Run embed_new.py to update LanceDB vectors.")


def stats():
    """Show current sync stats."""
    state = load_sync_state()
    print(f"Last sync: {state.get('last_sync', 'Never')}")
    print(f"Tracked files: {len(state.get('processed_files', {}))}")
    
    if OUTPUT_FILE.exists():
        df = pd.read_parquet(OUTPUT_FILE)
        print(f"\nClawdbot import file:")
        print(f"  Records: {len(df)}")
        print(f"  Date range: {df['created'].min()} to {df['created'].max()}")
        print(f"  User messages: {len(df[df['role'] == 'user'])}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        stats()
    elif len(sys.argv) > 1 and sys.argv[1] == "full":
        sync_all_sessions(incremental=False)
    else:
        sync_all_sessions(incremental=True)
