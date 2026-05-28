#!/usr/bin/env python3
"""
brain-mcp v1.0 — MCP server for the Bob protocol (https://apiiam.com/bob/init.0).

Reads Bob's local parquets under $BOB_HOME (default ~/.bob/) via duckdb. No
cloud calls, no embeddings, no auth. Phone-home is forbidden by Bob's protocol
contract — this server enforces that by never opening a network socket.

Layer priority:
  - L1 (~/.bob/turns.l1.parquet) preferred when present (noise-filtered)
  - L0 (~/.bob/turns.parquet) fallback (raw; warn in narration)

Register in ~/.claude/mcp.json as:
  "brain": {
    "command": "uvx",
    "args": ["brain-mcp"]
  }
or, for local dev:
  "brain": {
    "command": "python",
    "args": ["/path/to/Projects/bob/mcp/server.py"]
  }
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
from mcp.server.fastmcp import FastMCP

BOB_HOME = Path(os.environ.get("BOB_HOME", str(Path.home() / ".bob"))).expanduser()
L0_PATH = BOB_HOME / "turns.parquet"
L1_PATH = BOB_HOME / "turns.l1.parquet"
L1_DROPPED_PATH = BOB_HOME / "turns.l1.dropped.parquet"
L1_MANIFEST_PATH = BOB_HOME / "turns.l1.manifest.json"

mcp = FastMCP(
    "brain-mcp",
    instructions=(
        "Bob's memory layer. Tools read the user's own AI conversation history "
        "from ~/.bob/ (or $BOB_HOME). All read-only. No cloud. L1 preferred over L0 "
        "when available. Call bob_health first if uncertain about state."
    ),
)


def _active_layer() -> tuple[str, Path]:
    """Returns ('l1', path) if L1 exists, else ('l0', path) if L0 exists, else raises."""
    if L1_PATH.exists():
        return "l1", L1_PATH
    if L0_PATH.exists():
        return "l0", L0_PATH
    raise FileNotFoundError(
        f"Bob has not been installed. Run https://apiiam.com/bob/init.0 first "
        f"(expected file at {L0_PATH})."
    )


def _query(sql: str) -> list[dict]:
    """Run sql against the active layer parquet. Read-only."""
    con = duckdb.connect(":memory:")
    try:
        rows = con.execute(sql).fetchall()
        cols = [d[0] for d in con.description] if con.description else []
        return [dict(zip(cols, r)) for r in rows]
    finally:
        con.close()


def _esc(s: str) -> str:
    """SQL string-literal escape."""
    return s.replace("'", "''")


def _layer_table_expr() -> str:
    """Returns SQL expression naming the parquet to read."""
    _, path = _active_layer()
    return f"read_parquet('{path}')"


def _render(rows: list[dict], headline: str) -> str:
    """Render rows as a compact markdown text block. Keep under ~6KB."""
    if not rows:
        return f"{headline}\n\n(no matches)"
    lines = [headline, ""]
    for i, r in enumerate(rows, 1):
        ts = r.get("ts")
        src = r.get("src")
        role = r.get("role")
        conv = (r.get("conv_id") or "")[:18]
        text = (r.get("text_clean") or r.get("text") or "").strip()
        if len(text) > 380:
            text = text[:377] + "..."
        text = text.replace("\n", " ")
        lines.append(
            f"{i}. [{ts}] {src}:{role} conv={conv}\n   {text}"
        )
    return "\n".join(lines)


# ─── Tools ───────────────────────────────────────────────────────────────


@mcp.tool()
def bob_health() -> str:
    """
    Report the state of Bob's local memory. Use this first if you don't know
    whether Bob has been installed, what layer is active, or how many turns are
    available. Returns: paths checked, layer in use, per-source counts, L1
    manifest summary if present.
    """
    lines = [f"BOB_HOME: {BOB_HOME}", ""]
    have_l0 = L0_PATH.exists()
    have_l1 = L1_PATH.exists()
    lines.append(f"L0 (turns.parquet):           {'present' if have_l0 else 'MISSING'}")
    lines.append(f"L1 (turns.l1.parquet):        {'present' if have_l1 else 'absent (init.1 not run)'}")
    lines.append(f"L1 dropped audit:             {'present' if L1_DROPPED_PATH.exists() else 'absent'}")
    lines.append(f"L1 manifest:                  {'present' if L1_MANIFEST_PATH.exists() else 'absent'}")
    lines.append("")

    if not have_l0:
        lines.append("Bob is not installed. Run https://apiiam.com/bob/init.0 in a fresh LLM session.")
        return "\n".join(lines)

    layer, path = _active_layer()
    lines.append(f"Active layer: {layer.upper()} ({path})")
    rows = _query(f"SELECT src, count(*) AS n FROM {_layer_table_expr()} GROUP BY src ORDER BY n DESC")
    lines.append("\nPer-source counts:")
    for r in rows:
        lines.append(f"  {r['src']:30s} {r['n']:>8d}")

    total_row = _query(f"SELECT count(*) AS n, min(ts) AS first_ts, max(ts) AS last_ts FROM {_layer_table_expr()}")
    if total_row:
        t = total_row[0]
        lines.append(f"\nTotal turns: {t['n']:,}")
        lines.append(f"Range: {t['first_ts']} → {t['last_ts']}")

    if L1_MANIFEST_PATH.exists():
        try:
            m = json.loads(L1_MANIFEST_PATH.read_text())
            lines.append("\nL1 manifest:")
            for k in ("l0_in", "l1_kept", "l1_dropped", "user_yield_pct"):
                if k in m:
                    lines.append(f"  {k}: {m[k]}")
        except Exception as e:
            lines.append(f"\nL1 manifest exists but unreadable: {e}")

    return "\n".join(lines)


@mcp.tool()
def bob_search(query: str, src: Optional[str] = None, role: Optional[str] = None, limit: int = 12) -> str:
    """
    Keyword search across Bob's turns. Case-insensitive substring match against
    text (or text_clean when L1 is active). Filter by src ('claude-code',
    'codex', 'cursor', 'chatgpt-export', 'claude-desktop-export') and/or role
    ('user', 'assistant'). Returns up to `limit` most recent matches.

    Bob does NOT do semantic search — this is exact-string keyword. For "what
    do I think about X" patterns use bob_what_do_i_think; for "recent work on
    X" use bob_tunnel_state.
    """
    q = _esc(query)
    where = [f"(text ILIKE '%{q}%')"]
    if src:
        where.append(f"src = '{_esc(src)}'")
    if role:
        where.append(f"role = '{_esc(role)}'")
    sql = (
        f"SELECT ts, src, conv_id, role, "
        f"  CASE WHEN length(text) > 600 THEN substring(text, 1, 600) || '...' ELSE text END AS text "
        f"FROM {_layer_table_expr()} "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY ts DESC LIMIT {int(limit)}"
    )
    return _render(_query(sql), f"Search: {query!r} (src={src or 'any'}, role={role or 'any'}, layer={_active_layer()[0]})")


@mcp.tool()
def bob_recent(src: Optional[str] = None, role: Optional[str] = None, hours: int = 24, limit: int = 20) -> str:
    """
    Most recent turns within the last `hours`. Filter by src and/or role.
    Default: last 24h, any source, any role, up to 20 turns. Use to answer
    "what was I just working on?".
    """
    where = [f"ts >= now() - INTERVAL '{int(hours)} hours'"]
    if src:
        where.append(f"src = '{_esc(src)}'")
    if role:
        where.append(f"role = '{_esc(role)}'")
    sql = (
        f"SELECT ts, src, conv_id, role, "
        f"  CASE WHEN length(text) > 400 THEN substring(text, 1, 400) || '...' ELSE text END AS text "
        f"FROM {_layer_table_expr()} "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY ts DESC LIMIT {int(limit)}"
    )
    return _render(_query(sql), f"Recent (last {hours}h, src={src or 'any'}, role={role or 'any'})")


@mcp.tool()
def bob_conversations_by_date(date: str, src: Optional[str] = None) -> str:
    """
    List all conversations active on a given date (YYYY-MM-DD). For each
    conversation: conv_id, src, turn count, first/last timestamp, and a short
    excerpt from the first user turn. Use to browse by time.
    """
    d = _esc(date)
    where = [f"date_trunc('day', ts) = '{d}'::date"]
    if src:
        where.append(f"src = '{_esc(src)}'")
    sql = (
        f"WITH day AS ( "
        f"  SELECT * FROM {_layer_table_expr()} WHERE {' AND '.join(where)} "
        f") "
        f"SELECT src, conv_id, count(*) AS turns, min(ts) AS first_ts, max(ts) AS last_ts, "
        f"  (SELECT substring(text, 1, 240) FROM day d2 "
        f"   WHERE d2.conv_id = day.conv_id AND d2.src = day.src AND d2.role = 'user' "
        f"   ORDER BY ts ASC LIMIT 1) AS first_user_text "
        f"FROM day GROUP BY src, conv_id ORDER BY first_ts ASC"
    )
    rows = _query(sql)
    if not rows:
        return f"Conversations on {date} (src={src or 'any'})\n\n(none)"
    lines = [f"Conversations on {date} (src={src or 'any'}) — {len(rows)} convs", ""]
    for r in rows:
        lines.append(
            f"- {r['src']}:{(r['conv_id'] or '')[:30]} — {r['turns']} turns, "
            f"{r['first_ts'].strftime('%H:%M') if r['first_ts'] else '??'}"
            f"–{r['last_ts'].strftime('%H:%M') if r['last_ts'] else '??'}"
        )
        if r.get("first_user_text"):
            t = r["first_user_text"].replace("\n", " ").strip()[:200]
            lines.append(f"  {t}")
    return "\n".join(lines)


@mcp.tool()
def bob_tunnel_state(domain: str, limit: int = 15) -> str:
    """
    Reconstruct work-in-progress on a domain/topic. Returns the most recent
    user+assistant turns matching `domain` (keyword), grouped by conversation,
    so you can re-enter where you left off. Lighter than full conversations_by_date;
    focused on a single thread.
    """
    q = _esc(domain)
    sql = (
        f"WITH hits AS ( "
        f"  SELECT * FROM {_layer_table_expr()} "
        f"  WHERE text ILIKE '%{q}%' "
        f"), "
        f"last_seen AS ( "
        f"  SELECT conv_id, src, max(ts) AS last_ts FROM hits GROUP BY conv_id, src "
        f"  ORDER BY last_ts DESC LIMIT 5 "
        f") "
        f"SELECT h.ts, h.src, h.conv_id, h.role, "
        f"  CASE WHEN length(h.text) > 300 THEN substring(h.text, 1, 300) || '...' ELSE h.text END AS text "
        f"FROM hits h "
        f"JOIN last_seen ls ON ls.conv_id = h.conv_id AND ls.src = h.src "
        f"ORDER BY h.ts DESC LIMIT {int(limit)}"
    )
    return _render(_query(sql), f"Tunnel state for {domain!r} — most recent activity across top 5 convs")


@mcp.tool()
def bob_what_do_i_think(topic: str, limit: int = 10) -> str:
    """
    Surface the user's own positions on a topic. Filters to role='user' turns
    containing the keyword, deduplicated on first-200-char prefix, most recent
    first. Use when LLM is about to make a claim about what the user believes.
    """
    q = _esc(topic)
    sql = (
        f"SELECT ts, src, conv_id, role, text "
        f"FROM {_layer_table_expr()} "
        f"WHERE role = 'user' AND text ILIKE '%{q}%' "
        f"QUALIFY row_number() OVER (PARTITION BY substring(text, 1, 200) ORDER BY ts DESC) = 1 "
        f"ORDER BY ts DESC LIMIT {int(limit)}"
    )
    return _render(_query(sql), f"What user said about {topic!r} (user-turns only, deduped)")


@mcp.tool()
def bob_thinking_trajectory(topic: str, limit: int = 12) -> str:
    """
    Chronological pass on a topic — first mention to most recent. Shows how
    the user's framing has evolved. Returns user-role turns in ASCENDING ts
    order so the trajectory reads left-to-right.
    """
    q = _esc(topic)
    sql = (
        f"SELECT ts, src, conv_id, role, text "
        f"FROM {_layer_table_expr()} "
        f"WHERE role = 'user' AND text ILIKE '%{q}%' "
        f"ORDER BY ts ASC LIMIT {int(limit)}"
    )
    return _render(_query(sql), f"Thinking trajectory on {topic!r} (oldest → newest)")


@mcp.tool()
def bob_open_threads(limit: int = 15) -> str:
    """
    Surface unfinished work — user turns from the last 14 days ending in a
    question mark, or containing common open-thread tokens (TODO, next step,
    waiting on, blocked, follow up). Most recent first. Use when user says
    'what am I forgetting' or 'where am I stuck'.
    """
    sql = (
        f"SELECT ts, src, conv_id, role, text "
        f"FROM {_layer_table_expr()} "
        f"WHERE role = 'user' "
        f"  AND ts >= now() - INTERVAL '14 days' "
        f"  AND (text ILIKE '%TODO%' OR text ILIKE '%next step%' OR text ILIKE '%waiting on%' "
        f"       OR text ILIKE '%blocked%' OR text ILIKE '%follow up%' OR text ILIKE '%still need%' "
        f"       OR trim(text) LIKE '%?') "
        f"ORDER BY ts DESC LIMIT {int(limit)}"
    )
    return _render(_query(sql), f"Open threads (last 14 days, user-turns)")


@mcp.tool()
def bob_dropped(reason: Optional[str] = None, limit: int = 12) -> str:
    """
    Inspect what L1 filtered out. Optional `reason` filter (e.g. 'cc:build-log-paste').
    Useful for debugging the L1 noise filter or auditing a missing turn.
    Returns rows from turns.l1.dropped.parquet — fails if L1 was not run.
    """
    if not L1_DROPPED_PATH.exists():
        return (
            "L1 dropped audit not available. Run https://apiiam.com/bob/init.1 "
            "to produce the noise-filter audit trail."
        )
    where = ["1=1"]
    if reason:
        where.append(f"drop_reason = '{_esc(reason)}'")
    sql = (
        f"SELECT ts, src, conv_id, role, drop_reason AS role_tag, "
        f"  CASE WHEN length(text) > 400 THEN substring(text, 1, 400) || '...' ELSE text END AS text "
        f"FROM read_parquet('{L1_DROPPED_PATH}') "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY ts DESC LIMIT {int(limit)}"
    )
    rows = _query(sql)
    # Rename role_tag → role for the render helper to display reason
    for r in rows:
        r["role"] = r.pop("role_tag")
    return _render(rows, f"Dropped turns (reason={reason or 'any'}, layer=L1.dropped)")


if __name__ == "__main__":
    mcp.run()


def main() -> int:
    """Console-script entry point for `brain-mcp` (v1.0+). Starts the FastMCP server on stdio."""
    mcp.run()
    return 0
