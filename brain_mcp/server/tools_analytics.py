"""
brain-mcp — Analytics tools.

Query analytics across timeline, tool stacks, problem resolution,
spend, and conversation summary. These tools require interpretation
layer data (built by pipelines that analyze raw conversations).

All analytics are optional — the server works without them.
"""

import json
from pathlib import Path

from brain_mcp.config import get_config
from .db import get_conversations


def register(mcp):
    """Register analytics tools with the MCP server."""

    @mcp.tool()
    def query_analytics(
        view: str = "timeline",
        date: str = None,
        month: str = None,
        source: str = None,
        limit: int = 15,
    ) -> str:
        """
        Query analytics across timeline, tool stacks, problem resolution,
        spend, and conversation summary.

        Args:
            view: What to analyze:
                - "timeline" (default): What happened on a specific date
                - "stacks": Technology stack patterns over time
                - "problems": Debugging and problem resolution patterns
                - "spend": Cost breakdown by source/time
                - "summary": Comprehensive conversation analysis summary
            date: Date in YYYY-MM-DD format (used with view="timeline")
            month: YYYY-MM filter (used with stacks, problems, spend views)
            source: Source filter for spend (e.g., "openrouter", "claude_code")
            limit: Max results (default 15)
        """
        cfg = get_config()

        # Analytics data lives in data/interpretations/ and data/facts/
        interp_dir = cfg.data_dir / "interpretations"
        facts_dir = cfg.data_dir / "facts"

        if view == "timeline":
            if not date:
                return "Please provide a date (YYYY-MM-DD) for timeline view."
            return _query_timeline(date, facts_dir, interp_dir)
        elif view == "stacks":
            return _query_tool_stacks(month, limit, interp_dir)
        elif view == "problems":
            return _query_problem_resolution(month, limit, interp_dir)
        elif view == "spend":
            return _query_spend(month, source, facts_dir)
        elif view == "summary":
            return _query_conversation_summary(interp_dir)
        else:
            return (
                f"Unknown view: {view}. "
                "Use: timeline, stacks, problems, spend, summary"
            )

    def _query_timeline(
        date: str, facts_dir: Path, interp_dir: Path
    ) -> str:
        """What happened on a specific date across all sources."""
        temporal_path = facts_dir / "temporal_dim.parquet"

        if not temporal_path.exists():
            # Fallback: use conversations directly
            try:
                con = get_conversations()
                results = con.execute("""
                    SELECT source, conversation_title,
                           COUNT(*) as msg_count
                    FROM conversations
                    WHERE CAST(created AS DATE) = CAST(? AS DATE)
                    GROUP BY source, conversation_title
                    ORDER BY msg_count DESC
                    LIMIT 20
                """, [date]).fetchall()

                if not results:
                    return f"No activity found for {date}"

                output = [f"## Timeline for {date}\n"]
                total = sum(r[2] for r in results)
                output.append(f"**Total messages**: {total}\n")

                for source, title, count in results:
                    output.append(
                        f"- **{title or 'Untitled'}** "
                        f"({source}, {count} msgs)"
                    )
                return "\n".join(output)
            except FileNotFoundError as e:
                return str(e)

        import duckdb
        con = duckdb.connect()

        temporal = con.execute(f"""
            SELECT date, year, quarter, month, day_of_week, is_weekend,
                   messages_sent, convos_started, videos_watched,
                   commits_made, cost_total, tokens_used, was_active
            FROM '{temporal_path}'
            WHERE CAST(date AS VARCHAR) = ?
        """, [date]).fetchone()

        if not temporal:
            return f"No data for {date}"

        output = [f"## Timeline for {date}\n"]
        output.append(f"**{temporal[4]}** (Q{temporal[2]} {temporal[1]})")
        output.append(
            f"Weekend: {'Yes' if temporal[5] else 'No'}\n"
        )

        output.append("### Activity")
        output.append(f"  Messages: {temporal[6] or 0}")
        output.append(f"  Conversations: {temporal[7] or 0}")
        output.append(f"  Commits: {temporal[9] or 0}")

        if temporal[10]:
            output.append(f"\n### Spend")
            output.append(f"  Cost: ${temporal[10]:.2f}")
            output.append(f"  Tokens: {temporal[11]:,}")

        # Get focus keywords if available
        focus_path = interp_dir / "focus" / "v1" / "daily.parquet"
        if focus_path.exists():
            try:
                focus = con.execute(f"""
                    SELECT top_keyword, keywords, focus_score
                    FROM '{focus_path}'
                    WHERE CAST(date AS VARCHAR) = ?
                """, [date]).fetchone()

                if focus:
                    try:
                        kw_list = json.loads(focus[1])[:5]
                    except Exception:
                        kw_list = []
                    output.append(f"\n### Focus")
                    output.append(
                        f"  Top: {focus[0]} (score: {focus[2]:.2f})"
                    )
                    output.append(f"  Keywords: {', '.join(kw_list)}")
            except Exception:
                pass

        return "\n".join(output)

    def _query_tool_stacks(
        month: str = None, limit: int = 12, interp_dir: Path = None
    ) -> str:
        """Query technology stack patterns."""
        path = interp_dir / "tool_stacks" / "v2" / "monthly.parquet"
        if not path.exists():
            return (
                "Tool stack data not found. "
                "This requires running the tool_stacks analysis pipeline."
            )

        import duckdb
        con = duckdb.connect()

        if month:
            results = con.execute(f"""
                SELECT month_start, stack_count, dominant_stack,
                       new_adoptions, abandonments, all_techs
                FROM '{path}'
                WHERE CAST(month_start AS VARCHAR) LIKE ?
                ORDER BY month_start DESC
                LIMIT {limit}
            """, [f"{month}%"]).fetchall()
        else:
            results = con.execute(f"""
                SELECT month_start, stack_count, dominant_stack,
                       new_adoptions, abandonments, all_techs
                FROM '{path}'
                ORDER BY month_start DESC
                LIMIT {limit}
            """).fetchall()

        if not results:
            return f"No tool stack data found{' for ' + month if month else ''}"

        output = [
            f"## Tool Stack Evolution"
            f"{' for ' + month if month else ''}\n"
        ]
        for row in results:
            month_start, count, dominant, new_tech, dropped, _ = row
            output.append(f"**{month_start}** — {count} stacks identified")
            output.append(
                f"  Dominant: {dominant[:60]}"
                f"{'...' if dominant and len(dominant) > 60 else ''}"
            )
            if new_tech:
                output.append(f"  ➕ New: {new_tech}")
            if dropped:
                output.append(f"  ➖ Dropped: {dropped}")
            output.append("")

        return "\n".join(output)

    def _query_problem_resolution(
        month: str = None, limit: int = 12, interp_dir: Path = None
    ) -> str:
        """Query debugging and problem resolution patterns."""
        path = interp_dir / "problem_resolution" / "v2" / "monthly.parquet"
        if not path.exists():
            return (
                "Problem resolution data not found. "
                "This requires running the problem_resolution "
                "analysis pipeline."
            )

        import duckdb
        con = duckdb.connect()

        if month:
            results = con.execute(f"""
                SELECT month_start, chain_count, domains, difficulties,
                       debugging_patterns, hardest_problem, aha_quotes
                FROM '{path}'
                WHERE CAST(month_start AS VARCHAR) LIKE ?
                ORDER BY month_start DESC
                LIMIT {limit}
            """, [f"{month}%"]).fetchall()
        else:
            results = con.execute(f"""
                SELECT month_start, chain_count, domains, difficulties,
                       debugging_patterns, hardest_problem, aha_quotes
                FROM '{path}'
                ORDER BY month_start DESC
                LIMIT {limit}
            """).fetchall()

        if not results:
            return (
                f"No problem resolution data found"
                f"{' for ' + month if month else ''}"
            )

        output = [
            f"## Problem Resolution Patterns"
            f"{' for ' + month if month else ''}\n"
        ]
        for row in results:
            month_start, chains, domains, difficulties, _, hardest, aha = row
            output.append(
                f"**{month_start}** — {chains} resolution chains"
            )
            output.append(f"  Domains: {domains}")
            output.append(f"  Difficulties: {difficulties}")
            if hardest:
                output.append(
                    f"  Hardest: {hardest[:80]}"
                    f"{'...' if len(hardest) > 80 else ''}"
                )
            if aha:
                output.append(
                    f"  💡 Aha: {aha[:80]}"
                    f"{'...' if len(aha) > 80 else ''}"
                )
            output.append("")

        return "\n".join(output)

    def _query_spend(
        month: str = None,
        source: str = None,
        facts_dir: Path = None,
    ) -> str:
        """Query spend data from facts/spend layers."""
        monthly_path = facts_dir / "spend" / "monthly.parquet"
        daily_path = facts_dir / "spend" / "daily.parquet"

        if not monthly_path.exists():
            return (
                "Spend data not found. "
                "This requires running the spend analysis pipeline."
            )

        import duckdb
        con = duckdb.connect()
        output = []

        if month:
            query = f"""
                SELECT date, source, cost_usd, tokens_total
                FROM '{daily_path}'
                WHERE CAST(date AS VARCHAR) LIKE ?
                {'AND source = ?' if source else ''}
                ORDER BY date DESC
            """
            params = [f"{month}%"]
            if source:
                params.append(source)
            results = con.execute(query, params).fetchall()

            output.append(f"## Daily Spend for {month}\n")
            total_cost = 0
            for d, src, cost, tokens in results:
                output.append(
                    f"  {d} | {src}: ${cost:.2f} ({tokens:,} tokens)"
                )
                total_cost += cost or 0
            output.append(f"\n**Total**: ${total_cost:.2f}")
        else:
            query = f"""
                SELECT month, source, cost_usd, tokens_total
                FROM '{monthly_path}'
                {'WHERE source = ?' if source else ''}
                ORDER BY month DESC, cost_usd DESC
                LIMIT 30
            """
            params = [source] if source else []
            results = con.execute(query, params).fetchall()

            output.append("## Monthly Spend Summary\n")
            current_month = None
            for mo, src, cost, tokens in results:
                if mo != current_month:
                    current_month = mo
                    output.append(f"\n**{mo}**")
                output.append(
                    f"  {src}: ${cost:.2f} ({tokens:,} tokens)"
                )

        return "\n".join(output)

    def _query_conversation_summary(interp_dir: Path) -> str:
        """Comprehensive conversation analysis summary."""
        stats_path = (
            interp_dir / "conversation_stats" / "v1" / "conversations.parquet"
        )

        if not stats_path.exists():
            # Fall back to basic stats from conversations
            try:
                con = get_conversations()
                total = con.execute(
                    "SELECT COUNT(*) FROM conversations"
                ).fetchone()[0]
                convos = con.execute(
                    "SELECT COUNT(DISTINCT conversation_id) "
                    "FROM conversations"
                ).fetchone()[0]
                user_msgs = con.execute(
                    "SELECT COUNT(*) FROM conversations "
                    "WHERE role = 'user'"
                ).fetchone()[0]
                questions = con.execute(
                    "SELECT COUNT(*) FROM conversations "
                    "WHERE has_question = 1 AND role = 'user'"
                ).fetchone()[0]

                return (
                    f"## Conversation Summary\n\n"
                    f"**Total messages**: {total:,}\n"
                    f"**Conversations**: {convos:,}\n"
                    f"**User messages**: {user_msgs:,}\n"
                    f"**Questions asked**: {questions:,}\n\n"
                    f"_For detailed analysis, run the conversation "
                    f"analysis pipeline._"
                )
            except FileNotFoundError as e:
                return str(e)

        import duckdb
        con = duckdb.connect()

        output = ["## Conversation Analysis Summary\n"]

        totals = con.execute(f"""
            SELECT COUNT(*) as conversations,
                   SUM(message_count) as messages,
                   SUM(total_words) as words,
                   SUM(questions_asked) as questions,
                   SUM(code_messages) as code_msgs,
                   AVG(message_count) as avg_length
            FROM '{stats_path}'
        """).fetchone()

        output.append("### Totals")
        output.append(f"  Conversations: {totals[0]:,}")
        output.append(f"  Messages: {totals[1]:,}")
        output.append(f"  Total Words: {totals[2]:,}")
        output.append(f"  Questions Asked: {totals[3]:,}")
        output.append(f"  Code Messages: {totals[4]:,}")
        output.append(
            f"  Avg Conversation Length: {totals[5]:.0f} messages"
        )

        deep_path = (
            interp_dir / "conversation_threads" / "v1"
            / "deep_conversations.parquet"
        )
        if deep_path.exists():
            deep = con.execute(
                f"SELECT COUNT(*) FROM '{deep_path}'"
            ).fetchone()[0]
            output.append(f"\n### Deep Conversations")
            output.append(f"  100+ message conversations: {deep}")

        top = con.execute(f"""
            SELECT conversation_title, message_count
            FROM '{stats_path}'
            ORDER BY message_count DESC
            LIMIT 5
        """).fetchall()

        output.append(f"\n### Longest Conversations")
        for title, msgs in top:
            t = (title or "untitled")[:35]
            output.append(f"  {msgs:>5,} msgs — {t}")

        return "\n".join(output)
