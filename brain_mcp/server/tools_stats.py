"""
brain-mcp — Stats tools.

Brain statistics with multiple views: overview, domains, pulse,
conversations, embeddings, github, markdown.
"""

from brain_mcp.config import get_config
from .db import (
    get_conversations,
    get_summaries_db,
    get_github_db,
    get_markdown_db,
    get_lance_db,
    lance_count,
    parse_json_field,
)


def register(mcp):
    """Register stats tools with the MCP server."""

    @mcp.tool()
    def brain_stats(view: str = "overview") -> str:
        """
        Brain overview, domain distribution, and thinking pulse.

        Args:
            view: What to display:
                - "overview" (default): Stats across all data sources
                - "domains": Domain breakdown with counts, %, breakthroughs, top concepts
                - "pulse": Domain × thinking_stage matrix — what's crystallizing vs exploring
                - "conversations": Detailed conversation stats
                - "embeddings": Embedding coverage stats
                - "github": Repository and commit stats
                - "markdown": Document corpus stats
        """
        view = view.lower().strip()

        if view == "domains":
            return _domain_map_view()
        elif view == "pulse":
            return _thinking_pulse_view()
        elif view == "conversations":
            return _conversation_stats()
        elif view == "embeddings":
            return _embedding_stats()
        elif view == "github":
            return _github_stats()
        elif view == "markdown":
            return _markdown_stats()
        elif view == "overview":
            return _overview()
        else:
            return f"Unknown view: {view}. Use: overview, domains, pulse, conversations, embeddings, github, markdown"

    @mcp.tool()
    def unfinished_threads(domain: str = None, importance: str = "significant") -> str:
        """
        Find conversations worth revisiting: exploring/crystallizing stage with open questions.
        """
        sdb = get_summaries_db()
        if not sdb:
            return "Summary database not found."

        importance_filter = {
            "breakthrough": "importance = 'breakthrough'",
            "significant": "importance IN ('breakthrough', 'significant')",
            "routine": "1=1",
        }.get(importance, "importance IN ('breakthrough', 'significant')")

        where_parts = [
            "thinking_stage IN ('exploring', 'crystallizing')",
            importance_filter,
            "open_questions IS NOT NULL",
            "open_questions != '[]'",
            "open_questions NOT LIKE '%none identified%'",
        ]
        params = []
        if domain:
            where_parts.append("domain_primary = ?")
            params.append(domain)

        where_clause = " AND ".join(where_parts)

        results = sdb.execute(f"""
            SELECT conversation_id, title, source, domain_primary,
                   thinking_stage, importance, open_questions, summary, msg_count
            FROM summaries
            WHERE {where_clause}
            ORDER BY
                CASE importance WHEN 'breakthrough' THEN 0 WHEN 'significant' THEN 1 ELSE 2 END,
                msg_count DESC
            LIMIT 25
        """, params).fetchall()

        if not results:
            return f"No unfinished threads found{' in ' + domain if domain else ''}."

        filter_desc = f" in {domain}" if domain else ""
        output = [f"## Unfinished Threads{filter_desc} (importance >= {importance})\n"]

        for conv_id, title, source, dom, stage, imp, oq_raw, summary, msg_count in results:
            questions = parse_json_field(oq_raw)
            real_questions = [q for q in questions if q and "none identified" not in str(q).lower()]
            if not real_questions:
                continue

            imp_icon = {"breakthrough": "🔥", "significant": "⭐", "routine": "📝"}.get(imp, "📝")
            stage_icon = {"exploring": "🔍", "crystallizing": "💎"}.get(stage, "📝")

            output.append(f"### {imp_icon} {stage_icon} {title or 'Untitled'}")
            output.append(f"_Domain: {dom} | Source: {source} | {msg_count} msgs_")
            output.append(f"> {(summary or '')[:200]}...")
            output.append("**Open questions**:")
            for q in real_questions[:3]:
                output.append(f"  ❓ {q[:200]}")
            if len(real_questions) > 3:
                output.append(f"  _... and {len(real_questions) - 3} more_")
            output.append(f"_ID: {conv_id[:20]}..._\n")

        return "\n".join(output)

    # ═══════════════════════════════════════════════════════════════════
    # Internal view functions
    # ═══════════════════════════════════════════════════════════════════

    def _overview() -> str:
        cfg = get_config()
        output = ["## Brain Data Overview\n"]

        # Conversations
        try:
            con = get_conversations()
            total = con.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            user = con.execute("SELECT COUNT(*) FROM conversations WHERE role = 'user'").fetchone()[0]
            output.append(f"**Conversations**: {total:,} messages ({user:,} user)")
        except Exception:
            output.append("**Conversations**: unavailable")

        # Embeddings
        try:
            embedded = lance_count("message")
            if embedded:
                output.append(f"**Embeddings**: {embedded:,} vectors ({cfg.embedding.dim}d, LanceDB)")
            else:
                output.append("**Embeddings**: unavailable")
        except Exception:
            output.append("**Embeddings**: unavailable")

        # Summaries
        try:
            sdb = get_summaries_db()
            if sdb:
                v6_total = sdb.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
                v6_bt = sdb.execute("SELECT COUNT(*) FROM summaries WHERE importance = 'breakthrough'").fetchone()[0]
                v6_dec = sdb.execute("""
                    SELECT COUNT(*) FROM summaries
                    WHERE decisions IS NOT NULL AND decisions != '[]'
                    AND decisions NOT LIKE '%none identified%'
                """).fetchone()[0]
                v6_oq = sdb.execute("""
                    SELECT COUNT(*) FROM summaries
                    WHERE open_questions IS NOT NULL AND open_questions != '[]'
                    AND open_questions NOT LIKE '%none identified%'
                """).fetchone()[0]
                output.append(f"**Summaries**: {v6_total:,} conversations summarized")
                output.append(f"  - {v6_bt:,} breakthroughs | {v6_dec:,} with decisions | {v6_oq:,} with open questions")

                domains = sdb.execute("""
                    SELECT domain_primary, COUNT(*) as cnt FROM summaries
                    GROUP BY domain_primary ORDER BY cnt DESC LIMIT 10
                """).fetchall()
                if domains:
                    output.append("  - **Top domains**: " + ", ".join(f"{d[0]} ({d[1]})" for d in domains))
            else:
                output.append("**Summaries**: unavailable")
        except Exception:
            output.append("**Summaries**: unavailable")

        # GitHub
        try:
            if cfg.github_repos_parquet.exists():
                gdb = get_github_db()
                repos = gdb.execute("SELECT COUNT(*) FROM github_repos").fetchone()[0]
                commits = 0
                if cfg.github_commits_parquet.exists():
                    commits = gdb.execute("SELECT COUNT(*) FROM github_commits").fetchone()[0]
                output.append(f"**GitHub**: {repos} repos, {commits:,} commits")
        except Exception:
            pass

        # Markdown
        try:
            mdb = get_markdown_db()
            if mdb:
                docs = mdb.execute("SELECT COUNT(*) FROM markdown_docs").fetchone()[0]
                output.append(f"**Markdown Docs**: {docs:,}")
        except Exception:
            pass

        output.append("\n_Use brain_stats(view='...') for details._")
        return "\n".join(output)

    def _conversation_stats() -> str:
        con = get_conversations()
        total = con.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        total_convs = con.execute("SELECT COUNT(DISTINCT conversation_id) FROM conversations").fetchone()[0]
        date_range = con.execute("SELECT MIN(created), MAX(created) FROM conversations").fetchone()
        sources = con.execute("SELECT source, COUNT(*) FROM conversations GROUP BY source ORDER BY COUNT(*) DESC").fetchall()
        models = con.execute("SELECT model, COUNT(*) FROM conversations GROUP BY model ORDER BY COUNT(*) DESC LIMIT 10").fetchall()

        output = [
            "## Conversation Archive Statistics\n",
            f"**Total Messages**: {total:,}",
            f"**Total Conversations**: {total_convs:,}",
            f"**Date Range**: {date_range[0]} to {date_range[1]}\n",
            "### By Source:",
        ]
        for source, count in sources:
            output.append(f"- {source}: {count:,}")
        output.append("\n### Top Models:")
        for model, count in models:
            output.append(f"- {model}: {count:,}")
        return "\n".join(output)

    def _embedding_stats() -> str:
        cfg = get_config()
        if not cfg.lance_path.exists():
            return "Vector database not found. Run the embed pipeline first."

        db = get_lance_db()
        if not db:
            return "Could not connect to vector database."

        try:
            tbl = db.open_table("message")
            total = tbl.count_rows()
            df = tbl.to_pandas()[["year"]].value_counts().reset_index()
            df.columns = ["year", "count"]
            by_year = sorted(df.values.tolist(), key=lambda x: x[0])
        except Exception as e:
            return f"Error: {e}"

        output = [
            "## Embedding Statistics (LanceDB)\n",
            f"**Total embedded messages**: {total:,}",
            f"**Embedding model**: {cfg.embedding.model}",
            f"**Dimensions**: {cfg.embedding.dim}\n",
            "### By Year:",
        ]
        for year, count in by_year:
            output.append(f"- {int(year)}: {count:,} messages")
        return "\n".join(output)

    def _github_stats() -> str:
        cfg = get_config()
        if not cfg.github_repos_parquet.exists():
            return "GitHub data not found."

        con = get_github_db()
        stats = con.execute("""
            SELECT COUNT(*), SUM(CASE WHEN is_private THEN 1 ELSE 0 END),
                   SUM(CASE WHEN NOT is_private THEN 1 ELSE 0 END),
                   MIN(created_at), MAX(pushed_at)
            FROM github_repos
        """).fetchone()

        output = ["## GitHub Statistics\n"]
        output.append(f"**Total Repos**: {stats[0]}")
        output.append(f"**Private**: {stats[1]} | **Public**: {stats[2]}")
        output.append(f"**Range**: {str(stats[3])[:10]} to {str(stats[4])[:10]}\n")

        languages = con.execute("""
            SELECT language, COUNT(*) FROM github_repos
            WHERE language IS NOT NULL GROUP BY language ORDER BY COUNT(*) DESC LIMIT 10
        """).fetchall()
        output.append("### Languages")
        for lang, count in languages:
            output.append(f"- {lang}: {count}")

        if cfg.github_commits_parquet.exists():
            commit_stats = con.execute("""
                SELECT COUNT(*), COUNT(DISTINCT repo_name), MIN(timestamp), MAX(timestamp)
                FROM github_commits
            """).fetchone()
            output.append(f"\n### Commits")
            output.append(f"**Total**: {commit_stats[0]:,}")
            output.append(f"**Range**: {str(commit_stats[2])[:10]} to {str(commit_stats[3])[:10]}")

        return "\n".join(output)

    def _markdown_stats() -> str:
        db = get_markdown_db()
        if not db:
            return "Markdown corpus not found."

        stats = db.execute("""
            SELECT COUNT(*), SUM(word_count),
                   SUM(CASE WHEN voice = 'FIRST_PERSON' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN energy = 'BREAKTHROUGH' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN depth_score >= 70 THEN 1 ELSE 0 END),
                   SUM(decision_count), SUM(todos_open)
            FROM markdown_docs
        """).fetchone()

        output = ["## Markdown Corpus Statistics\n"]
        output.append(f"**Total Files**: {stats[0]:,}")
        output.append(f"**Total Words**: {stats[1]:,}")
        output.append(f"**First Person Docs**: {stats[2]:,}")
        output.append(f"**Breakthrough Docs**: {stats[3]:,}")
        output.append(f"**Deep Docs (70+)**: {stats[4]:,}")
        output.append(f"**Total Decisions**: {stats[5]:,}")
        output.append(f"**Open TODOs**: {stats[6]:,}")
        return "\n".join(output)

    def _domain_map_view() -> str:
        sdb = get_summaries_db()
        if not sdb:
            return "Summary database not found."

        domains = sdb.execute("""
            SELECT domain_primary, COUNT(*) as count,
                   COUNT(CASE WHEN importance = 'breakthrough' THEN 1 END) as breakthroughs,
                   COUNT(CASE WHEN importance = 'significant' THEN 1 END) as significant,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) as pct
            FROM summaries GROUP BY domain_primary ORDER BY count DESC
        """).fetchall()

        if not domains:
            return "No domain data available."

        total = sum(d[1] for d in domains)
        output = [f"## Domain Map ({total:,} conversations)\n"]

        for domain, count, bt, sig, pct in domains:
            bar = "█" * int(pct / 2)
            bt_str = f" 🔥{bt}" if bt else ""
            sig_str = f" ⭐{sig}" if sig else ""
            output.append(f"**{domain}**: {count:,} ({pct}%) {bar}{bt_str}{sig_str}")

        # Top concepts per top 5 domains
        output.append("\n### Top Concepts by Domain\n")
        for domain, count, _, _, _ in domains[:5]:
            rows = sdb.execute("""
                SELECT concepts FROM summaries
                WHERE domain_primary = ? AND concepts IS NOT NULL LIMIT 50
            """, [domain]).fetchall()
            concept_counts = {}
            for row in rows:
                for c in parse_json_field(row[0]):
                    if c:
                        concept_counts[c] = concept_counts.get(c, 0) + 1
            top_concepts = sorted(concept_counts.items(), key=lambda x: -x[1])[:8]
            if top_concepts:
                output.append(f"**{domain}**: {', '.join(c[0] for c in top_concepts)}")

        return "\n".join(output)

    def _thinking_pulse_view(domain: str = None) -> str:
        sdb = get_summaries_db()
        if not sdb:
            return "Summary database not found."

        if domain:
            stages = sdb.execute("""
                SELECT thinking_stage, COUNT(*) as cnt,
                       COUNT(CASE WHEN importance = 'breakthrough' THEN 1 END) as breakthroughs
                FROM summaries WHERE domain_primary = ? AND thinking_stage IS NOT NULL
                GROUP BY thinking_stage ORDER BY cnt DESC
            """, [domain]).fetchall()
            if not stages:
                return f"No data for domain: {domain}"
            output = [f"## Thinking Pulse: {domain}\n"]
            stage_icons = {"exploring": "🔍", "crystallizing": "💎", "refining": "🔧", "executing": "🚀"}
            for stage, cnt, bt in stages:
                icon = stage_icons.get(stage, "📝")
                bt_str = f" (🔥{bt} breakthroughs)" if bt else ""
                output.append(f"{icon} **{stage}**: {cnt} conversations{bt_str}")
            return "\n".join(output)

        crosstab = sdb.execute("""
            SELECT domain_primary,
                   COUNT(CASE WHEN thinking_stage = 'exploring' THEN 1 END) as exploring,
                   COUNT(CASE WHEN thinking_stage = 'crystallizing' THEN 1 END) as crystallizing,
                   COUNT(CASE WHEN thinking_stage = 'refining' THEN 1 END) as refining,
                   COUNT(CASE WHEN thinking_stage = 'executing' THEN 1 END) as executing,
                   COUNT(*) as total
            FROM summaries WHERE thinking_stage IS NOT NULL
            GROUP BY domain_primary ORDER BY total DESC LIMIT 25
        """).fetchall()

        if not crosstab:
            return "No thinking pulse data available."

        output = ["## Thinking Pulse (all domains)\n"]
        output.append(f"{'Domain':<25s} {'🔍 Expl':>8s} {'💎 Cryst':>8s} {'🔧 Refn':>8s} {'🚀 Exec':>8s} {'Total':>7s}")
        output.append("-" * 72)

        crystallizing_domains = []
        exploring_domains = []

        for domain, expl, cryst, refn, exec_, total in crosstab:
            output.append(f"{domain:<25s} {expl:>8d} {cryst:>8d} {refn:>8d} {exec_:>8d} {total:>7d}")
            if cryst > expl and cryst > 0:
                crystallizing_domains.append((domain, cryst))
            if expl > cryst + refn + exec_ and expl > 0:
                exploring_domains.append((domain, expl))

        if crystallizing_domains:
            output.append("\n### 💎 Crystallizing (ready to ship)")
            for domain, cnt in sorted(crystallizing_domains, key=lambda x: -x[1])[:5]:
                output.append(f"  - {domain} ({cnt} crystallizing)")

        if exploring_domains:
            output.append("\n### 🔍 Still Exploring")
            for domain, cnt in sorted(exploring_domains, key=lambda x: -x[1])[:5]:
                output.append(f"  - {domain} ({cnt} exploring)")

        return "\n".join(output)
