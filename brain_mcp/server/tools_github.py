"""
brain-mcp — GitHub integration tools.

Cross-reference GitHub repos, commits, and conversations.
These tools are optional and only work when GitHub data has been imported.
"""

import re
from typing import Optional

from brain_mcp.config import get_config
from .db import (
    get_conversations,
    get_embedding,
    get_github_db,
    get_lance_db,
    lance_search,
)


def register(mcp):
    """Register GitHub tools with the MCP server."""

    @mcp.tool()
    def github_search(
        query: str = "",
        project: str = None,
        mode: str = "timeline",
        limit: int = 10,
    ) -> str:
        """
        Search GitHub repos, commits, and cross-reference with conversations.

        Args:
            query: Search query (used for code semantic search or as conversation_id for validate mode)
            project: Project/repo name (used for timeline and conversations modes)
            mode: Search mode:
                - "timeline" (default): Project creation date, commits, activity windows
                - "conversations": Find conversations mentioning a project
                - "code": Semantic search across commits AND conversations
                - "validate": Check conversation date validity via GitHub evidence.
                              Pass conversation_id as query.
            limit: Max results (default 10)
        """
        cfg = get_config()

        if mode == "conversations":
            return _conversation_project_context(project or query, limit)
        elif mode == "code":
            return _code_to_conversation(query, limit)
        elif mode == "validate":
            return _validate_date_with_github(query)

        # Default: timeline mode
        project_name = project or query
        if not cfg.github_repos_parquet.exists():
            return "GitHub data not imported. Add GitHub data and run ingest first."

        con = get_github_db()
        if not con:
            return "GitHub database not available."

        pattern = f"%{project_name.lower()}%"

        try:
            repos = con.execute("""
                SELECT repo_name, created_at, pushed_at, description,
                       language, is_private, stars, url
                FROM github_repos
                WHERE LOWER(repo_name) LIKE ?
                ORDER BY created_at DESC
                LIMIT 5
            """, [pattern]).fetchall()
        except Exception as e:
            return f"Error querying GitHub repos: {e}"

        if not repos:
            return f"No GitHub project found matching '{project_name}'"

        output = [f"## GitHub Project Timeline: '{project_name}'\n"]

        for repo in repos:
            name, created, pushed, desc, lang, private, stars, url = repo
            output.append(f"### {name}")
            output.append(f"**Created**: {str(created)[:10]}")
            output.append(f"**Last pushed**: {str(pushed)[:10] if pushed else 'N/A'}")
            output.append(f"**Language**: {lang or 'N/A'}")
            output.append(f"**Private**: {'Yes' if private else 'No'}")
            output.append(f"**Stars**: {stars}")
            if desc:
                output.append(f"**Description**: {desc[:100]}")
            output.append(f"**URL**: {url}\n")

            # Get commits for this repo
            if cfg.github_commits_parquet.exists():
                try:
                    commits = con.execute("""
                        SELECT timestamp, message, author
                        FROM github_commits
                        WHERE repo_name = ?
                        ORDER BY timestamp DESC
                        LIMIT 10
                    """, [name]).fetchall()

                    if commits:
                        output.append("**Recent Commits**:")
                        for ts, msg, _ in commits:
                            msg_preview = msg.split('\n')[0][:60]
                            output.append(f"  - [{str(ts)[:10]}] {msg_preview}...")

                        monthly = con.execute("""
                            SELECT strftime(timestamp, '%Y-%m') as month,
                                   COUNT(*) as count
                            FROM github_commits
                            WHERE repo_name = ?
                            GROUP BY 1 ORDER BY 1
                        """, [name]).fetchall()

                        if monthly:
                            output.append("\n**Activity by Month**:")
                            for month, count in monthly[-6:]:
                                bar = "█" * min(count, 20)
                                output.append(f"  {month}: {bar} ({count})")
                except Exception:
                    pass

        return "\n".join(output)

    def _conversation_project_context(project: str, limit: int = 10) -> str:
        """Find conversations mentioning a specific GitHub project."""
        cfg = get_config()

        try:
            con = get_conversations()
        except FileNotFoundError as e:
            return str(e)

        pattern = f"%{project.lower()}%"

        results = con.execute("""
            SELECT conversation_title,
                   substr(content, 1, 250) as preview,
                   created, role, conversation_id
            FROM conversations
            WHERE (LOWER(content) LIKE ? OR LOWER(conversation_title) LIKE ?)
              AND role = 'user'
            ORDER BY created DESC
            LIMIT ?
        """, [pattern, pattern, limit]).fetchall()

        if not results:
            return f"No conversations found mentioning '{project}'"

        # Get project creation date for validation
        project_created = None
        if cfg.github_repos_parquet.exists():
            gh_db = get_github_db()
            if gh_db:
                try:
                    repo_result = gh_db.execute("""
                        SELECT repo_name, created_at
                        FROM github_repos
                        WHERE LOWER(repo_name) LIKE ?
                        LIMIT 1
                    """, [pattern]).fetchone()
                    if repo_result:
                        project_created = repo_result[1]
                except Exception:
                    pass

        output = [f"## Conversations about: '{project}'\n"]

        if project_created:
            output.append(f"_GitHub project created: {str(project_created)[:10]}_\n")

        for title, preview, created, _, conv_id in results:
            date_flag = ""
            if project_created:
                try:
                    if created < project_created:
                        date_flag = " ⚠️ PREDATES PROJECT"
                except Exception:
                    pass

            output.append(
                f"**[{str(created)[:10]}]** "
                f"{title or 'Untitled'}{date_flag}"
            )
            output.append(f"> {preview}...")
            output.append(f"_ID: {conv_id[:20]}..._\n")

        return "\n".join(output)

    def _code_to_conversation(query: str, limit: int = 10) -> str:
        """Semantic search across commits AND conversations."""
        cfg = get_config()

        embedding = get_embedding(query)
        if not embedding:
            return "Could not generate embedding for query."

        output = [f"## Code ↔ Conversation Search: '{query}'\n"]

        if cfg.lance_path.exists():
            db = get_lance_db()
            if db:
                try:
                    table_names = (
                        db.table_names()
                        if hasattr(db, "table_names")
                        else []
                    )
                    if "commit" in table_names:
                        tbl = db.open_table("commit")
                        commit_df = tbl.search(embedding).limit(
                            limit // 2
                        ).to_pandas()
                        if not commit_df.empty:
                            output.append("### Related Commits")
                            for _, row in commit_df.iterrows():
                                repo = row.get("repo_name", "unknown")
                                msg = str(row.get("message", ""))
                                ts = row.get("timestamp", "")
                                sim = 1 / (1 + row.get("_distance", 0))
                                msg_preview = msg.split("\n")[0][:80]
                                output.append(f"**[{repo}]** {msg_preview}")
                                output.append(
                                    f"  {str(ts)[:10]} | "
                                    f"Similarity: {sim:.3f}\n"
                                )

                    conv_results = lance_search(embedding, limit=limit // 2)
                    if conv_results:
                        output.append("### Related Conversations")
                        for title, content, year, month, sim in conv_results:
                            preview = content[:150]
                            output.append(
                                f"**[{year}-{month:02d}]** "
                                f"{title or 'Untitled'}"
                            )
                            output.append(f"> {preview}...")
                            output.append(f"Similarity: {sim:.3f}\n")
                except Exception as e:
                    output.append(f"_Search error: {e}_")

        if len(output) == 1:
            output.append(
                "_No embeddings found. Run the embed pipeline first._"
            )

        return "\n".join(output)

    def _validate_date_with_github(conversation_id: str) -> str:
        """Check conversation date validity via GitHub evidence."""
        cfg = get_config()

        try:
            con = get_conversations()
        except FileNotFoundError as e:
            return str(e)

        conv = con.execute("""
            SELECT conversation_title,
                   MIN(created) as first_msg,
                   MAX(created) as last_msg,
                   COUNT(*) as msg_count,
                   MAX(timestamp_is_fallback) as has_fallback
            FROM conversations
            WHERE conversation_id = ?
            GROUP BY conversation_title
        """, [conversation_id]).fetchone()

        if not conv:
            return f"Conversation not found: {conversation_id}"

        title, first_msg, _, msg_count, has_fallback = conv

        output = [f"## Date Validation: {title or 'Untitled'}\n"]
        output.append(f"**Conversation ID**: {conversation_id[:30]}...")
        output.append(f"**Recorded date**: {str(first_msg)[:10]}")
        output.append(f"**Messages**: {msg_count}")
        output.append(
            f"**Fallback timestamp**: "
            f"{'Yes ⚠️' if has_fallback else 'No ✓'}\n"
        )

        content = con.execute("""
            SELECT content FROM conversations
            WHERE conversation_id = ? AND role = 'user'
            LIMIT 50
        """, [conversation_id]).fetchall()

        all_content = " ".join([c[0] for c in content if c[0]])

        if not cfg.github_repos_parquet.exists():
            output.append(
                "_GitHub data not available for validation._"
            )
            return "\n".join(output)

        gh_db = get_github_db()
        if not gh_db:
            output.append("_GitHub database not available._")
            return "\n".join(output)

        try:
            repos = gh_db.execute("""
                SELECT repo_name, created_at FROM github_repos
            """).fetchall()
        except Exception:
            output.append("_Could not query GitHub repos._")
            return "\n".join(output)

        issues = []
        validations = []

        for repo_name, created_at in repos:
            if re.search(
                rf"\b{re.escape(repo_name)}\b", all_content, re.IGNORECASE
            ):
                if first_msg < created_at:
                    issues.append({
                        "project": repo_name,
                        "project_created": str(created_at)[:10],
                        "conv_date": str(first_msg)[:10],
                        "days_before": (created_at - first_msg).days,
                    })
                else:
                    validations.append({
                        "project": repo_name,
                        "project_created": str(created_at)[:10],
                    })

        if issues:
            output.append("### ⚠️ Date Conflicts Found")
            output.append(
                "_Conversation mentions projects that didn't exist yet_\n"
            )
            for issue in issues:
                output.append(
                    f"- **{issue['project']}** created "
                    f"{issue['project_created']}"
                )
                output.append(
                    f"  But conversation dated {issue['conv_date']} "
                    f"({issue['days_before']} days before!)"
                )

        if validations:
            output.append("\n### ✓ Valid Project References")
            for v in validations[:5]:
                output.append(
                    f"- {v['project']} (created {v['project_created']})"
                )

        if not issues and not validations:
            output.append(
                "_No GitHub project references found in this conversation._"
            )

        output.append("\n### Verdict")
        if issues:
            output.append(
                "🔴 **DATE LIKELY INCORRECT** — Conversation references "
                "projects that didn't exist yet."
            )
            output.append(
                f"   Earliest valid date: "
                f"{max(i['project_created'] for i in issues)}"
            )
        elif has_fallback:
            output.append(
                "🟡 **UNCERTAIN** — Uses fallback timestamp, "
                "but no conflicting evidence found."
            )
        else:
            output.append(
                "🟢 **LIKELY VALID** — No date conflicts detected."
            )

        return "\n".join(output)
