"""
brain-mcp — Conversation tools.

Tools for retrieving and browsing raw conversation data.
"""

from .db import get_conversations


def register(mcp):
    """Register conversation tools with the MCP server."""

    @mcp.tool()
    def search_conversations(term: str = "", limit: int = 15, role: str = None) -> str:
        """
        Full-text search across all conversation messages.

        Args:
            term: Search term (keyword). If empty with role="user", finds recent user questions.
            limit: Max results (default 15)
            role: Filter by role — "user" for your words, "assistant" for AI responses.
                  With role="user" and empty term, returns recent questions asked.
        """
        con = get_conversations()

        # Special mode: find recent user questions when no term and role=user
        if (not term or not term.strip()) and role == "user":
            results = con.execute("""
                SELECT substr(content, 1, 200) as question,
                       conversation_title, source, created
                FROM conversations
                WHERE has_question = 1 AND role = 'user'
                ORDER BY created DESC
                LIMIT ?
            """, [limit]).fetchall()

            output = ["## Recent Questions Asked\n"]
            for question, title, source, created in results:
                output.append(f"**[{created}]** {question}")
                output.append(f"  _From: {title or 'Untitled'} ({source})_\n")
            return "\n".join(output)

        pattern = f"%{term}%"

        if role:
            results = con.execute("""
                SELECT source, model, conversation_title, role,
                       substr(content, 1, 200) as preview,
                       created, conversation_id
                FROM conversations
                WHERE content ILIKE ? AND role = ?
                ORDER BY created DESC
                LIMIT ?
            """, [pattern, role, limit]).fetchall()
        else:
            results = con.execute("""
                SELECT source, model, conversation_title, role,
                       substr(content, 1, 200) as preview,
                       created, conversation_id
                FROM conversations
                WHERE content ILIKE ?
                ORDER BY created DESC
                LIMIT ?
            """, [pattern, limit]).fetchall()

        if not results:
            return f"No conversations found containing '{term}'"

        output = [f"## Conversations containing '{term}' ({len(results)} found)\n"]
        for source, model, title, msg_role, preview, created, conv_id in results:
            output.append(f"**[{created}]** {title or 'Untitled'}")
            output.append(f"  {msg_role}: {preview}...")
            output.append(f"  _ID: {conv_id[:20]}... | {source}/{model}_\n")
        return "\n".join(output)

    @mcp.tool()
    def get_conversation(conversation_id: str) -> str:
        """
        Get the full content of a specific conversation by ID.
        Use search_conversations() first to find conversation IDs.
        """
        con = get_conversations()
        messages = con.execute("""
            SELECT role, content, msg_timestamp
            FROM conversations
            WHERE conversation_id = ?
            ORDER BY msg_index ASC
        """, [conversation_id]).fetchall()

        if not messages:
            return f"Conversation not found: {conversation_id}"

        output = [f"## Conversation ({len(messages)} messages)\n"]
        for role, content, ts in messages[:20]:
            output.append(f"### {role.upper()} [{ts}]")
            output.append(str(content)[:1000] if content else "(empty)")
            if content and len(str(content)) > 1000:
                output.append(f"_... ({len(str(content))} chars total)_")
            output.append("")

        if len(messages) > 20:
            output.append(f"_... {len(messages) - 20} more messages_")

        return "\n".join(output)

    @mcp.tool()
    def conversations_by_date(date: str, limit: int = 30) -> str:
        """
        Get conversations from a specific date (YYYY-MM-DD format).
        """
        con = get_conversations()
        results = con.execute("""
            SELECT DISTINCT conversation_id, conversation_title, source, model, created
            FROM conversations
            WHERE CAST(created AS DATE) = CAST(? AS DATE)
            ORDER BY created DESC
            LIMIT ?
        """, [date, limit]).fetchall()

        if not results:
            return f"No conversations found on {date}"

        output = [f"## Conversations on {date} ({len(results)} found)\n"]
        for conv_id, title, source, model, _ in results:
            output.append(f"- **{title or 'Untitled'}**")
            output.append(f"  {source}/{model} | ID: {conv_id[:20]}...")
        return "\n".join(output)
