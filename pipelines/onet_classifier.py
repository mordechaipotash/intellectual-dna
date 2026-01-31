"""
O*NET Work Activity Classifier for Intellectual DNA

Classifies conversation messages into O*NET's 41 Generalized Work Activities
to show where AI is augmenting your work.
"""

import duckdb
from pathlib import Path
from collections import Counter
import json

# O*NET 41 Generalized Work Activities
ONET_WORK_ACTIVITIES = {
    # Information Input (4.A.1)
    "4.A.1.a.1": "Getting Information",
    "4.A.1.a.2": "Monitoring Processes, Materials, or Surroundings",
    "4.A.1.b.1": "Identifying Objects, Actions, and Events",
    "4.A.1.b.2": "Inspecting Equipment, Structures, or Materials",
    "4.A.1.b.3": "Estimating Quantifiable Characteristics",

    # Mental Processes (4.A.2)
    "4.A.2.a.1": "Judging Qualities of Objects, Services, or People",
    "4.A.2.a.2": "Processing Information",
    "4.A.2.a.3": "Evaluating Information for Compliance",
    "4.A.2.a.4": "Analyzing Data or Information",
    "4.A.2.b.1": "Making Decisions and Solving Problems",
    "4.A.2.b.2": "Thinking Creatively",
    "4.A.2.b.3": "Updating and Using Relevant Knowledge",
    "4.A.2.b.4": "Developing Objectives and Strategies",
    "4.A.2.b.5": "Scheduling Work and Activities",
    "4.A.2.b.6": "Organizing, Planning, and Prioritizing Work",

    # Work Output (4.A.3)
    "4.A.3.a.1": "Performing General Physical Activities",
    "4.A.3.a.2": "Handling and Moving Objects",
    "4.A.3.a.3": "Controlling Machines and Processes",
    "4.A.3.a.4": "Operating Vehicles or Equipment",
    "4.A.3.b.1": "Working with Computers",
    "4.A.3.b.2": "Drafting Technical Specifications",
    "4.A.3.b.4": "Repairing Mechanical Equipment",
    "4.A.3.b.5": "Repairing Electronic Equipment",
    "4.A.3.b.6": "Documenting/Recording Information",

    # Interacting with Others (4.A.4)
    "4.A.4.a.1": "Interpreting Information for Others",
    "4.A.4.a.2": "Communicating with Peers",
    "4.A.4.a.3": "Communicating Outside Organization",
    "4.A.4.a.4": "Establishing Interpersonal Relationships",
    "4.A.4.a.5": "Assisting and Caring for Others",
    "4.A.4.a.6": "Selling or Influencing Others",
    "4.A.4.a.7": "Resolving Conflicts and Negotiating",
    "4.A.4.a.8": "Working with the Public",
    "4.A.4.b.1": "Coordinating Work of Others",
    "4.A.4.b.2": "Developing and Building Teams",
    "4.A.4.b.3": "Training and Teaching Others",
    "4.A.4.b.4": "Guiding and Motivating Subordinates",
    "4.A.4.b.5": "Coaching and Developing Others",
    "4.A.4.b.6": "Providing Consultation and Advice",
    "4.A.4.c.1": "Performing Administrative Activities",
    "4.A.4.c.2": "Staffing Organizational Units",
    "4.A.4.c.3": "Monitoring and Controlling Resources",
}

# Keyword patterns for classification
ACTIVITY_KEYWORDS = {
    "4.A.1.a.1": ["search", "find", "look up", "research", "what is", "how does", "tell me", "show me", "get"],
    "4.A.2.a.4": ["analyze", "data", "pattern", "statistics", "metrics", "query", "sql", "duckdb"],
    "4.A.2.b.1": ["fix", "error", "bug", "debug", "problem", "solve", "issue", "broken", "not working"],
    "4.A.2.b.2": ["create", "design", "idea", "brainstorm", "imagine", "what if", "concept", "novel"],
    "4.A.2.b.3": ["learn", "understand", "explain", "how to", "tutorial", "documentation", "docs"],
    "4.A.2.b.4": ["plan", "strategy", "objective", "goal", "roadmap", "milestone"],
    "4.A.2.b.6": ["organize", "prioritize", "todo", "task", "schedule", "workflow"],
    "4.A.3.b.1": ["code", "script", "function", "implement", "build", "python", "javascript", "sql"],
    "4.A.3.b.2": ["api", "schema", "architecture", "design", "spec", "interface", "structure"],
    "4.A.3.b.6": ["document", "write", "readme", "markdown", "record", "log", "note"],
    "4.A.4.a.1": ["summarize", "explain to", "translate", "simplify", "clarify"],
    "4.A.4.a.6": ["pitch", "convince", "sell", "market", "promote", "reddit", "post"],
    "4.A.4.b.3": ["teach", "train", "guide", "tutorial", "lesson"],
    "4.A.4.b.6": ["advise", "recommend", "suggest", "consult", "help me decide", "what do you think"],
}

def classify_message(content: str) -> list[str]:
    """Classify a message into O*NET work activities based on keywords."""
    if not content:
        return []

    content_lower = content.lower()
    matches = []

    for activity_code, keywords in ACTIVITY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in content_lower:
                matches.append(activity_code)
                break

    # Default: if working with computers detected via common tech terms
    tech_indicators = ["python", "javascript", "git", "docker", "supabase", "claude", "api", "json", "parquet"]
    if not matches and any(ind in content_lower for ind in tech_indicators):
        matches.append("4.A.3.b.1")  # Working with Computers

    return list(set(matches))


def analyze_month(month: str = "2025-12"):
    """Analyze a month's messages and classify into O*NET activities."""
    base = Path("/Users/mordechai/intellectual_dna")
    parquet_path = base / "data" / "all_conversations.parquet"

    conn = duckdb.connect()

    # Get user messages from the specified month
    query = f"""
    SELECT
        message_id,
        content,
        conversation_title,
        created,
        source
    FROM read_parquet('{parquet_path}')
    WHERE role = 'user'
      AND content IS NOT NULL
      AND length(content) > 20
      AND strftime(created, '%Y-%m') = '{month}'
    ORDER BY created DESC
    LIMIT 5000
    """

    messages = conn.execute(query).fetchall()
    print(f"\n📊 Analyzing {len(messages)} messages from {month}\n")

    # Classify each message
    activity_counts = Counter()
    classified_messages = []

    for msg_id, content, title, created, source in messages:
        activities = classify_message(content)
        for act in activities:
            activity_counts[act] += 1
        if activities:
            classified_messages.append({
                "content": content[:200],
                "activities": activities,
                "title": title
            })

    # Calculate percentages
    total_classified = sum(activity_counts.values())

    print("=" * 60)
    print(f"🎯 O*NET WORK ACTIVITY HEATMAP - {month}")
    print("=" * 60)
    print(f"Total messages: {len(messages)}")
    print(f"Classified into activities: {total_classified} assignments")
    print()

    # Group by category
    categories = {
        "Information Input": [],
        "Mental Processes": [],
        "Work Output": [],
        "Interacting with Others": []
    }

    for code, count in sorted(activity_counts.items(), key=lambda x: -x[1]):
        name = ONET_WORK_ACTIVITIES.get(code, code)
        pct = (count / len(messages)) * 100

        if code.startswith("4.A.1"):
            categories["Information Input"].append((name, count, pct))
        elif code.startswith("4.A.2"):
            categories["Mental Processes"].append((name, count, pct))
        elif code.startswith("4.A.3"):
            categories["Work Output"].append((name, count, pct))
        elif code.startswith("4.A.4"):
            categories["Interacting with Others"].append((name, count, pct))

    for category, items in categories.items():
        if items:
            print(f"\n### {category}")
            print("-" * 50)
            for name, count, pct in sorted(items, key=lambda x: -x[1]):
                bar = "█" * int(pct / 2) + "░" * (25 - int(pct / 2))
                print(f"{bar} {pct:5.1f}% | {name}")

    # AI Augmentation insight
    print("\n" + "=" * 60)
    print("💡 AI AUGMENTATION INSIGHT")
    print("=" * 60)

    # Top 5 activities
    top5 = sorted(activity_counts.items(), key=lambda x: -x[1])[:5]
    print("\nTop 5 AI-assisted work activities:")
    for i, (code, count) in enumerate(top5, 1):
        name = ONET_WORK_ACTIVITIES.get(code, code)
        pct = (count / len(messages)) * 100
        print(f"  {i}. {name}: {pct:.1f}%")

    # Sample messages per top activity
    print("\n📝 Sample Messages by Activity:")
    for code, _ in top5[:3]:
        name = ONET_WORK_ACTIVITIES.get(code, code)
        print(f"\n  [{name}]")
        samples = [m for m in classified_messages if code in m["activities"]][:2]
        for s in samples:
            preview = s["content"][:80].replace("\n", " ")
            print(f"    • \"{preview}...\"")

    conn.close()
    return activity_counts


if __name__ == "__main__":
    import sys
    month = sys.argv[1] if len(sys.argv) > 1 else "2025-12"
    analyze_month(month)
