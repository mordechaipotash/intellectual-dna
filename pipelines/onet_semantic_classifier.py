"""
O*NET Semantic Classifier for Intellectual DNA

Uses nomic-embed-text embeddings to classify messages into O*NET work activities
by semantic similarity - much more accurate than keyword matching.
"""

import sys
from pathlib import Path
import numpy as np
from collections import Counter
import duckdb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.utils.embedding_utils import get_embeddings_batch, get_embedding

# O*NET Work Activities with rich descriptions for better semantic matching
ONET_ACTIVITIES = {
    # Information Input
    "getting_information": {
        "code": "4.A.1.a.1",
        "name": "Getting Information",
        "description": "Observing, receiving, and otherwise obtaining information from all relevant sources. Searching for data, researching topics, looking up documentation, asking questions to understand something, gathering facts and details.",
        "category": "Information Input"
    },
    "monitoring": {
        "code": "4.A.1.a.2",
        "name": "Monitoring Processes",
        "description": "Monitoring and reviewing information from materials, events, or the environment to detect or assess problems. Watching for errors, checking status, tracking progress, observing system behavior.",
        "category": "Information Input"
    },
    "identifying": {
        "code": "4.A.1.b.1",
        "name": "Identifying Objects and Events",
        "description": "Identifying information by categorizing, estimating, recognizing differences or similarities, and detecting changes. Classifying items, pattern recognition, categorization.",
        "category": "Information Input"
    },
    "inspecting": {
        "code": "4.A.1.b.2",
        "name": "Inspecting Equipment",
        "description": "Inspecting equipment, structures, or materials to identify causes of errors or problems. Code review, debugging, examining systems, checking for issues.",
        "category": "Information Input"
    },
    "estimating": {
        "code": "4.A.1.b.3",
        "name": "Estimating Quantities",
        "description": "Estimating sizes, distances, quantities, time, cost, resources, or materials needed. Planning resources, estimating effort, calculating requirements.",
        "category": "Information Input"
    },

    # Mental Processes
    "judging_qualities": {
        "code": "4.A.2.a.1",
        "name": "Judging Qualities",
        "description": "Judging the qualities of things, services, or people. Evaluating code quality, assessing options, comparing alternatives, rating performance, reviewing work.",
        "category": "Mental Processes"
    },
    "processing_info": {
        "code": "4.A.2.a.2",
        "name": "Processing Information",
        "description": "Compiling, coding, categorizing, calculating, tabulating, auditing, or verifying information or data. Data processing, parsing, transforming, formatting.",
        "category": "Mental Processes"
    },
    "evaluating_compliance": {
        "code": "4.A.2.a.3",
        "name": "Evaluating Compliance",
        "description": "Using relevant information and individual judgment to determine whether events or processes comply with laws, regulations, or standards. Validation, verification, testing against requirements.",
        "category": "Mental Processes"
    },
    "analyzing_data": {
        "code": "4.A.2.a.4",
        "name": "Analyzing Data",
        "description": "Identifying the underlying principles, reasons, or facts of information by breaking down data into separate parts. Data analysis, SQL queries, statistics, exploring patterns, understanding metrics.",
        "category": "Mental Processes"
    },
    "problem_solving": {
        "code": "4.A.2.b.1",
        "name": "Making Decisions and Solving Problems",
        "description": "Analyzing information and evaluating results to choose the best solution and solve problems. Debugging, fixing errors, troubleshooting, finding solutions, resolving issues.",
        "category": "Mental Processes"
    },
    "thinking_creatively": {
        "code": "4.A.2.b.2",
        "name": "Thinking Creatively",
        "description": "Developing, designing, or creating new applications, ideas, relationships, systems, or products. Brainstorming, innovation, inventing, imagining possibilities, creative design.",
        "category": "Mental Processes"
    },
    "updating_knowledge": {
        "code": "4.A.2.b.3",
        "name": "Updating Knowledge",
        "description": "Keeping up-to-date technically and applying new knowledge to your job. Learning new technologies, reading documentation, understanding new concepts, studying tutorials.",
        "category": "Mental Processes"
    },
    "developing_strategy": {
        "code": "4.A.2.b.4",
        "name": "Developing Objectives and Strategies",
        "description": "Establishing long-range objectives and specifying strategies and actions to achieve them. Strategic planning, goal setting, roadmapping, defining milestones.",
        "category": "Mental Processes"
    },
    "scheduling": {
        "code": "4.A.2.b.5",
        "name": "Scheduling Work",
        "description": "Scheduling events, programs, and activities. Time management, setting deadlines, planning timelines, coordinating schedules.",
        "category": "Mental Processes"
    },
    "organizing_work": {
        "code": "4.A.2.b.6",
        "name": "Organizing and Prioritizing",
        "description": "Developing specific goals and plans to prioritize, organize, and accomplish your work. Task management, todo lists, prioritization, workflow organization, planning next steps.",
        "category": "Mental Processes"
    },

    # Work Output
    "physical_activities": {
        "code": "4.A.3.a.1",
        "name": "Physical Activities",
        "description": "Performing physical activities that require considerable use of arms and legs and moving whole body. Physical labor, manual work, hands-on tasks.",
        "category": "Work Output"
    },
    "handling_objects": {
        "code": "4.A.3.a.2",
        "name": "Handling Objects",
        "description": "Using hands and arms in handling, installing, positioning, and moving materials. Manual manipulation, assembly, physical setup.",
        "category": "Work Output"
    },
    "controlling_machines": {
        "code": "4.A.3.a.3",
        "name": "Controlling Machines",
        "description": "Using control mechanisms or direct physical activity to operate machines or processes. Operating equipment, running automated systems.",
        "category": "Work Output"
    },
    "operating_vehicles": {
        "code": "4.A.3.a.4",
        "name": "Operating Vehicles",
        "description": "Running, maneuvering, navigating, or driving vehicles or mechanized equipment.",
        "category": "Work Output"
    },
    "working_with_computers": {
        "code": "4.A.3.b.1",
        "name": "Working with Computers",
        "description": "Using computers and computer systems to program, write software, set up functions, enter data, or process information. Coding, programming, scripting, building applications, implementing features.",
        "category": "Work Output"
    },
    "drafting_technical": {
        "code": "4.A.3.b.2",
        "name": "Drafting Technical Specifications",
        "description": "Providing documentation, detailed instructions, drawings, or specifications for others. Writing API docs, creating schemas, designing system architecture, technical specifications.",
        "category": "Work Output"
    },
    "repairing_mechanical": {
        "code": "4.A.3.b.4",
        "name": "Repairing Mechanical Equipment",
        "description": "Servicing, repairing, adjusting, and testing machines, devices, moving parts, and equipment.",
        "category": "Work Output"
    },
    "repairing_electronic": {
        "code": "4.A.3.b.5",
        "name": "Repairing Electronic Equipment",
        "description": "Servicing, repairing, calibrating, regulating, fine-tuning, or testing machines that use electrical or electronic circuits.",
        "category": "Work Output"
    },
    "documenting": {
        "code": "4.A.3.b.6",
        "name": "Documenting Information",
        "description": "Entering, transcribing, recording, storing, or maintaining information in written or electronic form. Writing documentation, logging, note-taking, creating records, markdown files.",
        "category": "Work Output"
    },

    # Interacting with Others
    "interpreting_info": {
        "code": "4.A.4.a.1",
        "name": "Interpreting Information for Others",
        "description": "Translating or explaining what information means and how it can be used. Summarizing, simplifying complex topics, explaining technical concepts, clarifying meaning.",
        "category": "Interacting with Others"
    },
    "communicating_peers": {
        "code": "4.A.4.a.2",
        "name": "Communicating with Peers",
        "description": "Providing information to supervisors, co-workers, and subordinates by telephone, email, or in person. Team communication, collaboration, discussing with colleagues.",
        "category": "Interacting with Others"
    },
    "communicating_external": {
        "code": "4.A.4.a.3",
        "name": "Communicating Externally",
        "description": "Communicating with people outside the organization. Client communication, public relations, external stakeholders, community engagement.",
        "category": "Interacting with Others"
    },
    "relationships": {
        "code": "4.A.4.a.4",
        "name": "Building Relationships",
        "description": "Developing constructive and cooperative working relationships with others. Networking, rapport building, maintaining professional relationships.",
        "category": "Interacting with Others"
    },
    "assisting_others": {
        "code": "4.A.4.a.5",
        "name": "Assisting Others",
        "description": "Providing personal assistance, emotional support, or other personal care. Helping colleagues, supporting team members, mentoring.",
        "category": "Interacting with Others"
    },
    "selling_influencing": {
        "code": "4.A.4.a.6",
        "name": "Selling or Influencing",
        "description": "Convincing others to buy merchandise or to change their minds or actions. Marketing, promoting, pitching ideas, persuading, Reddit posts, social media promotion.",
        "category": "Interacting with Others"
    },
    "resolving_conflicts": {
        "code": "4.A.4.a.7",
        "name": "Resolving Conflicts",
        "description": "Handling complaints, settling disputes, resolving grievances and conflicts, or negotiating with others.",
        "category": "Interacting with Others"
    },
    "working_with_public": {
        "code": "4.A.4.a.8",
        "name": "Working with Public",
        "description": "Performing for people or dealing directly with the public. Public speaking, presentations, customer service, public engagement.",
        "category": "Interacting with Others"
    },
    "coordinating_others": {
        "code": "4.A.4.b.1",
        "name": "Coordinating Work of Others",
        "description": "Getting members of a group to work together to accomplish tasks. Project management, team coordination, orchestrating efforts.",
        "category": "Interacting with Others"
    },
    "building_teams": {
        "code": "4.A.4.b.2",
        "name": "Building Teams",
        "description": "Encouraging and building mutual trust, respect, and cooperation among team members.",
        "category": "Interacting with Others"
    },
    "training_teaching": {
        "code": "4.A.4.b.3",
        "name": "Training and Teaching",
        "description": "Identifying educational needs of others, developing programs, and teaching. Creating tutorials, writing guides, educational content, explaining how-to.",
        "category": "Interacting with Others"
    },
    "guiding_motivating": {
        "code": "4.A.4.b.4",
        "name": "Guiding Subordinates",
        "description": "Providing guidance and direction to subordinates, including setting performance standards and monitoring performance.",
        "category": "Interacting with Others"
    },
    "coaching": {
        "code": "4.A.4.b.5",
        "name": "Coaching Others",
        "description": "Identifying developmental needs of others and coaching, mentoring, or helping others improve their knowledge or skills.",
        "category": "Interacting with Others"
    },
    "consulting_advising": {
        "code": "4.A.4.b.6",
        "name": "Consulting and Advising",
        "description": "Providing guidance and expert advice to management or other groups. Consulting, giving recommendations, suggesting approaches, advising on decisions.",
        "category": "Interacting with Others"
    },
    "administrative": {
        "code": "4.A.4.c.1",
        "name": "Administrative Activities",
        "description": "Performing day-to-day administrative tasks such as maintaining files and processing paperwork. Admin work, filing, organizing records.",
        "category": "Interacting with Others"
    },
    "staffing": {
        "code": "4.A.4.c.2",
        "name": "Staffing",
        "description": "Recruiting, interviewing, selecting, hiring, and promoting employees. HR activities, hiring decisions.",
        "category": "Interacting with Others"
    },
    "controlling_resources": {
        "code": "4.A.4.c.3",
        "name": "Controlling Resources",
        "description": "Monitoring and controlling resources and overseeing spending. Budget management, resource allocation, cost tracking.",
        "category": "Interacting with Others"
    },
}

# Cache for activity embeddings
_activity_embeddings = None


def get_activity_embeddings():
    """Embed all O*NET activity descriptions (cached)."""
    global _activity_embeddings
    if _activity_embeddings is not None:
        return _activity_embeddings

    print("Embedding O*NET activity descriptions...")
    descriptions = [a["description"] for a in ONET_ACTIVITIES.values()]
    embeddings = get_embeddings_batch(descriptions)

    _activity_embeddings = {
        key: np.array(emb)
        for key, emb in zip(ONET_ACTIVITIES.keys(), embeddings)
    }
    print(f"Embedded {len(_activity_embeddings)} O*NET activities")
    return _activity_embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def classify_message_semantic(content: str, top_k: int = 3) -> list[tuple[str, float]]:
    """
    Classify a message into O*NET activities using semantic similarity.

    Returns top-k activities with similarity scores.
    """
    if not content or len(content) < 20:
        return []

    # Get message embedding
    msg_embedding = get_embedding(content)
    if msg_embedding is None:
        return []

    msg_vec = np.array(msg_embedding)
    activity_embeddings = get_activity_embeddings()

    # Compute similarities
    similarities = []
    for key, act_vec in activity_embeddings.items():
        sim = cosine_similarity(msg_vec, act_vec)
        similarities.append((key, sim))

    # Sort by similarity and return top-k
    similarities.sort(key=lambda x: -x[1])
    return similarities[:top_k]


def analyze_month_semantic(month: str = "2025-12", sample_size: int = 500):
    """Analyze a month's messages using semantic classification."""
    base = Path("/Users/mordechai/intellectual_dna")
    parquet_path = base / "data" / "all_conversations.parquet"

    conn = duckdb.connect()

    # Get user messages from the specified month
    query = f"""
    SELECT
        message_id,
        content,
        conversation_title,
        created
    FROM read_parquet('{parquet_path}')
    WHERE role = 'user'
      AND content IS NOT NULL
      AND length(content) > 50
      AND strftime(created, '%Y-%m') = '{month}'
    ORDER BY random()
    LIMIT {sample_size}
    """

    messages = conn.execute(query).fetchall()
    print(f"\n📊 Semantic Analysis: {len(messages)} messages from {month}\n")

    # Pre-warm the embedding model
    get_activity_embeddings()

    # Classify each message
    activity_scores = Counter()
    activity_counts = Counter()
    classified = []

    print("Classifying messages...")
    for i, (msg_id, content, title, created) in enumerate(messages):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(messages)}")

        results = classify_message_semantic(content, top_k=2)
        for activity_key, score in results:
            if score > 0.3:  # Threshold for relevance
                activity_scores[activity_key] += score
                activity_counts[activity_key] += 1
                classified.append({
                    "content": content[:150],
                    "activity": activity_key,
                    "score": score
                })

    print("\n" + "=" * 70)
    print(f"🎯 O*NET SEMANTIC CLASSIFICATION - {month}")
    print("=" * 70)
    print(f"Messages analyzed: {len(messages)}")
    print(f"Classifications made: {sum(activity_counts.values())}")
    print()

    # Group by category
    categories = {
        "Information Input": [],
        "Mental Processes": [],
        "Work Output": [],
        "Interacting with Others": []
    }

    for key, count in sorted(activity_counts.items(), key=lambda x: -x[1])[:20]:
        act = ONET_ACTIVITIES[key]
        avg_score = activity_scores[key] / count if count > 0 else 0
        pct = (count / len(messages)) * 100
        categories[act["category"]].append((act["name"], count, pct, avg_score))

    for category, items in categories.items():
        if items:
            print(f"\n### {category}")
            print("-" * 60)
            for name, count, pct, avg_score in sorted(items, key=lambda x: -x[1]):
                bar = "█" * int(pct / 2) + "░" * (25 - int(pct / 2))
                print(f"{bar} {pct:5.1f}% (sim:{avg_score:.2f}) | {name}")

    # Top activities
    print("\n" + "=" * 70)
    print("💡 TOP AI-AUGMENTED WORK ACTIVITIES")
    print("=" * 70)

    top10 = sorted(activity_counts.items(), key=lambda x: -x[1])[:10]
    for i, (key, count) in enumerate(top10, 1):
        act = ONET_ACTIVITIES[key]
        pct = (count / len(messages)) * 100
        print(f"  {i:2}. {act['name']}: {pct:.1f}%")

    # Sample high-confidence classifications
    print("\n📝 HIGH-CONFIDENCE CLASSIFICATIONS:")
    high_conf = sorted(classified, key=lambda x: -x["score"])[:10]
    for item in high_conf:
        act = ONET_ACTIVITIES[item["activity"]]
        preview = item["content"][:70].replace("\n", " ")
        print(f"\n  [{act['name']}] (sim: {item['score']:.3f})")
        print(f"    \"{preview}...\"")

    conn.close()
    return activity_counts, activity_scores


if __name__ == "__main__":
    month = sys.argv[1] if len(sys.argv) > 1 else "2025-12"
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    analyze_month_semantic(month, sample)
