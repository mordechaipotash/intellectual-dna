"""
Intellectual DNA - Data Pipelines

Central location for all data import and processing pipelines.

Usage:
    python -m pipelines status              # Show brain status
    python -m pipelines import-claude       # Import Claude Code conversations
    python -m pipelines embed               # Run embedding pipeline
    python -m pipelines embed --all         # Embed all remaining messages
"""

from .status import show_status
from .import_claude_code import import_claude_code
from .embed_messages import run_embedding_pipeline
from .embed_all import embed_all

__all__ = [
    'show_status',
    'import_claude_code',
    'run_embedding_pipeline',
    'embed_all',
]
