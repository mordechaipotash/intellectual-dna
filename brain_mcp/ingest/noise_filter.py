"""
brain-mcp — Noise filter for embedding pipeline.

Filters out messages with no semantic value before embedding.
These are confirmations, single words, tool results, etc.

Usage:
    from ingest.noise_filter import is_noise_message

    if not is_noise_message(content):
        # embed this message
"""

import re

NOISE_PATTERNS = [
    # System noise
    r"^\[Tool Result\]$",
    r"^\[Request interrupted",
    r"^\[Image: User uploaded",
    r"^<local-command",
    r"^Warmup$",

    # Continuations
    r"^cont\s*$",
    r"^Cont\s*$",
    r"^continue\s*$",
    r"^Continue\s*$",

    # Yes/No/Ok confirmations
    r"^(yes|Yes|YES|y|yeah|yep|yea)\s*$",
    r"^(no|No|NO|n|nope|nah)\s*$",
    r"^(ok|Ok|OK|okay|Okay)\s*$",
    r"^(sure|Sure)\s*$",
    r"^(right|Right)\s*$",
    r"^(exactly|Exactly)\s*$",
    r"^(correct|Correct)\s*$",
    r"^(perfect|Perfect)\s*$",
    r"^(great|Great)\s*$",
    r"^(good|Good)\s*$",
    r"^(nice|Nice)\s*$",
    r"^(cool|Cool)\s*$",
    r"^(done|Done)\s*$",

    # Generic commands (too vague for semantic search)
    r"^(more|More|MORE)\s*$",
    r"^(next|Next)\s*$",
    r"^(go|Go)\s*$",
    r"^go for it\s*$",
    r"^do it\s*$",
    r"^run it\s*$",
    r"^please run it\s*$",
    r"^try again\s*$",
    r"^impl\s*$",
    r"^shorter\s*$",
    r"^all\s*$",
    r"^help\s*$",
    r"^hi\s*$",
    r"^hello\s*$",
    r"^thanks\s*$",
    r"^thank you\s*$",
    r"^full code\s*$",
    r"^do better\s*$",
    r"^now\??\s*$",
    r"^dont\s*$",
    r"^stop\s*$",

    # Single chars/numbers
    r"^[a-zA-Z]$",
    r"^[0-9]+x?$",
    r"^kill [0-9]+$",
]

# Pre-compile for performance
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]


def is_noise_message(content: str) -> bool:
    """
    Check if a message is noise that should be filtered from embeddings.

    Args:
        content: The message content to check

    Returns:
        True if the message is noise (should NOT be embedded)
        False if the message has value (should be embedded)
    """
    content = str(content).strip()

    for pattern in _COMPILED_PATTERNS:
        if pattern.search(content):
            return True

    return False
