#!/usr/bin/env python3
"""
Build phrases/v1 interpretation: Extract signature phrases using n-grams.

Finds recurring multi-word phrases in conversations to identify
verbal fingerprints and signature expressions.
"""

import re
import json
import duckdb
from pathlib import Path
from collections import Counter

BASE_DIR = Path("/Users/mordechai/intellectual_dna")
DATA_DIR = BASE_DIR / "data"
FACTS_DIR = DATA_DIR / "facts"
INTERP_DIR = DATA_DIR / "interpretations" / "phrases" / "v1"

# Stopwords for filtering
STOPWORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
    'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
    'is', 'are', 'was', 'were', 'been', 'being', 'has', 'had', 'does', 'did',
    'just', 'like', 'can', 'could', 'should', 'would', 'might', 'must',
    'im', "i'm", 'its', "it's", 'thats', "that's", 'dont', "don't",
    'youre', "you're", 'weve', "we've", 'ive', "i've",
    # Code/tech stopwords
    'const', 'let', 'var', 'function', 'return', 'import', 'export',
    'async', 'await', 'true', 'false', 'null', 'undefined', 'string',
    'number', 'boolean', 'type', 'interface', 'class', 'extends',
}

# Words that indicate system/code content (skip entire phrase if contains)
SYSTEM_WORDS = {
    # File paths
    'users', 'mordechai', 'wotcfy', 'apps', 'documents', 'desktop',
    'library', 'downloads', 'node_modules', 'src', 'dist', 'build',
    # File extensions
    'tsx', 'jsx', 'json', 'yaml', 'yml', 'parquet', 'csv', 'md',
    'py', 'js', 'ts', 'css', 'html', 'sql', 'sh', 'env',
    # Code artifacts
    'localhost', 'http', 'https', 'www', 'api', 'endpoint',
    'npm', 'yarn', 'pip', 'git', 'commit', 'branch', 'merge',
    # Environment/IDE
    'vscode', 'cursor', 'terminal', 'console', 'stdout', 'stderr',
    'environment', 'config', 'settings', 'workspace',
    # Timestamps/system
    'utc', 'gmt', 'timezone', 'timestamp', 'datetime',
    'jerusalem', 'asia', 'tokens', 'context', 'window',
    # AI/tool artifacts
    'claude', 'anthropic', 'openai', 'assistant', 'tool_result',
    'function_call', 'parameters', 'schema', 'prompt',
    # Git output
    'objects', 'remote', 'counting', 'compressing', 'receiving',
    'resolving', 'deltas', 'unpacking', 'enumerating', 'cloning',
    'fetching', 'pushing', 'pulling', 'rebasing', 'stashing',
    # Shell output
    'drwxr', 'drwx', 'rwxr', 'xr', 'staff', 'wheel', 'root',
    'total', 'lrwxr', 'crw', 'brw',
    # Railway/deployment
    'railway', 'vercel', 'heroku', 'netlify', 'docker',
    'container', 'kubernetes', 'pod', 'deployment',
    # More code terms
    'undefined', 'null', 'nan', 'inf', 'error', 'exception',
    'traceback', 'stacktrace', 'stderr', 'stdout',
    # Common framework/lib terms
    'react', 'vue', 'angular', 'svelte', 'nextjs', 'nuxt',
    'tailwind', 'webpack', 'vite', 'esbuild', 'rollup',
    'typescript', 'javascript', 'python', 'rust', 'golang',
    # File/path parts
    'png', 'jpg', 'jpeg', 'gif', 'svg', 'ico', 'webp',
    'txt', 'log', 'tmp', 'temp', 'bak', 'cache',
    'index', 'main', 'app', 'page', 'component', 'module',
    # Tool output
    'executed', 'command', 'output', 'result', 'returned',
    'success', 'failed', 'completed', 'running', 'starting',
    'installing', 'building', 'compiling', 'bundling',
    # AI/system instructions
    'tool', 'message', 'proceeding', 'assess', 'must',
    'instructions', 'following', 'provided', 'specified',
    'required', 'optional', 'default', 'value',
}

# Regex patterns to skip lines entirely
LINE_SKIP_PATTERNS = [
    r'^[\s]*$',                          # Empty lines
    r'^[/\\]',                           # Paths starting with /
    r'^\s*[{}\[\]]',                     # JSON/code blocks
    r'^\s*(const|let|var|function|class|import|export|return)\s',  # Code
    r'^\s*[<>]',                         # HTML/XML tags
    r'^\d{4}-\d{2}-\d{2}',              # Dates
    r'^\d+:\d+',                         # Times
    r'^https?://',                       # URLs
    r'^[A-Z_]{3,}:',                     # ENV vars like API_KEY:
    r'^\s*#',                            # Comments
    r'^\s*//',                           # Comments
    r'environment_details',              # System context blocks
    r'visible_files',                    # IDE context
    r'open_tabs',                        # IDE context
    r'tokens?\s*used',                   # Token usage
    r'context\s*window',                 # Context info
    r'file[_\s]?path',                   # File paths
    r'tool\s*(has|may|must|can|should)', # AI tool instructions
    r'before\s*proceeding',              # AI instructions
    r'assess\s*the\s*(first|tool)',      # AI instructions
    r'only\s*one\s*tool',                # AI instructions
    r'per\s*message',                    # AI instructions
    r'command\s*executed',               # Shell output
    r'requirements\.txt',                # File references
    r'package\.json',                    # File references
    r'\.png|\.jpg|\.gif|\.svg',          # Image files
    r'deleted\s*app',                    # Git diff output
    r'app\s*signatures',                 # Path artifacts
    r'react[-_\s]?dom',                  # Framework names
    # AI assistant instruction phrases
    r'attempt\s*completion',             # Claude/AI instructions
    r'act\s*mode',                       # Claude modes
    r'consider\s*their\s*input',         # AI instructions
    r'continue\s*the\s*task',            # AI instructions
    r'toggle\s*to',                      # UI instructions
    r'current\s*state\s*of',             # State tracking
    r'final\s*version',                  # File state
    r'full\s*updated',                   # File state
    r'content\s*reflects',               # File state
    r'search\s*replace',                 # Editor operations
    r'single\s*quotes',                  # Syntax instructions
    r'ensure\s*the\s*file',              # File instructions
    r'important\s*for\s*any',            # Boilerplate
    r'final_file_content',               # Claude artifacts
    r'shown\s*here',                     # UI references
    r'make\s*sure\s*to',                 # Instructions
    r'please\s*note',                    # Instructions
    r'above\s*code',                     # Code references
    r'below\s*code',                     # Code references
    # More system prompt patterns
    r'messages\s*below',                 # System context
    r'were\s*generated',                 # System context
    r'generated\s*by\s*the',             # System context
    r'do\s*not\s*respond',               # System instructions
    r'otherwise\s*consider',             # System instructions
    r'unless\s*the\s*user',              # System instructions
    r'explicitly\s*asks',                # System instructions
    r'in\s*your\s*response',             # System instructions
    r'caveat\s*the',                     # System boilerplate
    r'user\s*while',                     # System context
    # More system/tool noise
    r'configuration\s*updated',          # Tool output
    r'updated\s*successfully',           # Tool output
    r'macbook\s*(air|pro)',              # Device names
    r'mordechai(s|potash)',              # Username
    r'supabase\s*mcp',                   # Tool names
    r'ide_selection',                    # IDE context
    r'denied\s*this\s*operation',        # Permission messages
    r'found\s*for\s*this\s*service',     # Service messages
    r'may\s*not\s*be\s*related',         # System context
    r'related\s*to\s*the\s*current',     # System context
]


def is_natural_language_line(line: str) -> bool:
    """Check if a line looks like natural human language."""
    if len(line) < 15:
        return False

    # Must have spaces (natural language has word boundaries)
    words = line.split()
    if len(words) < 3:
        return False

    # Check word characteristics
    natural_words = 0
    for word in words:
        word_clean = word.strip('.,!?;:\'"()[]{}')
        # Natural words: lowercase, no underscores, no camelCase, short
        if (word_clean.islower() and
            '_' not in word_clean and
            len(word_clean) >= 2 and
            len(word_clean) <= 15 and
            word_clean.isalpha()):
            natural_words += 1

    # At least 50% of words should look natural
    return natural_words / len(words) >= 0.5


def clean_text_for_phrases(text: str) -> str:
    """
    Pre-process text to extract only conversational content.
    Removes system context, code, file paths, etc.
    """
    if not text:
        return ""

    lines = text.split('\n')
    clean_lines = []

    for line in lines:
        line_lower = line.lower().strip()

        # Skip empty or very short lines
        if len(line_lower) < 15:
            continue

        # Skip lines matching system patterns
        skip = False
        for pattern in LINE_SKIP_PATTERNS:
            if re.search(pattern, line_lower):
                skip = True
                break

        if skip:
            continue

        # Skip lines with too many path separators (likely file paths)
        if line.count('/') > 2 or line.count('\\') > 2:
            continue

        # Skip lines with underscores (code identifiers)
        if line.count('_') > 2:
            continue

        # Skip lines with code-like patterns (brackets, semicolons ratio)
        special_chars = sum(1 for c in line if c in '{}[]();=><_')
        if len(line) > 0 and special_chars / len(line) > 0.05:
            continue

        # Only keep lines that look like natural language
        if not is_natural_language_line(line):
            continue

        clean_lines.append(line)

    return ' '.join(clean_lines)


def extract_ngrams(text: str, n: int) -> list:
    """Extract n-grams from text."""
    if not text:
        return []

    # Clean and tokenize
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()

    # Filter very short words and numbers
    words = [w for w in words if len(w) >= 2 and not w.isdigit()]

    # Generate n-grams
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])

        # Skip if starts or ends with stopword
        if ngram[0] in STOPWORDS or ngram[-1] in STOPWORDS:
            continue

        # Skip if ANY word is a system word
        if any(w in SYSTEM_WORDS for w in ngram):
            continue

        # Skip if contains numbers (likely IDs, versions)
        if any(any(c.isdigit() for c in w) for w in ngram):
            continue

        ngrams.append(' '.join(ngram))

    return ngrams


def build_phrases():
    """Build phrase extraction from all user messages."""
    print("Building phrases/v1 interpretation...")

    con = duckdb.connect()
    content_path = FACTS_DIR / "brain" / "content.parquet"

    if not content_path.exists():
        print(f"  ERROR: {content_path} not found")
        return

    # Load SHORT user messages (more likely to be actual human input)
    # Long messages often contain pasted context, code, or AI output
    print("  Loading short user messages (< 500 chars)...")
    messages = con.execute(f"""
        SELECT full_content
        FROM '{content_path}'
        WHERE event_type = 'message'
          AND full_content IS NOT NULL
          AND LENGTH(full_content) < 500
          AND LENGTH(full_content) > 10
    """).fetchall()

    print(f"  Processing {len(messages)} messages...")

    # Extract n-grams of different sizes
    bigrams = Counter()
    trigrams = Counter()
    fourgrams = Counter()

    for (text,) in messages:
        # Clean text to remove system/code content first
        clean_text = clean_text_for_phrases(text)
        if len(clean_text) < 20:  # Skip if too little content after cleaning
            continue

        bigrams.update(extract_ngrams(clean_text, 2))
        trigrams.update(extract_ngrams(clean_text, 3))
        fourgrams.update(extract_ngrams(clean_text, 4))

    # Filter to phrases appearing 5+ times
    min_count = 5

    significant_phrases = []

    for phrase, count in bigrams.most_common(500):
        if count >= min_count:
            significant_phrases.append({
                'phrase': phrase,
                'ngram_size': 2,
                'count': count,
                'per_1000': round(count * 1000 / len(messages), 2)
            })

    for phrase, count in trigrams.most_common(500):
        if count >= min_count:
            significant_phrases.append({
                'phrase': phrase,
                'ngram_size': 3,
                'count': count,
                'per_1000': round(count * 1000 / len(messages), 2)
            })

    for phrase, count in fourgrams.most_common(200):
        if count >= min_count:
            significant_phrases.append({
                'phrase': phrase,
                'ngram_size': 4,
                'count': count,
                'per_1000': round(count * 1000 / len(messages), 2)
            })

    # Sort by count
    significant_phrases.sort(key=lambda x: -x['count'])

    print(f"  Found {len(significant_phrases)} significant phrases (5+ occurrences)")

    # Write to parquet
    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERP_DIR / "phrases.parquet"

    con.execute("""
        CREATE TABLE phrases (
            phrase VARCHAR,
            ngram_size INTEGER,
            count INTEGER,
            per_1000 DOUBLE
        )
    """)

    for p in significant_phrases:
        con.execute("""
            INSERT INTO phrases VALUES (?, ?, ?, ?)
        """, [p['phrase'], p['ngram_size'], p['count'], p['per_1000']])

    con.execute(f"COPY phrases TO '{output_path}' (FORMAT PARQUET)")
    print(f"  Wrote to {output_path}")

    # Show top phrases by size
    print("\n  Top Bigrams (2 words):")
    for p in [x for x in significant_phrases if x['ngram_size'] == 2][:10]:
        print(f"    \"{p['phrase']}\" ({p['count']}x)")

    print("\n  Top Trigrams (3 words):")
    for p in [x for x in significant_phrases if x['ngram_size'] == 3][:10]:
        print(f"    \"{p['phrase']}\" ({p['count']}x)")

    print("\n  Top Fourgrams (4 words):")
    for p in [x for x in significant_phrases if x['ngram_size'] == 4][:10]:
        print(f"    \"{p['phrase']}\" ({p['count']}x)")

    # Create config
    config = {
        "name": "phrases/v1",
        "description": "Signature phrase extraction using n-grams",
        "algorithm": "ngram_frequency",
        "parameters": {
            "min_count": min_count,
            "ngram_sizes": [2, 3, 4],
            "max_phrases_per_size": [500, 500, 200]
        },
        "created": "2025-12-25"
    }

    config_path = INTERP_DIR / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return len(significant_phrases)


if __name__ == '__main__':
    build_phrases()
