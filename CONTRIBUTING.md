# Contributing to Brain MCP

Thanks for your interest in contributing! Brain MCP is a local-first tool for turning AI conversations into a searchable second brain, and we welcome contributions of all kinds.

## Getting Started

1. **Fork and clone** the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/brain-mcp.git
   cd brain-mcp
   ```

2. **Set up the dev environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Run the tests**:
   ```bash
   pytest tests/ -v
   ```

## Making Changes

1. Create a feature branch: `git checkout -b my-feature`
2. Make your changes
3. Run tests: `pytest tests/ -v`
4. Commit with a descriptive message
5. Push and open a Pull Request

## Code Style

- Python 3.11+ — use modern features (type hints, f-strings, dataclasses)
- Keep functions focused and well-documented
- Docstrings on all public functions
- No hardcoded paths — everything goes through `config.py`

## What to Contribute

- **New ingesters** — support more AI chat formats (Copilot, Gemini, etc.)
- **Bug fixes** — especially around edge cases in parsing
- **Tests** — more test coverage is always welcome
- **Docs** — improve the getting-started experience
- **Performance** — faster embedding, smarter incremental sync

## Adding a New Ingester

1. Create `brain_mcp/ingest/my_source.py`
2. Use `make_record()` and `finalize_conversation()` from `brain_mcp.ingest.schema`
3. Add sample fixture data in `tests/fixtures/`
4. Add tests in `tests/test_basic.py`
5. Register in `brain_mcp/ingest/__init__.py`

## Reporting Issues

- Include your Python version and OS
- Include the full error traceback
- Describe what you expected vs. what happened

## Code of Conduct

Be kind. Be constructive. We're all here to build something useful.
