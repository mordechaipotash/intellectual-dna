FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY brain_mcp/ ./brain_mcp/

# Install brain-mcp
RUN pip install --no-cache-dir .

# Default: run the MCP server via stdio
ENTRYPOINT ["brain-mcp"]
CMD ["serve"]
