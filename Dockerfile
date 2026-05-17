FROM python:3.13-slim AS base

WORKDIR /app

# System deps for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e .

# Copy data and models (if present)
COPY data/ data/
COPY models/ models/

# Create dirs for runtime data
RUN mkdir -p data/predictions data/betting data/line_movement data/accuracy

EXPOSE 8000

COPY start.sh .
RUN chmod +x start.sh

# Health check — API responds on /api/v1/health
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/v1/health || exit 1

# Default: run predictions then start API
CMD ["bash", "start.sh"]
