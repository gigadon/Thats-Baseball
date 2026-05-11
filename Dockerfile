FROM python:3.13-slim

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
RUN mkdir -p data/predictions data/betting

EXPOSE 8000

# Default: run the API server
CMD uvicorn mlb.api.app:app --host 0.0.0.0 --port ${PORT:-8000}
