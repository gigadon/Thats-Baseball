FROM python:3.13-slim

WORKDIR /app

# System deps for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e ".[dev]"

# Copy data and models (if present)
COPY data/ data/
COPY models/ models/
COPY alembic.ini .
COPY alembic/ alembic/

EXPOSE 8000

# Default: run the API server
CMD ["uvicorn", "mlb.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
