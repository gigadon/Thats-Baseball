#!/bin/bash
set -e

echo "=== Running daily predictions ==="
python -m mlb.etl.daily_runner --date "$(date +%Y-%m-%d)" || echo "Daily runner warning — starting API anyway"

echo "=== Starting scheduler in background ==="
python -m mlb.etl.scheduler &

echo "=== Starting API server ==="
exec uvicorn mlb.api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
