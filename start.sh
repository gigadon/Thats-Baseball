#!/bin/bash
set -e

echo "=== Starting API server ==="
uvicorn mlb.api.app:app --host 0.0.0.0 --port "${PORT:-8000}" &
API_PID=$!

echo "=== Running daily predictions in background ==="
python -m mlb.etl.daily_runner --date "$(date +%Y-%m-%d)" &

echo "=== Starting scheduler in background ==="
python -m mlb.etl.scheduler &

# Wait for the API server (main process)
wait $API_PID
