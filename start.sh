#!/bin/bash
set -e

echo "=== Starting API server ==="
uvicorn mlb.api.app:app --host 0.0.0.0 --port "${PORT:-8000}" &
API_PID=$!

# Let the scheduler handle daily_runner via --run-now to avoid race conditions
echo "=== Starting scheduler (runs daily pipeline on startup) ==="
python -m mlb.etl.scheduler --run-now 2>&1 &

# Wait for the API server (main process)
wait $API_PID
