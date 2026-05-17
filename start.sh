#!/bin/bash
set -e

echo "=== Starting API server ==="
uvicorn mlb.api.app:app --host 0.0.0.0 --port "${PORT:-8000}" &
API_PID=$!

echo "=== Running daily predictions in background ==="
python -m mlb.etl.daily_runner --date "$(date +%Y-%m-%d)" 2>&1 | tee /tmp/daily_runner.log &
RUNNER_PID=$!

# Wait briefly for runner and log result
(wait $RUNNER_PID && echo "=== Daily runner completed ===" || echo "=== Daily runner FAILED (exit $?) — see logs above ===") &

echo "=== Starting scheduler in background ==="
python -m mlb.etl.scheduler 2>&1 &

# Wait for the API server (main process)
wait $API_PID
