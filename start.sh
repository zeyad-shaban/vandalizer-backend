#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting Redis server in the background..."
redis-server --protected-mode no &

echo "Starting Celery worker in the background..."
celery -A tasks worker --loglevel=info &

echo "Starting FastAPI endpoint in the foreground..."
exec uvicorn main:app --host 0.0.0.0 --port 7860