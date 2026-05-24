#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

echo "Resetting runtime job files..."
mkdir -p uploads outputs /tmp/redis
rm -rf uploads/* outputs/*
rm -f dump.rdb /tmp/redis/dump.rdb /var/lib/redis/dump.rdb

echo "Starting Redis server in the background..."
redis-server --protected-mode no --save "" --appendonly no --dir /tmp/redis &

echo "Waiting for Redis..."
until redis-cli ping >/dev/null 2>&1; do
    sleep 0.2
done
redis-cli FLUSHALL >/dev/null

echo "Starting Celery worker in the background..."
celery -A tasks worker --loglevel=info --pool=solo --concurrency=1 &

echo "Starting FastAPI endpoint in the foreground..."
exec uvicorn main:app --host 0.0.0.0 --port 7860
