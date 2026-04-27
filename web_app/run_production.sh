#!/bin/bash
# Run SeismicX Web Interface with Gunicorn (production WSGI server)
# Usage: ./run_production.sh [--port PORT] [--workers NUM]

PORT=${PORT:-5010}
WORKERS=${WORKERS:-1}  # Use single worker process to avoid multiprocessing issues

# Check if gunicorn is installed
if ! command -v gunicorn &> /dev/null; then
    echo "❌ Gunicorn not found. Install with: pip install gunicorn"
    exit 1
fi

echo "Starting SeismicX Web Interface with Gunicorn..."
echo "  Server: http://localhost:$PORT"
echo "  Workers: $WORKERS (process-based, thread-safe)"

# Run with single worker process (avoids threading issues with sentence-transformers)
gunicorn \
    --bind "0.0.0.0:$PORT" \
    --workers "$WORKERS" \
    --threads 1 \
    --timeout 300 \
    --access-logfile - \
    --error-logfile - \
    app:app
