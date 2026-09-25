#!/usr/bin/env bash
# ==============================================================================
# run_embeddings_background.sh
# Run BGE-M3 news article embedding pipeline in background with logging
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/news_embedding_pipeline.log"
PID_FILE="$PROJECT_ROOT/logs/embedding_pipeline.pid"

mkdir -p "$PROJECT_ROOT/logs"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Pipeline is already running with PID: $OLD_PID"
        echo "Follow progress with: tail -f \"$LOG_FILE\""
        exit 0
    else
        rm -f "$PID_FILE"
    fi
fi

echo "Starting BGE-M3 news embedding pipeline in background..."
echo "Logs will be written to: $LOG_FILE"

nohup python3 "$SCRIPT_DIR/embed_news_articles.py" \
    --zip "$PROJECT_ROOT/archive.zip" \
    --db "$PROJECT_ROOT/data/news_bge_m3_embeddings.db" \
    --npz "$PROJECT_ROOT/data/news_bge_m3_embeddings.npz" \
    --csv "$PROJECT_ROOT/data/news_bge_m3_embeddings.csv" \
    --agg "$PROJECT_ROOT/data/news_embeddings_by_date_region.csv" \
    --batch-size 8 \
    --max-words 512 \
    > "$LOG_FILE" 2>&1 &

NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

echo "Pipeline started successfully with PID: $NEW_PID"
echo "You can monitor live progress with:"
echo "  tail -f \"$LOG_FILE\""
