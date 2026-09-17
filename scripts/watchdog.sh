#!/bin/bash
# Yellow Phoenix API Watchdog
# Restarts API if it dies. Logs everything.

YP_ROOT="$HOME/yellow_phoenix"
API_LOG_DIR="$YP_ROOT/logs"
API_SCRIPT="$YP_ROOT/scripts/flask_api.py"
PID_FILE="/tmp/yp_api.pid"
PYTHON="$YP_ROOT/.venv/bin/python3"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$API_LOG_DIR/watchdog.log"
}

start_api() {
    log "Starting API..."
    cd "$YP_ROOT"
    nohup "$PYTHON" "$API_SCRIPT" > "$API_LOG_DIR/api_$(date +%Y%m%d_%H%M).log" 2>&1 &
    NEW_PID=$!
    echo $NEW_PID > "$PID_FILE"
    log "API started with PID $NEW_PID"
    sleep 40  # Wait for engine load
}

# Initial start
if [ ! -f "$PID_FILE" ] || ! kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
    start_api
fi

log "Watchdog running. Checking every 5 seconds."

while true; do
    sleep 5
    if [ ! -f "$PID_FILE" ]; then
        log "PID file missing. Restarting..."
        start_api
        continue
    fi
    
    PID=$(cat "$PID_FILE")
    if ! kill -0 "$PID" 2>/dev/null; then
        log "API (PID $PID) dead. Restarting..."
        start_api
    fi
done
