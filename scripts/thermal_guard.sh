#!/bin/bash
# Yellow Phoenix Thermal Guard
# Kills API if Mac is overheating, on battery with lid closed, or thermally throttled.

YP_ROOT="$HOME/yellow_phoenix"
PID_FILE="/tmp/yp_api.pid"
LOG="$YP_ROOT/logs/thermal_guard.log"

kill_api() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        kill "$PID" 2>/dev/null
        echo "[$(date)] THERMAL KILL: API (PID $PID) stopped — $1" >> "$LOG"
    fi
}

log() {
    echo "[$(date)] $1" >> "$LOG"
}

log "Thermal guard started."

while true; do
    sleep 30

    # 1. Check thermal throttling (pmset -g therm)
    THERM=$(pmset -g therm 2>/dev/null)
    if echo "$THERM" | grep -q "CPU_Speed_Limit.*[^1]00"; then
        # Speed limit under 100% = throttling
        LIMIT=$(echo "$THERM" | grep "CPU_Speed_Limit" | awk '{print $3}')
        if [ "$LIMIT" != "100" ] && [ "$LIMIT" != "" ]; then
            kill_api "CPU throttled to ${LIMIT}%"
            continue
        fi
    fi

    # 2. Check if lid is closed AND on battery
    LID=$(ioreg -r -k AppleClamshellState | grep AppleClamshellState | head -1 | awk '{print $NF}')
    POWER=$(pmset -g ps | head -1 | grep -i "Battery Power")
    if [ "$LID" = "Yes" ] && [ -n "$POWER" ]; then
        kill_api "Lid closed on battery power"
        continue
    fi

    # 3. Check CPU load (over 90% sustained = hot)
    LOAD=$(uptime | awk -F'load averages:' '{print $2}' | awk '{print $1}')
    # Compare load to CPU count
    NCPU=$(sysctl -n hw.ncpu)
    THRESHOLD=$(echo "$NCPU * 0.9" | bc -l 2>/dev/null || echo "$NCPU")
    if [ "$(echo "$LOAD > $THRESHOLD" | bc -l 2>/dev/null)" = "1" ]; then
        kill_api "CPU load ${LOAD} > threshold ${THRESHOLD}"
        continue
    fi

    # All clear
    echo "[$(date)] CHECK OK: load=${LOAD}, lid=${LID}, power=$(pmset -g ps | head -1)" >> "$LOG"
done
