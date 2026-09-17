#!/bin/bash
# 🛑 YELLOW PHOENIX — MASTER STOP SCRIPT
cd "$(dirname "$0")/.." || exit 1

echo "╔══════════════════════════════════════════╗"
echo "║     🛑 YELLOW PHOENIX MASTER STOP        ║"
echo "╚══════════════════════════════════════════╝"

for pidfile in .pid_api .pid_yp_autonomic_soak .pid_telemetry .pid_bench_10k .pid_bench_para .pid_bench_unified; do
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if ps -p "$pid" >/dev/null 2>&1; then
            echo "Stopping $pidfile (PID $pid)..."
            kill "$pid" 2>/dev/null
            sleep 1
        fi
        rm -f "$pidfile"
    fi
done

echo "✅ All Yellow processes stopped."
echo "   Soak is OFF. Autonomic brain halted."
echo "   To restart: ./scripts/start_yellow.sh"
