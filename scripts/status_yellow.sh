#!/bin/bash
# 📊 YELLOW PHOENIX — STATUS CHECK
cd "$(dirname "$0")/.." || exit 1

echo "╔══════════════════════════════════════════╗"
echo "║     📊 YELLOW PHOENIX STATUS             ║"
echo "╚══════════════════════════════════════════╝"

# API
if lsof -i :5001 >/dev/null 2>&1; then
    echo "✅ API:      RUNNING on :5001"
else
    echo "❌ API:      STOPPED"
fi

# Soak
if [ -f .pid_yp_autonomic_soak ] && ps -p "$(cat .pid_yp_autonomic_soak)" >/dev/null 2>&1; then
    echo "✅ Soak:     RUNNING (PID $(cat .pid_yp_autonomic_soak))"
else
    echo "❌ Soak:     STOPPED"
fi

# Telemetry
if [ -f .pid_telemetry ] && ps -p "$(cat .pid_telemetry)" >/dev/null 2>&1; then
    echo "✅ Telemetry: RUNNING (PID $(cat .pid_telemetry))"
else
    echo "❌ Telemetry: STOPPED"
fi

# DB
if [ -f data/phoenix_arxiv_1m.db ]; then
    size=$(du -sh data/phoenix_arxiv_1m.db | cut -f1)
    echo "✅ DB:       $size (unified 1M+12K)"
else
    echo "❌ DB:       MISSING"
fi

# HNSW
if [ -f data/binary_hnsw_arxiv1m_m16.bin ]; then
    size=$(du -sh data/binary_hnsw_arxiv1m_m16.bin | cut -f1)
    echo "✅ HNSW:     $size"
else
    echo "❌ HNSW:     MISSING"
fi

echo ""
echo "Commands:"
echo "   Start:  ./scripts/start_yellow.sh"
echo "   Stop:   ./scripts/stop_yellow.sh"
echo "   Kill:   ./scripts/kill_switch.sh"
