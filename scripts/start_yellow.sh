#!/bin/bash
# 🟡 YELLOW PHOENIX — MASTER START SCRIPT
# Usage: ./scripts/start_yellow.sh [api] [soak] [telemetry]
#   ./scripts/start_yellow.sh          → starts soak + api + telemetry
#   ./scripts/start_yellow.sh soak     → starts soak only
#   ./scripts/start_yellow.sh api      → starts api only
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate

MODE="${1:-all}"
START_API=false
START_SOAK=false
START_TELEMETRY=false

case "$MODE" in
    all) START_API=true; START_SOAK=true; START_TELEMETRY=true ;;
    api) START_API=true ;;
    soak) START_SOAK=true ;;
    telemetry) START_TELEMETRY=true ;;
    *) echo "Usage: $0 [all|api|soak|telemetry]"; exit 1 ;;
esac

echo "╔══════════════════════════════════════════╗"
echo "║     🟡 YELLOW PHOENIX MASTER START       ║"
echo "╚══════════════════════════════════════════╝"

# ── 1. Flask API (port 5001) ──
if $START_API; then
    if lsof -i :5001 >/dev/null 2>&1; then
        echo "✅ API already running on :5001"
    else
        echo "🚀 Starting Flask API on :5001..."
        nohup .venv/bin/python3 scripts/flask_api.py > logs/api_$(date +%Y%m%d_%H%M).log 2>&1 &
        echo $! > .pid_api
        sleep 2
        echo "   PID: $(cat .pid_api)"
    fi
fi

# ── 2. Autonomic Soak ──
if $START_SOAK; then
    if [ -f .pid_yp_autonomic_soak ] && ps -p "$(cat .pid_yp_autonomic_soak)" >/dev/null 2>&1; then
        echo "✅ Soak already running (PID $(cat .pid_yp_autonomic_soak))"
    else
        echo "🧠 Starting autonomic soak..."
        nohup .venv/bin/python3 -m yp_autonomic.orchestrator start > logs/soak/autonomic_$(date +%Y%m%d_%H%M).log 2>&1 &
        echo $! > .pid_yp_autonomic_soak
        sleep 3
        echo "   PID: $(cat .pid_yp_autonomic_soak)"
    fi
fi

# ── 3. Live Telemetry Dashboard ──
if $START_TELEMETRY; then
    echo "📊 Starting live telemetry..."
    nohup .venv/bin/python3 -c "
import time, subprocess, os
log_dir = 'logs/soak'
print('=== YELLOW LIVE TELEMETRY ===')
print('Press Ctrl+C to stop')
print('')
while True:
    try:
        newest = sorted([f for f in os.listdir(log_dir) if f.startswith('autonomic_')], reverse=True)[0]
        result = subprocess.run(['tail', '-n', '6', f'{log_dir}/{newest}'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        os.system('clear')
        print('=== YELLOW LIVE TELEMETRY ===')
        print(f'Log: {newest}')
        print('-' * 60)
        for line in lines[-5:]:
            print(line)
        print('-' * 60)
        time.sleep(5)
    except KeyboardInterrupt:
        break
    except:
        time.sleep(5)
" > logs/telemetry_$(date +%Y%m%d_%H%M).log 2>&1 &
    echo $! > .pid_telemetry
    echo "   PID: $(cat .pid_telemetry)"
fi

echo ""
echo "🟡 YELLOW IS LIVE"
echo "   API:      http://localhost:5001"
echo "   Soak log: logs/soak/autonomic_*.log"
echo "   Status:   ./scripts/status_yellow.sh"
echo "   Stop:     ./scripts/stop_yellow.sh"
echo "   Kill:     ./scripts/kill_switch.sh"
