#!/bin/bash
# 🔴 ENTERPRISE KILL SWITCH — Halts all Yellow Phoenix autonomic processes immediately
echo "[KILL SWITCH] Halting all YP autonomic processes..."
pkill -f "yp_autonomic.orchestrator"
pkill -f "yp_autonomic.agent"
pkill -f "yp_bridge"
pkill -f "bench_10k"
pkill -f "bench_10k_paraphrase"
rm -f .pid_yp_autonomic .pid_yp_autonomic_soak .pid_bench_10k .pid_bench_para .halt_ingestion
echo "[KILL SWITCH] All processes halted. Soak is OFF."
echo "[KILL SWITCH] To restart: cd ~/yellow_phoenix && .venv/bin/python3 -m yp_autonomic.orchestrator start"
