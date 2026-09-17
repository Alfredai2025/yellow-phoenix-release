#!/bin/bash
# Overnight runner: Experiment B (kill-ceiling graphs, 1M) — Experiment A
# already completed in-session (2M pairs, Entry 61) and is NOT re-run here.
# Idempotent-ish: each phase checks for its output and skips if present.
# Robust: per-phase error capture; a phase failure does not block later phases.
set -uo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
export YP_GENESIS_PHRASE=yp_phoenix_2026
cd /Users/mac/yellow_phoenix

LOG=logs/overnight_ab.log
D=/tmp/killceiling
PY=/Users/mac/yp_venv/bin/python
mkdir -p "$D"
echo "=== overnight run start $(date) ===" >> "$LOG"

phase() {
    local name="$1"; shift
    echo "--- phase $name start $(date) ---" >> "$LOG"
    if "$@" >> "$LOG" 2>&1; then
        echo "--- phase $name OK $(date) ---" >> "$LOG"
    else
        echo "--- phase $name FAILED (rc=$?) $(date); continuing ---" >> "$LOG"
    fi
}

# 1. subset prep (1M)
if [ ! -f "$D/subset_meta.json" ]; then
    phase subset_prep "$PY" scripts/b_subset_prep.py 1000000 "$D"
else
    echo "subset present, skipping" >> "$LOG"
fi

# 2. cargo bins
phase cargo_build cargo build --release --bin build_hnsw_from_ism --bin vamana_build --bin eval_graph

# 3. HNSW baseline graph
if [ ! -f "$D/hnsw_1m.bin" ]; then
    phase hnsw_build env HNSW_M=12 HNSW_EF_CONSTRUCTION=200 HNSW_EF_SEARCH=128 \
        ./target/release/build_hnsw_from_ism "$D/subset.ism" "$D/hnsw_1m.bin"
fi

# 4. vamana plain
if [ ! -f "$D/vamana_1m.bin" ]; then
    phase vamana_build bash -c "./target/release/vamana_build '$D/subset.ism' '$D/vamana_1m.bin' 12 60 1200 0 2>>'$LOG' | tail -1 > '$D/build_vamana.json'"
fi

# 5. vamana paged (lambda_x10=300 start, auto-tune)
if [ ! -f "$D/vamana_paged_1m.bin" ]; then
    phase vamana_paged_build bash -c "./target/release/vamana_build '$D/subset.ism' '$D/vamana_paged_1m.bin' 12 60 1200 300 paged 2>>'$LOG' | tail -1 > '$D/build_vamana_paged.json'"
fi

# 6. evals (always re-run if graphs exist and eval jsons missing)
[ -f "$D/eval_hnsw.json" ] || phase eval_hnsw ./target/release/eval_graph hnsw "$D/hnsw_1m.bin" "$D/queries.ism" "$D/gt.bin" "$D/eval_hnsw.json"
[ -f "$D/eval_vamana.json" ] || phase eval_vamana ./target/release/eval_graph vamana "$D/vamana_1m.bin" "$D/queries.ism" "$D/gt.bin" "$D/eval_vamana.json"
[ -f "$D/eval_vamana_paged.json" ] || phase eval_vamana_paged ./target/release/eval_graph vamana "$D/vamana_paged_1m.bin" "$D/queries.ism" "$D/gt.bin" "$D/eval_vamana_paged.json"

# 7. report + Entry 62
phase report "$PY" scripts/killceiling_report.py

echo "=== overnight run end $(date) ===" >> "$LOG"
SUMMARY="overnight_ab done $(date): decision=$(python3 -c "import json;d=json.load(open('benchmark_results/killceiling_graph_1m_$(date +%Y%m%d).json'));print(d['decision'])" 2>/dev/null || echo unknown)"
echo "$SUMMARY" > logs/overnight_ab_DONE
echo "$SUMMARY" >> "$LOG"
