#!/bin/bash
echo "[cleanup] Starting..."

# NEVER DELETE (the true seed)
# - src/
# - yp_engine.py, yp_bridge.py
# - data/itq_model_512.npz
# - data/paper_metadata.jsonl
# - .git/
# - yp_venv/

# DELETE (fast to rebuild)
echo "[cleanup] Deleting Rust build artifacts..."
rm -rf target.bak
cargo clean 2>/dev/null

# DELETE (test/scratch data)
echo "[cleanup] Deleting synthetic test folders..."
rm -rf synthetic_ladder synthetic_20m synthetic_10m synthetic_5m
rm -f synthetic_*.npy synthetic_*.bin 2>/dev/null

# DELETE (old benchmark meshes)
echo "[cleanup] Deleting old bench meshes..."
rm -rf bench_mesh

# SMART DELETE: Keep newest HNSW index, delete older ones
echo "[cleanup] Keeping newest HNSW index, deleting old ones..."
LATEST_INDEX=$(ls -t data/hnsw_*.index 2>/dev/null | head -1)
for f in data/hnsw_*.index; do
    if [ "$f" != "$LATEST_INDEX" ] && [ -f "$f" ]; then
        rm -f "$f" && echo "  deleted: $f"
    else
        echo "  kept: $f"
    fi
done

# SMART DELETE: Keep newest .npy in data/, delete old
echo "[cleanup] Keeping newest data cache, deleting old..."
for pattern in data/*.npy data/*.bin; do
    # Skip if it matches golden build files
    case "$pattern" in
        *itq_model_512.npz*) continue ;;
        *paper_metadata*) continue ;;
    esac
    # Delete .npy/.bin files older than 7 days
    find data -name "*.npy" -o -name "*.bin" -mtime +7 -not -name "itq_model_512.npz" -not -name "*paper_metadata*" -exec rm -f {} \; 2>/dev/null
done

# Clean git debris
echo "[cleanup] Git garbage collection..."
git gc --prune=now --aggressive 2>/dev/null

# Truncate logs
echo "[cleanup] Truncating logs..."
tail -n 500 logs/benchmark_autoshard_quick.log > /tmp/tmp_log 2>/dev/null && mv /tmp/tmp_log logs/benchmark_autoshard_quick.log 2>/dev/null
tail -n 500 logs/benchmark_production_e2e.log > /tmp/tmp_log 2>/dev/null && mv /tmp/tmp_log logs/benchmark_production_e2e.log 2>/dev/null

echo ""
echo "[cleanup] DONE. Space remaining:"
df -h /Users/mac | tail -1
echo ""
echo "Golden build status:"
cat GOLDEN_BUILD.md 2>/dev/null || echo "  No golden build marked yet. Run: ./scripts/mark_golden.sh"
