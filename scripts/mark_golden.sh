#!/bin/bash
# Mark current build as golden — run this AFTER a successful benchmark
echo "[golden] Marking current build as golden..."
date > GOLDEN_BUILD.md
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" >> GOLDEN_BUILD.md
echo "Files protected from cleanup:" >> GOLDEN_BUILD.md

# Protect these files (the ones that take 1+ hour to rebuild)
for f in data/itq_model_512.npz data/paper_metadata.jsonl data/hnsw_*.index mesh_itq_512.bin; do
    if [ -f "$f" ]; then
        ls -lh "$f" >> GOLDEN_BUILD.md
        echo "  PROTECTED: $f"
    fi
done

echo "[golden] Done. See GOLDEN_BUILD.md"
cat GOLDEN_BUILD.md
