#!/bin/bash
set -e
cd ~/yellow_phoenix
export YP_GENESIS_PHRASE="YP_DEV_BUILD_2026_08_19"
export PATH="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$HOME/.cargo/bin:$PATH"

BUILDER="./target/release/build_hnsw_from_ism"
if [ ! -x "$BUILDER" ]; then
    echo "Building build_hnsw_from_ism binary..."
    cargo build --release --bin build_hnsw_from_ism --features flat-array
fi

echo "Building HNSW variants for iPhone memory testing..."
echo ""

build_variant() {
    local m="$1"
    local ef="$2"
    local out="$3"
    if [ ! -f "$out" ]; then
        echo "=== M=$m (ef_construction=$ef) → $out ==="
        HNSW_M="$m" HNSW_EF_CONSTRUCTION="$ef" "$BUILDER" data/synth_10m.ism "$out"
    else
        echo "Already exists: $out"
    fi
    echo ""
}

# Full (m, ef_construction) matrix used by the 10M variant batch.
build_variant 16 200 data/synth_10m_hnsw.bin
build_variant 12 50  data/synth_10m_hnsw_m12.bin
build_variant 8  50  data/synth_10m_hnsw_m8.bin
build_variant 6  50  data/synth_10m_hnsw_m6.bin
build_variant 4  50  data/synth_10m_hnsw_m4.bin

echo "Done. Files:"
ls -lh data/synth_10m_hnsw*.bin
