#!/bin/bash
# Rebuild shootout HNSW files if they are missing or not in BinaryHNSW YPH5v4 format.

set -euo pipefail

cd ~/yellow_phoenix
export YP_GENESIS_PHRASE="YP_DEV_BUILD_2026_08_19"
export PATH="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$HOME/.cargo/bin:$PATH"

BUILDER="./target/release/build_hnsw_from_ism"
if [ ! -x "$BUILDER" ]; then
    echo "Building build_hnsw_from_ism binary..."
    cargo build --release --bin build_hnsw_from_ism --features flat-array
fi

needs_rebuild() {
    local f="$1"
    if [ ! -f "$f" ]; then
        return 0
    fi
    # Read magic + version: expect "YPH5" (0x59504835) followed by 0x04.
    local header
    header=$(xxd -l 5 -p "$f" 2>/dev/null || true)
    if [ "$header" != "5950483504" ]; then
        return 0
    fi
    return 1
}

build() {
    local ism="$1"
    local out="$2"
    local m="${3:-16}"
    local ef="${4:-200}"
    if needs_rebuild "$out"; then
        echo "=== Building $out (M=$m, ef=$ef) ==="
        HNSW_M="$m" HNSW_EF_CONSTRUCTION="$ef" "$BUILDER" "$ism" "$out"
    else
        echo "OK (YPH5v4): $out"
    fi
}

# Main shootout sizes. 5M/10M/20M already use M=16; keep that for consistency.
build data/synth_106k.ism  data/synth_106k_hnsw.bin  16 200
build data/synth_1m.ism    data/synth_1m_hnsw.bin    16 200
build data/synth_5m.ism    data/synth_5m_hnsw.bin    16 200
build data/synth_10m.ism   data/synth_10m_hnsw.bin   16 200
build data/synth_20m.ism   data/synth_20m_hnsw.bin   16 200

echo ""
echo "Done. Files:"
ls -lh data/synth_*_hnsw.bin
